#!/usr/bin/env python3
"""Materialize and validate deterministic action-branch canaries.

The branch inventory produced by the profile review is a design input.  This
tool is the deliberately separate execution boundary: it replays an authored
scenario with a concrete intervention, overlays the resulting kinematics on
the immutable render-ready background, and asks the existing objective truth
pipeline to derive predicates, events, and graph artifacts.

Only actions that can be resolved from an authored action bundle and typed
arguments are executable here.  Unknown generators fail closed; they are not
silently replaced by a template trajectory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "Dataset" / "tools"
SUMO_SCRIPTS_ROOT = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scripts"
for _path in (PROJECT_ROOT, TOOLS_ROOT, SUMO_SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from Dataset.semantic_truth.objective_pipeline import (  # noqa: E402
    ObjectiveEpisodeArtifacts,
    ObjectivePipelineError,
    build_episode_objective_artifacts,
    write_episode_outputs,
)
from Dataset.semantic_truth.provenance import canonical_json, digest_object  # noqa: E402
from batch_generate import (  # noqa: E402
    EpisodeStateEngine,
    PRESERVED_ENTITY_FIELDS,
    RUNTIME_STATE_FIELDS,
    build_event_realizations,
    enrich_event_log_validation_fields,
    preserved_fields_from,
)
from donghu_core.event_script_interpreter import EventScriptInterpreter  # noqa: E402
from branch_continuation import (  # noqa: E402
    CausalContinuationError,
    causal_suppression_closure,
    suppress_event_triggers,
)


DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT
    / "artifacts/branches/aero_branch_candidate_review_20260902"
    / "generated"
    / "final_branches"
)
DEFAULT_CAPTURE_ROOT = (
    PROJECT_ROOT / "aw_data" / "render_ready_episodes_capture_filtered"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/branches/aero210_action_branch_rollouts_20260903"
FORMAL_DURATION_TICKS = 900
BRANCH_SCHEMA_NAME = "aeroworld_deterministic_action_branch"
BRANCH_SCHEMA_VERSION = "1.0.0"
TOOL_VERSION = "action_branch_rollout_v1"


class BranchRolloutError(RuntimeError):
    """A branch cannot be materialized without inventing authority."""


class UnsupportedBranch(BranchRolloutError):
    """The design does not resolve to a registered deterministic action."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchRolloutError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BranchRolloutError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BranchRolloutError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise BranchRolloutError(f"cannot read JSONL: {path}: {exc}") from exc
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(dict(value)) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(dict(row)) + "\n")


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise BranchRolloutError(f"required regular file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value))
    return safe.strip("_") or "arm"


def _scenario_script(scenario_id: str) -> Path:
    matches: list[Path] = []
    for path in sorted((PROJECT_ROOT / "Dataset" / "scenarios").rglob("event_script.json")):
        try:
            value = _read_json(path)
        except BranchRolloutError:
            continue
        if str(value.get("scenario_id") or path.parent.name) == scenario_id:
            matches.append(path)
    if len(matches) != 1:
        raise BranchRolloutError(
            f"scenario {scenario_id!r} must resolve to exactly one authored event_script.json; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _profile_design(candidate_root: Path, profile_id: str) -> dict[str, Any]:
    path = candidate_root / profile_id / "normalized_design.json"
    if not path.is_file():
        raise BranchRolloutError(f"normalized candidate design is missing: {path}")
    design = _read_json(path)
    if design.get("profile_id") and design.get("profile_id") != profile_id:
        raise BranchRolloutError(f"design/profile mismatch: {path}")
    return design


def _profile_arm(candidate_root: Path, profile_id: str, arm_id: str) -> dict[str, Any]:
    arm_dir = candidate_root / profile_id / "arms"
    for path in sorted(arm_dir.glob("*.json")):
        arm_record = _read_json(path)
        arm = arm_record.get("decision_arm")
        if isinstance(arm, dict) and str(arm.get("arm_id")) == arm_id:
            return arm_record
    raise BranchRolloutError(
        f"arm {arm_id!r} is not present in candidate profile {profile_id!r}"
    )


def _seed_anchor(design: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    anchor = design.get("anchor")
    if not isinstance(anchor, Mapping):
        raise BranchRolloutError("candidate design lacks anchor")
    for item in anchor.get("seed_anchors") or []:
        if isinstance(item, Mapping) and str(item.get("episode_id")) == episode_id:
            result = dict(item)
            if not isinstance(result.get("dispatch_tick"), int):
                raise BranchRolloutError(f"anchor dispatch_tick is not an integer: {result}")
            return result
    raise BranchRolloutError(f"design has no seed anchor for {episode_id}")


def _seed_role_bindings(design: Mapping[str, Any], seed_id: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for binding in design.get("role_bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        role = str(binding.get("role") or "")
        by_seed = binding.get("entity_ids_by_seed")
        if role and isinstance(by_seed, Mapping):
            ids = by_seed.get(seed_id)
            if isinstance(ids, list) and all(isinstance(item, str) and item for item in ids):
                result[role] = list(ids)
    return result


def _authored_events(script: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("event_id")): dict(item)
        for item in script.get("events") or []
        if isinstance(item, Mapping) and item.get("event_id")
    }


def _authored_triggers(script: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("trigger_id")): dict(item)
        for item in script.get("triggers") or []
        if isinstance(item, Mapping) and item.get("trigger_id")
    }


def _continuation_policy(design: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    ref = str(arm.get("continuation_policy_ref") or "")
    for policy in design.get("continuation_policies") or []:
        if isinstance(policy, Mapping) and str(policy.get("policy_id")) == ref:
            return dict(policy)
    raise BranchRolloutError(f"continuation policy is missing: {ref}")


def _suppressed_event_ids(
    design: Mapping[str, Any], arm: Mapping[str, Any], script: Mapping[str, Any]
) -> list[str]:
    events = _authored_events(script)
    source_ids = {
        str(step.get("source_event_id"))
        for step in arm.get("mechanism_steps") or []
        if isinstance(step, Mapping)
        and step.get("step_kind") == "suppress_source_event"
        and step.get("source_event_id")
    }
    if not source_ids:
        raise UnsupportedBranch("suppression arm has no source_event_id")
    policy = _continuation_policy(design, arm)
    affected: set[str] = set(source_ids)
    for rule in policy.get("channel_rules") or []:
        if not isinstance(rule, Mapping):
            continue
        if str(rule.get("policy") or "") != "cancel_if_predecessor_absent":
            continue
        for event_id in rule.get("affected_event_ids") or []:
            if event_id:
                affected.add(str(event_id))
    unknown = sorted(affected - set(events))
    if unknown:
        raise UnsupportedBranch(f"suppression names non-authored events: {unknown}")
    # The policy's explicit channel list is retained as provenance, but it is
    # not sufficient to prevent an authored downstream event from recreating
    # the suppressed state.  Resolve the event graph from the immutable source
    # script and take the union for every declared source event.  A malformed
    # graph is a hard branch rejection; silently falling back to the policy
    # list would make the resulting Arm non-causal.
    try:
        for source_id in source_ids:
            affected.update(
                causal_suppression_closure(
                    script,
                    source_id,
                )
            )
    except CausalContinuationError as exc:
        raise UnsupportedBranch(str(exc)) from exc
    unknown = sorted(affected - set(events))
    if unknown:
        raise UnsupportedBranch(f"suppression names non-authored events: {unknown}")
    return sorted(affected)


def _typed_arguments(step: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in step.get("arguments") or []:
        if not isinstance(item, Mapping) or not item.get("name"):
            raise UnsupportedBranch("generator argument is malformed")
        if "value" not in item:
            raise UnsupportedBranch(f"generator argument has no value: {item.get('name')}")
        values[str(item["name"])] = copy.deepcopy(item["value"])
    return values


def _role_entity_ids(
    step: Mapping[str, Any], role_bindings: Mapping[str, list[str]], fallback: str
) -> list[str]:
    role_args = step.get("role_arguments") or []
    for item in role_args:
        if not isinstance(item, Mapping):
            continue
        parameter = str(item.get("parameter") or "")
        role = str(item.get("role") or "")
        ids = role_bindings.get(parameter) or role_bindings.get(role)
        if ids:
            return list(ids)
    return [fallback] if fallback else []


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _apply_move_arguments(action: dict[str, Any], arguments: Mapping[str, Any]) -> None:
    """Apply only typed, unambiguous movement arguments to an authored action."""

    velocity_keys = (
        "velocity_mps",
        "ascent_velocity_mps",
        "transit_velocity_mps",
        "rth_velocity_mps",
        "descent_velocity_mps",
        "correction_speed_mps",
    )
    for key in velocity_keys:
        if key in arguments:
            try:
                velocity = float(arguments[key])
            except (TypeError, ValueError) as exc:
                raise UnsupportedBranch(f"invalid movement velocity {key}") from exc
            if not math.isfinite(velocity) or velocity <= 0.0:
                raise UnsupportedBranch(f"movement velocity must be positive: {key}")
            action["velocity_mps"] = velocity
            break

    altitude_keys = (
        "target_altitude_m",
        "primary_target_altitude_m",
        "bypass_altitude_m",
        "climb_altitude_m",
        "target_height_m",
    )
    altitude: float | None = None
    for key in altitude_keys:
        if key in arguments:
            try:
                altitude = float(arguments[key])
            except (TypeError, ValueError) as exc:
                raise UnsupportedBranch(f"invalid altitude argument {key}") from exc
            break
    if altitude is not None:
        if not math.isfinite(altitude) or altitude <= 0.0:
            raise UnsupportedBranch("target altitude must be positive")
        points = action.get("waypoints_enu_m")
        if not isinstance(points, list) or not points:
            raise UnsupportedBranch("altitude intervention lacks authored waypoints")
        rewritten: list[list[float]] = []
        for index, point in enumerate(points):
            vector = _vector3(point)
            if vector is None:
                raise UnsupportedBranch("authored waypoint is not numeric")
            if index == 0 and vector[2] <= 5.0:
                rewritten.append(vector)
            else:
                vector[2] = altitude
                rewritten.append(vector)
        action["waypoints_enu_m"] = rewritten

    waypoint_values: list[tuple[str, list[float]]] = []
    for key, value in arguments.items():
        vector = _vector3(value)
        if vector is not None and ("waypoint" in key or key.endswith("_enu_m")):
            waypoint_values.append((key, vector))
    if waypoint_values:
        waypoint_values.sort(key=lambda pair: pair[0])
        action["waypoints_enu_m"] = [vector for _, vector in waypoint_values]

    hold_position = None
    for key in ("hold_pose_enu_m", "hold_position_enu_m", "position_enu_m"):
        if key in arguments:
            hold_position = _vector3(arguments[key])
            if hold_position is None:
                raise UnsupportedBranch(f"invalid hold position: {key}")
            break
    if hold_position is not None:
        action["waypoints_enu_m"] = [hold_position, hold_position]
        action["velocity_mps"] = max(0.1, float(action.get("velocity_mps") or 0.1))


def _build_replacement_actions(
    design: Mapping[str, Any],
    arm: Mapping[str, Any],
    script: Mapping[str, Any],
    role_bindings: Mapping[str, list[str]],
    anchor_event_id: str,
) -> tuple[list[dict[str, Any]], int | None, list[str]]:
    events = _authored_events(script)
    anchor_event = events.get(anchor_event_id)
    if anchor_event is None:
        raise BranchRolloutError(f"anchor event is not authored: {anchor_event_id}")
    steps = [
        step
        for step in arm.get("mechanism_steps") or []
        if isinstance(step, Mapping) and step.get("step_kind") == "invoke_action_generator"
    ]
    if len(steps) != 1:
        raise UnsupportedBranch(
            "only one typed invoke_action_generator step is executable in the canary"
        )
    step = steps[0]
    action_types = {str(value) for value in step.get("action_types") or []}
    if "set_runtime_state" in action_types and "move_entity" not in action_types:
        raise UnsupportedBranch(
            "runtime-only generator has no authored state patch mapping"
        )
    if "move_entity" not in action_types:
        raise UnsupportedBranch(
            f"generator action types are not kinematic: {sorted(action_types)}"
        )
    authored_moves = [
        dict(action)
        for action in anchor_event.get("actions") or []
        if isinstance(action, Mapping) and action.get("type") == "move_entity"
    ]
    if not authored_moves:
        raise UnsupportedBranch("anchor event has no authored move_entity action")
    arguments = _typed_arguments(step)
    fallback_entity = str(authored_moves[0].get("entity_id") or "")
    target_ids = _role_entity_ids(step, role_bindings, fallback_entity)
    if not target_ids:
        raise UnsupportedBranch("generator has no resolvable target entity")
    generated: list[dict[str, Any]] = []
    for index, entity_id in enumerate(target_ids):
        source = next(
            (action for action in authored_moves if str(action.get("entity_id")) == entity_id),
            authored_moves[min(index, len(authored_moves) - 1)],
        )
        action = copy.deepcopy(source)
        action["entity_id"] = entity_id
        action["action_id"] = f"branch_generated_{_slug(str(arm.get('arm_id')))}_{index}"
        _apply_move_arguments(action, arguments)
        generated.append(action)
    dispatch_tick: int | None = None
    for key in ("start_tick", "initial_dispatch_tick", "delayed_dispatch_tick"):
        if key in arguments:
            try:
                dispatch_tick = int(arguments[key])
            except (TypeError, ValueError) as exc:
                raise UnsupportedBranch(f"invalid dispatch tick: {key}") from exc
            if dispatch_tick < 0 or dispatch_tick > FORMAL_DURATION_TICKS:
                raise UnsupportedBranch(f"dispatch tick outside formal range: {dispatch_tick}")
            break
    return generated, dispatch_tick, [str(step.get("generator_id") or "")]


def build_branch_script(
    design: Mapping[str, Any],
    arm_record: Mapping[str, Any],
    source_script: Mapping[str, Any],
    *,
    seed_id: str,
    anchor_event_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a derived script plus an explicit branch ledger."""

    script = copy.deepcopy(dict(source_script))
    arm = arm_record.get("decision_arm")
    if not isinstance(arm, Mapping):
        raise BranchRolloutError("arm record lacks decision_arm")
    arm_kind = str(arm.get("arm_kind") or "")
    operation = str(arm.get("event_operation") or "")
    events = _authored_events(script)
    triggers = _authored_triggers(script)
    if anchor_event_id not in events:
        raise BranchRolloutError(f"anchor event is absent from source script: {anchor_event_id}")

    ledger: dict[str, Any] = {
        "schema_name": "branch_event_ledger",
        "schema_version": "1.0.0",
        "seed_id": seed_id,
        "anchor_event_id": anchor_event_id,
        "fork_tick": int(
            next(
                item["dispatch_tick"]
                for item in (design.get("anchor", {}).get("seed_anchors") or [])
                if isinstance(item, Mapping) and item.get("seed_id") == seed_id
            )
        ),
        "operation": operation,
        "arm_kind": arm_kind,
        "status": "planned",
        "suppressed_event_ids": [],
        "generated_action_ids": [],
        "generator_ids": [],
    }

    if arm_kind == "factual" and operation == "reuse_factual_causal_event":
        ledger["status"] = "applied"
        ledger["factual_preserved"] = True
        return script, ledger

    role_bindings = _seed_role_bindings(design, seed_id)
    if operation == "suppress_causal_event":
        suppressed = _suppressed_event_ids(design, arm, script)
        try:
            originals = suppress_event_triggers(
                script,
                suppressed,
                never_tick=FORMAL_DURATION_TICKS + 1,
            )
        except CausalContinuationError as exc:
            raise UnsupportedBranch(str(exc)) from exc
        ledger["suppressed_event_ids"] = suppressed
        ledger["suppressed_event_original_triggers"] = originals
        ledger["suppression_basis"] = "authored_trigger_and_causal_predecessor_closure"
        ledger["suppression_scope"] = "whole_event_bundle_and_branch_ledger"
        ledger["status"] = "applied"
        return script, ledger

    if operation not in {"replace_causal_event", "add_decision_event"}:
        raise UnsupportedBranch(f"unsupported branch operation: {operation}")
    generated, dispatch_tick, generator_ids = _build_replacement_actions(
        design, arm, script, role_bindings, anchor_event_id
    )
    anchor = next(
        item
        for item in script.get("events") or []
        if isinstance(item, Mapping) and str(item.get("event_id")) == anchor_event_id
    )
    anchor["actions"] = generated
    if dispatch_tick is not None:
        trigger_id = str(anchor.get("trigger_ref") or "")
        trigger = triggers.get(trigger_id)
        if not isinstance(trigger, Mapping) or trigger.get("type") != "tick":
            raise UnsupportedBranch(
                "typed delayed action can only retime an authored tick trigger"
            )
        for item in script.get("triggers") or []:
            if isinstance(item, Mapping) and str(item.get("trigger_id")) == trigger_id:
                item["tick"] = dispatch_tick
                break
    ledger["generated_action_ids"] = [str(action.get("action_id")) for action in generated]
    ledger["generator_ids"] = generator_ids
    ledger["status"] = "applied"
    return script, ledger


def _run_engine(
    script_path: Path,
    scene: Mapping[str, Any],
    episode_id: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    script = _read_json(script_path)
    engine = EpisodeStateEngine(dict(scene), script, script_path, FORMAL_DURATION_TICKS)
    interpreter = EventScriptInterpreter(script_path, episode_id=episode_id)
    current_tick = [0]

    handlers = {
        "move_entity": engine._handle_move_entity,
        "set_visual_state": engine._handle_set_visual_state,
        "set_pedestrian_activity": engine._handle_set_pedestrian_activity,
        "set_weather": engine._handle_set_weather,
        "set_runtime_state": engine._handle_set_runtime_state,
        "spawn_entity": engine._handle_spawn_entity,
        "capture_screenshot": engine._handle_capture_screenshot,
        "remove_entity": engine._handle_remove_entity,
        "sequence": engine._handle_noop,
        "play_animation": engine._handle_noop,
        "spawn_crowd": engine._handle_noop,
    }
    for action_type, handler in handlers.items():
        interpreter.register_handler(
            action_type,
            lambda action, handler=handler: handler(action, current_tick[0]),
        )

    for tick in range(FORMAL_DURATION_TICKS + 1):
        current_tick[0] = tick
        engine._apply_pending_runtime_state_patches(tick)
        for payload in engine.weather_transitions.get(tick, []):
            engine.weather = payload
        rows = engine._record_tick_rows(tick)
        for row in rows:
            interpreter.update_entity_state(
                row["entity_id"], row["pos_enu"], {}, row["vel_mps"]
            )
            if str(row.get("label_class") or "") == "pedestrian":
                interpreter.update_entity_activity(
                    row["entity_id"], str(row.get("activity_type") or "")
                )
        interpreter.update_weather_state(engine.weather_rows[-1])
        for entry in interpreter.tick(tick):
            result = dict(entry.get("result") or {})
            if result.get("status") == "error":
                raise BranchRolloutError(
                    f"{episode_id}: action failed at tick {tick}: {entry}"
                )

    event_log = interpreter.get_event_log()
    enrich_event_log_validation_fields(event_log, script, str(script.get("scenario_id") or ""))
    event_realizations = build_event_realizations(
        scenario_id=str(script.get("scenario_id") or ""),
        script=script,
        event_log=event_log,
        executed_actions=engine.executed_actions,
        trajectory_rows=engine.trajectory_rows,
        params=engine.params,
    )
    roster: list[dict[str, Any]] = []
    for entity_id, entity in sorted(engine.entities.items()):
        item: dict[str, Any] = {
            "entity_id": entity_id,
            "label_class": entity["label_class"],
            "asset_id": entity["asset_id"],
            "initial_yaw_deg": round(float(entity.get("initial_yaw_deg") or 0.0), 6),
        }
        item.update(preserved_fields_from(entity))
        if str(entity.get("spawn_policy") or "") == "event_script_only":
            item["activation_tick"] = int(entity["active_from"])
        if entity.get("inactive_from") is not None:
            item["deactivation_tick"] = int(entity["inactive_from"])
        roster.append(item)
    return engine.trajectory_rows, engine.weather_rows, event_log, event_realizations, roster


def _copy_if_file(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _overlay_branch_input(
    *,
    canonical_capture_episode: Path,
    source_scene_setup: Path,
    episode_id: str,
    branch_trajectories: Sequence[Mapping[str, Any]],
    branch_event_trace: Sequence[Mapping[str, Any]],
    branch_event_realizations: Sequence[Mapping[str, Any]],
    branch_ledger: Mapping[str, Any],
) -> tempfile.TemporaryDirectory[str]:
    """Create a strict objective input overlay in a disposable directory."""

    temp = tempfile.TemporaryDirectory(prefix="aeroworld_branch_input_")
    root = Path(temp.name) / episode_id
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "episode_manifest.json",
        "global_entity_roster.json",
        "weather_meta.jsonl",
        "scene_occupancy_manifest.json",
        "scenario_package.json",
        "render_host_config.json",
    ):
        _copy_if_file(canonical_capture_episode / name, root / name)
    _copy_if_file(source_scene_setup, root / "scene_setup.json")

    # The canonical trajectory stream is intentionally compact and carries
    # only per-tick kinematics plus lightweight labels.  Static contract and
    # runtime fields live in the immutable global roster/truth frame.  Use
    # that authoritative roster when a branch producer emits a richer row;
    # never invent a missing field or silently drop an authored intervention.
    parent_roster_by_entity: dict[str, dict[str, Any]] = {}
    parent_roster_path = canonical_capture_episode / "global_entity_roster.json"
    if parent_roster_path.is_file():
        roster_value = _read_json(parent_roster_path)
        for item in roster_value.get("entities") or [] if isinstance(roster_value, Mapping) else []:
            if isinstance(item, Mapping) and item.get("entity_id"):
                parent_roster_by_entity[str(item["entity_id"])] = dict(item)

    branch_by_key = {
        (int(row["tick"]), str(row["entity_id"])): dict(row)
        for row in branch_trajectories
        if row.get("tick") is not None and row.get("entity_id")
    }
    with (canonical_capture_episode / "trajectories.jsonl").open(
        encoding="utf-8-sig"
    ) as source, (root / "trajectories.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            row = json.loads(line)
            replacement = branch_by_key.get((int(row["tick"]), str(row["entity_id"])))
            if replacement is not None:
                for key in (
                    "pos_enu",
                    "vel_mps",
                    "yaw_deg",
                    "state",
                    "activity_type",
                    "animation_hint",
                    "posture",
                    "social_state",
                    "control_state",
                ):
                    if key in replacement:
                        row[key] = copy.deepcopy(replacement[key])
                for key in PRESERVED_ENTITY_FIELDS:
                    if key in replacement:
                        if key not in row:
                            roster_row = parent_roster_by_entity.get(str(row.get("entity_id") or ""), {})
                            if key in roster_row:
                                row[key] = copy.deepcopy(roster_row[key])
                            elif key in RUNTIME_STATE_FIELDS:
                                raise BranchRolloutError(
                                    f"parent sources lack runtime field {key!r} for {row.get('entity_id')} at tick {row.get('tick')}"
                                )
                            else:
                                # Static provenance fields such as task_id
                                # may exist only on the branch producer's
                                # entity object.  They are not truth inputs;
                                # leave the compact parent trajectory shape
                                # unchanged when neither immutable source
                                # carries them.
                                continue
                        row[key] = copy.deepcopy(replacement[key])
            target.write(canonical_json(row) + "\n")

    with (canonical_capture_episode / "truth_frames.jsonl").open(
        encoding="utf-8-sig"
    ) as source, (root / "truth_frames.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            frame = json.loads(line)
            tick = int(frame["tick"])
            for entity in frame.get("entities") or []:
                if not isinstance(entity, dict):
                    continue
                replacement = branch_by_key.get((tick, str(entity.get("entity_id"))))
                if replacement is None:
                    continue
                position = _vector3(replacement.get("pos_enu"))
                velocity = _vector3(replacement.get("vel_mps"))
                if position is None or velocity is None:
                    raise BranchRolloutError(
                        f"branch trajectory lacks numeric pose/velocity at {tick}: {replacement}"
                    )
                pose = entity.get("truth_pose")
                if not isinstance(pose, dict):
                    raise BranchRolloutError(
                        f"parent truth frame lacks truth_pose for {entity.get('entity_id')} at tick {tick}"
                    )
                pose["position_enu_m"] = position
                pose["velocity_enu_mps"] = velocity
                rotation = pose.get("rotation_deg")
                if not isinstance(rotation, dict):
                    raise BranchRolloutError(
                        f"parent truth frame lacks rotation mapping for {entity.get('entity_id')} at tick {tick}"
                    )
                if "yaw_deg" in replacement:
                    rotation["yaw_deg"] = float(replacement["yaw_deg"])
                if "state" not in entity:
                    raise BranchRolloutError(
                        f"parent truth frame lacks state for {entity.get('entity_id')} at tick {tick}"
                    )
                state = str(replacement["state"]) if "state" in replacement else str(entity["state"])
                annotations = entity.get("annotations")
                if not isinstance(annotations, dict) or "activity_type" not in annotations:
                    raise BranchRolloutError(
                        f"parent annotations/activity missing for {entity.get('entity_id')} at tick {tick}"
                    )
                activity = str(replacement["activity_type"]) if "activity_type" in replacement else str(annotations["activity_type"])
                entity["state"] = state
                annotations["activity_type"] = activity
                annotations["speed_mps"] = math.sqrt(sum(value * value for value in velocity))
                state_facets = annotations.get("state_facets")
                activity_facets = state_facets.get("activity") if isinstance(state_facets, dict) else None
                if not isinstance(activity_facets, dict):
                    raise BranchRolloutError(
                        f"parent annotations/state_facets.activity missing for {entity.get('entity_id')} at tick {tick}"
                    )
                activity_facets["activity_type"] = activity
                for key in ("animation_hint", "posture", "social_state"):
                    if key in replacement:
                        activity_facets[key] = replacement[key]
                for key in ("control_state", "uav_corridor", "assigned_altitude_m"):
                    if key in replacement:
                        entity[key] = copy.deepcopy(replacement[key])
                # Runtime-state fields are semantic input, not display-only
                # metadata.  The engine persists them in each branch
                # trajectory row (notably ``communication_state``), so the
                # truth-frame overlay must carry the complete preserved field
                # set before the objective pipeline recomputes predicates.
                # Omitting this made a valid branch intervention disappear at
                # the objective boundary while leaving the trajectory altered.
                for key in PRESERVED_ENTITY_FIELDS:
                    if key in replacement:
                        if key not in entity:
                            if key in RUNTIME_STATE_FIELDS:
                                raise BranchRolloutError(
                                    f"parent truth frame lacks runtime field {key!r} for {entity.get('entity_id')} at tick {tick}"
                                )
                            # Static provenance is optional in the compact
                            # truth-frame schema.  It is not a semantic
                            # runtime input, so absence is explicit rather
                            # than filled from a guessed default.
                            continue
                        entity[key] = copy.deepcopy(replacement[key])
                motion_state = frame.get("entity_motion_state")
                if isinstance(motion_state, dict) and str(entity["entity_id"]) in motion_state:
                    motion_state[str(entity["entity_id"])] = state
            target.write(canonical_json(frame) + "\n")

    _write_jsonl(root / "event_trace.jsonl", branch_event_trace)
    _write_jsonl(root / "event_realization.jsonl", branch_event_realizations)
    _write_jsonl(root / "branch_event_ledger.jsonl", [branch_ledger])
    return temp


def _target_binding_ids(
    design: Mapping[str, Any], episode_id: str, *, keys: Iterable[str]
) -> set[str]:
    seed_id = episode_id.rsplit("__seed", 1)[-1]
    seed_id = "seed" + seed_id
    roles = _seed_role_bindings(design, seed_id)
    wanted: set[str] = set()
    for key in keys:
        wanted.update(roles.get(key, []))
    return wanted


def _binding_role_candidates(
    key: str, role_bindings: Mapping[str, list[str]]
) -> set[str]:
    """Resolve an objective binding key to the design's role identities."""

    normalized = str(key).strip().casefold()
    aliases: list[str] = [normalized]
    if normalized in {"aircraft", "actor", "subject_entity", "target_aircraft"}:
        aliases.extend(
            role
            for role in role_bindings
            if "aircraft" in role.casefold() or role.casefold() == "mission_uav"
        )
    elif normalized in {"first_aircraft", "second_aircraft"}:
        aliases.append(normalized)
        aliases.append("mission_uav")
    elif normalized in {"home_pad", "landing_pad", "pad"}:
        aliases.extend(role for role in role_bindings if "pad" in role.casefold())
    elif normalized in {"station", "facility", "target_facility"}:
        aliases.extend(
            role
            for role in role_bindings
            if any(token in role.casefold() for token in ("station", "facility", "charger"))
        )
    result: set[str] = set()
    for alias in aliases:
        result.update(role_bindings.get(alias, []))
    return result


def _row_matches_roles(
    row: Mapping[str, Any], role_bindings: Mapping[str, list[str]]
) -> bool:
    bindings = row.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        return False
    for key, value in bindings.items():
        if not isinstance(value, str):
            return False
        candidates = _binding_role_candidates(str(key), role_bindings)
        if candidates and value not in candidates:
            return False
    return True


def _load_objective_rows(objective_root: Path, name: str) -> list[dict[str, Any]]:
    path = objective_root / name
    if not path.is_file():
        raise BranchRolloutError(f"objective artifact is missing: {path}")
    return _read_jsonl(path)


def _target_summary(
    *,
    objective_root: Path,
    design: Mapping[str, Any],
    episode_id: str,
    fork_tick: int,
    end_tick: int,
) -> dict[str, Any]:
    goal = design.get("semantic_goal") or {}
    predicate = dict(goal.get("primary_predicate") or {})
    predicate_id = str(predicate.get("predicate_id") or "")
    objective_event = goal.get("objective_event")
    event_family = (
        str(objective_event.get("event_family_id") or "")
        if isinstance(objective_event, Mapping)
        else ""
    )
    event_type = (
        str(objective_event.get("event_type_id") or "")
        if isinstance(objective_event, Mapping)
        else ""
    )
    # The objective contract names role keys in the bindings.  Resolve each
    # row's role through the design binding map; this avoids confusing the
    # mission UAV with the observer or unrelated global-flow aircraft.
    seed_id = "seed" + episode_id.rsplit("__seed", 1)[-1]
    role_bindings = _seed_role_bindings(design, seed_id)
    predicate_rows = [
        row
        for row in _load_objective_rows(objective_root, "predicate_truth.jsonl")
        if str(row.get("predicate_id") or "") == predicate_id
        and fork_tick <= int(row.get("tick", -1)) <= end_tick
        and _row_matches_roles(row, role_bindings)
    ]
    true_rows = [row for row in predicate_rows if row.get("value") == "true"]
    transition_rows = [
        row
        for row in _load_objective_rows(objective_root, "predicate_transitions.jsonl")
        if str(row.get("predicate_id") or "") == predicate_id
        and fork_tick <= int(row.get("to_tick", row.get("tick", -1))) <= end_tick
        and _row_matches_roles(row, role_bindings)
    ]
    # A predicate-only goal has no objective event.  Do not fall back to an
    # unrelated landing/altitude occurrence merely because it shares one of
    # the bound entities; doing so makes an alternative look semantically
    # different for the wrong reason.  Event rows are selected only when the
    # design names a registered objective event family and type.
    event_rows = []
    if event_family or event_type:
        event_rows = [
            row
            for row in _load_objective_rows(objective_root, "event_occurrences.jsonl")
            if (not event_family or str(row.get("event_family_id") or "") == event_family)
            and (not event_type or str(row.get("event_type_id") or "") == event_type)
            and fork_tick
            <= int(row.get("trigger_tick", row.get("start_tick", -1)))
            <= end_tick
            and _row_matches_roles(row, role_bindings)
        ]
    return {
        "predicate_id": predicate_id,
        "objective_event_family_id": event_family,
        "objective_event_type_id": event_type,
        "bound_entity_ids": sorted(
            {
                value
                for row in predicate_rows + event_rows
                for value in (row.get("bindings") or {}).values()
                if isinstance(value, str)
            }
        ),
        "predicate_true_count": len(true_rows),
        "predicate_true_ticks": sorted({int(row["tick"]) for row in true_rows}),
        "predicate_transition_count": len(transition_rows),
        "predicate_transition_directions": [
            str(row.get("direction") or "") for row in transition_rows
        ],
        "predicate_transitions": [
            {
                "direction": row.get("direction"),
                "from_tick": row.get("from_tick"),
                "to_tick": row.get("to_tick"),
                "transition_id": row.get("transition_id"),
            }
            for row in transition_rows
        ],
        "event_occurrence_count": len(event_rows),
        "event_trigger_ticks": sorted(
            {int(row.get("trigger_tick", row.get("start_tick", -1))) for row in event_rows}
        ),
        "event_ids": sorted(str(row.get("event_id") or "") for row in event_rows),
    }


def _plot_branch(
    *,
    trajectory_path: Path,
    output_path: Path,
    entity_ids: Sequence[str],
    fork_tick: int,
    end_tick: int,
    title: str,
    predicate_id: str = "",
    separation_model: Mapping[str, Any] | None = None,
    separation_margin_m: float | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment gate
        raise BranchRolloutError("matplotlib is required for branch-specific plots") from exc

    wanted = set(entity_ids)
    rows = [
        row
        for row in _read_jsonl(trajectory_path)
        if str(row.get("entity_id")) in wanted
        and fork_tick <= int(row.get("tick", -1)) <= end_tick
    ]
    if not rows:
        raise BranchRolloutError("cannot render a branch plot without selected trajectory rows")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["entity_id"]), []).append(row)
    pair_plot = (
        predicate_id == "safety.aircraft_pair_below_separation"
        and len(grouped) == 2
        and isinstance(separation_model, Mapping)
    )
    figure, axes = plt.subplots(
        1,
        3 if pair_plot else 2,
        figsize=(15 if pair_plot else 11, 4.5),
        constrained_layout=True,
    )
    axes = list(axes) if isinstance(axes, (list, tuple)) else list(axes.flat)
    for entity_id, entity_rows in sorted(grouped.items()):
        entity_rows.sort(key=lambda row: int(row["tick"]))
        axes[0].plot(
            [float(row["pos_enu"][1]) for row in entity_rows],
            [float(row["pos_enu"][0]) for row in entity_rows],
            label=entity_id,
        )
        axes[-1].plot(
            [int(row["tick"]) for row in entity_rows],
            [float(row["pos_enu"][2]) for row in entity_rows],
            label=entity_id,
        )
    axes[0].set_xlabel("ENU Y (m)")
    axes[0].set_ylabel("ENU X (m)")
    axes[0].set_title("Branch trajectory")
    if pair_plot:
        ordered_ids = sorted(grouped)
        # Build an explicit pair map so every distance is derived from the two
        # measured poses, never from a template value.
        pair_rows: dict[int, dict[str, dict[str, Any]]] = {}
        for entity_id in ordered_ids:
            for row in grouped[entity_id]:
                pair_rows.setdefault(int(row["tick"]), {})[entity_id] = row
        separation_metric = str(separation_model.get("metric") or "")
        horizontal_limit = (
            float(separation_model["horizontal_limit_m"])
            if separation_metric != "euclidean"
            else None
        )
        vertical_limit = (
            float(separation_model["vertical_limit_m"])
            if separation_metric != "euclidean"
            else None
        )
        margin = float(separation_margin_m if separation_margin_m is not None else separation_model.get("separation_margin_m", 15.0))
        distance_ticks: list[int] = []
        equivalent: list[float] = []
        horizontal: list[float] = []
        vertical: list[float] = []
        for tick, rows_by_id in sorted(pair_rows.items()):
            if set(rows_by_id) != set(ordered_ids):
                continue
            first = _vector3(rows_by_id[ordered_ids[0]].get("pos_enu"))
            second = _vector3(rows_by_id[ordered_ids[1]].get("pos_enu"))
            if first is None or second is None:
                continue
            h = math.dist(first[:2], second[:2])
            v = abs(first[2] - second[2])
            distance_ticks.append(tick)
            horizontal.append(h)
            vertical.append(v)
            if separation_metric == "euclidean":
                equivalent.append(math.dist(first, second))
            else:
                equivalent.append(max(h / horizontal_limit, v / vertical_limit) * margin)
        axes[1].plot(distance_ticks, equivalent, color="#b24d35", label="equivalent separation")
        axes[1].plot(distance_ticks, horizontal, color="#14718e", alpha=0.55, label="horizontal")
        axes[1].plot(distance_ticks, vertical, color="#7b5aa6", alpha=0.55, label="vertical")
        axes[1].axhline(margin, color="#b24d35", linestyle=":", linewidth=1.0, label=f"margin {margin:g} m")
        axes[1].set_xlabel("tick")
        axes[1].set_ylabel("governed distance (m)")
        axes[1].set_title("Pair separation metric")
    axes[-1].set_xlabel("tick")
    axes[-1].set_ylabel("ENU Z (m)")
    axes[-1].set_title("Altitude/state geometry")
    # Fork/end markers belong only on time axes.  Drawing tick values on the
    # XY axis would silently turn them into spatial coordinates.
    for axis in axes[1:]:
        axis.axvline(fork_tick, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(end_tick, color="grey", linestyle=":", linewidth=0.8)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, loc="best")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _separation_model_for_entities(
    script: Mapping[str, Any], entity_ids: Sequence[str]
) -> dict[str, Any] | None:
    wanted = set(entity_ids)
    if len(wanted) != 2:
        return None
    for trigger in script.get("triggers") or []:
        if not isinstance(trigger, Mapping) or trigger.get("type") != "entity_proximity":
            continue
        if {str(trigger.get("entity_a") or ""), str(trigger.get("entity_b") or "")} != wanted:
            continue
        horizontal = trigger.get("horizontal_distance_m")
        vertical = trigger.get("vertical_distance_m")
        if horizontal is None or vertical is None:
            continue
        return {
            "horizontal_limit_m": float(horizontal),
            "vertical_limit_m": float(vertical),
            "trigger_id": str(trigger.get("trigger_id") or ""),
        }
    return None


def _governed_parameter(objective_root: Path, name: str) -> float | None:
    summary = _read_json(objective_root / "world_truth_summary.json")
    values = summary.get("governed_parameters")
    if not isinstance(values, Mapping) or name not in values:
        return None
    try:
        return float(values[name])
    except (TypeError, ValueError):
        return None


def _objective_digest(root: Path) -> str:
    def stable_manifest_bytes(path: Path) -> bytes:
        """Remove only pipeline-generated temporary/output prefixes.

        The objective manifest records useful provenance, but its absolute
        temporary input and output directories change on every invocation.
        Hashing those prefixes would make an identical rollout look different.
        All semantic fields and per-artifact digests remain in the normalized
        manifest included below.
        """

        manifest = json.loads(path.read_text(encoding="utf-8-sig"))

        def scrub(value: Any, key: str = "") -> Any:
            if isinstance(value, dict):
                return {str(k): scrub(v, str(k)) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(item, key) for item in value]
            if isinstance(value, str):
                if key in {"episode_root", "output_dir"}:
                    return f"<volatile:{key}>"
                if value.startswith("/tmp/"):
                    return "<volatile:tmp>/" + Path(value).name
                # The manifest can carry an absolute output path in nested
                # record references; preserve the basename while removing
                # the invocation-specific root.
                marker = "/aero210_action_branch_rollouts_"
                if marker in value:
                    return "<volatile:rollout-root>" + value.split(marker, 1)[1].split("/", 1)[-1]
            return value

        return (canonical_json(scrub(manifest)) + "\n").encode("utf-8")

    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.name == "manifest.json":
                digest = hashlib.sha256(stable_manifest_bytes(path)).hexdigest()
                entries[path.relative_to(root).as_posix()] = "sha256:" + digest
            else:
                entries[path.relative_to(root).as_posix()] = _sha256_file(path)
    return digest_object(entries)


def _write_raw_branch(
    branch_root: Path,
    *,
    episode_id: str,
    scenario_id: str,
    source_script: Path,
    source_scene: Path,
    derived_script: Mapping[str, Any],
    ledger: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
    weather: Sequence[Mapping[str, Any]],
    event_trace: Sequence[Mapping[str, Any]],
    event_realizations: Sequence[Mapping[str, Any]],
    roster: Sequence[Mapping[str, Any]],
    arm_id: str,
    arm_kind: str,
    candidate_design_path: Path,
    semantic_goal: Mapping[str, Any],
) -> None:
    branch_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(branch_root / "trajectories.jsonl", trajectories)
    _write_jsonl(branch_root / "weather_meta.jsonl", weather)
    _write_jsonl(branch_root / "event_trace.jsonl", event_trace)
    _write_jsonl(branch_root / "event_realization.jsonl", event_realizations)
    _write_jsonl(branch_root / "branch_event_ledger.jsonl", [ledger])
    _write_json(branch_root / "global_entity_roster.json", {"entities": list(roster)})
    _write_json(branch_root / "event_script.json", dict(derived_script))
    manifest = {
        "schema_name": BRANCH_SCHEMA_NAME,
        "schema_version": BRANCH_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "branch_id": f"{episode_id}::{arm_id}",
        "arm_id": arm_id,
        "arm_kind": arm_kind,
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "semantic_goal": copy.deepcopy(dict(semantic_goal)),
        "candidate_design_sha256": _sha256_file(candidate_design_path),
        "source_authority": {
            "event_script": str(source_script.relative_to(PROJECT_ROOT)),
            "event_script_sha256": _sha256_file(source_script),
            "scene_setup": str(source_scene.relative_to(PROJECT_ROOT)),
            "scene_setup_sha256": _sha256_file(source_scene),
        },
        "derived_script_sha256": _sha256_file(branch_root / "event_script.json"),
        "ledger_sha256": _sha256_file(branch_root / "branch_event_ledger.jsonl"),
        "execution": {
            "engine": "Dataset/tools/batch_generate.py::EpisodeStateEngine",
            "duration_ticks": FORMAL_DURATION_TICKS,
            "ticks": "0..900 inclusive, 1 tick engine grid",
            "fired_event_count": len(event_trace),
            "realized_event_count": len(event_realizations),
        },
        "artifacts": {
            name: {
                "path": name,
                "sha256": _sha256_file(branch_root / name),
            }
            for name in (
                "trajectories.jsonl",
                "weather_meta.jsonl",
                "event_trace.jsonl",
                "event_realization.jsonl",
                "branch_event_ledger.jsonl",
                "global_entity_roster.json",
                "event_script.json",
            )
        },
    }
    _write_json(branch_root / "branch_manifest.json", manifest)


def _compare_target(
    factual: Mapping[str, Any],
    alternative: Mapping[str, Any],
    goal_mode: str,
    arm_kind: str,
) -> dict[str, Any]:
    factual_events = int(factual.get("event_occurrence_count") or 0)
    alternative_events = int(alternative.get("event_occurrence_count") or 0)
    factual_true = int(factual.get("predicate_true_count") or 0)
    alternative_true = int(alternative.get("predicate_true_count") or 0)
    factual_event_ticks = list(factual.get("event_trigger_ticks") or [])
    alternative_event_ticks = list(alternative.get("event_trigger_ticks") or [])
    factual_transition_ticks = [
        int(row.get("to_tick"))
        for row in factual.get("predicate_transitions") or []
        if row.get("to_tick") is not None
    ]
    alternative_transition_ticks = [
        int(row.get("to_tick"))
        for row in alternative.get("predicate_transitions") or []
        if row.get("to_tick") is not None
    ]
    factual_ticks = factual_event_ticks or factual_transition_ticks or list(
        factual.get("predicate_true_ticks") or []
    )
    alternative_ticks = alternative_event_ticks or alternative_transition_ticks or list(
        alternative.get("predicate_true_ticks") or []
    )
    contract_issue: str | None = None
    if goal_mode == "E_to_absent":
        if arm_kind == "delay_recover":
            # A delayed/recovered event is intentionally still observable.
            # The profile-level E_to_absent declaration is therefore
            # incompatible with this arm and must be repaired by the design
            # author; never reinterpret it as absence.
            passed = False
            classification = "design_goal_conflict_delay_arm"
            contract_issue = (
                "arm_kind=delay_recover produced a later target observation, "
                "but the candidate semantic_goal declares E_to_absent"
            )
        else:
            passed = (
                (factual_events > 0 or factual_true > 0)
                and alternative_events == 0
                and alternative_true == 0
            )
            classification = "event_absent" if passed else "unproven"
    elif goal_mode in {"event_timing_or_recovery", "predicate_only"}:
        # Compare the same observable layer on both sides.  A missing
        # transition is meaningful for a predicate-only recovery arm; using
        # all true ticks as a fallback would incorrectly turn a stable true
        # predicate into a timing claim.
        if factual_event_ticks or alternative_event_ticks:
            passed = bool(
                factual_event_ticks
                and alternative_event_ticks
                and factual_event_ticks != alternative_event_ticks
            )
        else:
            passed = factual_transition_ticks != alternative_transition_ticks and bool(
                factual_transition_ticks or alternative_transition_ticks
            )
        classification = "timing_or_recovery_changed" if passed else "state_only_or_unproven"
    else:
        passed = False
        classification = "unsupported_goal_mode"
    return {
        "status": "PASS" if passed else "FAIL",
        "classification": classification,
        "goal_mode": goal_mode,
        "factual_event_occurrence_count": factual_events,
        "alternative_event_occurrence_count": alternative_events,
        "factual_predicate_true_count": factual_true,
        "alternative_predicate_true_count": alternative_true,
        "factual_observed_ticks": factual_ticks,
        "alternative_observed_ticks": alternative_ticks,
        "factual_event_ticks": factual_event_ticks,
        "alternative_event_ticks": alternative_event_ticks,
        "factual_transition_ticks": factual_transition_ticks,
        "alternative_transition_ticks": alternative_transition_ticks,
        "contract_issue": contract_issue,
    }


def build_selected_canary(
    *,
    profile_id: str,
    episode_id: str,
    arm_ids: Sequence[str],
    candidate_root: Path,
    capture_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise BranchRolloutError(f"refusing to overwrite existing output root: {output_root}")
    if not arm_ids or len(set(arm_ids)) != len(arm_ids):
        raise BranchRolloutError("canary arm selection must be non-empty and duplicate-free")
    design = _profile_design(candidate_root, profile_id)
    selected_arm_records = [_profile_arm(candidate_root, profile_id, arm_id) for arm_id in arm_ids]
    factual_count = sum(
        isinstance(record.get("decision_arm"), Mapping)
        and record["decision_arm"].get("arm_kind") == "factual"
        for record in selected_arm_records
    )
    if factual_count != 1:
        raise BranchRolloutError(
            f"canary selection must contain exactly one factual comparator; got {factual_count}"
        )
    anchor = _seed_anchor(design, episode_id)
    scenario_id = str(episode_id.rsplit("__seed", 1)[0])
    seed_id = "seed" + episode_id.rsplit("__seed", 1)[-1]
    source_script = _scenario_script(scenario_id)
    source_scene = source_script.with_name("scene_setup.json")
    if not source_scene.is_file():
        raise BranchRolloutError(f"authored scene_setup.json is missing: {source_scene}")
    capture_episode = capture_root / episode_id
    if not capture_episode.is_dir():
        raise BranchRolloutError(f"canonical capture episode is missing: {capture_episode}")
    source_script_payload = _read_json(source_script)
    scene = _read_json(source_scene)
    output_root.mkdir(parents=True, exist_ok=False)
    branch_root = output_root / "branches" / profile_id / episode_id
    records: list[dict[str, Any]] = []
    objective_summaries: dict[str, dict[str, Any]] = {}

    for arm_id in arm_ids:
        arm_record = _profile_arm(candidate_root, profile_id, arm_id)
        arm = arm_record["decision_arm"]
        arm_root = branch_root / _slug(arm_id)
        try:
            derived_script, ledger = build_branch_script(
                design,
                arm_record,
                source_script_payload,
                seed_id=seed_id,
                anchor_event_id=str(anchor["event_id"]),
            )
            with tempfile.TemporaryDirectory(prefix="aeroworld_branch_script_") as script_temp:
                derived_script_path = Path(script_temp) / "event_script.json"
                derived_script_path.write_text(
                    canonical_json(derived_script) + "\n", encoding="utf-8"
                )
                trajectories, weather, event_trace, event_realizations, roster = _run_engine(
                    derived_script_path, scene, episode_id
                )
            raw_root = arm_root / "raw"
            _write_raw_branch(
                raw_root,
                episode_id=episode_id,
                scenario_id=scenario_id,
                source_script=source_script,
                source_scene=source_scene,
                derived_script=derived_script,
                ledger=ledger,
                trajectories=trajectories,
                weather=weather,
                event_trace=event_trace,
                event_realizations=event_realizations,
                roster=roster,
                arm_id=arm_id,
                arm_kind=str(arm.get("arm_kind") or ""),
                candidate_design_path=candidate_root / profile_id / "normalized_design.json",
                semantic_goal=design.get("semantic_goal") or {},
            )
            # Objective input must have the canonical episode directory name;
            # it is disposable and never enters the formal aw_data tree.
            input_temp = _overlay_branch_input(
                canonical_capture_episode=capture_episode,
                source_scene_setup=source_scene,
                episode_id=episode_id,
                branch_trajectories=trajectories,
                branch_event_trace=event_trace,
                branch_event_realizations=event_realizations,
                branch_ledger=ledger,
            )
            try:
                input_root = Path(input_temp.name) / episode_id
                objective_root = arm_root / "objective"
                artifacts: ObjectiveEpisodeArtifacts = build_episode_objective_artifacts(
                    input_root, objective_root
                )
                write_episode_outputs(artifacts)
                objective_summaries[arm_id] = _target_summary(
                    objective_root=objective_root,
                    design=design,
                    episode_id=episode_id,
                    fork_tick=int(anchor["dispatch_tick"]),
                    end_tick=int((design.get("window_policy") or {}).get("end_tick") or FORMAL_DURATION_TICKS),
                )
                entity_ids = objective_summaries[arm_id].get("bound_entity_ids") or []
                _plot_branch(
                    trajectory_path=input_root / "trajectories.jsonl",
                    output_path=arm_root / "branch_plot.png",
                    entity_ids=entity_ids,
                    fork_tick=int(anchor["dispatch_tick"]),
                    end_tick=int((design.get("window_policy") or {}).get("end_tick") or FORMAL_DURATION_TICKS),
                    title=f"{episode_id} · {arm_id}",
                    predicate_id=str(
                        ((design.get("semantic_goal") or {}).get("primary_predicate") or {}).get(
                            "predicate_id", ""
                        )
                    ),
                    separation_model=_separation_model_for_entities(
                        derived_script, entity_ids
                    ),
                    separation_margin_m=_governed_parameter(
                        objective_root, "separation_margin_m"
                    ),
                )
            finally:
                input_temp.cleanup()
            records.append(
                {
                    "profile_id": profile_id,
                    "episode_id": episode_id,
                    "arm_id": arm_id,
                    "arm_kind": arm.get("arm_kind"),
                    "status": "EXECUTED",
                    "formal_objective_closure_status": artifacts.closure.get("status"),
                    "branch_root": str(arm_root.relative_to(output_root)),
                    "objective_digest": _objective_digest(arm_root / "objective"),
                    "plot_sha256": _sha256_file(arm_root / "branch_plot.png"),
                    "target": objective_summaries[arm_id],
                }
            )
        except (BranchRolloutError, ObjectivePipelineError, OSError, ValueError) as exc:
            records.append(
                {
                    "profile_id": profile_id,
                    "episode_id": episode_id,
                    "arm_id": arm_id,
                    "arm_kind": arm.get("arm_kind"),
                    "status": "REJECTED_FAIL_CLOSED",
                    "reason": str(exc),
                }
            )

    factual_record = next(
        (record for record in records if record.get("arm_kind") == "factual" and record.get("status") == "EXECUTED"),
        None,
    )
    goal_mode = str((design.get("semantic_goal") or {}).get("event_change_mode") or "")
    for record in records:
        if record.get("arm_kind") == "factual" or record.get("status") != "EXECUTED":
            continue
        record["comparison"] = _compare_target(
            factual_record.get("target", {}) if factual_record else {},
            record.get("target", {}),
            goal_mode,
            str(record.get("arm_kind") or ""),
        )
    executed_alternatives = [
        record
        for record in records
        if record.get("arm_kind") != "factual" and record.get("status") == "EXECUTED"
    ]
    passing_alternatives = [
        record for record in executed_alternatives if record.get("comparison", {}).get("status") == "PASS"
    ]
    # A canary is an explicit selection contract: a partial execution is not
    # a successful canary merely because the surviving alternatives happen to
    # compare well.  Missing/rejected arms must remain visible and close the
    # release gate.
    all_requested_executed = len(records) == len(arm_ids) and all(
        record.get("status") == "EXECUTED" for record in records
    )
    release_status = (
        "PASS"
        if all_requested_executed
        and factual_record is not None
        and executed_alternatives
        and len(passing_alternatives) == len(executed_alternatives)
        else "NO-GO"
    )
    formal_closure_pass = bool(records) and all(
        record.get("status") == "EXECUTED"
        and record.get("formal_objective_closure_status") == "PASS"
        for record in records
    )
    contract_issues = [
        {
            "arm_id": record.get("arm_id"),
            "issue": record.get("comparison", {}).get("contract_issue"),
        }
        for record in records
        if record.get("comparison", {}).get("contract_issue")
    ]
    report = {
        "schema_name": "aeroworld_action_branch_canary_report",
        "schema_version": "1.0.0",
        "tool_version": TOOL_VERSION,
        "scope": {
            "profile_id": profile_id,
            "episode_id": episode_id,
            "arm_ids_requested": list(arm_ids),
            "source_candidate_design": str((candidate_root / profile_id / "normalized_design.json").relative_to(PROJECT_ROOT)),
            "source_event_script": str(source_script.relative_to(PROJECT_ROOT)),
        },
        "authority": {
            "candidate_design_sha256": _sha256_file(candidate_root / profile_id / "normalized_design.json"),
            "event_script_sha256": _sha256_file(source_script),
            "scene_setup_sha256": _sha256_file(source_scene),
            "canonical_capture_episode_sha256": _sha256_file(capture_episode / "episode_manifest.json"),
        },
        "execution": {
            "command": "Dataset/tools/build_action_branch_rollouts.py canary",
            "engine": "Dataset/tools/batch_generate.py::EpisodeStateEngine",
            "objective_pipeline": "Dataset/semantic_truth/objective_pipeline.py",
            "window_fork_tick": int(anchor["dispatch_tick"]),
            "window_end_tick": int((design.get("window_policy") or {}).get("end_tick") or FORMAL_DURATION_TICKS),
        },
        "status": release_status,
        "frontend_release_status": release_status,
        # Branch alternatives may intentionally remove downstream episode
        # stages.  They are reviewable branch truth, but are not a formal
        # training corpus until every objective closure is PASS.
        "training_release_status": "PASS" if formal_closure_pass else "NO-GO",
        "contract_issues": contract_issues,
        "records": records,
        "counts": {
            "requested": len(arm_ids),
            "executed": sum(record.get("status") == "EXECUTED" for record in records),
            "rejected_fail_closed": sum(record.get("status") == "REJECTED_FAIL_CLOSED" for record in records),
            "passing_alternatives": len(passing_alternatives),
        },
    }
    _write_json(output_root / "canary_report.json", report)
    return report


def refresh_branch_plots(
    *, rollout_root: Path, candidate_root: Path
) -> dict[str, Any]:
    """Re-render each recorded arm plot from its persisted raw trajectory.

    This is intentionally separate from objective generation: changing the
    visualization cannot change truth rows, and the updated PNG digest is
    written back to the derived canary report for the validator to recheck.
    """

    report_path = rollout_root / "canary_report.json"
    report = _read_json(report_path)
    scope = report.get("scope") or {}
    profile_id = str(scope.get("profile_id") or "")
    episode_id = str(scope.get("episode_id") or "")
    design = _profile_design(candidate_root, profile_id)
    source_script = _read_json(_scenario_script(episode_id.rsplit("__seed", 1)[0]))
    for record in report.get("records") or []:
        if not isinstance(record, Mapping) or record.get("status") != "EXECUTED":
            continue
        branch_root = rollout_root / str(record.get("branch_root") or "")
        if not branch_root.is_dir():
            raise BranchRolloutError(f"branch root is missing: {branch_root}")
        target = record.get("target") or {}
        entity_ids = [str(value) for value in target.get("bound_entity_ids") or []]
        objective_root = branch_root / "objective"
        _plot_branch(
            trajectory_path=branch_root / "raw" / "trajectories.jsonl",
            output_path=branch_root / "branch_plot.png",
            entity_ids=entity_ids,
            fork_tick=int((design.get("window_policy") or {}).get("fork_tick") or report.get("execution", {}).get("window_fork_tick") or 0),
            end_tick=int((design.get("window_policy") or {}).get("end_tick") or report.get("execution", {}).get("window_end_tick") or FORMAL_DURATION_TICKS),
            title=f"{episode_id} · {record.get('arm_id')}",
            predicate_id=str(((design.get("semantic_goal") or {}).get("primary_predicate") or {}).get("predicate_id", "")),
            separation_model=_separation_model_for_entities(source_script, entity_ids),
            separation_margin_m=_governed_parameter(objective_root, "separation_margin_m") if objective_root.is_dir() else None,
        )
        record["plot_sha256"] = _sha256_file(branch_root / "branch_plot.png")
    _write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    canary = sub.add_parser("canary", help="execute a selected deterministic canary")
    canary.add_argument("--profile", default="L1-1_v1")
    canary.add_argument("--episode", default="L1-1_v1__seed00")
    canary.add_argument(
        "--arm",
        action="append",
        dest="arms",
        default=None,
        help="candidate arm id; repeat for an explicit selection",
    )
    canary.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    canary.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    canary.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plots = sub.add_parser("refresh-plots", help="re-render plots from persisted branch trajectories")
    plots.add_argument("--rollout-root", type=Path, required=True)
    plots.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    args = parser.parse_args(argv)
    if args.command == "canary":
        # argparse's default list is replaced, rather than appended, when the
        # caller supplies --arm.  This keeps the requested scope explicit.
        report = build_selected_canary(
            profile_id=str(args.profile),
            episode_id=str(args.episode),
            arm_ids=list(
                args.arms
                or [
                    "arm_factual",
                    "arm_hold_on_pad_suppress_takeoff",
                    "arm_delayed_takeoff_dispatch",
                ]
            ),
            candidate_root=Path(args.candidate_root).resolve(),
            capture_root=Path(args.capture_root).resolve(),
            output_root=Path(args.output_root).resolve(),
        )
        print(canonical_json(report))
        return 0 if report.get("status") == "PASS" else 2
    if args.command == "refresh-plots":
        report = refresh_branch_plots(
            rollout_root=Path(args.rollout_root).resolve(),
            candidate_root=Path(args.candidate_root).resolve(),
        )
        print(canonical_json(report))
        return 0
    raise BranchRolloutError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BranchRolloutError as exc:
        print(f"action branch rollout failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
