"""Requested branch controls, without outcome text or fabricated dispatches.

The first implementation supports explicit motion/runtime candidate generators.
Unsupported generators, unresolved parameters and ambiguous single-waypoint
motion fail closed. It does not reconstruct trajectories or infer legal action
execution. Source-script checks are separate from neural input serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .branch_candidates import AuthoredBranchCandidate
from .command_intent import CONTROL_FIELDS, requested_parameters
from .contract import ModelDataError
from Dataset.tools.structured_branch_source import PROJECT_ROOT, project_path


OPERATIONS = (
    "reuse_factual_causal_event", "suppress_causal_event",
    "delay_causal_event", "replace_causal_event",
)
# Producer (run_motion_family_delta_grounding) preserves authored spawn only for
# these typed divert arm kinds—not for suppress/hold or every replace_causal_event.
_DIVERT_SPAWN_PREREQ_ARM_KINDS = frozenset({
    "divert_lateral", "divert_vertical", "divert_altitude",
})
_ACTION_IDENTITY_FIELDS = frozenset({"type", "action_id", "entity_id", "ped_id", "camera_id", "vehicle_id"})
_STEP_FIELDS = frozenset({
    "step_kind", "generator_id", "action_types", "action_type", "actor_id",
    "velocity_mps", "waypoints_enu_m", "state_patch", "description",
    "source_event_id", "delay_ticks",
})
_GENERATOR_ACTION_TYPES = frozenset({"move_entity", "set_runtime_state"})


@dataclass(frozen=True)
class PlannedControl:
    action_type: str
    scope: str
    actor_id: str | None
    planned_start_tick: int
    parameters: Mapping[str, Any]

    def input_record(self, available_tick: int) -> dict[str, Any]:
        return {
            "action_type": self.action_type, "scope": self.scope,
            "actor_id": self.actor_id, "known_at_tick": available_tick,
            "planned_start_tick": self.planned_start_tick,
            "planned_duration_ticks": None,
            "parameters": requested_parameters(self.action_type, self.parameters),
        }


@dataclass(frozen=True)
class BranchCommandPlan:
    episode_id: str
    decision_tick: int
    observation_cutoff_tick: int | None
    predicate_id: str
    operation: str
    commands: tuple[PlannedControl, ...]
    source_script_sha256: str
    candidate_sha256: str
    observation_phase: str = "strict_prefix"

    def input_records(self) -> list[dict[str, Any]]:
        if self.operation not in OPERATIONS:
            raise ModelDataError("unsupported branch control operation")
        return [command.input_record(self.decision_tick) for command in self.commands]


def _command(action: Mapping[str, Any], dispatch_tick: int) -> PlannedControl:
    kind = action["type"]
    if kind not in CONTROL_FIELDS:
        raise ModelDataError(f"unsupported planned action type: {kind}")
    scope = "global" if kind == "set_weather" else "sensor" if kind == "capture_screenshot" else "entity"
    if scope == "global":
        actor = None
    else:
        fields = ("camera_id",) if scope == "sensor" else ("entity_id", "ped_id", "vehicle_id")
        handles = [action[field] for field in fields if field in action]
        if len(handles) != 1 or not isinstance(handles[0], str) or not handles[0]:
            raise ModelDataError("planned command requires one explicit actor handle")
        actor = handles[0]
    parameters = requested_parameters(kind, {key: value for key, value in action.items()
                                              if key not in _ACTION_IDENTITY_FIELDS})
    start = dispatch_tick
    if kind == "set_runtime_state":
        # batch_generate._handle_set_runtime_state schedules at tick+max(1,delay).
        # The handler's omitted-delay default is zero; no absent field is added
        # to the requested parameters and this is not an observed completion.
        start += max(1, parameters.get("delay_ticks", 0))
    if type(start) is not int or not 0 <= start <= 900:
        raise ModelDataError("planned command starts outside the formal episode")
    return PlannedControl(kind, scope, actor, start, parameters)


def _generated_commands(step, dispatch_tick, actor_kinds):
    unknown = set(step) - _STEP_FIELDS
    if unknown:
        raise ModelDataError(f"unclassified generator fields: {sorted(unknown)}")
    actor = step["actor_id"]
    if not isinstance(actor, str) or not actor:
        raise ModelDataError("generator lacks explicit actor_id")
    types = step["action_types"]
    if not isinstance(types, list) or not types or any(
        kind not in _GENERATOR_ACTION_TYPES for kind in types
    ):
        raise ModelDataError("unsupported candidate generator action_types")
    if step["action_type"] not in types:
        raise ModelDataError("generator action_type disagrees with action_types")
    actions = []
    if "move_entity" in types:
        speed, points = step["velocity_mps"], step["waypoints_enu_m"]
        if type(speed) not in {int, float} or not math.isfinite(speed) or speed < 0:
            raise ModelDataError("generator velocity is not finite and nonnegative")
        if not isinstance(points, list) or not points or any(
            not isinstance(point, list) or len(point) != 3 or any(
                type(value) not in {int, float} or not math.isfinite(value) for value in point
            ) for point in points
        ):
            raise ModelDataError("generator waypoints are not explicit finite xyz triples")
        if speed == 0:
            # _move_from_step emits a hold, never motion toward placeholder xyz.
            if actor not in actor_kinds or actor_kinds[actor] not in {
                "uav", "pedestrian", "vehicle", "facility", "traffic_light", "prop",
            }:
                raise ModelDataError("hold translation lacks explicit initial actor type")
            kind = actor_kinds[actor]
            if kind in {"vehicle", "facility", "traffic_light", "prop"}:
                actions.append({"type": "set_runtime_state", "entity_id": actor, "delay_ticks": 0,
                                "state_patch": {"control_state": {"hold_active": True}}})
            else:
                actions.append({"type": "set_visual_state", "entity_id": actor,
                                "visual_state": {"mode": "hold" if kind == "uav" else "stopped"}})
        else:
            length = sum(math.dist(left, right) for left, right in zip(points[:-1], points[1:]))
            if len(points) < 2 or length <= 1e-6:
                raise ModelDataError("candidate motion requires a nondegenerate multi-waypoint path; no guessed current pose")
            actions.append({"type": "move_entity", "entity_id": actor,
                            "velocity_mps": speed, "waypoints_enu_m": points})
    if "set_runtime_state" in types:
        actions.append({"type": "set_runtime_state", "entity_id": actor, "delay_ticks": 0,
                        "state_patch": step["state_patch"]})
    return [_command(action, dispatch_tick) for action in actions]


def compile_branch_plan(
    authority: AuthoredBranchCandidate,
    *,
    actor_kinds: Mapping[str, str],
    project_root: Path = PROJECT_ROOT,
) -> BranchCommandPlan:
    """Compile explicit source requests; do not consume branch outcome evidence.

    actor_kinds must come from explicit initial/prefix authority, not ID spelling
    or future trajectory summaries. Only hold translation needs these classes;
    the neural encoder still independently requires actual graph-bound handles.
    """
    record = authority.record
    decision = record["decision_arm"]
    operation = authority.operation
    if operation not in OPERATIONS:
        raise ModelDataError(f"unsupported candidate operation: {operation}")
    authority.require_available(authority.decision_tick)
    path = project_path(authority.source.manifest["parent_script_ref"], project_root=project_root)
    data = path.read_bytes()
    script = json.loads(data)
    if script["scenario_id"] != record["profile_id"]:
        raise ModelDataError("candidate source script has a different profile")
    anchor = record["anchor"]["event_id"]
    events = [event for event in script["events"] if event["event_id"] == anchor]
    if len(events) != 1:
        raise ModelDataError("candidate source event is not declared exactly once")
    source_actions = events[0]["actions"]
    if not isinstance(source_actions, list) or not source_actions:
        raise ModelDataError("pending source bundle has no explicit commands")
    source_controls = [_command(action, authority.decision_tick) for action in source_actions]
    source_actors = {command.actor_id for command in source_controls if command.scope == "entity"}
    steps = decision["mechanism_steps"]
    if not isinstance(steps, list) or not steps:
        raise ModelDataError("candidate has no declared mechanism steps")
    for step in steps:
        if not isinstance(step, dict) or set(step) - _STEP_FIELDS:
            raise ModelDataError("candidate contains unclassified mechanism fields")
        if step["step_kind"] in {"reuse_source_event", "postpone_source_event"}:
            if step["source_event_id"] != anchor:
                raise ModelDataError("candidate references another pending source bundle")
            matches = [command for command in source_controls
                       if command.actor_id == step["actor_id"] and command.action_type == step["action_type"]]
            if not matches:
                raise ModelDataError("candidate source action has no matching actor/type")
            # postpone.delay_ticks is the ARM delay, not the source action's delay_ticks.
            keys = (set(step) & CONTROL_FIELDS[step["action_type"]]) - {"delay_ticks"}
            if keys and not any(all(key in command.parameters and command.parameters[key] == step[key]
                           for key in keys) for command in matches):
                raise ModelDataError("candidate supplied controls differ from its source bundle")
        elif step["step_kind"] != "invoke_action_generator":
            raise ModelDataError(f"unsupported candidate step: {step['step_kind']}")
    commands = []
    if operation == "reuse_factual_causal_event":
        if any(step["step_kind"] != "reuse_source_event" for step in steps):
            raise ModelDataError("factual candidate has non-reuse steps")
        commands = source_controls
    else:
        if operation == "delay_causal_event":
            delays = {step["delay_ticks"] for step in steps if step["step_kind"] == "postpone_source_event"}
            if len(delays) != 1:
                raise ModelDataError("delay candidate requires a single declared delay")
            delay = next(iter(delays))
            if type(delay) is not int or delay <= 0:
                raise ModelDataError("delay candidate has invalid ticks")
            commands = [_command(action, authority.decision_tick + delay) for action in source_actions]
        elif any(step["step_kind"] != "invoke_action_generator" for step in steps):
            raise ModelDataError("replacement/suppression candidate contains a source replay step")
        generated = []
        for step in steps:
            if step["step_kind"] == "invoke_action_generator":
                if operation in {"suppress_causal_event", "delay_causal_event"} and (
                    "move_entity" in step["action_types"] and step["velocity_mps"] != 0
                ):
                    raise ModelDataError("suppression/delay continuation is not an explicit hold")
                generated.extend(_generated_commands(step, authority.decision_tick, actor_kinds))
        if not generated:
            if operation == "delay_causal_event" and not source_actors:
                generated = []
            else:
                raise ModelDataError("candidate has no explicit continuation/replacement commands")
        generated_actors = {command.actor_id for command in generated if command.scope == "entity"}
        if source_actors and not source_actors <= generated_actors:
            raise ModelDataError("candidate omits source actors; coupled holds must be explicit")
        # Mirror producer preserve_authored_spawn_prerequisite_for_diversion:
        # only typed divert_* kinds, and only spawn_entity from the predecision
        # parent_script_ref source bundle (source_controls above)—never from the
        # already-repaired saved branch event_script, and never for suppress/hold.
        arm_kind = decision.get("arm_kind")
        if not isinstance(arm_kind, str) or not arm_kind:
            arm_kind = record.get("arm_kind")
        if (
            operation == "replace_causal_event"
            and arm_kind in _DIVERT_SPAWN_PREREQ_ARM_KINDS
        ):
            spawn_prereqs = [
                command for command in source_controls
                if command.action_type == "spawn_entity"
                and command.scope == "entity"
                and command.actor_id in generated_actors
            ]
            if spawn_prereqs:
                generated = [*spawn_prereqs, *generated]
        # Same-tick command order follows mechanism_steps / source_actions order.
        # Do not sort by actor_id: diversion index signs and repeated same-actor
        # actions are order-sensitive, and lexical reordering would swap physics.
        commands = [*generated, *commands]
    return BranchCommandPlan(
        episode_id=record["anchor"]["episode_id"], decision_tick=authority.decision_tick,
        observation_cutoff_tick=authority.observation_cutoff_tick, predicate_id=authority.predicate_id,
        operation=operation, commands=tuple(commands), source_script_sha256=hashlib.sha256(data).hexdigest(),
        candidate_sha256=authority.candidate_sha256,
        observation_phase=getattr(authority, "observation_phase", "strict_prefix"),
    )

def _saved_hold_required(plan: BranchCommandPlan, authority: AuthoredBranchCandidate) -> bool:
    # Continuation follows typed steps and compiled entity commands, not names or rain flags.
    if plan.operation == "suppress_causal_event":
        return True
    if plan.operation != "delay_causal_event":
        return False
    steps = authority.record["decision_arm"]["mechanism_steps"]
    if not isinstance(steps, list):
        raise ModelDataError("candidate has no declared mechanism steps")
    if any(isinstance(step, dict) and step.get("step_kind") == "invoke_action_generator" for step in steps):
        return True
    return any(command.scope == "entity" for command in plan.commands)


def _coupled_hold_event_id(plan: BranchCommandPlan, authority: AuthoredBranchCandidate, ledger: Mapping[str, Any]) -> str | None:
    required = _saved_hold_required(plan, authority)
    if "coupled_hold" not in ledger:
        if required:
            raise ModelDataError("saved branch ledger lacks required coupled hold")
        return None
    hold = ledger["coupled_hold"]
    if not isinstance(hold, dict):
        raise ModelDataError("saved branch coupled hold is malformed")
    injected = hold.get("injected")
    event_id = hold.get("event_id")
    action_count = hold.get("action_count")
    claimed = injected is True or (isinstance(event_id, str) and event_id) or (
        type(action_count) is int and action_count > 0)
    if required or claimed:
        if injected is not True:
            raise ModelDataError("saved branch coupled hold was not injected")
        if not isinstance(event_id, str) or not event_id:
            raise ModelDataError("saved branch coupled hold lacks event_id")
        if type(action_count) is not int or action_count <= 0:
            raise ModelDataError("saved branch coupled hold has invalid action_count")
        return event_id
    if injected is False and event_id is None and action_count == 0:
        return None
    raise ModelDataError("saved branch coupled hold is malformed")


def verify_saved_script_controls(plan: BranchCommandPlan, authority: AuthoredBranchCandidate) -> None:
    """Check instruction alignment only; never use saved instructions as inputs.

    This is not proof of realized motion, actual dispatch time, action legality
    or shared simulator checkpoint. Those belong to independent evidence/scoring.
    """
    script = json.loads(authority.source.raw_file("event_script.json").read_text())
    ledger = json.loads(authority.source.raw_file("branch_event_ledger.jsonl").read_text())
    if not isinstance(script, dict) or not isinstance(ledger, dict):
        raise ModelDataError("saved branch script or ledger is not a JSON object")
    ids = []
    hold_id = None
    hold_count = None
    if plan.operation in {"suppress_causal_event", "delay_causal_event"}:
        hold_id = _coupled_hold_event_id(plan, authority, ledger)
        if hold_id is not None:
            ids.append((hold_id, plan.decision_tick))
            hold_count = ledger["coupled_hold"]["action_count"]
    if plan.operation != "suppress_causal_event":
        if plan.operation == "delay_causal_event":
            delay = ledger.get("delay_ticks")
            if type(delay) is not int or delay <= 0:
                raise ModelDataError("saved branch ledger has invalid delay_ticks")
        else:
            delay = 0
        ids.append((authority.record["anchor"]["event_id"], plan.decision_tick + delay))
    expected = []
    for event_id, dispatch in ids:
        events = [event for event in script["events"] if event["event_id"] == event_id]
        if len(events) != 1:
            raise ModelDataError("saved branch script lacks one selected control bundle")
        actions = events[0]["actions"]
        if event_id == hold_id and hold_count != len(actions):
            raise ModelDataError("saved branch coupled hold action_count disagrees with the selected bundle")
        expected.extend(_command(action, dispatch) for action in actions)
    if tuple(expected) != plan.commands:
        raise ModelDataError("compiled candidate controls differ from saved branch instructions")
