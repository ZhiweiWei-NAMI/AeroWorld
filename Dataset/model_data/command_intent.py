"""Origin-known command intents, separate from event labels and realized effects.

Entity IDs are opaque binding handles, not text features. A downstream model
must bind them to its origin-authorized entity roster. These records are not a
complete process belief, proof of legal action execution, or a future plan.

Field selection follows Dataset/tools/batch_generate.py action handlers. Only
explicitly classified generator diagnostics are removed; new fields fail closed.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .action_contract import ActionCommandRecord
from .contract import ModelDataError
from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_METADATA_FIELDS,
    invalid_runtime_state_value_paths,
    unconsumed_runtime_state_paths,
)


CONTROL_FIELDS = {
    "move_entity": frozenset({
        "waypoints_enu_m", "velocity_mps", "activity_type", "post_activity_type",
        "terminal_feasibility",
    }),
    "set_runtime_state": frozenset({"state_patch", "delay_ticks"}),
    "set_visual_state": frozenset({"visual_state", "mode"}),
    "set_weather": frozenset({"profile", "overrides"}),
    "spawn_entity": frozenset({"position_enu_m", "asset_id", "rotation_deg", "visual_state"}),
    "remove_entity": frozenset(),
    "capture_screenshot": frozenset(),
    "set_pedestrian_activity": frozenset({"activity_type"}),
}
DIAGNOSTIC_FIELDS = {
    "move_entity": frozenset({"continuity_repair", "uav_corridor_validation"}),
    "set_visual_state": frozenset({"expected_state"}),
    "set_runtime_state": frozenset({
        "runtime_state_schedule_profile_id", "runtime_state_schedule_semantics",
    }),
}
TERMINAL_DIAGNOSTIC_FIELDS = frozenset({
    "available_ticks", "dispatch_upper_bound_tick", "home_pose_origin_z_m",
    "max_speed_mps", "model", "required_speed_mps", "route_length_m",
    "touchdown_altitude_tolerance_m", "touchdown_dwell_ticks",
})


def _validate_json(value: Any, path: str) -> None:
    if value is None or type(value) is bool:
        return
    if type(value) in {int, float}:
        if not math.isfinite(value):
            raise ModelDataError(f"nonfinite command value: {path}")
        return
    if isinstance(value, str):
        if value.startswith("$param."):
            raise ModelDataError(f"unresolved command parameter: {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}/{index}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ModelDataError(f"nontext command key: {path}")
            _validate_json(item, f"{path}/{key}")
        return
    raise ModelDataError(f"non-JSON command value: {path}")


def _vector3(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3 or any(
        type(item) not in {int, float} or not math.isfinite(item) for item in value
    ):
        raise ModelDataError(f"command requires a finite 3-vector: {name}")


def execution_parameters(command: ActionCommandRecord) -> dict[str, Any]:
    return requested_parameters(command.action_type, command.parameters)


def requested_parameters(kind: str, raw_parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Retain requested controls, not their measured outcome or semantic label.

    Shared by issued commands and explicit prospective plans; a prospective
    request is never fabricated as an actual dispatch record. Spawn rotation is
    an authored request, not proof of execution. Absent controls remain absent.
    """
    if kind not in CONTROL_FIELDS:
        raise ModelDataError(f"unsupported command intent action: {kind}")
    if not isinstance(raw_parameters, Mapping):
        raise ModelDataError("command parameters are not an object")
    unknown = set(raw_parameters) - CONTROL_FIELDS[kind] - DIAGNOSTIC_FIELDS.get(kind, frozenset())
    if unknown:
        raise ModelDataError(f"unclassified command parameters for {kind}: {sorted(unknown)}")
    parameters = {key: value for key, value in raw_parameters.items() if key in CONTROL_FIELDS[kind]}
    _validate_json(parameters, "parameters")
    if kind == "move_entity":
        points = parameters.get("waypoints_enu_m")
        if not isinstance(points, list) or not points:
            raise ModelDataError("move intent lacks explicit waypoints")
        for point in points:
            _vector3(point, "waypoints_enu_m")
        speed = parameters.get("velocity_mps")
        if type(speed) not in {int, float} or speed <= 0:
            raise ModelDataError("move intent lacks explicit positive velocity_mps")
        if "terminal_feasibility" in parameters:
            terminal = parameters["terminal_feasibility"]
            if not isinstance(terminal, Mapping):
                raise ModelDataError("terminal_feasibility is not an object")
            unknown = set(terminal) - TERMINAL_DIAGNOSTIC_FIELDS - {"landing_reference_enu_m"}
            if unknown:
                raise ModelDataError(f"unclassified terminal control fields: {sorted(unknown)}")
            if "landing_reference_enu_m" not in terminal:
                raise ModelDataError("terminal control lacks landing_reference_enu_m")
            _vector3(terminal["landing_reference_enu_m"], "landing_reference_enu_m")
            parameters["terminal_feasibility"] = {
                "landing_reference_enu_m": terminal["landing_reference_enu_m"],
            }
    if kind == "set_runtime_state":
        patch = parameters.get("state_patch")
        if not isinstance(patch, Mapping) or not patch:
            raise ModelDataError("runtime intent lacks an explicit state_patch")
        if any(not isinstance(fields, Mapping) or not fields for fields in patch.values()):
            raise ModelDataError("runtime intent has an invalid state family")
        invalid = unconsumed_runtime_state_paths(patch) + invalid_runtime_state_value_paths(patch)
        if invalid:
            raise ModelDataError(f"invalid runtime intent paths: {sorted(invalid)}")
        parameters["state_patch"] = {
            family: {field: value for field, value in fields.items()
                     if field not in RUNTIME_STATE_METADATA_FIELDS}
            for family, fields in patch.items()
        }
        if any(not fields for fields in parameters["state_patch"].values()):
            raise ModelDataError("runtime intent contains a metadata-only state family")
        if "delay_ticks" in parameters and (
            type(parameters["delay_ticks"]) is not int or parameters["delay_ticks"] < 0
        ):
            raise ModelDataError("runtime intent has invalid delay_ticks")
    if "position_enu_m" in parameters:
        _vector3(parameters["position_enu_m"], "position_enu_m")
    if "rotation_deg" in parameters:
        rotation = parameters["rotation_deg"]
        if not isinstance(rotation, Mapping) or set(rotation) != {"yaw_deg"} or type(rotation["yaw_deg"]) not in {int, float}:
            raise ModelDataError("unclassified authored rotation_deg representation")
    # A detached JSON tree prevents source records being mutated via the view.
    return json.loads(json.dumps(parameters, sort_keys=True, allow_nan=False))


def command_intent(command: ActionCommandRecord) -> dict[str, Any]:
    """Project one dispatch record; callers must enforce its availability cutoff."""
    return {
        "action_type": command.action_type,
        "scope": command.scope,
        "actor_id": command.actor_id,
        "issued_tick": command.issued_tick,
        "planned_start_tick": command.planned_start_tick,
        "planned_duration_ticks": command.planned_duration_ticks,
        "parameters": execution_parameters(command),
    }


def serialize_origin_command_intents(
    commands: Sequence[ActionCommandRecord], *, episode_id: str, origin_tick: int,
    observation_phase: str = "strict_prefix",
) -> bytes:
    """Return all commands known by origin, including older ongoing intentions.

    Command history is not truncated to the sensor window. A previous command
    remains a historical intention, not a claim that its process is still active.
    Future actual commands, event IDs, event-derived targets/roles, trigger
    labels, provenance digests, process phases and outcomes never enter the view.
    Same-tick source order is retained because command order may affect execution.

    observation_phase=pre_dispatch at origin_tick T excludes commands whose
    availability is T: those are dispatched by interpreter.tick(T) after the
    recorded pre-dispatch row.
    """
    if type(origin_tick) is not int or not 0 <= origin_tick <= 900:
        raise ModelDataError("command intent origin must be an integer tick in 0..900")
    if observation_phase not in {"strict_prefix", "pre_dispatch"}:
        raise ModelDataError("unsupported command observation phase")
    if not isinstance(episode_id, str) or not episode_id:
        raise ModelDataError("command intent episode identity is missing")
    selected = []
    seen = set()
    for command in commands:
        if command.episode_id != episode_id:
            raise ModelDataError("command intent mixes episode identities")
        if command.action_id in seen:
            raise ModelDataError("command intent repeats an action identity")
        seen.add(command.action_id)
        if observation_phase == "pre_dispatch":
            known = command.availability_tick < origin_tick
        else:
            known = command.availability_tick <= origin_tick
        if known:
            selected.append(command)
    selected.sort(key=lambda value: value.issued_tick)
    return json.dumps(
        {"origin_tick": origin_tick, "commands": [command_intent(c) for c in selected]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
