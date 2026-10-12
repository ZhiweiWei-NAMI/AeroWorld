"""Closed record structure for X ARM raw action audit rows.

Source family ``arm_actions``; profiles X1_rain_to_c2loss_to_forced_landing,
X2_gnss_spoof_to_geofence_violation, X3_pedestrian_fall_to_emergency_response,
X4_fog_to_uav_conflict, X5_comm_failure_to_pad_contention,
X6_crowd_evacuation_to_airspace_lockdown (``grp_cross_layer`` windows).
Producer chain: ``Dataset/tools/x_arm_pipeline.py::run_arm`` replays the
compiled branch through ``Dataset/tools/l6_v2/engine.py::execute`` with
exactly ``(scene, script, episode_id, end_tick, snapshot_ticks)`` — the call
passes no ``injections``, no ``omit_action_ids``, no ``cancel_event_ids`` and
no ``retime`` — then filters ``run['audit']`` to the window ticks and writes
each row verbatim to ``raw/actions.jsonl``.  The 42 X configs (key set
``{arm_id, definition, episode_id, gates, group, hypothesis,
intervention_actions, mode, profile, removed_event_ids, scope, seed, sources,
window}``, ``intervention_actions: []`` in every factual) carry no mechanism
key that could feed those arguments, and the dedicated branch entries are
loaded as modules, never invoked as injection writers.  With no omission
source, ``dispatch_source`` (``l6_v2/engine.py``) takes only its non-ok and
its dispatch path, and the non-ok path raises before a row is appended; every
persisted row is therefore ``origin: 'source_script'`` with a handler ok
result, built as ``{'tick': clock tick, 'origin': 'source_script', 'action':
deep-copied authored payload, 'result': handler result}``, extended by
``motion_schedule`` exactly when ``motion_effect`` sees a ``move_entity``
action with an ok result.  The engine's intervention loop, omission stamp and
interpreter-level ``error``/``skipped``/``sequence`` results all terminate the
run instead of producing an X record.  Action effects are a separate derived
source (``raw/action_effects.jsonl``) and are not declared here.

The record space partitions by the dispatched action type, which fixes the
action payload, the result shape and the ``motion_schedule`` presence at once,
so the root is a flat oneOf of eight complete-row branches, mutually exclusive
by ``action.type``:

- ok ``move_entity`` requires the ``motion_schedule`` sibling (terminal
  keyframe of the realised schedule); every other branch rejects it.
- every other type carries exactly its handler's ok result.

Per-branch key authority is ``EpisodeStateEngine._handle_*`` in
``Dataset/tools/batch_generate.py`` (registered by ``l6_v2.engine._handlers``):
keys the handler tolerates (default velocity, default pedestrian activity,
delay-free patches, scene-position spawn, the ``runtime_state`` alias) are
declared optional where applicable; keys the handler
always writes (``capture_id``, ``activity_type`` in the pedestrian result) are
required.  ``state_patch``/``runtime_state`` is closed against the executable
contract in ``Dataset/tools/runtime_state_contract.py``: the ten declared
families, their consumed fields, boolean/enum/command value rules, facility
requester lists and non-negative numbers, plus the ``rule_id``/
``mechanism_id`` metadata fields.  The shared string fragments encode the
producer's stripped lowercase enum membership and nonblank checks while
keeping the written string.  The consumed patch uses the ``engine_whole``
source guard; an ignored coexisting alias does not.  ``set_weather``
profiles are closed to ``WEATHER_PROFILES`` (``batch_generate.py``) and
overrides to ``Dataset/tools/weather_fields.py``.  ``terminal_feasibility`` is
closed against ``regenerate_boundary_scenarios.enforce_uav_terminal_feasibility``
(full ten-key stamp) and the X branch entries that author the reduced
landing-reference form; ``uav_corridor_validation`` is closed against
``_replace_uav_action_waypoints``.  The ``set_runtime_state`` result's
``state_families`` is the writer's ``sorted(patch)`` of validated families
(``_handle_set_runtime_state``), not an arbitrary string list.

The ``set_runtime_state`` patch pair follows ``runtime_state_patch_from_action``
(lines 956-968): ``state_patch`` is read first and a non-null value is always
validated, the ``runtime_state`` alias is read only when ``state_patch`` is
absent or ``None``, and a row with neither usable value raises.  Current X
source scripts and branch builders write ``state_patch`` only; the declared
alias and explicit-null forms therefore have no X instance validation.
The shared handler can ignore an arbitrary alias value beside a valid
``state_patch``, but the fixed X source chain cannot author that shape.
It is outside this module's declared source range.

These records carry no integrity metadata (the producer path writes no
digest/hash/sha256 field), so no publication-exclusion path is required.

The sibling root ``arm_actions_l6_structure.py`` declares the L6 producer's
record space over the same engine; it keeps intervention and omission branches
because the L6 producer passes those mechanism arguments, so it is not reused
as-is.  Registry wiring into ``structure_closure.py`` belongs to the
integration stage; this module only exports the pure schema constructor and
scans no data at import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
    _CONTROL_ENUMS,
    _MISSION_ENUMS,
    _SECURITY_ENUMS,
)
from Dataset.tools.weather_fields import (
    WEATHER_ALIASES,
    WEATHER_BOOLEAN_FIELDS,
    WEATHER_NUMERIC_FIELDS,
)
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

__all__ = ["arm_actions_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# Types registered by l6_v2.engine._handlers; every other type produces no
# handler result and never reaches an X audit row.  remove_entity is declared
# for the registered handler, but current X scripts do not author it;
# its branch has no X instance validation.
_SOURCE_SCRIPT_TYPES = (
    "move_entity", "set_visual_state", "set_runtime_state", "capture_screenshot",
    "spawn_entity", "set_pedestrian_activity", "set_weather", "remove_entity",
)
# EpisodeStateEngine._replace_uav_action_waypoints writes exactly these two;
# both appear in current X rows.
_CORRIDOR_STATUSES = (
    "low_altitude_waypoints_preserved_and_building_clear",
    "waypoints_repaired_for_building_clearance",
)
# enforce_uav_terminal_feasibility stamps this model on the full variant; the
# reduced variant reuses the same model string.
_TERMINAL_MODEL = "constant_speed_path_with_explicit_landing_reference_v2"
_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _number3() -> dict:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints() -> dict:
    return {"type": "array", "minItems": 1, "items": _number3()}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state patch field, from the executable
    producer contract (validate_runtime_state_patch plus
    invalid_runtime_state_value_paths in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return nonblank_python_string_rule()
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        member = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
        return {"oneOf": [
            dict(member),
            {"type": "array", "minItems": 1, "items": member},
        ]}
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        return nonblank_python_string_rule()
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        return runtime_enum_string_rule(RUNTIME_STATE_ENUM_FIELDS[(family, field)])
    raise ValueError(
        "runtime state field is outside the executable producer contract: "
        f"{family}.{field}")


def _runtime_state_family_patch(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
    }


def _runtime_state_patch(*, consumed: bool) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {family: _runtime_state_family_patch(family)
                       for family in sorted(RUNTIME_STATE_FIELDS)},
        **(runtime_state_source_guard_rule("engine_whole") if consumed else {}),
    }


def _visual_state() -> dict:
    # The X event scripts author exactly {"mode": ...}; the handler keeps the
    # payload as the activity label.  An empty mode is handler-tolerated but
    # has no X instance and no X branch producer, so mode stays required.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode"],
        "properties": {
            "mode": {"type": "string", "minLength": 1},
            "lights_on": {"type": "boolean"},
        },
    }


def _terminal_feasibility() -> dict:
    # enforce_uav_terminal_feasibility writes the full ten-key stamp; the X
    # branch entries author the reduced landing-reference form directly.  Both
    # variants appear on X rows.
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": ["model", "landing_reference_enu_m",
                         "touchdown_altitude_tolerance_m", "touchdown_dwell_ticks"],
            "properties": {
                "model": {"const": _TERMINAL_MODEL},
                "landing_reference_enu_m": _number3(),
                "touchdown_altitude_tolerance_m": _float_leaf(),
                "touchdown_dwell_ticks": {"type": "integer", "minimum": 0},
            },
        },
        {
            "type": "object", "additionalProperties": False,
            "required": [
                "model", "dispatch_upper_bound_tick", "route_length_m",
                "available_ticks", "required_speed_mps", "max_speed_mps",
                "home_pose_origin_z_m", "landing_reference_enu_m",
                "touchdown_altitude_tolerance_m", "touchdown_dwell_ticks",
            ],
            "properties": {
                "model": {"const": _TERMINAL_MODEL},
                "dispatch_upper_bound_tick": {"type": "integer", "minimum": 0},
                "route_length_m": _float_leaf(),
                "available_ticks": {"type": "integer", "minimum": 1},
                "required_speed_mps": _float_leaf(),
                "max_speed_mps": _float_leaf(),
                "home_pose_origin_z_m": _float_leaf(),
                "landing_reference_enu_m": _number3(),
                "touchdown_altitude_tolerance_m": _float_leaf(),
                "touchdown_dwell_ticks": {"type": "integer", "minimum": 0},
            },
        },
    ]}


def _move_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "waypoints_enu_m"],
        "properties": {
            "type": {"const": "move_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_move_entity reads velocity through _to_float with
            # default 1.0 and raises on an empty waypoint list; every X row
            # writes it.  The audit row keeps the authored payload.
            "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
            "waypoints_enu_m": _waypoints(),
            # Activity labels stay producer-side strings: pedestrian labels
            # are normalized by pedestrian_activity_catalog and vehicle/uav
            # states come from the scene contract.
            "activity_type": _identifier(),
            "post_activity_type": _identifier(),
            "terminal_feasibility": _terminal_feasibility(),
            "uav_corridor_validation": {
                "type": "object", "additionalProperties": False,
                "required": ["status", "assigned_altitude_m",
                             "original_waypoints_enu_m", "preserved_action_type"],
                "properties": {
                    "status": {"type": "string", "enum": list(_CORRIDOR_STATUSES)},
                    "assigned_altitude_m": _float_leaf(),
                    "original_waypoints_enu_m": _waypoints(),
                    "preserved_action_type": {"const": "move_entity"},
                },
            },
        },
    }


def _set_visual_state_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "visual_state"],
        "properties": {
            "type": {"const": "set_visual_state"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            "visual_state": _visual_state(),
        },
    }


def _set_runtime_state_action() -> dict:
    # Production basis (batch_generate.py): ``_handle_set_runtime_state``
    # resolves the patch through ``runtime_state_patch_from_action`` (lines
    # 956-968), which reads ``action['state_patch']`` and falls back to
    # ``action['runtime_state']`` only when that value is ``None``; the event
    # script interpreter forwards each authored action unchanged
    # (donghu_core/event_script_interpreter.py lines 474-479), so the two keys
    # may coexist and each may be written as explicit ``null``.  At least one
    # key must carry a validated patch mapping: ``{"state_patch": None}``
    # alone raises ``set_runtime_state requires state_patch or runtime_state``.
    # A non-null ``state_patch`` mapping is always validated; a
    # ``runtime_state`` beside it is never read.  The fixed X source chain
    # writes ``state_patch`` alone; an arbitrary ignored alias is outside
    # this module's source range.  Alias and explicit-null forms have no X
    # instance validation.  The pair logic lives in one nested oneOf whose branches
    # each declare both patch keys, because a branch may not re-type a key the
    # base object already declares; the exclusive earlier form is gone.
    return {
        "type": "object",
        "unevaluatedProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "oneOf": [
            {
                "required": ["state_patch"],
                "properties": {
                    "state_patch": _runtime_state_patch(consumed=True),
                    # Coexisting alias is accepted and never read; a written
                    # null is likewise ignored.  No X instance uses the alias
                    # in either form.
                    "runtime_state": {"oneOf": [
                        _runtime_state_patch(consumed=False), {"type": "null"}]},
                },
            },
            {
                # Fallback branch: ``state_patch`` absent or an explicit null,
                # the alias mapping carries the patch.
                "required": ["runtime_state"],
                "properties": {
                    "runtime_state": _runtime_state_patch(consumed=True),
                    "state_patch": {"type": "null"},
                },
            },
        ],
        "properties": {
            "type": {"const": "set_runtime_state"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_set_runtime_state rejects negative values and defaults
            # the key to 0; every X row writes 0, 1 or 5.
            "delay_ticks": {"type": "integer", "minimum": 0},
            # Authored schedule identifiers carried beside the patch; the
            # engine dispatches the handler, the formal schedule contracts
            # consume the authored payload.
            "runtime_state_schedule_profile_id": _identifier(),
            "runtime_state_schedule_semantics": _identifier(),
        },
    }


def _spawn_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "properties": {
            "type": {"const": "spawn_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_spawn_entity keeps the value only when truthy; every X
            # action writes it.
            "asset_id": _identifier(),
            # _handle_spawn_entity defaults to the declared scene position.
            "position_enu_m": _number3(),
            # Authored beside the position; the engine handler itself does not
            # read it and the scene yaw consumer reads only yaw_deg.  The X
            # scripts author exactly that key.
            "rotation_deg": {
                "type": "object", "additionalProperties": False,
                "required": ["yaw_deg"],
                "properties": {
                    "yaw_deg": {"type": "number"},
                    "pitch_deg": {"type": "number"},
                    "roll_deg": {"type": "number"},
                },
            },
            "visual_state": _visual_state(),
        },
    }


def _set_pedestrian_activity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "properties": {
            "type": {"const": "set_pedestrian_activity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_set_pedestrian_activity defaults to "waiting".
            "activity_type": _identifier(),
        },
    }


def _set_weather_action() -> dict:
    overrides = {name: {"type": "number"}
                 for name in sorted(WEATHER_NUMERIC_FIELDS | set(WEATHER_ALIASES))}
    for name in sorted(WEATHER_BOOLEAN_FIELDS):
        overrides[name] = {"type": "boolean"}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id"],
        "properties": {
            "type": {"const": "set_weather"},
            "action_id": _identifier(),
            # weather_payload resolves the profile through WEATHER_PROFILES
            # and raises on any other name; X authors exactly "rain"/"fog".
            "profile": {"type": "string", "minLength": 1,
                        "enum": ["clear", "dusk", "fog", "heat", "light smoke",
                                 "rain", "wind"]},
            # weather_payload rejects unknown keys, non-boolean booleans and
            # non-finite numbers, and conflicting canonical/alias pairs; the
            # empty-override default form has no X instance.
            "overrides": {
                "type": "object", "additionalProperties": False,
                "properties": overrides,
            },
        },
    }


def _remove_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "properties": {
            "type": {"const": "remove_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
        },
    }


def _capture_screenshot_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id"],
        "properties": {
            "type": {"const": "capture_screenshot"},
            "action_id": _identifier(),
            # _handle_capture_screenshot falls back to the empty action_id
            # when neither key resolves; every X action writes both.
            "camera_id": _identifier(),
        },
    }


_ACTION_BRANCHES = {
    "move_entity": _move_entity_action,
    "set_visual_state": _set_visual_state_action,
    "set_runtime_state": _set_runtime_state_action,
    "spawn_entity": _spawn_entity_action,
    "set_pedestrian_activity": _set_pedestrian_activity_action,
    "set_weather": _set_weather_action,
    "remove_entity": _remove_entity_action,
    "capture_screenshot": _capture_screenshot_action,
}


def _result_ok(*fields: str) -> dict:
    """One ok result shape from _append_action_result(status, tick=..., ...)."""
    rules: dict[str, dict] = {
        "status": {"const": "ok"},
        "tick": {"type": "integer", "minimum": 0},
        "entity_id": _identifier(),
        "mode": _identifier(),
        "activity_type": _identifier(),
        "profile": _identifier(),
        "capture_id": _identifier(),
        "scheduled_tick": {"type": "integer", "minimum": 0},
        "effective_tick": {"type": "integer", "minimum": 1},
        "path_length_m": {"type": "number", "minimum": 0,
                          "x-source-value-types": ["float"]},
        # _handle_set_runtime_state writes state_families=sorted(patch) where
        # patch is validate_runtime_state_patch's output: families are unique
        # (dict keys), sorted, and drawn from RUNTIME_STATE_FIELDS only.
        "state_families": {"type": "array", "minItems": 1,
                           "uniqueItems": True,
                           "items": {"type": "string", "minLength": 1,
                                     "enum": sorted(RUNTIME_STATE_FIELDS)}},
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status", *fields],
        "properties": {name: rules[name] for name in ("status", *fields)},
    }


_OK_RESULT_BRANCHES = {
    "move_entity": lambda: _result_ok("tick", "scheduled_tick", "entity_id",
                                      "path_length_m"),
    "set_visual_state": lambda: _result_ok("tick", "entity_id", "mode"),
    "set_runtime_state": lambda: _result_ok("tick", "scheduled_tick",
                                            "effective_tick", "entity_id",
                                            "state_families"),
    "spawn_entity": lambda: _result_ok("tick", "entity_id"),
    "set_pedestrian_activity": lambda: _result_ok("tick", "entity_id",
                                                  "activity_type"),
    "set_weather": lambda: _result_ok("tick", "profile"),
    "remove_entity": lambda: _result_ok("tick", "effective_tick", "entity_id"),
    "capture_screenshot": lambda: _result_ok("tick", "capture_id"),
}


def _motion_schedule() -> dict:
    # l6_v2.engine.motion_effect stamps the terminal keyframe of the realised
    # move schedule on every ok move_entity row.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["terminal_tick", "endpoint_enu_m"],
        "properties": {
            "terminal_tick": {"type": "integer", "minimum": 0},
            # keyframes_for stores vector3 positions, always floats.
            "endpoint_enu_m": {"type": "array", "minItems": 3, "maxItems": 3,
                               "items": _float_leaf()},
        },
    }


def _row(action_type: str) -> dict:
    """One complete-row branch: one action type, its ok result, the motion
    sibling exactly on move_entity.

    ``dispatch_source`` fixes the result shape from the dispatched type and
    ``motion_effect`` fixes ``motion_schedule`` presence, so the branch is
    exact rather than per-field permissive.
    """
    properties: dict[str, dict] = {
        "tick": {"type": "integer", "minimum": 0},
        "origin": {"const": "source_script"},
        "action": _ACTION_BRANCHES[action_type](),
        "result": _OK_RESULT_BRANCHES[action_type](),
    }
    required = ["tick", "origin", "action", "result"]
    if action_type == "move_entity":
        properties["motion_schedule"] = _motion_schedule()
        required.append("motion_schedule")
    return {
        "type": "object", "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def arm_actions_schema() -> dict:
    """Return the closed record schema for one ``arm_actions`` X JSONL row.

    The root is a flat oneOf of eight complete-row branches — one per
    registered handler type at ``origin: 'source_script'`` with its ok
    result — because the X producer wires no mechanism into the engine: the
    ``run_arm`` call passes no injections, omissions, retiming or
    cancellations, the configs carry no such key, ``dispatch_source`` raises
    on any non-ok handler result before a row is appended, and the
    interpreter-level ``error``/``skipped`` results never reach an audit row.
    The earlier ``intervention`` and ``omitted_by_configuration`` branches are
    removed accordingly.  ``action.type`` pins each branch, so exactly one
    branch matches each record.  Present X records fill seven of the eight
    branches (``remove_entity`` has no X row); ``move_entity`` rows all carry
    ``motion_schedule``.

    Branches with a producer basis but no current X instance are declared and
    remain 尚无实例验证:

    - the ``remove_entity`` branch (registered handler, no current X instance);
    - the ``runtime_state`` patch alias (alone or beside ``state_patch``) and
      the explicit ``null`` form of either patch key (current X authors write
      ``state_patch`` alone);
    - ``lights_on`` in ``visual_state``, ``pitch_deg``/``roll_deg`` rotation
      members, scene-position spawn (``position_enu_m`` absent), default
      velocity, default pedestrian activity, delay-free ``set_runtime_state``,
      empty ``set_weather`` overrides, and weather profiles outside the
      authored ``rain``/``fog`` pair.

    Both corridor statuses and both ``terminal_feasibility`` variants are
    represented by the declared branches; selected real rows were validated.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_actions_x.record.schema.json",
        "title": "X ARM raw action audit record",
        "oneOf": [_row(name) for name in _SOURCE_SCRIPT_TYPES],
    }
