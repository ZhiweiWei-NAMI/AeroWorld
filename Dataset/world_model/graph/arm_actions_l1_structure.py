"""Closed record structure for L1 ARM raw action audit rows.

Source family ``arm_actions``; profiles L1-1_v1, L1-1_v2, L1-2_v1, L1-3_v1,
L1-3_v2, L1-4_v1, L1-4_v2.  Producer: ``Dataset/tools/l1_window.py::run_engine``
collects one audit row per dispatched action and persists the window slice to
``raw/actions.jsonl``.  A row is ``{tick, origin, action, result}`` plus one
extra key ``created_schedule`` exactly when the dispatched action is an ok
``move_entity``: the dispatcher deep-copies the schedule dict that
``EpisodeStateEngine._handle_move_entity`` appended
(``batch_generate.py::_handle_move_entity``, consumed by ``keyframes_for``).
The action is the deep-copied payload as resolved by
``donghu_core.event_script_interpreter`` (source-script rows) or authored by
``scripts/arms/<profile>/...`` and bounded by ``rules/arm_config.schema.json``
(intervention rows); the result comes from the registered
``EpisodeStateEngine._handle_*`` handler.

``origin`` splits the record space the way production does:

- ``source_script`` rows are dispatched through the eight handlers registered
  from ``l1_window.DISPATCH``; the dispatcher writes
  ``{'status': 'omitted_by_explicit_intervention'}`` instead of a handler
  result when the window omits the action at or after the fork tick
  (``plan['omit_action_ids']``), so ok rows and omitted rows are separate
  branches.  The omission check is origin-independent in shape: any of the
  eight types can be omitted, and the row keeps the authored payload.
- ``intervention`` rows come from ``config['execution']['actions']``, which
  ``rules/arm_config.schema.json`` restricts to ``move_entity`` and
  ``set_runtime_state``.  The dispatcher raises on any non-ok result, so they
  always carry the handler's ok shape.  The config contract also narrows the
  payload per type (``additionalProperties: false`` plus per-type ``if``/
  ``then`` requirements), so the intervention action branches are declared
  separately from the source-script ones: a ``move_entity`` intervention
  requires ``velocity_mps`` and ``waypoints_enu_m``, admits only the
  single-reference ``terminal_feasibility`` variant, and cannot carry
  ``uav_corridor_validation``; a ``set_runtime_state`` intervention requires
  ``state_patch`` and cannot carry the ``runtime_state`` alias,
  ``delay_ticks`` or the scheduling triple.  Neither spelling reaches a row
  in any other form because ``require_plan_matches_config`` proves the entry
  plan byte-identical to the validated YAML.

Per-branch key authority is ``EpisodeStateEngine._handle_*``: optional keys the
handler tolerates (default velocity on source-script rows, scene-position
spawn, ``visual_state`` extras such as ``lights_on``) are declared optional
even where the current L1 corpus always writes them.  ``state_patch`` is
closed against the executable contract in
``Dataset/tools/runtime_state_contract.py``: the ten declared families, their
consumed fields, boolean/enum/command value rules, facility requester lists
and non-negative numbers, plus the ``rule_id``/``mechanism_id`` metadata
fields.  Shared raw-string rules retain the written spelling while checking
the producer's ``strip().lower()`` enum membership.  The consumed patch
carries the producer's ``engine_whole`` source guard.  ``set_weather`` overrides are closed
against ``Dataset/tools/weather_fields.py``; ``uav_corridor_validation``
against ``regenerate_boundary_scenarios.py::_replace_uav_action_waypoints``
(the two corridor statuses and the preserved move_entity type).  The
``created_schedule`` closure is the schedule writer in ``_handle_move_entity``
(fixed keys; ``source`` is the single literal ``"event_script.move_entity"``
and waypoints are the realized path).

The seven fixed L1 source scripts write only non-null ``state_patch`` on
``set_runtime_state`` actions.  ``regenerate_boundary_scenarios.py`` writes
that key explicitly; no current L1 source writer produces the generic
``runtime_state`` handler alias or a null patch.  The L1 intervention config
also requires ``state_patch`` and excludes the alias.

Scheduling on source-script ``set_runtime_state`` rows: ``delay_ticks``
defaults to 0 (``_to_int`` fallback) and the handler stamps
``scheduled_tick = effective_tick = tick + max(1, delay_ticks)``; the L1
event-state stamp writer (``regenerate_boundary_scenarios.py`` lines
12793-12795) always co-writes ``delay_ticks`` with
``runtime_state_schedule_profile_id`` (the profile's own id,
``l1_l3_runtime_state_schedule``) and ``runtime_state_schedule_semantics``
(the literal ``"queued_after_event_actions_delay_ticks_0_next_tick"``), so
the grammar requires the three keys together and checks their fixed values.
Both stamped and delay-free rows have L1 instances.

``state_families`` in an ok ``set_runtime_state`` result is the writer's
``sorted(patch)`` (``batch_generate.py`` line 2309) of the validated family
names, and ``validate_runtime_state_patch`` requires at least one allowed
family (line 892), so it is closed as a non-repeating array drawn from the
ten declared family names.  The producer's ascending-string order is a write
convention the record grammar cannot express; the written values stay closed.

Action effects, units, coordinates and semantic truth are later-stage
concerns and are not encoded here.  Registry wiring into
``structure_closure.py`` belongs to the integration stage; this module only
exports the pure schema constructor and scans no data at import.
"""

from __future__ import annotations

from Dataset.tools.runtime_state_contract import (
    RUNTIME_STATE_BOOLEAN_FIELDS,
    RUNTIME_STATE_COMMAND_FIELDS,
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_ENUM_FIELDS,
    RUNTIME_STATE_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
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

# The eight handler types l1_window.DISPATCH registers for the event-script
# interpreter; every other source-script type is rejected before a row exists.
_SOURCE_SCRIPT_TYPES = (
    "move_entity", "set_visual_state", "set_runtime_state", "capture_screenshot",
    "spawn_entity", "set_pedestrian_activity", "set_weather", "remove_entity",
)
# rules/arm_config.schema.json restricts intervention payloads to these two.
_INTERVENTION_TYPES = ("move_entity", "set_runtime_state")
# regenerate_boundary_scenarios.py::_replace_uav_action_waypoints writes one
# of these two statuses on every stamped move_entity action.
_CORRIDOR_STATUSES = (
    "low_altitude_waypoints_preserved_and_building_clear",
    "waypoints_repaired_for_building_clearance",
)
# enforce_uav_terminal_feasibility stamps this model on the full variant.
_TERMINAL_MODEL = "constant_speed_path_with_explicit_landing_reference_v2"
# The single schedule source _handle_move_entity writes.
_SCHEDULE_SOURCE = "event_script.move_entity"


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
        allowed = RUNTIME_STATE_ENUM_FIELDS[(family, "mode")]
        return {"oneOf": [
            runtime_enum_string_rule(allowed),
            {"type": "array", "minItems": 1,
             "items": runtime_enum_string_rule(allowed)},
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


def _runtime_state_patch() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {family: _runtime_state_family_patch(family)
                       for family in sorted(RUNTIME_STATE_FIELDS)},
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _visual_state() -> dict:
    # visual(action_id, entity_id, mode, **state) writes mode plus booleans
    # such as lights_on; the L1 corpus uses exactly the mode key.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode"],
        "properties": {
            "mode": {"type": "string", "minLength": 1},
            "lights_on": {"type": "boolean"},
        },
    }


def _terminal_feasibility(intervention: bool) -> dict:
    """``enforce_uav_terminal_feasibility`` writes the full ten-key variant on
    source-script actions; ``rules/arm_config.schema.json`` admits only the
    single-reference variant on intervention payloads."""
    full = {
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
            "route_length_m": {"type": "number", "minimum": 0},
            "available_ticks": {"type": "integer", "minimum": 1},
            "required_speed_mps": {"type": "number", "minimum": 0},
            "max_speed_mps": _float_leaf(),
            "home_pose_origin_z_m": _float_leaf(),
            "landing_reference_enu_m": _number3(),
            "touchdown_altitude_tolerance_m": _float_leaf(),
            "touchdown_dwell_ticks": {"type": "integer", "minimum": 0},
        },
    }
    if intervention:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["landing_reference_enu_m"],
            "properties": {"landing_reference_enu_m": _number3()},
        }
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": ["landing_reference_enu_m"],
            "properties": {"landing_reference_enu_m": _number3()},
        },
        full,
    ]}


def _move_entity_action(intervention: bool) -> dict:
    """Handler keys plus the origin's own authoring contract.  The source
    script keeps the handler tolerances (``velocity_mps`` defaults to 1.0,
    ``uav_corridor_validation`` is stamped by the boundary regenerator); the
    intervention config requires velocity and waypoints, admits only the
    single-reference terminal variant and cannot carry the corridor stamp."""
    rule = {
        "type": "object",
        "unevaluatedProperties": False,
        "required": ["type", "action_id", "entity_id", "waypoints_enu_m"],
        "properties": {
            "type": {"const": "move_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_move_entity reads velocity with default 1.0; the
            # authored payload keeps pre-coercion numbers and the
            # intervention contract admits any positive number.
            "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
            "waypoints_enu_m": _waypoints(),
            "activity_type": _identifier(),
            "post_activity_type": _identifier(),
            "terminal_feasibility": _terminal_feasibility(intervention),
        },
    }
    if not intervention:
        rule["properties"]["uav_corridor_validation"] = {
            "type": "object", "additionalProperties": False,
            "required": ["status", "assigned_altitude_m",
                         "original_waypoints_enu_m", "preserved_action_type"],
            "properties": {
                "status": {"type": "string", "enum": list(_CORRIDOR_STATUSES)},
                "assigned_altitude_m": _float_leaf(),
                "original_waypoints_enu_m": _waypoints(),
                "preserved_action_type": {"const": "move_entity"},
            },
        }
    if intervention:
        rule["required"] = [*rule["required"], "velocity_mps"]
    return rule


def _created_schedule() -> dict:
    # _handle_move_entity writes exactly these keys: the realized schedule the
    # keyframe consumer reads.  start_pos_enu is _entity_pos_at's vector3
    # triple; waypoints_enu_m is the varied realized path (possibly empty).
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "type", "tick", "waypoints_enu_m", "velocity_mps", "activity_type",
            "post_activity_type", "source", "source_event_tick", "action_id",
            "start_pos_enu",
        ],
        "properties": {
            "type": {"const": "move"},
            "tick": {"type": "integer", "minimum": 0},
            "waypoints_enu_m": {
                "type": "array",
                "items": _number3(),
            },
            "velocity_mps": _float_leaf(),
            "activity_type": _identifier(),
            "post_activity_type": _identifier(),
            "source": {"const": _SCHEDULE_SOURCE},
            "source_event_tick": {"type": "integer", "minimum": 0},
            "action_id": _identifier(),
            "start_pos_enu": _number3(),
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


def _set_runtime_state_action(intervention: bool) -> dict:
    """The fixed L1 scripts and intervention config both write a valid
    ``state_patch``.  Only source-script rows can carry the schedule stamp."""
    if intervention:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["type", "action_id", "entity_id", "state_patch"],
            "properties": {
                "type": {"const": "set_runtime_state"},
                "action_id": _identifier(),
                "entity_id": _identifier(),
                "state_patch": _runtime_state_patch(),
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "state_patch"],
        "properties": {
            "type": {"const": "set_runtime_state"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            "state_patch": _runtime_state_patch(),
            # _handle_set_runtime_state defaults to 0 (effective next tick).
            "delay_ticks": {"type": "integer", "minimum": 0},
            "runtime_state_schedule_profile_id": {"const": "l1_l3_runtime_state_schedule"},
            "runtime_state_schedule_semantics": {
                "const": "queued_after_event_actions_delay_ticks_0_next_tick"},
        },
        # The fixed scripts and stamp writer co-write this exact triple.
        "if": {"anyOf": [
            {"required": ["delay_ticks"]},
            {"required": ["runtime_state_schedule_profile_id"]},
            {"required": ["runtime_state_schedule_semantics"]},
        ]},
        "then": {"required": [
            "delay_ticks", "runtime_state_schedule_profile_id",
            "runtime_state_schedule_semantics"]},
    }


def _spawn_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "properties": {
            "type": {"const": "spawn_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            "asset_id": _identifier(),
            # _handle_spawn_entity defaults to the declared scene position.
            "position_enu_m": _number3(),
            # Authored by rot(); the engine handler itself does not read it.
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
            # _handle_set_weather defaults to "clear".
            "profile": _identifier(),
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
            "camera_id": _identifier(),
        },
    }


_SOURCE_ACTION_BRANCHES = {
    "move_entity": lambda: _move_entity_action(False),
    "set_visual_state": _set_visual_state_action,
    "set_runtime_state": lambda: _set_runtime_state_action(False),
    "spawn_entity": _spawn_entity_action,
    "set_pedestrian_activity": _set_pedestrian_activity_action,
    "set_weather": _set_weather_action,
    "remove_entity": _remove_entity_action,
    "capture_screenshot": _capture_screenshot_action,
}

_INTERVENTION_ACTION_BRANCHES = {
    "move_entity": lambda: _move_entity_action(True),
    "set_runtime_state": lambda: _set_runtime_state_action(True),
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
        # str(action_id or camera_id or "") can legitimately write "".
        "capture_id": {"type": "string"},
        "scheduled_tick": {"type": "integer", "minimum": 0},
        "effective_tick": {"type": "integer", "minimum": 1},
        "path_length_m": {"type": "number", "minimum": 0,
                          "x-source-value-types": ["float"]},
        # _handle_set_runtime_state writes sorted(patch); the patch always
        # carries at least one family, so the array is a non-repeating
        # selection of the declared family names.  The ascending write order
        # is a producer convention the record grammar cannot express.
        "state_families": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(RUNTIME_STATE_FIELDS)},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status", *fields],
        "properties": {name: rules[name] for name in ("status", *fields)},
    }


_OK_RESULT_FIELDS = {
    "move_entity": ("tick", "scheduled_tick", "entity_id", "path_length_m"),
    "set_visual_state": ("tick", "entity_id", "mode"),
    "set_runtime_state": ("tick", "scheduled_tick", "effective_tick",
                          "entity_id", "state_families"),
    "spawn_entity": ("tick", "entity_id"),
    "set_pedestrian_activity": ("tick", "entity_id", "activity_type"),
    "set_weather": ("tick", "profile"),
    "remove_entity": ("tick", "effective_tick", "entity_id"),
    "capture_screenshot": ("tick", "capture_id"),
}


def _omitted_result() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"const": "omitted_by_explicit_intervention"}},
    }


def _ok_row(origin: str, name: str, *, with_schedule: bool = False) -> dict:
    """One ok audit row: the dispatcher writes handler results with status ok
    for both origins, and stamps created_schedule on ok move_entity rows only
    (l1_window.run_engine.dispatch)."""
    properties = {
        "tick": {"type": "integer", "minimum": 0},
        "origin": {"const": origin},
        "action": _ACTION_BRANCHES[origin][name](),
        "result": _result_ok(*_OK_RESULT_FIELDS[name]),
    }
    required = ["tick", "origin", "action", "result"]
    if with_schedule:
        properties["created_schedule"] = _created_schedule()
        required.append("created_schedule")
    return {
        "type": "object", "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _omitted_row(origin: str) -> dict:
    """One omitted audit row: the dispatcher substitutes the omission result
    for the handler result and writes no created_schedule."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["tick", "origin", "action", "result"],
        "properties": {
            "tick": {"type": "integer", "minimum": 0},
            "origin": {"const": origin},
            "action": {"oneOf": [
                _SOURCE_ACTION_BRANCHES[name]() for name in _SOURCE_SCRIPT_TYPES
            ]},
            "result": _omitted_result(),
        },
    }


_ACTION_BRANCHES = {
    "source_script": _SOURCE_ACTION_BRANCHES,
    "intervention": _INTERVENTION_ACTION_BRANCHES,
}


def arm_actions_schema() -> dict:
    """Return the closed record schema for one ``arm_actions`` L1 JSONL row.

    Branches with a producer basis but no current L1 instance are declared
    and remain unverified by instances: source-script ``spawn_entity``,
    ``set_weather``, ``set_pedestrian_activity``, ``remove_entity`` and
    ``capture_screenshot`` rows, intervention ``set_runtime_state`` rows, the
    scene-position spawn, default
    velocity, ``visual_state`` on ``spawn_entity``, ``visual_state.lights_on``,
    empty ``set_weather`` overrides and an empty ``capture_id``.  The
    intervention ``set_runtime_state`` branch is additionally bounded by
    ``rules/arm_config.schema.json`` (``state_patch`` required, no alias, no
    ``delay_ticks``, no scheduling keys), so none of its declared shapes
    outside the required patch has any production writer today.

    Instance-verified branches cover source-script ``move_entity``/
    ``set_visual_state``/``set_runtime_state`` (ok and omitted),
    intervention ``move_entity`` ok rows, ``created_schedule`` on ok move rows
    of both origins, both ``terminal_feasibility`` variants (ten-key on source
    scripts, single-reference on interventions), both corridor statuses,
    delay-free and stamped source-script ``set_runtime_state``, and the
    ``state_families`` closure over the ten declared families.
    """
    source_script = {"oneOf": [
        *(_ok_row("source_script", name, with_schedule=name == "move_entity")
          for name in _SOURCE_SCRIPT_TYPES),
        _omitted_row("source_script"),
    ]}
    intervention = {"oneOf": [
        *(_ok_row("intervention", name, with_schedule=name == "move_entity")
          for name in _INTERVENTION_TYPES),
    ]}
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_actions_l1.record.schema.json",
        "title": "L1 ARM raw action audit record",
        "oneOf": [source_script, intervention],
    }
