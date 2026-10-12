"""Closed record structure for L3 ARM raw action audit rows.

Source family ``arm_actions``; profiles L3-1_v1, L3-2_v1, L3-2_v2, L3-3_v1,
L3-3_v2.  Producer: ``Dataset/tools/l3_v2/arm_runtime.py::run_arm`` writes
``raw/actions.jsonl`` verbatim from the audit list that
``Dataset/tools/l3_v2/engine.py::execute`` collects.  One row is appended per
dispatched action and is exactly ``{tick, origin, action, result}``: the
action is the deep-copied authored payload (``$param`` refs already resolved
by the interpreter or dropped from failed lookups in ``resolve_param``) and
the result comes from the registered handler in
``Dataset/tools/batch_generate.py::EpisodeStateEngine``.

Row branches are bound per action type, the way production dispatches them
(``engine.py::dispatch_source`` and the injection loop both call
``handler_map[action["type"]]``):

- ``source_script`` rows are dispatched through the event-script interpreter
  and may carry ``{'status': 'omitted_by_configuration'}`` when the declared
  mechanism omits them (``engine.py`` ``dispatch_source``; the omission status
  differs from the L2 producer's own omission string).  A handler failure
  raises through ``arm_runtime`` before any arm is written, so a persisted row
  carries either the ok result of its own type's handler or this omission
  result.  The omission result is not tied to one action type: the row keeps
  the authored payload of whatever type was omitted.
- ``intervention`` rows come from ``ArmSpec.inject_actions``, dispatched
  directly through the same ``handler_map`` at the declared tick; the
  dispatcher raises on any non-ok result, so they always carry their own
  type's ok result shape.  ``rules/arm_config.schema.json`` does not constrain
  L3 — L3 configuration identity is the ``ArmSpec`` dataclass itself — so the
  only production type guard is that handler map, which admits all eight
  registered types.  Both row branches therefore accept all eight types, each
  bound to its own result shape.

Per-branch key authority is ``EpisodeStateEngine._handle_*``: optional keys
the handler tolerates (default velocity, scene-position spawn,
``runtime_state`` alias, delay-free ``set_runtime_state``, default activity)
are declared optional even where the current L3 corpus always writes them.
``action_id`` is required only on ``source_script`` branches: every
script-authoring and scenario-generation path writes it (the interpreter
reads ``resolved.get("action_id", "")`` for its trace row), while the L3
injection loop hands a mapping to the handler, which derives a fallback
(``action.get("action_id") or …`` in each ``_handle_*``), so intervention
branches leave it optional.  Only the source-script builder requires a nested
``visual_state.mode`` and rewrites the pedestrian target to ``entity_id``;
the intervention dispatch also accepts the handler's top-level ``mode`` and
``ped_id`` aliases.
The source-script builder does not write those aliases, while the L3
injection loop passes its mapping directly to the handler without a nested
action schema.  ``state_patch``/``runtime_state`` is closed against the executable
contract in ``Dataset/tools/runtime_state_contract.py``: the ten declared
families, their consumed fields, boolean/enum/command value rules, facility
requester lists and non-negative numbers, plus the ``rule_id``/
``mechanism_id`` metadata fields.  ``runtime_state_patch_from_action`` reads
``state_patch`` when present and not ``None``, falls back to the
``runtime_state`` alias on an explicit null or an absent key, and raises only
when both are ``None`` (batch_generate.py ``runtime_state_patch_from_action``;
the audit row keeps the authored payload verbatim).  The pair choice is a
``oneOf`` of two branches, each requiring one key and pinning the shape of
both patch keys: a validated ``state_patch`` branch that also admits the
coexisting alias as an unread patch or a written ``null``, and the fallback
``runtime_state`` branch that pins ``state_patch`` to ``null``.  Key closure
uses ``unevaluatedProperties`` because branch properties do not close the
object under ``additionalProperties``.  The single-key forms fall out of the
branch requirements, and the two-null form that raises matches neither
branch.  The enum and command patterns match the producer's
``strip().lower()`` check while retaining the written string.  The source
guard applies only to the patch selected by the handler; a co-written alias
beside a non-null ``state_patch`` is not consumed.  ``set_weather`` overrides are closed against
``Dataset/tools/weather_fields.py``.

L3-specific declarations: ``terminal_feasibility`` keeps only the full
ten-key stamp written by
``Dataset/tools/regenerate_boundary_scenarios.py::enforce_uav_terminal_feasibility``
— the handler reads only ``landing_reference_enu_m`` from it, and no L3
producer or configuration path writes the single-reference variant the L2
window schema admits.  ``uav_corridor_validation`` keeps the two statuses the
same generator stamps via ``_replace_uav_action_waypoints``.

``state_families`` in a ``set_runtime_state`` result is declared
``{enum, minItems 1, uniqueItems}``: the members are the surviving patch
families and the handler writes them sorted
(``_handle_set_runtime_state``, ``state_families=sorted(patch)``).  The
grammar compiler has no keyword that can pin element order (no ``contains``;
``uniqueItems`` constrains membership only), so the ascending sort remains a
writer invariant outside this schema.

Action effects (keyframes, realised motion, semantic truth) are later-stage
concerns and are not encoded here.  These rows carry no integrity metadata
fields, so no publication-excluded path applies.  Registry wiring into
``structure_closure.py`` belongs to the integration stage; ``arm_actions`` is
currently validated by source projection (``field_scan_defined``).  This
module only exports the pure schema constructor and scans no data at import.
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

# Types registered by engine.execute; every other type is rejected by the
# handler map before an audit row exists.
_HANDLER_TYPES = (
    "move_entity", "set_visual_state", "set_runtime_state", "spawn_entity",
    "set_pedestrian_activity", "set_weather", "remove_entity", "capture_screenshot",
)
# _replace_uav_action_waypoints writes exactly these two statuses.
_CORRIDOR_STATUSES = (
    "low_altitude_waypoints_preserved_and_building_clear",
    "waypoints_repaired_for_building_clearance",
)
# enforce_uav_terminal_feasibility stamps this model on the full variant.
_TERMINAL_MODEL = "constant_speed_path_with_explicit_landing_reference_v2"
# engine.dispatch_source writes this exact omission status (engine.py L97-99).
_OMITTED_STATUS = "omitted_by_configuration"


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
        member = runtime_enum_string_rule(
            RUNTIME_STATE_ENUM_FIELDS[(family, "mode")])
        return {"oneOf": [
            member,
            {"type": "array", "minItems": 1,
             "items": runtime_enum_string_rule(
                 RUNTIME_STATE_ENUM_FIELDS[(family, "mode")])},
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


def _runtime_state_family_patch(family: str, *, consumed: bool) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **(runtime_state_source_guard_rule("engine_whole") if consumed else {}),
    }


def _runtime_state_patch(*, consumed: bool) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {family: _runtime_state_family_patch(family, consumed=consumed)
                       for family in sorted(RUNTIME_STATE_FIELDS)},
        **(runtime_state_source_guard_rule("engine_whole") if consumed else {}),
    }


def _visual_state() -> dict:
    # The L3 action payloads read so far write mode-only visual_state (the
    # authoring sources dispatch set_visual_state with ``{'mode': ...}``
    # payloads, and both handlers read only ``visual_state["mode"]``, with
    # the action-level ``mode`` as fallback).  ``lights_on`` appears only in
    # the authoring-layer convention documented on
    # ``spec_compiler.EntitySpec.visual_state``; it is kept as an optional
    # key and no L3 instance read carries it.
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
    # enforce_uav_terminal_feasibility is the only L3 writer and always
    # stamps the full ten-key variant (regenerate_boundary_scenarios.py).
    return {
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


def _move_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "waypoints_enu_m"],
        "properties": {
            "type": {"const": "move_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_move_entity reads velocity as
            # _to_float(action.get("velocity_mps", 1.0), 1.0), so an absent or
            # explicit-null value dispatches as 1.0.  The audit row keeps the
            # authored payload pre-coercion and pre-clamp (the engine clamps
            # the scheduled velocity to >= 0.1), so the written null is
            # declared; the generator rejects non-positive numbers at
            # authoring time.
            "velocity_mps": {"oneOf": [
                {"type": "number", "exclusiveMinimum": 0},
                {"type": "null"},
            ]},
            # _handle_move_entity raises on an empty waypoint list.
            "waypoints_enu_m": _waypoints(),
            # Read for pedestrian moves; normalised defaults "walking" /
            # "waiting" stay producer-side.
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
    # runtime_state_patch_from_action uses ``is None`` fallbacks: a present
    # non-null state_patch is validated and wins, an explicit null or an
    # absent key falls back to the runtime_state alias, and two nulls (or two
    # absences) raise.  The pair choice lives in one nested oneOf whose
    # branches each require one key and declare both patch keys, because a
    # branch may not re-type a key the base object already declares and the
    # base ``properties`` therefore declares neither.  Key closure comes from
    # ``unevaluatedProperties`` (keys validated inside the branches count as
    # evaluated in Draft 2020-12; ``additionalProperties`` would not see
    # them).  Each executable row satisfies exactly one branch: the primary
    # branch also admits the coexisting alias as a patch or an explicit null
    # (never read once the primary is non-null), the fallback branch pins the
    # primary to null, and the two-null form that raises matches neither.
    return {
        "type": "object", "unevaluatedProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "oneOf": [
            {
                "required": ["state_patch"],
                "properties": {
                    "state_patch": _runtime_state_patch(consumed=True),
                    # Coexisting alias is never read once the primary is
                    # non-null; the written null is likewise ignored.
                    "runtime_state": {"oneOf": [
                        _runtime_state_patch(consumed=False), {"type": "null"}]},
                },
            },
            {
                # Fallback branch: primary absent or an explicit null, the
                # alias mapping carries the validated patch.
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
            # _handle_set_runtime_state defaults to 0 and rejects negatives.
            "delay_ticks": {"type": "integer", "minimum": 0},
            # Written together by the scenario generator for scheduled
            # transitions; the dispatch path itself reads neither.
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
            # _handle_spawn_entity keeps the declared asset when absent.
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
            # Read for the spawned activity label and preserved fields.
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
            # weather_payload rejects keys outside WEATHER_OVERRIDE_FIELDS.
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
            # The handler derives capture_id without reading camera_id; the
            # generator writes the default "demo_high_overview".
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
        # set_runtime_state sorts the surviving patch families; the raw
        # grammar keeps the written family names within the declared set.
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
        "properties": {"status": {"const": _OMITTED_STATUS}},
    }


def _action_rule(names: tuple[str, ...]) -> dict:
    return {"oneOf": [_ACTION_BRANCHES[name]() for name in names]}


def _intervention_action_rule(names: tuple[str, ...]) -> dict:
    """The injection loop sends a mapping straight to the handler.  It does
    not run the source-script builder's action normalization."""
    return {"oneOf": [_intervention_action(name) for name in names]}


def _without_required_action_id(branch: dict) -> dict:
    return {**branch, "required": [
        name for name in branch["required"] if name != "action_id"]}


def _intervention_action(name: str) -> dict:
    branch = _without_required_action_id(_ACTION_BRANCHES[name]())
    if name == "set_visual_state":
        branch["required"] = [key for key in branch["required"]
                              if key != "visual_state"]
        branch["properties"]["mode"] = {"oneOf": [
            {"type": "string"}, {"type": "null"}]}
        branch["properties"]["visual_state"] = {"oneOf": [
            {"type": "object", "additionalProperties": False,
             "properties": {
                 "mode": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                 "lights_on": {"type": "boolean"},
             }},
            {"type": "null"},
        ]}
    elif name == "set_pedestrian_activity":
        branch["required"] = [key for key in branch["required"]
                              if key != "entity_id"]
        branch["properties"]["ped_id"] = _identifier()
        branch["anyOf"] = [{"required": ["entity_id"]},
                           {"required": ["ped_id"]}]
    return branch


def _intervention_result(name: str) -> dict:
    result = _result_ok(*_OK_RESULT_FIELDS[name])
    if name == "set_visual_state":
        # The handler writes an empty mode when neither action-level nor
        # nested mode is supplied; unlike the script builder, injection does
        # not require either spelling.
        result["properties"]["mode"] = {"type": "string"}
    return result


def _row(origin: str, action: dict, result: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["tick", "origin", "action", "result"],
        "properties": {
            "tick": {"type": "integer", "minimum": 0},
            "origin": {"const": origin},
            "action": action,
            "result": result,
        },
    }


def _type_branches() -> list[dict]:
    """One full row branch per action type: the dispatch path calls the
    handler of the row's own action.type, so the result shapes that can
    accompany a type are its handler's ok shape plus, on source-script rows
    only, the omission status (which keeps the authored payload of whatever
    type was omitted and is therefore not tied to a single type)."""
    return [
        _row(
            "source_script",
            _action_rule((name,)),
            {"oneOf": [_result_ok(*_OK_RESULT_FIELDS[name]), _omitted_result()]},
        )
        for name in _HANDLER_TYPES
    ] + [
        _row("intervention", _intervention_action_rule((name,)),
             _intervention_result(name))
        for name in _HANDLER_TYPES
    ]


def arm_actions_schema() -> dict:
    """Return the closed record schema for one ``arm_actions`` L3 JSONL row.

    The root splits into sixteen branches: eight source-script and eight
    intervention branches, each binding one action.type to the result shape
    its ``EpisodeStateEngine`` handler writes (``dispatch_source`` and the
    injection loop both dispatch through ``handler_map[action["type"]]``).
    Each source-script branch also admits the configuration-omission result.

    Branches with a producer basis but no current L3 instance are declared
    and remain unverified by instances: source-script ``set_weather`` and
    ``set_pedestrian_activity`` rows, the ``runtime_state`` patch alias, the
    co-written dual-key form and its explicit-null spellings
    (``{"state_patch": None, "runtime_state": <patch>}`` and the mirror),
    intervention top-level ``mode`` and ``ped_id`` aliases, an intervention
    ``set_visual_state`` with an absent, empty or null visual state,
    explicit-null ``velocity_mps``, scene-position spawn, default
    velocity, ``rotation_deg`` ``pitch_deg``/``roll_deg``, ``lights_on``
    inside a ``visual_state`` (read so far only in the mode-only spellings),
    empty or present ``set_weather`` overrides, and intervention rows of
    every type other than ``move_entity``, ``set_visual_state`` and
    ``set_runtime_state``.  Instance-verified branches: all six dispatched
    source-script types with their handler results, the L3 omission result,
    both ``uav_corridor_validation`` statuses, the full ``terminal_feasibility``
    stamp and the ``state_families`` closed membership (``minItems 1`` over
    the ten family names); the grammar has no keyword pinning element
    order, so the ascending sort stays a producer-side writer invariant
    (module header note), not a schema-verified claim.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_actions_l3.record.schema.json",
        "title": "L3 ARM raw action audit record",
        "oneOf": _type_branches(),
    }
