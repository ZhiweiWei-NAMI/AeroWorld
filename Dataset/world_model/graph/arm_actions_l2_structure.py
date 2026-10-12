"""Record structure for L2 ARM raw action audit rows.

Source family ``arm_actions``; profiles L2-1_v1, L2-1_v2, L2-2_v1, L2-2_v2,
L2-3_v1, L2-3_v2, L2-4_v1, L2-4_v2, L2-5_v1.  Producer:
``Dataset/tools/l2_window.py::execute_engine`` collects one audit row per
dispatched action (lines 148-158: ``audit.append({'tick': ..., 'origin': ...,
'action': deepcopy(action), 'result': deepcopy(result)})``) and ``_execute``
writes the list verbatim to ``raw/actions.jsonl`` (line 441).  A row is
``{tick, origin, action, result}``; the action is the deep-copied authored
payload, the result comes from the registered handler in
``Dataset/tools/batch_generate.py::EpisodeStateEngine``.

Branches couple the action type with the result that the handler registered
for that type returns: each row branch is the pair (one action-object shape,
the result shape of that handler).  The grammar therefore rejects the
cross-handler combinations the producer cannot write (for example a
``capture_screenshot`` action with a ``state_families`` result, or a
``set_visual_state`` action with a ``profile`` result).  ``origin`` splits the
record space the way production does:

- ``source_script`` rows are dispatched by the event-script interpreter for
  the eight handler types ``l2_window.py`` registers (lines 133-142 and
  160-161; the dispatcher raises ``unimplemented action type`` on anything
  else).  A scripted action at or after the fork tick whose ``action_id`` is
  listed in ``config['execution']['omit_action_ids']`` is not handed to any
  handler; its result is exactly ``{'status':
  'omitted_by_explicit_intervention'}`` (line 148-151), so each scripted type
  also carries an omission branch whose action keeps the authored shape.
- ``intervention`` rows come from ``config['execution']['actions']``
  (dispatched at line 194), which ``rules/l2_window.schema.json`` restricts to
  ``move_entity``, ``set_runtime_state`` and ``set_visual_state``; the
  dispatcher raises on any non-ok result (line 152-154), so an intervention
  row always carries the handler's ok shape.  The intervention action object
  follows the formal config schema (``execution.actions[].action``): it has
  one shared set of optional fields for all three types, including fields a
  selected handler ignores.  The schema excludes ``runtime_state`` aliases,
  schedule metadata and ``uav_corridor_validation``; its
  ``terminal_feasibility`` has one landing reference and its ``visual_state``
  has one nested ``mode``.  The config excludes ``state_patch`` from
  ``move_entity`` and ``set_visual_state``; all its other shared optional
  fields remain declared for those handler types.

The L2 dispatcher raises on any non-ok result, so a row's ``result`` always
carries the branch's ok shape or the omission status; per-branch key authority
is ``EpisodeStateEngine._handle_*``.  Optional keys the handler tolerates
(default velocity, scene-position spawn, ``runtime_state`` alias, delay-free
schedules) are declared optional even where the current L2 corpus always writes
them.  ``state_patch``/``runtime_state`` is closed against the executable
contract in ``Dataset/tools/runtime_state_contract.py``: the ten declared
families, their consumed fields, boolean/enum/command value rules, facility
requester lists and non-negative numbers, plus the ``rule_id``/``mechanism_id``
metadata fields.  Enum membership and case normalisation stay producer-side;
the grammar keeps the written string.  The consumed patch object carries the
``engine_whole`` source guard for forbidden semantic tokens and path strings;
an alias ignored beside ``state_patch`` does not carry that guard.
``state_families`` in an ok
``set_runtime_state`` result is the writer's ``sorted(patch)`` of validated
family names, so it is closed as a non-repeating array of the ten declared
family names (``batch_generate.py::_handle_set_runtime_state``, lines 2309
and 2287).  ``set_weather`` overrides are closed against
``Dataset/tools/weather_fields.py``; a ``null`` or empty ``profile`` string
falls back to the ``"clear"`` default the handler and ``weather_payload``
apply, so ``profile`` stays optional.

The source-script visual action has three disjoint branches: nonempty nested
mode, nonempty top-level fallback mode, or no effective mode.  An absent or
explicitly empty mode takes the fallback; if both spellings are absent or
empty, the handler writes ``mode: ""`` in an ok result.
Handler fallbacks the source-script object declares from the handler read
paths (none of which has an L2 instance on the fallback path):
``_handle_set_visual_state`` (batch_generate.py lines 2199-2200) reads the
mode from ``visual_state`` first and falls back to a top-level ``action['mode']``
string; on both origins, an action with neither spelling still returns ok
with the written ``mode: ""``.  ``_handle_set_pedestrian_activity``
(lines 2236-2239) falls back to
``action['ped_id']`` when ``entity_id`` is absent; ``_handle_set_weather``
(lines 2271-2272) and ``weather_payload`` default ``profile`` to ``"clear"``.

The ``runtime_state`` alias spelling: ``runtime_state_patch_from_action``
(batch_generate.py lines 956-968) reads ``action['state_patch']`` first and
falls back to ``action['runtime_state']`` only when ``state_patch`` is absent
or ``None``; the upstream event-script interpreter (``donghu_core/
event_script_interpreter.py`` lines 432-438) forwards each authored action
unchanged, so the two keys may legally coexist and either may be written as
explicit ``null``.  The grammar keeps the per-branch pair semantics: at least
one key must carry a valid patch mapping (``{"state_patch": None}`` alone
raises), a non-null ``state_patch`` mapping is always validated, and a
coexisting ``runtime_state`` is never read.  For that unread alias the only
declared forms are a valid patch mapping or the written ``null``; any other
value (for example a string or a mapping with unknown families) is ignored by
the producer but rejected here, because no production branch or contract
closes its shape — this is the unclosed path
``action.set_runtime_state.runtime_state`` for values outside the patch
contract when ``state_patch`` is non-null.  The alias has no L2 instance in
either form.  The same discipline applies to the visual-mode fallback: the
handler (line 2200) reads a nested ``visual_state.mode`` first and ignores a
top-level ``mode`` beside it, and no contract closes that ignored value, so
the top-level spelling is declared only on the fallback branch without a
nested mode and the coexistence is the unclosed path
``action.set_visual_state.mode`` beside a nested ``visual_state.mode``.

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
from Dataset.world_model.graph.structure_fragments import runtime_state_source_guard_rule

__all__ = ["arm_actions_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# Types registered by l2_window.execute_engine (lines 133-142); every other
# type is rejected by the dispatcher before an audit row exists.
_SOURCE_SCRIPT_TYPES = (
    "move_entity", "set_visual_state", "set_runtime_state", "capture_screenshot",
    "spawn_entity", "set_pedestrian_activity", "set_weather", "remove_entity",
)
# rules/l2_window.schema.json restricts intervention payloads to these three.
_INTERVENTION_TYPES = ("move_entity", "set_visual_state", "set_runtime_state")
# EpisodeStateEngine._replace_uav_action_waypoints writes exactly these two.
_CORRIDOR_STATUSES = (
    "low_altitude_waypoints_preserved_and_building_clear",
    "waypoints_repaired_for_building_clearance",
)
# enforce_uav_terminal_feasibility stamps this model on the full variant.
_TERMINAL_MODEL = "constant_speed_path_with_explicit_landing_reference_v2"


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
        return {"type": "string", "minLength": 1}
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        return {"oneOf": [
            {"type": "string", "minLength": 1},
            {"type": "array", "minItems": 1,
             "items": {"type": "string", "minLength": 1}},
        ]}
    if family == "facility_state" and field == "requester_ids":
        # The producer accepts an empty list; only non-string members fail.
        return {"type": "array", "items": {"type": "string", "minLength": 1}}
    if family == "facility_state" and field in ("capacity", "request_count"):
        return {"type": "number", "minimum": 0}
    if family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id"):
        return {"type": "string", "minLength": 1}
    if family == "communication_state" and field in (
            "availability", "link_availability"):
        return {"type": "number", "minimum": 0, "maximum": 1}
    if family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m"):
        return {"type": "number", "minimum": 0}
    if (family, field) in RUNTIME_STATE_ENUM_FIELDS:
        # Membership (case-insensitively) is checked by the producer contract;
        # the raw grammar keeps the written string.
        return {"type": "string", "minLength": 1}
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
    # validate_runtime_state_patch: every family value must be a non-empty
    # mapping and the whole patch must include at least one allowed family.
    # A coexisting runtime_state alias is ignored when state_patch is present.
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {family: _runtime_state_family_patch(family)
                       for family in sorted(RUNTIME_STATE_FIELDS)},
        **(runtime_state_source_guard_rule("engine_whole") if consumed else {}),
    }


def _visual_state() -> dict:
    # visual(action_id, entity_id, mode, **state) writes mode plus booleans
    # such as lights_on; the L2 corpus uses exactly these two keys.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode"],
        "properties": {
            "mode": {"type": "string", "minLength": 1},
            "lights_on": {"type": "boolean"},
        },
    }


def _visual_state_fallback() -> dict:
    # An absent or empty nested mode makes the handler try the top-level
    # spelling.  Unknown authored keys ignored by preserved_fields_from
    # still need separate authority.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"const": ""},
            "lights_on": {"type": "boolean"},
        },
    }


def _terminal_feasibility() -> dict:
    # enforce_uav_terminal_feasibility writes the full ten-key variant;
    # single-reference instances also occur on source-script rows.
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": ["landing_reference_enu_m"],
            "properties": {"landing_reference_enu_m": _number3()},
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
                "route_length_m": {"type": "number", "minimum": 0},
                "available_ticks": {"type": "integer", "minimum": 1},
                "required_speed_mps": {"type": "number", "minimum": 0},
                "max_speed_mps": _float_leaf(),
                "home_pose_origin_z_m": _float_leaf(),
                "landing_reference_enu_m": _number3(),
                "touchdown_altitude_tolerance_m": _float_leaf(),
                "touchdown_dwell_ticks": {"type": "integer", "minimum": 0},
            },
        },
    ]}


def _corridor_validation() -> dict:
    # Only _replace_uav_action_waypoints (source-script path) writes this key.
    return {
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


def _intervention_action_properties(action_type: str) -> dict[str, dict]:
    """Shared optional fields in rules/l2_window.schema.json action rows.

    The config schema has one action object for all three handler types;
    l2_window.dispatch copies that object into the audit row unchanged.
    ``state_patch`` is included only for its consuming handler.  The config
    rejects that key on move and visual actions while preserving every other
    shared optional field for all three types.
    """
    properties: dict[str, dict] = {
        "type": {"const": action_type},
        "action_id": _identifier(),
        "entity_id": _identifier(),
        "visual_state": {
            "type": "object", "additionalProperties": False,
            "required": ["mode"],
            "properties": {"mode": _identifier()},
        },
        "delay_ticks": {"type": "integer", "minimum": 0},
        "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
        "waypoints_enu_m": _waypoints(),
        "activity_type": _identifier(),
        "post_activity_type": _identifier(),
        "terminal_feasibility": {
            "type": "object", "additionalProperties": False,
            "required": ["landing_reference_enu_m"],
            "properties": {"landing_reference_enu_m": _number3()},
        },
    }
    if action_type == "set_runtime_state":
        properties["state_patch"] = _runtime_state_patch(consumed=True)
    return properties


def _move_entity_action(*, source_script: bool) -> dict:
    """move_entity action object.

    ``source_script=True``: authored script payload; the corridor validation
    stamp is written only on this path and both terminal_feasibility variants
    occur.  ``source_script=False``: use all representable optional fields
    from the shared action object in rules/l2_window.schema.json.
    """
    properties: dict[str, dict] = {
        "type": {"const": "move_entity"},
        "action_id": _identifier(),
        "entity_id": _identifier(),
        # _handle_move_entity reads velocity with default 1.0; the config
        # schema requires a positive number and the authored scripts always
        # write it, but the handler does not require the key.
        "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
        "waypoints_enu_m": _waypoints(),
        "activity_type": _identifier(),
        "post_activity_type": _identifier(),
    }
    if source_script:
        properties["terminal_feasibility"] = _terminal_feasibility()
        properties["uav_corridor_validation"] = _corridor_validation()
    else:
        properties = _intervention_action_properties("move_entity")
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "waypoints_enu_m"],
        "properties": properties,
    }


def _set_visual_state_action(*, source_script: bool) -> dict:
    """set_visual_state action object.

    ``source_script=True``: _handle_set_visual_state reads the mode from
    ``visual_state`` first and falls back to a top-level ``mode`` string
    (batch_generate.py lines 2199-2200).  The branches live in one nested
    oneOf (a branch may not re-type a key the base object already declares):
    branch 1 is the nonempty nested
    spelling and declares no top-level ``mode`` — a top-level ``mode``
    beside a nested one is ignored by the handler and no contract closes its
    value, so that coexistence stays outside the grammar (unclosed path in
    the module docstring); branch 2 is the fallback — a top-level mode with
    ``visual_state`` absent or carrying no effective nested ``mode`` (the handler
    treats an absent key as an empty mapping via
    ``action.get("visual_state") or {}``, line 2199, and still passes it
    through ``preserved_fields_from``, line 2223, which accepts empty
    mappings.  The declared shapes without an effective nested mode are an
    absent key, ``lights_on``, an empty mapping, or an explicit empty mode).
    Branch 1 requires a nonempty nested ``mode`` and branches 2 and 3 only
    permit an absent or empty nested mode, so the branches are disjoint
    without an
    ``unevaluatedProperties`` inside the branches; the outer object carries
    the property closure.
    Branch 3 has neither effective mode spelling and returns ok with
    ``mode: ""``.  ``source_script=False``: the shared config action object
    leaves ``visual_state`` optional and admits ``mode`` only inside it.
    """
    properties: dict[str, dict] = {
        "type": {"const": "set_visual_state"},
        "action_id": _identifier(),
        "entity_id": _identifier(),
    }
    if source_script:
        return {
            "type": "object",
            "unevaluatedProperties": False,
            "required": ["type", "action_id", "entity_id"],
            "oneOf": [
                {
                    "required": ["visual_state"],
                    "properties": {"visual_state": _visual_state()},
                },
                {
                    "required": ["mode"],
                    "properties": {
                        "mode": _identifier(),
                        "visual_state": _visual_state_fallback(),
                    },
                },
                {
                    # Either spelling may be absent or explicitly empty;
                    # neither produces an effective mode.
                    "properties": {
                        "mode": {"const": ""},
                        "visual_state": _visual_state_fallback(),
                    },
                },
            ],
            "properties": properties,
        }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "properties": _intervention_action_properties("set_visual_state"),
    }


def _set_runtime_state_action(*, source_script: bool) -> dict:
    """set_runtime_state action object.

    ``source_script=True``: production basis (batch_generate.py) is
    ``_handle_set_runtime_state`` resolving the patch through
    ``runtime_state_patch_from_action`` (lines 956-968), which reads
    ``action['state_patch']`` and falls back to ``action['runtime_state']``
    only when that value is ``None``; the event script interpreter forwards
    each authored action unchanged (donghu_core/event_script_interpreter.py
    lines 432-438), so the two keys may coexist and each may be written as
    explicit ``null``.  At least one key must carry a validated patch mapping:
    ``{"state_patch": None}`` alone raises ``set_runtime_state requires
    state_patch or runtime_state``.  A non-null ``state_patch`` mapping is
    always validated; a ``runtime_state`` beside it is never read, so only its
    two authority-backed forms (a valid patch mapping, or the written
    ``null``) are declared here and any other ignored alias value stays an
    unclosed path reported in the module docstring.  The pair logic lives in
    one nested oneOf whose branches each declare both patch keys, because a
    branch may not re-type a key the base object already declares.  The
    schedule metadata keys ride along on authored script actions.
    ``source_script=False``: the config object has a consumed
    ``state_patch`` alongside its other optional fields, but no alias,
    null-pair or schedule keys.
    """
    properties: dict[str, dict] = {
        "type": {"const": "set_runtime_state"},
        "action_id": _identifier(),
        "entity_id": _identifier(),
        "delay_ticks": {"type": "integer", "minimum": 0},
    }
    if source_script:
        return {
            "type": "object",
            "unevaluatedProperties": False,
            "required": ["type", "action_id", "entity_id"],
            "oneOf": [
                {
                    "required": ["state_patch"],
                    "properties": {
                        "state_patch": _runtime_state_patch(consumed=True),
                        # Coexisting alias is accepted and never read; a
                        # written null is likewise ignored.  No L2 instance
                        # uses the alias in either form.
                        "runtime_state": {"oneOf": [
                            _runtime_state_patch(consumed=False), {"type": "null"}]},
                    },
                },
                {
                    # Fallback branch: ``state_patch`` absent or an explicit
                    # null, the alias mapping carries the patch.
                    "required": ["runtime_state"],
                    "properties": {
                        "runtime_state": _runtime_state_patch(consumed=True),
                        "state_patch": {"type": "null"},
                    },
                },
            ],
            "properties": {
                **properties,
                "runtime_state_schedule_profile_id": _identifier(),
                "runtime_state_schedule_semantics": _identifier(),
            },
        }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "state_patch"],
        "properties": _intervention_action_properties("set_runtime_state"),
    }


def _spawn_entity_action() -> dict:
    # _handle_spawn_entity defaults the position to the declared scene
    # position and reads visual_state only as the spawned activity hint.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "properties": {
            "type": {"const": "spawn_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            "asset_id": _identifier(),
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
    # _handle_set_pedestrian_activity (batch_generate.py lines 2236-2239)
    # resolves the entity from ``entity_id`` falling back to ``ped_id``.
    # The scripted writer (regenerate_boundary_scenarios.ped_activity, line
    # 1542) only writes ``entity_id`` and the L2 corpus has no instance of
    # the fallback, so the alias branch is declared producer-evidenced but
    # unverified by instances.  The pair lives in a nested oneOf whose
    # branches each declare both spellings: branch 1 is the written form
    # (a ``ped_id`` beside ``entity_id`` is ignored by the handler and only
    # its written ``null`` form is declared), branch 2 is the fallback with
    # the top-level entity key absent — required makes the branches
    # disjoint, and the outer object carries the property closure.
    base = {
        "type": {"const": "set_pedestrian_activity"},
        "action_id": _identifier(),
        "activity_type": _identifier(),
    }
    return {
        "type": "object",
        "unevaluatedProperties": False,
        "required": ["type", "action_id"],
        "oneOf": [
            {
                "required": ["entity_id"],
                "properties": {"entity_id": _identifier(),
                               "ped_id": {"type": "null"}},
            },
            {
                "required": ["ped_id"],
                "properties": {"ped_id": _identifier()},
            },
        ],
        "properties": base,
    }


def _set_weather_action() -> dict:
    # _handle_set_weather (lines 2271-2272) applies
    # ``str(action.get("profile") or "clear")``: an absent, null or empty
    # profile falls back to the "clear" default, so the key stays optional
    # and the written value may be the explicit null.  The authored spelling
    # is kept (profile key membership and case normalisation are checked
    # producer-side by weather_payload); the row is written before any
    # result, so a row with an unknown profile cannot exist — but the raw
    # grammar keeps the string unconstrained, matching the L1 sibling.
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
            "profile": {"oneOf": [{"type": "string"}, {"type": "null"}]},
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


def _result_ok(*fields: str) -> dict:
    """One ok result shape from _append_action_result(status, tick=..., ...)."""
    rules: dict[str, dict] = {
        "status": {"const": "ok"},
        "tick": {"type": "integer", "minimum": 0},
        "entity_id": _identifier(),
        # The visual-state handler writes "" when neither mode spelling is
        # present; the raw value is always a string.
        "mode": {"type": "string"},
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


# One ok result shape per handler, from the _append_action_result calls in
# each _handle_* body; the per-type pairs below couple them with the matching
# action object.
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

_SOURCE_ACTION_BRANCHES = {
    "move_entity": lambda: _move_entity_action(source_script=True),
    "set_visual_state": lambda: _set_visual_state_action(source_script=True),
    "set_runtime_state": lambda: _set_runtime_state_action(source_script=True),
    "spawn_entity": _spawn_entity_action,
    "set_pedestrian_activity": _set_pedestrian_activity_action,
    "set_weather": _set_weather_action,
    "remove_entity": _remove_entity_action,
    "capture_screenshot": _capture_screenshot_action,
}

_INTERVENTION_ACTION_BRANCHES = {
    "move_entity": lambda: _move_entity_action(source_script=False),
    "set_visual_state": lambda: _set_visual_state_action(source_script=False),
    "set_runtime_state": lambda: _set_runtime_state_action(source_script=False),
}


def _omitted_result() -> dict:
    # l2_window.py lines 148-151: the dispatcher writes exactly this mapping
    # and no handler result keys survive.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"const": "omitted_by_explicit_intervention"}},
    }


def _source_row(action_type: str) -> dict:
    """Pair a source action with its handler result or explicit omission."""
    action = _SOURCE_ACTION_BRANCHES[action_type]()

    def row(action_schema: dict, result_schema: dict) -> dict:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["tick", "origin", "action", "result"],
            "properties": {
                "tick": {"type": "integer", "minimum": 0},
                "origin": {"const": "source_script"},
                "action": action_schema,
                "result": result_schema,
            },
        }

    if action_type == "set_visual_state":
        # The three action branches have a nonempty nested mode, a nonempty
        # top-level mode, or no effective mode.  Only the third returns an
        # empty result mode.
        ok_rows = []
        for index, case in enumerate(action["oneOf"]):
            result = _OK_RESULT_BRANCHES[action_type]()
            result["properties"]["mode"] = (
                {"const": ""} if index == 2 else _identifier()
            )
            ok_rows.append(row({**action, "oneOf": [case]}, result))
    else:
        ok_rows = [row(action, _OK_RESULT_BRANCHES[action_type]())]
    return {"oneOf": [*ok_rows, row(action, _omitted_result())]}


def _intervention_rows(action_type: str) -> list[dict]:
    """Pair each config-schema action shape with its handler ok result.

    A visual action may omit ``visual_state`` under the config schema.  The
    handler then succeeds with ``mode: ""``; a present ``visual_state`` has
    a required nonempty mode and produces a nonempty result mode.
    """
    action = _INTERVENTION_ACTION_BRANCHES[action_type]()

    def row(action_schema: dict, result_schema: dict) -> dict:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["tick", "origin", "action", "result"],
            "properties": {
                "tick": {"type": "integer", "minimum": 0},
                "origin": {"const": "intervention"},
                "action": action_schema,
                "result": result_schema,
            },
        }

    if action_type != "set_visual_state":
        return [row(action, _OK_RESULT_BRANCHES[action_type]())]

    with_visual = {**action, "required": [*action["required"], "visual_state"]}
    without_visual = {
        **action,
        "properties": {key: rule for key, rule in action["properties"].items()
                       if key != "visual_state"},
    }
    empty_result = _OK_RESULT_BRANCHES[action_type]()
    empty_result["properties"]["mode"] = {"const": ""}
    nonempty_result = _OK_RESULT_BRANCHES[action_type]()
    nonempty_result["properties"]["mode"] = _identifier()
    return [
        row(with_visual, nonempty_result),
        row(without_visual, empty_result),
    ]


def arm_actions_schema() -> dict:
    """Return the declared record schema for one ``arm_actions`` L2 JSONL row.

    The record space splits by ``origin`` into per-type row branches that
    couple the action object with the result shape its handler returns.  All
    eight source-script handler types and all three intervention types are
    declared even where the current L2 corpus has no instance for a branch;
    those branches stay marked ``尚无实例验证`` in the hand-off: source-script
    ``set_weather``, ``set_pedestrian_activity`` and ``remove_entity`` rows,
    the handler fallback spellings (top-level or absent visual ``mode``, ``ped_id``,
    null/empty weather ``profile``), the absent-``visual_state`` intervention
    ``set_visual_state`` row (schema-legal since the config schema does not
    require the key), the ``runtime_state`` patch alias (alone
    or beside ``state_patch``), the explicit ``null`` form of either patch
    key, delay-free ``set_runtime_state``, scene-position spawn, default
    velocity, ``visual_state`` on ``spawn_entity``, empty ``set_weather``
    overrides.  Both origins can write a visual result ``mode: ""`` when
    their action has no mode spelling.
    Both ``terminal_feasibility`` variants are instance-verified on
    source-script rows.  The formal config schema rejects ``state_patch``
    on move and visual intervention actions; their other shared optional
    fields remain declared on each action branch.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_actions_l2.record.schema.json",
        "title": "L2 ARM raw action audit record",
        "oneOf": [
            *(branch for name in _SOURCE_SCRIPT_TYPES
              for branch in _source_row(name)["oneOf"]),
            *(branch for name in _INTERVENTION_TYPES
              for branch in _intervention_rows(name)),
        ],
    }
