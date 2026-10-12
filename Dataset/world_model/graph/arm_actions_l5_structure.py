"""Closed record structure for L5 ARM raw action audit rows.

Source family ``arm_actions``; profiles L5-1_v1, L5-1_v2, L5-1_v3, L5-2_v1,
L5-2_v2, L5-3_v1, L5-3_v2.  Producer: ``Dataset/tools/l5_arm_pipeline.py``
collects the audit rows built by ``Dataset/tools/l5_arm_engine.py`` and writes
the window slice (``start <= tick <= end``) verbatim to ``raw/actions.jsonl``.
A row is ``{tick, sequence, origin, action, result, effect}``; ``sequence`` is
``len(audit)`` at append time, ``action`` is the deep-copied authored payload,
``result`` comes from the registered ``EpisodeStateEngine._handle_*`` in
``Dataset/tools/batch_generate.py`` and ``effect`` is assembled by the
``dispatch`` closure in ``l5_arm_engine.py``.

``origin`` splits the record space the way production does:

- ``source_script`` rows are dispatched by the event-script interpreter and
  may carry ``{'status': 'omitted_by_explicit_intervention'}`` when the arm
  window omits their ``action_id`` at or after the fork tick.  The omission
  check runs before the handler, so the row keeps the base effect keys only
  — with one exception the dispatch closure derives from the action type,
  not the result: an omitted ``set_weather`` row still carries real
  ``weather_before``/``weather_after`` snapshots (l5_arm_engine.py builds
  them for every ``set_weather`` kind, handler or not), so omitted weather
  rows get their own branch.  Omitted rows always carry
  ``keyframes_changed: false`` — the omission path skips the only mutator
  between the before/after keyframe reads.
- ``intervention`` rows come from ``config['execution']['actions']``, whose
  steps ``rules/l5_arm_config.schema.json`` bound to ``tick`` 1..899 and an
  ``action`` of ``type`` ``set_weather`` with required non-empty
  ``overrides``; source-script weather actions can carry empty overrides.
  The dispatcher raises on any non-ok result, so intervention rows always
  carry the handler's ok shape plus the weather effect keys.

``set_pedestrian_activity`` is registered in ``l5_arm_engine.dispatch`` and
its handler (``batch_generate._handle_set_pedestrian_activity``) appends an
activity schedule without ``stop_motion``, but the effect assembly reads
``schedule['stop_motion']`` for every ok row of the three schedule-bearing
types (l5_arm_engine.py: ``schedule['type']=='activity' and
schedule['stop_motion'] is True``).  An ok ``set_pedestrian_activity`` row
therefore raises ``KeyError`` before the audit append; the row shape is
unwritable until the producer is fixed.  The grammar declares the type only
where it is reachable — the omission stamp (checked before the handler) —
and keeps no ok branch for it.  ``spawn_entity`` rows are declared: the
handler appends an ``activity`` schedule without ``stop_motion``, but
``spawn_entity`` is not one of the three types whose effect reads it, so
the row is writable; the 14 authored L5 spawns dispatch before the window
filter, so no window row exists yet.

The effect object is part of the audit row, so its structure is declared
here even though action semantics belong to later business verification:
the three base keys are always written, ``weather_before``/``weather_after``
are non-null exactly for ``set_weather`` rows (including omitted ones) and
carry the ``Dataset/tools/weather_fields.py`` payload shape, and ok rows of
``move_entity`` and ``set_visual_state`` add ``added_schedule``,
``keyframes_after`` and ``previous_motion_owner`` (copy of the last
schedule, of ``engine.keyframes[entity]`` and of the motion-ownership
entry).  The effect assembly queries ``set_pedestrian_activity`` on the
same three-type condition, but its rows cannot exist (producer bug above),
so the scheduled effect shape stays bound to the two writable types.
``move_entity`` schedules additionally carry ``motion_model``.  ``added_schedule.tick`` mirrors ``result.scheduled_tick``
for ``move`` schedules and the row tick for ``activity`` schedules;
``keyframes_after`` entries are ``[tick, position_enu_m, state_label]``
triples with strictly increasing integer ticks and a free-form label
string.  The motion-ownership entry gains ``stopping_activity: true``
exactly when the owning schedule is an activity with ``stop_motion``
true — writable only through ``set_visual_state`` stop labels, because the
``set_pedestrian_activity`` row is unwritable (above) — so the field is
declared as an optional const-true member.  ``weather`` and the weather
overrides are closed against ``Dataset/tools/weather_fields.py``;
``state_patch`` is closed against the executable contract in
``Dataset/tools/runtime_state_contract.py``: the ten declared families,
their consumed fields, boolean/enum/command value rules, facility requester
lists and non-negative numbers, plus the ``rule_id``/``mechanism_id``
metadata fields.  The grammar accepts the producer's raw enum/command
spellings with the shared Python-strip rule and checks the consumed patch
with the engine's recursive source guard.  ``state_families`` in an ok
``set_runtime_state`` result is the writer's ``sorted(patch)`` of validated
family names over a patch with at least one field, so it is closed as a
non-repeating array of the ten declared family names.

The ``source_script`` runtime rows require a non-null ``state_patch``:
the current L5 source scripts write that key for every ``set_runtime_state``
action, and the L5 branch entries do not introduce an alias-only action.
For ok rows, ``l5_arm_pipeline.action_effects`` also reads
``action['state_patch']`` directly.  Omitted rows bypass that read, but keep
the same authored source payload.  The ``runtime_state`` alias stays declared
beside the primary patch because the handler resolves a both-keys row with
``state_patch`` winning (``runtime_state_patch_from_action``) — JSON Schema
cannot express that precedence, so the double-key spelling stays admitted
under the ``state_patch`` branch — but an alias without a usable primary
patch has no producer path and is rejected.

Optional keys the handlers tolerate (default velocity, missing
``camera_id``, default ``activity_type``, default weather profile,
scene-position spawn) are declared optional even where the current L5
corpus always writes them; the default weather profile and the empty
``overrides`` object are additionally unreachable on intervention rows
because the config schema requires both; source-script actions admit empty
overrides.  ``visual_state`` payloads keep their open
builder contract: ``preserved_fields_from`` copies every
``PRESERVED_ENTITY_FIELDS`` member it finds in the payload (or under its
``initial_state``/``visual_state`` carriers) onto the entity, and
``_runtime_state_patches_from_carrier`` additionally accepts the ten
runtime-state families plus a nested ``runtime_state`` mapping inside
``visual_state`` — all validated by the executable contract before a row
exists, so the payload schema names ``mode``, ``lights_on``, the ten
families and the two carriers, and closes every other key's value with the
same leaf rules.  ``continuity_repair`` on
``move_entity`` is authored into script payloads by
``Dataset/tools/regenerate_boundary_scenarios.py`` but tick-0 dispatches fall
before the window filter, so it is declared from the writer with no
dispatched L5 instance.

The L2 sibling ``arm_actions_l2_structure.py`` keeps its own module: the L5
row root carries ``sequence`` and ``effect``, which the L2 producer does not
write.  Registry wiring into ``structure_closure.py`` belongs to the
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
_ROW_ID = "https://aeroworld.local/schema/arm_actions_l5.record.schema.json"

# regenerate_boundary_scenarios.py stamps exactly these two corridor
# statuses onto authored move payloads.
_CORRIDOR_STATUSES = (
    "low_altitude_waypoints_preserved_and_building_clear",
    "waypoints_repaired_for_building_clearance",
)
# regenerate_boundary_scenarios.py stamps this model name on the ten-key form.
_TERMINAL_MODEL = "constant_speed_path_with_explicit_landing_reference_v2"

# l5_arm_engine.dispatch writes the three base keys on every row.
_BASE_EFFECT_KEYS = ("keyframes_changed", "weather_before", "weather_after")
_SCHEDULE_EFFECT_KEYS = _BASE_EFFECT_KEYS + (
    "added_schedule", "keyframes_after", "previous_motion_owner",
)
# WEATHER_PROFILES payloads carry exactly these keys (condition plus 13
# numeric/boolean fields); the "light smoke" profile is the only one that adds
# a key, "dust", declared below as the one optional member.
_WEATHER_KEYS = (
    "condition", "rain", "wetness", "fog", "fog_density", "wind_speed",
    "wind_direction_deg", "visibility_m", "visibility", "temperature_c",
    "illumination_lux", "hazard_concentration_ppm", "hazard_radius_m",
    "hazard_source_active",
)
# l5_arm_engine motion_model keys for move schedules.
_MOTION_KEYS = (
    "configured_speed_mps", "segment_rounding", "nominal_duration_ticks",
    "nominal_completion_tick", "continuous_lower_bound_ticks", "segments",
)
# Keys of a motion-ownership entry; dispatch adds stopping_activity only on
# entries created by a stop-motion activity schedule.
_OWNER_KEYS = (
    "action_id", "origin", "dispatch_tick", "scheduled_tick",
    "nominal_completion_tick",
)
# The event-script producer treats both aliases as visibility spellings.
_WEATHER_OVERRIDE_ALIASES = ("fog", "visibility")


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _number3() -> dict:
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _waypoints() -> dict:
    return {"type": "array", "minItems": 1, "items": _number3()}


def _weather_object() -> dict:
    """One ``engine.weather`` snapshot as written by
    ``l5_arm_engine.dispatch`` (``copy.deepcopy(engine.weather)``).

    The snapshot is ``weather_payload(profile, overrides)``: a
    ``WEATHER_PROFILES`` payload with ``condition`` stamped and overrides
    merged.  Every profile payload carries the 13 numeric/boolean fields plus
    ``condition``; ``light smoke`` alone adds ``dust``, so the object is
    closed with one optional named member instead of a loose pattern.
    """
    return {
        "type": "object", "additionalProperties": False,
        "required": list(_WEATHER_KEYS),
        "properties": {
            "condition": _identifier(),
            "rain": {"type": "number"}, "wetness": {"type": "number"},
            "fog": {"type": "number"}, "fog_density": {"type": "number"},
            "wind_speed": {"type": "number"},
            "wind_direction_deg": {"type": "number"},
            "visibility_m": {"type": "number"},
            "visibility": {"type": "number"},
            "temperature_c": {"type": "number"},
            "illumination_lux": {"type": "number"},
            "hazard_concentration_ppm": {"type": "number"},
            "hazard_radius_m": {"type": "number"},
            "hazard_source_active": {"type": "boolean"},
            # Only the "light smoke" profile payload carries this key.
            "dust": {"type": "number"},
        },
    }


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


def _runtime_state_family_patch(family: str) -> dict:
    fields = sorted(RUNTIME_STATE_CONSUMED_FIELDS[family] | RUNTIME_STATE_METADATA_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {name: _runtime_state_field_rule(family, name)
                       for name in fields},
        **runtime_state_source_guard_rule("engine_whole"),
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


def _builder_leaf_value() -> dict:
    """Value rule for a builder-written key without a named contract."""
    return {"oneOf": [
        {"type": "string", "minLength": 1},
        {"type": "boolean"},
        {"type": "number"},
        {"type": "array", "minItems": 1,
         "items": {"type": "string", "minLength": 1}},
    ]}


def _runtime_state_carrier() -> dict:
    """One nested runtime-state carrier level (``initial_state`` or a nested
    ``visual_state`` inside a visual payload): preserved builder leaves plus
    the ten family patches plus the ``runtime_state`` mapping — exactly the
    members ``preserved_fields_from`` and
    ``_runtime_state_patches_from_carrier`` read one level deep."""
    return {
        "type": "object",
        "properties": {
            **{family: _runtime_state_family_patch(family)
               for family in sorted(RUNTIME_STATE_FIELDS)},
            "runtime_state": _runtime_state_patch(),
        },
        "patternProperties": {
            "^(?!" + "|".join(sorted(RUNTIME_STATE_FIELDS)) + "$|runtime_state$)"
            "[A-Za-z][A-Za-z0-9_]*$": _builder_leaf_value(),
        },
        "additionalProperties": False,
    }


def _visual_state() -> dict:
    """``visual(action_id, entity_id, mode, **state)`` payloads.

    ``_handle_set_visual_state`` reads ``mode`` as the activity label and
    hands the payload to ``preserved_fields_from``: every
    ``PRESERVED_ENTITY_FIELDS`` member found in the payload (or one level
    down, under the ``initial_state``/``visual_state`` carriers) is copied
    onto the entity, and ``_runtime_state_patches_from_carrier`` accepts the
    ten runtime families plus a nested ``runtime_state`` mapping at the top
    level and inside both carriers, each validated by the executable
    contract before a row can exist.  The key set is therefore closed by
    producer contract, not by the corpus key set: named members keep their
    value rules and every other builder-written key falls to the contract
    leaf alternatives.
    """
    return {
        "type": "object",
        "required": ["mode"],
        "properties": {
            "mode": {"type": "string", "minLength": 1},
            "lights_on": {"type": "boolean"},
            # Runtime families the payload may carry directly; each is a
            # full validated family patch with at least one field.
            **{family: _runtime_state_family_patch(family)
               for family in sorted(RUNTIME_STATE_FIELDS)},
            "initial_state": _runtime_state_carrier(),
            "visual_state": _runtime_state_carrier(),
            "runtime_state": _runtime_state_patch(),
        },
        "patternProperties": {
            "^(?!mode$|lights_on$|initial_state$|visual_state$|runtime_state$|"
            + "|".join(sorted(RUNTIME_STATE_FIELDS)) + "$)"
            "[A-Za-z][A-Za-z0-9_]*$": _builder_leaf_value(),
        },
        "additionalProperties": False,
    }


def _terminal_feasibility() -> dict:
    # regenerate_boundary_scenarios.py writes the full ten-key variant; the
    # single-reference variant stays declarable from the writer's contract.
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


def _move_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "waypoints_enu_m"],
        "properties": {
            "type": {"const": "move_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_move_entity reads velocity with default 1.0; every L5
            # producer writes it, the engine does not require it.  The audit
            # row keeps the authored payload pre-coercion and the intervention
            # config producer admits any number.
            "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
            "waypoints_enu_m": _waypoints(),
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
            # Authored by regenerate_boundary_scenarios.py for entry moves;
            # tick-0 dispatches fall before the L5 window filter, so no L5
            # audit row carries it.
            "continuity_repair": {
                "type": "object", "additionalProperties": False,
                "required": ["policy", "original_start_enu_m", "start_gap_m"],
                "properties": {
                    "policy": {"const":
                               "replace_first_waypoint_with_current_entity_position_v1"},
                    "original_start_enu_m": _number3(),
                    "start_gap_m": {"type": "number", "minimum": 0},
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
    """``set_runtime_state`` payloads as the producer chain admits them.

    Current L5 source scripts write ``state_patch`` for every action of this
    type, including one later omitted by the dispatch closure.  For ok rows,
    ``l5_arm_pipeline.action_effects`` also reads the key directly.  The
    ``runtime_state`` alias stays declared beside the primary patch because
    ``runtime_state_patch_from_action`` resolves a both-keys row with
    ``state_patch`` winning — JSON Schema cannot express that precedence, so
    the double-key spelling is admitted under this branch — while the
    alias-only spelling is rejected because no current L5 source writes it.
    """
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "state_patch"],
        "properties": {
            "type": {"const": "set_runtime_state"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_set_runtime_state reads delay with default 0 and
            # requires non-negative; the effective tick must stay within the
            # episode duration.
            "delay_ticks": {"type": "integer", "minimum": 0},
            # Authored schedule provenance written by the scenario builder.
            "runtime_state_schedule_profile_id": _identifier(),
            "runtime_state_schedule_semantics": _identifier(),
            "state_patch": _runtime_state_patch(),
            # runtime_state_patch_from_action accepts the alias spelling;
            # state_patch wins when both are present.
            "runtime_state": _runtime_state_patch(),
        },
    }


def _spawn_visual_state() -> dict:
    # _handle_spawn_entity reads only visual_state["mode"] as the spawn
    # activity label; the preserved-fields/carrier contract of the visual
    # handler does not run on spawn rows.  No L5 builder writes visual_state
    # on a spawn at all, so every key stays optional and the mode-required
    # contract of _visual_state does not apply here.
    return {
        "type": "object", "additionalProperties": False,
        "properties": {"mode": {"type": "string", "minLength": 1}},
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
            "visual_state": _spawn_visual_state(),
        },
    }


def _set_pedestrian_activity_action() -> dict:
    """``set_pedestrian_activity`` payloads.

    No ok row of this type is writable today: the engine effect assembly
    reads ``schedule['stop_motion']``, which this handler's schedule lacks
    (producer bug in the module docstring).  The shape stays declared for the
    omission branch — the omission stamp is applied before the handler — so
    an omitted ``set_pedestrian_activity`` row keeps its authored payload.
    ``ped_id`` is the accepted alias for ``entity_id`` in the handler.  The
    dispatcher and handler both prefer ``entity_id`` if both are present, so
    the keys may coexist in an omitted authored payload.
    """
    return {
        "type": "object", "additionalProperties": False,
        "anyOf": [
            {"required": ["type", "action_id", "entity_id"]},
            {"required": ["type", "action_id", "ped_id"]},
        ],
        "properties": {
            "type": {"const": "set_pedestrian_activity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            "ped_id": _identifier(),
            # _handle_set_pedestrian_activity defaults to "waiting".
            "activity_type": _identifier(),
        },
    }


def _set_weather_action(*, intervention: bool = False) -> dict:
    """``set_weather`` payloads.

    ``profile`` is required: ``rules/l5_arm_config.schema.json`` requires it
    on every intervention action and every L5 script row writes it.
    ``overrides`` is required and non-empty on intervention rows (config
    schema ``minProperties`` 1).  The source-script handler also accepts an
    empty object, while every current L5 script action writes the key.  Keys close
    against ``WEATHER_OVERRIDE_FIELDS`` (boolean flag, aliases, numeric
    fields); value types and alias conflicts stay producer-side checks in
    ``weather_payload``.
    """
    overrides = {name: {"type": "number"}
                 for name in sorted(WEATHER_NUMERIC_FIELDS | set(WEATHER_ALIASES))}
    for name in sorted(WEATHER_BOOLEAN_FIELDS):
        overrides[name] = {"type": "boolean"}
    override_rule = {
        "type": "object", "additionalProperties": False,
        "properties": overrides,
    }
    if intervention:
        override_rule["minProperties"] = 1
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "profile", "overrides"],
        "properties": {
            "type": {"const": "set_weather"},
            "action_id": _identifier(),
            # _handle_set_weather defaults to "clear" and checks membership in
            # WEATHER_PROFILES; the grammar keeps the written string.
            "profile": _identifier(),
            "overrides": override_rule,
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
            # _handle_capture_screenshot falls back to capture_id when absent.
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

# Types registered by l5_arm_engine; every other type is rejected by the
# dispatcher before an audit row exists.
_SOURCE_SCRIPT_TYPES = tuple(_ACTION_BRANCHES)
# rules/l5_arm_config.schema.json restricts intervention payloads to this one.
_INTERVENTION_TYPES = ("set_weather",)


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
        "state_families": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"enum": sorted(RUNTIME_STATE_FIELDS)},
        },
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
    # set_pedestrian_activity has no ok shape: the handler's activity
    # schedule lacks stop_motion, which the effect assembly reads (producer
    # bug in the module docstring).  Its action stays declared for the
    # omission branch only.
    "set_weather": lambda: _result_ok("tick", "profile"),
    "remove_entity": lambda: _result_ok("tick", "effective_tick", "entity_id"),
    "capture_screenshot": lambda: _result_ok("tick", "capture_id"),
}


def _omitted_result() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"const": "omitted_by_explicit_intervention"}},
    }


def _motion_model() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": list(_MOTION_KEYS),
        "properties": {
            "configured_speed_mps": {"type": "number", "exclusiveMinimum": 0},
            "segment_rounding": {"const":
                                 "ceil_per_nonzero_segment"},
            "nominal_duration_ticks": {"type": "integer", "minimum": 0},
            "nominal_completion_tick": {"type": "integer", "minimum": 1},
            "continuous_lower_bound_ticks": {"type": "number", "minimum": 0},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["distance_m", "ticks", "velocity_mps"],
                    "properties": {
                        "distance_m": {"type": "number"},
                        "ticks": {"type": "integer", "minimum": 1},
                        "velocity_mps": _number3(),
                    },
                },
            },
        },
    }


def _move_schedule() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": [
            "type", "tick", "action_id", "activity_type", "post_activity_type",
            "source", "source_event_tick", "waypoints_enu_m", "velocity_mps",
            "start_pos_enu",
        ],
        "properties": {
            "type": {"const": "move"},
            "tick": {"type": "integer", "minimum": 0},
            "action_id": _identifier(),
            "activity_type": {"oneOf": [_identifier(), {"type": "null"}]},
            "post_activity_type": {"oneOf": [_identifier(), {"type": "null"}]},
            "source": {"const": "event_script.move_entity"},
            "source_event_tick": {"type": "integer", "minimum": 0},
            "waypoints_enu_m": _waypoints(),
            "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
            "start_pos_enu": _number3(),
        },
    }


def _activity_schedule() -> dict:
    """An appended ``activity`` schedule as written by the two producers.

    ``set_visual_state`` writes ``start_pos_enu`` and ``stop_motion``;
    ``spawn_entity`` does not (its handler appends only the five base keys).
    ``source`` names the writer, so the shape closes on the writer constant.
    Branches are self-contained objects (the compiler evaluates each oneOf
    branch as an independent term).
    """
    return {"oneOf": [
        {
            "type": "object", "additionalProperties": False,
            "required": ["type", "tick", "activity_type", "source",
                         "start_pos_enu", "stop_motion", "action_id"],
            "properties": {
                "type": {"const": "activity"},
                "tick": {"type": "integer", "minimum": 0},
                "activity_type": _identifier(),
                "source": {"const": "event_script.set_visual_state"},
                "start_pos_enu": _number3(),
                "stop_motion": {"type": "boolean"},
                "action_id": _identifier(),
            },
        },
        {
            "type": "object", "additionalProperties": False,
            "required": ["type", "tick", "activity_type", "source",
                         "action_id"],
            "properties": {
                "type": {"const": "activity"},
                "tick": {"type": "integer", "minimum": 0},
                "activity_type": _identifier(),
                "source": {"const": "event_script.spawn_entity"},
                "action_id": _identifier(),
            },
        },
    ]}


def _previous_motion_owner() -> dict:
    """A motion-ownership entry as stamped by ``l5_arm_engine.dispatch``.

    Move schedules write the five base keys; a stop-motion activity schedule
    adds ``stopping_activity: true`` (engine: ``schedule['stop_motion']
    is True``), which only ``set_visual_state`` stop labels can produce while
    the ``set_pedestrian_activity`` row is unwritable (producer bug above).
    """
    return {
        "type": "object", "additionalProperties": False,
        "required": list(_OWNER_KEYS),
        "properties": {
            "action_id": _identifier(),
            # Only move/visual actions stamp ownership.  L5 interventions
            # dispatch set_weather alone, so they cannot own earlier motion.
            "origin": {"const": "source_script"},
            "dispatch_tick": {"type": "integer", "minimum": 0},
            "scheduled_tick": {"type": "integer", "minimum": 0},
            "nominal_completion_tick": {"type": "integer", "minimum": 0},
            # Present exactly on entries created by a stop-motion activity.
            "stopping_activity": {"const": True},
        },
    }


def _keyframes_after() -> dict:
    # engine.keyframes values: sorted list of [tick, position_enu_m, label].
    return {
        "type": "array", "minItems": 1, "uniqueItems": True,
        "items": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "prefixItems": [
                {"type": "integer", "minimum": 0},
                _number3(),
                {"type": "string"},
            ],
            "items": False,
        },
    }


def _base_effect(*, omitted: bool = False) -> dict:
    # dispatch writes weather_before/weather_after as None unless the row is
    # a set_weather action, which gets its own effect shape below.
    return {
        "type": "object", "additionalProperties": False,
        "required": list(_BASE_EFFECT_KEYS),
        "properties": {
            "keyframes_changed": {"const": False} if omitted else {"type": "boolean"},
            "weather_before": {"const": None},
            "weather_after": {"const": None},
        },
    }


def _weather_effect(*, omitted: bool = False) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["keyframes_changed", "weather_before", "weather_after"],
        "properties": {
            "keyframes_changed": {"const": False} if omitted else {"type": "boolean"},
            "weather_before": _weather_object(),
            "weather_after": _weather_object(),
        },
    }


def _schedule_effect(with_motion_model: bool) -> dict:
    """Effect shape for ok rows that append a schedule (dispatch writes
    ``added_schedule``, ``keyframes_after`` and ``previous_motion_owner``
    for move_entity and set_visual_state; ``set_pedestrian_activity`` is
    queried on the same condition but cannot produce a row).
    ``motion_model`` exists exactly when the appended schedule is a move."""
    rule = {
        "type": "object", "additionalProperties": False,
        "required": list(_SCHEDULE_EFFECT_KEYS),
        "properties": {
            "keyframes_changed": {"type": "boolean"},
            "weather_before": {"const": None},
            "weather_after": {"const": None},
            "added_schedule": {"oneOf": [_move_schedule(),
                                         _activity_schedule()]},
            "keyframes_after": _keyframes_after(),
            "previous_motion_owner": {
                "oneOf": [_previous_motion_owner(), {"type": "null"}]},
        },
    }
    if with_motion_model:
        rule["properties"]["motion_model"] = _motion_model()
        rule["required"] = list(rule["required"]) + ["motion_model"]
    return rule


# The four distinct effect shapes l5_arm_engine.dispatch writes, bound to
# action types by _TYPE_EFFECT at the row-variant level.
_EFFECT_SHAPES = {
    "set_weather": _weather_effect,
    "plain": _base_effect,
    "move_entity": lambda: _schedule_effect(True),
    "scheduled": lambda: _schedule_effect(False),
}
# action type -> effect shape it produces.
_TYPE_EFFECT = {
    "move_entity": "move_entity",
    "set_visual_state": "scheduled",
    "set_pedestrian_activity": "scheduled",
    "set_runtime_state": "plain",
    "spawn_entity": "plain",
    "remove_entity": "plain",
    "capture_screenshot": "plain",
    "set_weather": "set_weather",
}


def _row(origin: str, action: dict, result: dict, effect: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["tick", "sequence", "origin", "action", "result", "effect"],
        "properties": {
            "tick": {"type": "integer", "minimum": 0},
            "sequence": {"type": "integer", "minimum": 0},
            "origin": {"const": origin},
            "action": action,
            "result": result,
            "effect": effect,
        },
    }


def arm_actions_schema() -> dict:
    """Return the closed record schema for one ``arm_actions`` L5 JSONL row.

    The source-script space is one complete-row variant per dispatched action
    type: the producer binds the result and effect shapes to the action type
    deterministically in ``l5_arm_engine.dispatch``, so the variant's
    ``action.type`` const keeps exactly one branch per row.  Two exceptions
    come from the producer chain itself.  An ok ``set_pedestrian_activity``
    row has no writable shape (the handler's schedule lacks the
    ``stop_motion`` key the effect assembly reads), so it declares no ok
    branch.  Omitted rows keep the authored action; the omission branch
    carries the base effect keys, except ``set_weather`` — the dispatch
    closure writes real weather snapshots for every ``set_weather`` kind —
    which gets its own omission branch with the weather effect shape.

    Branches with a producer basis but no current L5 instance are declared
    and remain unverified by instances: ``intervention`` ``set_weather``
    rows, the omission results, ``spawn_entity`` and
    ``set_pedestrian_activity`` source-script rows, the ``runtime_state``
    patch alias, default velocity, missing ``camera_id``, scene-position
    spawn, ``visual_state`` on ``spawn_entity``, the single-reference
    ``terminal_feasibility`` variant, ``continuity_repair`` on dispatched
    rows and the ``dust`` weather snapshot member.  ``move_entity``
    corridors, the ten-key terminal-feasibility stamp, single-waypoint
    settle moves, the motion-model effect, the 14-key weather snapshot,
    ok ``set_visual_state`` rows with their activity schedule and
    ``stop_motion``, and ok ``capture_screenshot`` rows with the
    ``action_id``-derived ``capture_id`` are instance-verified (L5-3_v1
    seed00 factual rows).
    """
    effect_rules = {name: _EFFECT_SHAPES[_TYPE_EFFECT[name]]()
                    for name in _SOURCE_SCRIPT_TYPES}
    # One complete-row variant per writable dispatched action type.
    variants = [
        _row("source_script", _ACTION_BRANCHES[name](),
             _OK_RESULT_BRANCHES[name](), effect_rules[name])
        for name in _SOURCE_SCRIPT_TYPES if name in _OK_RESULT_BRANCHES
    ]
    # Omitted rows keep the authored action; any registered type may be
    # omitted.  Base effect keys only, except set_weather rows, which carry
    # the real weather snapshots the dispatch closure writes for them.
    variants.append(_row(
        "source_script",
        {"oneOf": [_ACTION_BRANCHES[name]()
                   for name in _SOURCE_SCRIPT_TYPES
                   if name != "set_weather"]},
        _omitted_result(),
        _base_effect(omitted=True),
    ))
    variants.append(_row(
        "source_script",
        _ACTION_BRANCHES["set_weather"](),
        _omitted_result(),
        _weather_effect(omitted=True),
    ))
    variants.append(_row(
        "intervention",
        _set_weather_action(intervention=True),
        _OK_RESULT_BRANCHES["set_weather"](),
        _EFFECT_SHAPES["set_weather"](),
    ))
    return {
        "$schema": _SCHEMA,
        "$id": _ROW_ID,
        "title": "L5 ARM raw action audit record",
        "oneOf": variants,
    }
