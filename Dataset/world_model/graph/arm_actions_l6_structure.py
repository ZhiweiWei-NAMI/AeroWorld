"""Closed record structure for L6 ARM raw action audit rows.

Source family ``arm_actions``; profiles L6-1_v1, L6-1_v2, L6-2_v1, L6-2_v2,
L6-3_v1, L6-3_v2, L6-4_v1, L6-4_v2, L6-5_v1, L6-5_v2.  Producer:
``Dataset/tools/l6_v2/arm_runtime.py::run_arm`` executes one arm through
``Dataset/tools/l6_v2/engine.py::execute`` and writes the window-scoped audit
slice verbatim to ``raw/actions.jsonl``.  The engine appends one audit row per
dispatched action and binds the complete row shape deterministically:

- ``dispatch_source`` handles event-script actions as
  ``{tick, origin: 'source_script', action, result}``.  For an action id in
  ``omit_action_ids`` at or after the fork tick it writes
  ``result = {'status': 'omitted_by_configuration'}`` and keeps the
  deep-copied authored payload (corridor or terminal-feasibility stamps
  included); otherwise ``result`` is the registered handler's output.
- the injection loop handles ``ArmSpec.inject_actions`` (mirrored in
  ``mechanism.injections`` and checked for equality by ``run_arm``) as
  ``{tick, origin: 'intervention', action, result}`` with the same handler
  output; it raises on any non-ok result, so intervention rows are always ok
  and never omitted.

``motion_effect`` deep-copies the entity's terminal keyframe into the sibling
key ``motion_schedule`` = ``{terminal_tick, endpoint_enu_m}`` exactly when the
row is an ok ``move_entity`` dispatch, regardless of origin; the omission stamp
and every non-move type never carry it.  One ok ``move_entity`` row always
carries the sibling, so each origin contributes one motion-carrying branch,
and the per-type motion-free branches forbid the key by closure.

The result shapes per action type are shared between the origins because both
dispatch paths call the same ``EpisodeStateEngine._handle_*`` in
``Dataset/tools/batch_generate.py``.  Each row variant therefore pins
``action.type`` to one const and pairs it with exactly the results its handler
writes (its ok shape, plus the omission stamp for source-script rows), so no
cross-type action/result combination can validate.  The engine registers all
eight handler types (``engine._handlers``), but no L6 script authors
``spawn_entity``, ``set_weather``, ``set_pedestrian_activity`` or
``remove_entity``; their ok and omitted row forms stay declared with a
producer basis and no L6 instance.  The intervention payloads come from the
``aero_l6_explicit_arm_v3`` configs (enforced by
``Dataset/tools/validate_config_format.py``); all 90 declared intervention
configs inject only ``move_entity``, ``set_visual_state`` and
``set_runtime_state`` — 60 move rows (none carrying a stamp key), 36 visual
rows and 45 runtime rows, the runtime ones always with ``delay_ticks`` and
never with the schedule keys — so the intervention branch is restricted to
those three types; the payload key space stays the shared handler space
because the ``ArmSpec`` contract does not constrain inject payload keys.

Per-branch key authority is ``EpisodeStateEngine._handle_*``: optional keys the
handler tolerates (default velocity, default delay, missing ``camera_id``,
scene-position spawn, default ``activity_type``, default weather profile,
empty ``set_weather`` overrides, ``visual_state`` on ``spawn_entity``) are
declared optional even where the current L6 corpus always writes them.
``state_patch`` is closed against the executable contract in
``Dataset/tools/runtime_state_contract.py``: the ten declared families, their
consumed fields, boolean/enum/command value rules, facility requester lists
and non-negative numbers, plus the ``rule_id``/``mechanism_id`` metadata
fields.  ``runtime_state_patch_from_action`` reads ``state_patch`` first and
falls back to the ``runtime_state`` alias when ``state_patch`` is absent or
``None``; a valid ``state_patch`` wins and the alias value is then never read,
so each key may be ``null`` as long as the other key carries a non-null patch
and at least one non-null patch exists.  The grammar admits a validated patch
or ``null`` under either key and requires one non-null patch.  The alias and
``null``/dual-key forms carry 「尚无实例验证」 because no declared L6 script or
config writes them.  With a valid ``state_patch``, the generic handler would
also ignore an arbitrary ``runtime_state`` value; no declared L6 authoring
path supplies that sidecar, so it is outside this finite source contract.
``set_weather`` overrides are closed against
``Dataset/tools/weather_fields.py``.

``uav_corridor_validation`` comes from
``Dataset/tools/regenerate_boundary_scenarios.py::_replace_uav_action_waypoints``;
its two statuses are the only values the corridor contract stamps, and both
have L6 instances (234 rows ``waypoints_repaired_for_building_clearance``,
18 rows ``low_altitude_waypoints_preserved_and_building_clear``).
``terminal_feasibility`` carries the full ten-key stamp from
``enforce_uav_terminal_feasibility``: every L6 script authors it, but only on
landing actions outside the declared windows, so the branch is declared with
「尚无实例验证」 in the L6 rows.  The single-reference and six-key variants of
the L4 configs have no L6 producer path and are not declared.

L6-1_v1 through L6-5_v1 and L6-1_v2 through L6-5_v2 are written by the same
producer (``Dataset/tools/l6_v2/arm_runtime.py``); the v1/v2 profile pairs
differ only in scene, script and entity identifiers, not in row structure.

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

__all__ = ["arm_actions_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"
_ROW_ID = "https://aeroworld.local/schema/arm_actions_l6.record.schema.json"

# Types registered by l6_v2.engine._handlers; every other type is rejected by
# the dispatcher before an audit row exists.
_SOURCE_SCRIPT_TYPES = (
    "move_entity", "set_visual_state", "set_runtime_state", "capture_screenshot",
    "spawn_entity", "set_pedestrian_activity", "set_weather", "remove_entity",
)
# Every declared aero_l6_explicit_arm_v3 intervention config injects only
# these three; the injection loop raises on a non-ok result.  The dispatch
# handler is shared with source-script rows, so the action and result shapes
# per type stay identical across origins.
_INTERVENTION_TYPES = ("move_entity", "set_visual_state", "set_runtime_state")
# regenerate_boundary_scenarios.py::_replace_uav_action_waypoints writes
# exactly these two statuses.
_CORRIDOR_STATUSES = (
    "low_altitude_waypoints_preserved_and_building_clear",
    "waypoints_repaired_for_building_clearance",
)
# enforce_uav_terminal_feasibility stamps this model on the full variant.
_TERMINAL_MODEL = "constant_speed_path_with_explicit_landing_reference_v2"


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _number3() -> dict:
    # vector3() coerces realized waypoints to float, but authored corridor
    # originals and config waypoints keep written ints; both are JSON numbers
    # from the same producer path.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number"},
    }


def _waypoints() -> dict:
    return {"type": "array", "minItems": 1, "items": _number3()}


def _float_leaf() -> dict:
    return {"type": "number", "x-source-value-types": ["float"]}


def _string_or_commands() -> list[dict]:
    """Branch alternatives for one command field.

    The compiler records ``oneOf`` branch leaves under separate variants, so
    the mixed string/array value rule is expressed as a variant selector with
    one typed alternative per branch; the closed record keeps the
    ``x-source-value-types`` spelling of the written string form.
    """
    return [
        {"type": "string", "minLength": 1},
        {"type": "array", "minItems": 1,
         "items": {"type": "string", "minLength": 1}},
    ]


def _runtime_state_field_rule(family: str, field: str) -> dict:
    """Value rule for one runtime-state patch field, from the executable
    producer contract (validate_runtime_state_patch plus
    invalid_runtime_state_value_paths in runtime_state_contract.py)."""
    if field in RUNTIME_STATE_METADATA_FIELDS:
        return {"type": "string", "minLength": 1}
    if field in RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset()):
        return {"type": "boolean"}
    if (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
        return {"oneOf": _string_or_commands()}
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


def _runtime_state_patch() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {family: _runtime_state_family_patch(family)
                       for family in sorted(RUNTIME_STATE_FIELDS)},
    }


def _visual_state() -> dict:
    # visual(action_id, entity_id, mode, **state) writes mode plus booleans
    # such as lights_on; every current L6 row uses mode only.
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
    # enforce_uav_terminal_feasibility writes the full ten-key variant; the
    # L6 scripts author it only on landing actions outside the declared
    # windows, so no L6 row carries it (「尚无实例验证」).
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
            "max_speed_mps": {"type": "number", "exclusiveMinimum": 0},
            "home_pose_origin_z_m": _float_leaf(),
            "landing_reference_enu_m": _number3(),
            "touchdown_altitude_tolerance_m": _float_leaf(),
            "touchdown_dwell_ticks": {"type": "integer", "minimum": 0},
        },
    }


def _motion_schedule() -> dict:
    # engine.motion_effect deep-copies the terminal keyframe of the scheduled
    # move: an integer tick and the float endpoint position.  Neither key is
    # optional and no other action type or result status produces the sibling.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["terminal_tick", "endpoint_enu_m"],
        "properties": {
            "terminal_tick": {"type": "integer", "minimum": 0},
            "endpoint_enu_m": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "items": _float_leaf(),
            },
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
            # _handle_move_entity reads velocity with default 1.0 and raises
            # on an empty waypoint list.  Every L6 row writes it.
            "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
            "waypoints_enu_m": _waypoints(),
            # Activity labels stay producer-side strings: vehicle states come
            # from the scene contract and pedestrian labels are normalized by
            # pedestrian_activity_catalog.normalize_activity_type.
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
    # _handle_set_visual_state also tolerates a top-level ``mode`` fallback,
    # but no authoring path (the visual() helper and the v3 configs both
    # always write visual_state) produces it and the empty-mode dispatch
    # would contradict the written result contract (mode minLength 1); the
    # fallback spelling is not declared.
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
    # runtime_state_patch_from_action reads state_patch first and falls back
    # to the runtime_state alias when the first key is absent or None; a
    # valid state_patch wins and the alias is then never read.  Each key may
    # therefore be null as long as the other carries a non-null patch, and
    # both keys may carry independent patches (state_patch wins).  The
    # action-level anyOf carries one complete closed variant per required
    # key — the patch rule stays inside the branch so the compiler keeps the
    # leaf variants separate and the branches never cross-merge — and each
    # branch requires a non-null selected patch, reproducing the producer's
    # raise when both keys are null or absent.  The first branch bounds the
    # ignored alias to a validated patch or null; an arbitrary ignored
    # sidecar has no declared L6 authoring path.  The alias and null/dual-key
    # forms have no current L6 instance (「尚无实例验证」).
    shared = {
        "type": {"const": "set_runtime_state"},
        "action_id": _identifier(),
        "entity_id": _identifier(),
        # _handle_set_runtime_state treats a missing delay as 0 and
        # schedules at tick + max(1, delay).
        "delay_ticks": {"type": "integer", "minimum": 0},
        # The handler never reads the schedule keys; the L6 source scripts
        # always write both, the intervention configs never do.
        "runtime_state_schedule_profile_id": _identifier(),
        "runtime_state_schedule_semantics": _identifier(),
    }
    null_only = {"type": "null"}
    patch = _runtime_state_patch()
    # The compiler keeps anyOf branch schemas fully separate (no cross-branch
    # property merging), so the two producer-readable spellings of the patch
    # carrier are complete variants here and the parent level intentionally
    # declares no properties of its own: every property lives inside a
    # branch, and the parent boundary stays closed by each branch's own
    # additionalProperties rule.
    return {
        "type": "object",
        "required": ["type", "action_id", "entity_id"],
        "anyOf": [
            {
                "type": "object", "additionalProperties": False,
                "required": ["type", "action_id", "entity_id", "state_patch"],
                "properties": {**shared,
                               "state_patch": patch,
                               # With a valid state_patch the alias is never
                               # read, so any spelling including null is
                               # accepted beside it.
                               "runtime_state": {"oneOf": [patch, null_only]}},
            },
            {
                "type": "object", "additionalProperties": False,
                "required": ["type", "action_id", "entity_id", "runtime_state"],
                "properties": {**shared,
                               "runtime_state": patch,
                               # Alias-fallback branch: state_patch is either
                               # absent or null (the producer reads the alias
                               # only in these two cases).
                               "state_patch": null_only},
            },
        ],
    }

def _spawn_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id"],
        "properties": {
            "type": {"const": "spawn_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # _handle_spawn_entity defaults to the declared scene asset.
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
            # _handle_capture_screenshot falls back to action_id when the
            # camera key is missing; every L6 row writes camera_id.
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


def _omitted_result() -> dict:
    # l6_v2.engine.dispatch_source stamps the fork-tick configuration
    # omission with this status; the L2/L4 producer writes
    # omitted_by_explicit_intervention instead, and the L6 injection loop
    # cannot produce an omission at all.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"const": "omitted_by_configuration"}},
    }


def _row(origin: str, action: dict, result: dict, *,
         with_motion_schedule: bool) -> dict:
    """One complete-row variant for one origin.

    ``with_motion_schedule`` is true exactly for ok ``move_entity`` rows:
    ``motion_effect`` appends the sibling to those rows only, so the variant
    requires it beside the ok move result, while the closed motion-free
    variants reject the key for every other row.
    """
    row = {
        "type": "object", "additionalProperties": False,
        "required": ["tick", "origin", "action", "result"],
        "properties": {
            "tick": {"type": "integer", "minimum": 0},
            "origin": {"const": origin},
            "action": action,
            "result": result,
        },
    }
    if with_motion_schedule:
        row["required"] = [*row["required"], "motion_schedule"]
        row["properties"]["motion_schedule"] = _motion_schedule()
    return row


def arm_actions_schema() -> dict:
    """Return the closed record schema for one ``arm_actions`` L6 JSONL row.

    Every branch is one complete row: ``action.type`` is pinned to one const
    and paired with exactly the results its producer writes — the handler's
    ok shape for that type (identical for both origins, which share the
    registered handler) plus, for source-script rows, the
    ``omitted_by_configuration`` stamp that keeps the authored payload.  The
    two ok ``move_entity`` branches (one per origin) require the
    ``motion_schedule`` sibling; all other branches forbid it by closure.
    Branches with a producer basis but no current L6 instance stay declared
    and are marked 「尚无实例验证」 in the delivery notes: source-script
    ``spawn_entity``/``set_weather``/``set_pedestrian_activity``/
    ``remove_entity`` rows in their ok and omitted forms, the ``runtime_state``
    patch alias and the bounded ``null``/dual-key ``set_runtime_state`` shapes, the
    schedule keys on intervention ``set_runtime_state`` payloads, stamp keys
    on intervention ``move_entity`` payloads, default velocity, delay-free
    ``set_runtime_state``, missing ``camera_id``, ``lights_on`` in
    ``visual_state``, scene-position spawn, ``pitch_deg``/``roll_deg``
    rotation members, empty ``set_weather`` overrides and the
    ``terminal_feasibility`` stamp.
    """
    variants: list[dict] = []
    for name in _SOURCE_SCRIPT_TYPES:
        action = _ACTION_BRANCHES[name]()
        if name == "move_entity":
            # ok move rows carry the motion_effect sibling; the omission
            # stamp keeps the authored payload (corridor stamp included) and
            # drops the effect.
            variants.append(_row("source_script", action,
                                 _OK_RESULT_BRANCHES[name](),
                                 with_motion_schedule=True))
            variants.append(_row("source_script", action, _omitted_result(),
                                 with_motion_schedule=False))
        else:
            variants.append(_row(
                "source_script", action,
                {"oneOf": [_OK_RESULT_BRANCHES[name](), _omitted_result()]},
                with_motion_schedule=False))
    for name in _INTERVENTION_TYPES:
        variants.append(_row(
            "intervention", _ACTION_BRANCHES[name](),
            _OK_RESULT_BRANCHES[name](),
            with_motion_schedule=(name == "move_entity")))
    return {
        "$schema": _SCHEMA,
        "$id": _ROW_ID,
        "title": "L6 ARM raw action audit record",
        "oneOf": variants,
    }
