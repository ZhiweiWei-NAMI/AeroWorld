"""Bounded record structure for L4 ARM raw action audit rows.

Source family ``arm_actions``; profiles L4-1_v1, L4-1_v2, L4-2_v1, L4-2_v2,
L4-3_v1, L4-3_v2, L4-3_v3, L4-4_v1, L4-4_v2, L4-5_v1, L4-5_v2, L4-5_v3,
L4-6_v1, L4-6_v2, L4-7_v1, L4-7_v2, L4-8_v1, L4-8_v2, L4-9_v1, L4-9_v2,
L4-10_v1, L4-10_v2, L4-11_v1, L4-11_v2.  Producer:
``Dataset/tools/l4_window.py::run_engine.dispatch`` appends one audit row per
dispatched action as ``{tick, origin, action, result}`` and
``execute_config`` writes the window-scoped slice verbatim to
``raw/actions.jsonl``; ``l4_repair_delivery.process`` re-executes and rejects
any divergence from that file.  UE delivery writes sibling ``ue_*`` files only
and never rewrites this one.  The action is the deep-copied authored payload;
the result comes from the registered ``EpisodeStateEngine`` handler in
``Dataset/tools/batch_generate.py``.

The root is a oneOf over complete-row branches, each fixing one origin, one
action payload type and the result shape the L4 dispatcher binds to that type
(``EpisodeStateEngine._handle_*``).  Coupling is per-row, as production is:
``source_script`` rows carry the ok result of the dispatched type, or
``{'status': 'omitted_by_explicit_intervention'}`` exactly when
``tick_now[0] >= window['fork_tick']`` and the authored ``action_id`` is in
``plan['omit_action_ids']`` (mirrored into ``config['execution']`` by
``require_plan_matches_config``); the omission is decided before the handler
runs, so an omitted row never carries an ok result and an ok row never
carries the omission stamp.  ``intervention`` rows come from
``config['execution']['actions']``, which ``rules/l4_arm_config.schema.json``
restricts to ``move_entity``, ``set_runtime_state`` and ``set_visual_state``;
the dispatcher raises on any non-ok result, so an intervention row always
carries the handler's ok shape.  The two origins also have different payload
contracts, so each keeps its own action schema:

- ``source_script`` payloads are authored by the profile event scripts
  (``Dataset/scenarios/L4_agents/.../event_script.json``); for the 24 scripts
  the 201 declared L4 configs reference, every action key set is a literal
  member of the builder products in ``regenerate_boundary_scenarios.py``
  (``action``/``move``/``visual``/``spawn``/``ped_activity``/``screenshot``/
  ``remove``/``set_weather``).  Builder-required keys stay required even
  where a shared handler has a default.  The ``spawn`` builder's ``**extra``
  can carry a bounded optional ``visual_state``.  Two
  tolerated spellings stay excluded because no L4 authoring path can emit
  them: ``set_weather.overrides: null`` (the only writer is the
  ``set_weather`` builder whose ``**overrides`` is always a Mapping) and the
  ``set_visual_state`` missing/null ``visual_state`` + top-level ``mode``
  fallback (``visual()`` writes ``visual_state`` with a positional mode and
  no builder, script action or config property writes a top-level ``mode``
  key — zero in the 24 scripts).
- ``intervention`` payloads are bounded by ``rules/l4_arm_config.schema.json``:
  one shared ``properties`` list for all three types (``type``, ``action_id``,
  ``entity_id``, ``velocity_mps``, ``waypoints_enu_m``, ``state_patch``,
  ``visual_state``, ``terminal_feasibility``) whose per-type ``if/then``
  blocks add required keys and forbid ``state_patch`` on ``move_entity`` and
  ``set_visual_state``.  Those handlers ignore that field, so their grammar
  branches omit it.  Other shared keys retain the config's value rules and
  each type's required keys.  The config
  ``visual_state`` is reduced to ``mode``, ``state_patch`` is the only patch
  key (no schedule metadata, no delay), and ``move_entity`` carries no
  activity/corridor keys.

Reachability of the runtime-state spellings: the 24 fixed L4 source scripts
write non-null ``state_patch`` and the three schedule keys together; none
writes ``runtime_state`` or a null patch.  Their schedule profile and
semantics are fixed.  The intervention config also excludes the alias and
requires ``state_patch`` on ``set_runtime_state``.  Consumed patch payloads
are closed against
the executable contract in ``Dataset/tools/runtime_state_contract.py``: the
ten declared families, their consumed fields, boolean/enum/command value
rules, facility requester lists and non-negative numbers, plus the
``rule_id``/``mechanism_id`` metadata fields.  Shared raw-string rules retain
the written spelling while checking ``strip().lower()`` enum membership;
the consumed patch carries the producer's ``engine_whole`` source guard.
``set_weather`` overrides are closed against ``Dataset/tools/weather_fields.py``;
an explicit null ``overrides`` is excluded because the handler's
``dict(action.get("overrides") or {})`` accepts ``None``, but the only
writer, the ``set_weather`` builder, always passes a Mapping.

``uav_corridor_validation`` comes from ``regenerate_boundary_scenarios.py::
_replace_uav_action_waypoints``; its two statuses are the only values the
corridor contract stamps.  ``continuity_repair`` is the three-key stamp
``enforce_move_start_continuity`` writes on a first waypoint that exceeds
the motion-class start tolerance but stays within the repair limit (the
pass runs unconditionally in ``make_bundle``, so the shape is
producer-reachable for every profile; no row needs to carry it for the
shape to be closed — the writer bounds every member).  ``terminal_feasibility`` is origin-bound: source
scripts carry the full ten-key stamp that ``enforce_uav_terminal_feasibility``
writes on every landing move (model string and dispatch-time bounds are
producer-pinned), interventions carry the ``rules/l4_arm_config.schema.json``
member set (all members optional there, the verified rows instantiate the
six-key control-reference stamp, and the written ``model`` string is that
schema's own open-string rule).  The pre-v2 landing-reference-only stub is a
shape no current L4 producer writes, so it is not declared.

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

# Types registered by l4_window.run_engine::DISPATCH; every other type is
# rejected by the dispatcher before an audit row exists.
_ACTION_TYPES = (
    "move_entity", "set_visual_state", "set_runtime_state", "spawn_entity",
    "remove_entity", "set_pedestrian_activity", "set_weather",
    "capture_screenshot",
)
# rules/l4_arm_config.schema.json restricts intervention payloads to these
# three; the corpus confirms the shapes.
_INTERVENTION_TYPES = ("move_entity", "set_visual_state", "set_runtime_state")
# regenerate_boundary_scenarios.py::_replace_uav_action_waypoints writes
# exactly these two statuses.
_CORRIDOR_STATUSES = (
    "low_altitude_waypoints_preserved_and_building_clear",
    "waypoints_repaired_for_building_clearance",
)
# enforce_uav_terminal_feasibility pins this model string on the full stamp.
_TERMINAL_MODEL = "constant_speed_path_with_explicit_landing_reference_v2"
# enforce_move_start_continuity: a gap above the smallest motion-class
# tolerance (ground; UAV allows up to 1.5) is the only one that can reach a
# stamped row.
_MOVE_START_TOLERANCE_GROUND_M = 0.75


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


def _script_visual_state() -> dict:
    """``visual(action_id, entity_id, mode, **state)`` payloads.

    The 24 fixed L4 scripts write ``mode`` and sometimes ``lights_on``.
    Generic ``**state`` handling is not authority for other key names in
    these source files.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode"],
        "properties": {
            "mode": {"type": "string", "minLength": 1},
            "lights_on": {"type": "boolean"},
        },
    }


def _intervention_visual_state() -> dict:
    # rules/l4_arm_config.schema.json: additionalProperties false, only mode,
    # required.
    return {
        "type": "object", "additionalProperties": False,
        "required": ["mode"],
        "properties": {"mode": {"type": "string", "minLength": 1}},
    }


def _script_terminal_feasibility() -> dict:
    """The full ten-key stamp enforce_uav_terminal_feasibility writes on
    every source-script landing move; the model string and dispatch-time
    bounds are pinned by that writer."""
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


def _intervention_terminal_feasibility() -> dict:
    """The authored stamp space of rules/l4_arm_config.schema.json: exactly
    the six members, the config's own bounds, every member optional (the 24
    verified intervention rows instantiate the six-key control-reference
    stamp); the config deliberately keeps ``model`` an open string."""
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "model": {"type": "string", "minLength": 1},
            "max_speed_mps": {"type": "number", "exclusiveMinimum": 0},
            "home_pose_origin_z_m": {"type": "number"},
            "landing_reference_enu_m": _number3(),
            "touchdown_altitude_tolerance_m": {"type": "number",
                                               "exclusiveMinimum": 0},
            "touchdown_dwell_ticks": {"type": "integer", "minimum": 1},
        },
    }


def _shared_intervention_properties(action_type: str) -> dict:
    """The bounded shared ``action`` properties of
    rules/l4_arm_config.schema.json.

    ``execution.actions`` declares ONE properties list for all three
    intervention types; the per-type ``if/then`` blocks require their own
    keys and forbid ``state_patch`` on the two non-runtime types.  The runtime
    handler validates its consumed patch, so that branch uses the closed
    executable patch contract.  Other shared fields use the config's value
    rules.
    """
    properties = {
        "type": {"enum": ["move_entity", "set_runtime_state",
                          "set_visual_state"]},
        "action_id": _identifier(),
        "entity_id": _identifier(),
        "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
        "waypoints_enu_m": _waypoints(),
        "visual_state": _intervention_visual_state(),
        "terminal_feasibility": _intervention_terminal_feasibility(),
    }
    if action_type == "set_runtime_state":
        properties["state_patch"] = _runtime_state_patch()
    return properties


def _intervention_action(action_type: str, required: tuple[str, ...]) -> dict:
    """One intervention branch over the shared config properties.

    Each branch pins ``type`` as a const, which makes the config's
    conditional per-type ``required`` an unconditional one for that type.
    The config forbids ``state_patch`` on the two non-runtime types; other
    shared keys follow the formal config.
    """
    properties = _shared_intervention_properties(action_type)
    properties["type"] = {"const": action_type}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", *required],
        "properties": properties,
    }


def _script_continuity_repair() -> dict:
    """The stamp ``enforce_move_start_continuity`` writes on a
    ``move_entity`` whose authored first waypoint exceeds the motion-class
    start tolerance (MOVE_START_TOLERANCE_GROUND_M=0.75 m, UAV 1.5 m) but
    stays inside the repair limit; it runs unconditionally in
    ``make_bundle`` for every bundle, L4 included, and only ever replaces
    waypoint zero.  The first waypoint after the repair is the entity's
    current scene position, so a start-gap value below the producer-side
    tolerance can never be repaired (and never reaches a row).  The policy
    string and the three-key object are producer-pinned;
    ``original_start_enu_m`` is the pre-repair ``q()``-quantized waypoint."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["policy", "original_start_enu_m", "start_gap_m"],
        "properties": {
            "policy": {"const": "replace_first_waypoint_with_current_entity_position_v1"},
            "original_start_enu_m": _number3(),
            "start_gap_m": {
                "type": "number",
                "exclusiveMinimum": _MOVE_START_TOLERANCE_GROUND_M,
            },
        },
    }


def _script_move_entity() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "velocity_mps",
                     "waypoints_enu_m"],
        "properties": {
            "type": {"const": "move_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # The L4 move builder requires velocity; every fixed script writes it.
            "velocity_mps": {"type": "number", "exclusiveMinimum": 0},
            "waypoints_enu_m": _waypoints(),
            # Activity labels stay producer-side strings: vehicle states come
            # from the scene contract and pedestrian labels are normalized by
            # pedestrian_activity_catalog.normalize_activity_type.
            "activity_type": _identifier(),
            "post_activity_type": _identifier(),
            "terminal_feasibility": _script_terminal_feasibility(),
            "continuity_repair": _script_continuity_repair(),
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


def _intervention_move_entity() -> dict:
    # velocity_mps and waypoints_enu_m are the config's conditional required
    # pair for move_entity; the handler's default velocity stays outside the
    # config contract.
    return _intervention_action("move_entity",
                                ("velocity_mps", "waypoints_enu_m"))


def _script_set_visual_state() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "visual_state"],
        "properties": {
            "type": {"const": "set_visual_state"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            "visual_state": _script_visual_state(),
        },
    }


def _intervention_set_visual_state() -> dict:
    return _intervention_action("set_visual_state", ("visual_state",))


def _script_set_runtime_state() -> dict:
    """The 24 fixed L4 scripts write a validated ``state_patch`` and the
    same three schedule keys on every runtime-state action."""
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "state_patch",
                     "delay_ticks", "runtime_state_schedule_profile_id",
                     "runtime_state_schedule_semantics"],
        "properties": {
            "type": {"const": "set_runtime_state"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            "state_patch": _runtime_state_patch(),
            "delay_ticks": {"type": "integer", "minimum": 0},
            "runtime_state_schedule_profile_id": {"const": "l4_runtime_state_schedule"},
            "runtime_state_schedule_semantics": {
                "const": "queued_after_event_actions_delay_ticks_0_next_tick"},
        },
    }


def _intervention_set_runtime_state() -> dict:
    # state_patch is required per the config's if/then block and closed by
    # the shared executable contract (see _shared_intervention_properties).
    return _intervention_action("set_runtime_state", ("state_patch",))


def _spawn_entity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "asset_id",
                     "position_enu_m", "rotation_deg"],
        "properties": {
            "type": {"const": "spawn_entity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # The L4 spawn builder writes the asset and placement.
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
            # Handler tolerance: ``(action.get("visual_state") or {})`` —
            # absent, null, empty or mode-free payloads all dispatch ok
            # (the activity label falls back to the entity state, then
            # "spawned").  The ``**extra`` writers in
            # regenerate_boundary_scenarios.py (spawn / spawn_from_scene
            # call sites) emit only ``visual_state`` and only as
            # ``{"mode": ...}`` or ``{"mode": ..., "lights_on": ...}``; the
            # corpus spawn rows carry none of it.  The value space stays
            # closed: null, the empty object, or the script visual-state
            # contract.
            "visual_state": {
                "oneOf": [
                    {"type": "null"},
                    {"type": "object", "additionalProperties": False},
                    _script_visual_state(),
                ],
            },
        },
    }


def _set_pedestrian_activity_action() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["type", "action_id", "entity_id", "activity_type"],
        "properties": {
            "type": {"const": "set_pedestrian_activity"},
            "action_id": _identifier(),
            "entity_id": _identifier(),
            # The L4 ped_activity builder always writes the key.
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
        "required": ["type", "action_id", "profile", "overrides"],
        "properties": {
            "type": {"const": "set_weather"},
            "action_id": _identifier(),
            # The L4 set_weather builder writes both profile and overrides.
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
        "required": ["type", "action_id", "camera_id"],
        "properties": {
            "type": {"const": "capture_screenshot"},
            "action_id": _identifier(),
            # The L4 screenshot builder writes camera_id.
            "camera_id": _identifier(),
        },
    }


_SCRIPT_ACTION_BRANCHES = {
    "move_entity": _script_move_entity,
    "set_visual_state": _script_set_visual_state,
    "set_runtime_state": _script_set_runtime_state,
    "spawn_entity": _spawn_entity_action,
    "set_pedestrian_activity": _set_pedestrian_activity_action,
    "set_weather": _set_weather_action,
    "remove_entity": _remove_entity_action,
    "capture_screenshot": _capture_screenshot_action,
}

_INTERVENTION_ACTION_BRANCHES = {
    "move_entity": _intervention_move_entity,
    "set_visual_state": _intervention_set_visual_state,
    "set_runtime_state": _intervention_set_runtime_state,
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
        # sorted(patch): the producer writes each applied family exactly once,
        # and only RUNTIME_STATE_FIELDS families survive
        # validate_runtime_state_patch, so the list is a non-empty set of the
        # declared family names in sorted order.
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
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"const": "omitted_by_explicit_intervention"}},
    }


def _row(origin: str, action: dict, result: dict) -> dict:
    """One complete-row branch: one origin, one action type, one result shape.

    ``l4_window.dispatch`` fixes the result from the dispatched type (ok) or
    the pre-handler omission stamp, so a branch is exact instead of pairing a
    per-type action oneOf with an independent result oneOf.
    """
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


def arm_actions_schema() -> dict:
    """Bounded oneOf over L4 action and result pairings.

    The branches are complete rows (origin + action type + result shape), so a
    ``move_entity`` action cannot pair with another handler's ok result, and
    only ``source_script`` rows can carry the omission stamp.  The formal
    config forbids ``state_patch`` on intervention ``move_entity`` and
    ``set_visual_state`` actions, matching their closed branches here.
    """
    branches: list[dict] = []
    for name in _ACTION_TYPES:
        # Ok source-script rows carry the dispatched handler's result.
        branches.append(_row("source_script", _SCRIPT_ACTION_BRANCHES[name](),
                             _OK_RESULT_BRANCHES[name]()))
        # The omission stamp is pre-handler (l4_window.dispatch): the row
        # keeps the authored payload and never carries a handler result.
        branches.append(_row("source_script", _SCRIPT_ACTION_BRANCHES[name](),
                             _omitted_result()))
    for name in _INTERVENTION_TYPES:
        # l4_arm_config.schema.json restricts payloads to these three, and
        # the dispatcher raises on any non-ok result.
        branches.append(_row("intervention",
                             _INTERVENTION_ACTION_BRANCHES[name](),
                             _OK_RESULT_BRANCHES[name]()))
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/arm_actions_l4.record.schema.json",
        "title": "L4 ARM raw action audit record",
        "oneOf": branches,
    }
