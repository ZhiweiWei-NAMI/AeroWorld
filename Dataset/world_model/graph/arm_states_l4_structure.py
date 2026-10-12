"""Whole-record closed grammar for L4 arm_states rows.

Grammar target
    raw/states.jsonl of the declared L4 profiles, i.e. the window rows
    produced by
    Dataset/tools/l4_window.py::execute_config, which slices
    EpisodeStateEngine.trajectory_rows to the arm window and the configured
    entity_ids, then writes the rows verbatim via write_jsonl(out /
    "raw/states.jsonl") (Dataset/tools/l4_window.py:657).

Row origin (all leaves of this grammar trace to a single producer chain)
    Dataset/tools/batch_generate.py
      - EpisodeStateEngine._row_for_entity (2395): fixed core keys,
        activity payload, then row.update(preserved_fields_from(entity)) -
        preserved keys exist on a row only when the engine entity carried
        them, so every key introduced here is optional and never null.
      - preserved_fields_from (1737) over PRESERVED_ENTITY_FIELDS (215),
        reading the scene entity directly (initial_state / visual_state
        nested first, then top level).
      - runtime-state families attach to the entity through
        _apply_pending_runtime_state_patches (2312) after
        validate_runtime_state_patch (841) forwarded each set_runtime_state
        action; a present family is a nonempty mapping whose members are
        contract-consumed booleans, governed enums and commands, or the
        metadata strings rule_id / mechanism_id.
      - event-actor rows: build_entities copies the scene entity verbatim
        (1966), so spawn_policy / event_actor_motion_contract carry the
        scene dict written by Dataset/tools/regenerate_boundary_scenarios.py
        stage_event_actor (1410) and spawn_event_actor_at_dispatch (1516).
      - UAV rows: _set_uav_altitude_metadata (7973) writes uav_corridor /
        assigned_altitude_m / uav_corridor_role / ground_reference_z_m;
        scene["lifecycle"] (7231) + altitude repair (8021-8027) write the
        lifecycle block; terminal_flight_state is stamped only in
        _stage_terminal_event (9914, L4-4 profiles).
    Dataset/tools/l4_window.py
      - window scoping (523): only rows with start_tick <= tick <= end_tick
        and entity_id in config["entity_ids"] are written; the window bounds
        are manifest facts, not row facts, so no tick enumeration here.
    Dataset/tools/runtime_state_contract.py
      - family names, consumed field names, value domains and metadata
        fields come from its executable tables. The raw-string fragments
        preserve the producer's strip/lower enum and nonblank checks; the
        engine_whole source guard applies to each consumed family object.

Producer branches (record root oneOf; a row belongs to exactly one)
    1. pedestrian       - pedestrian.cityops.basic.v1 rows; the activity
      payload uses the activity catalog (posture/social_state enum).
    2. uav_mission      - mission_uav UAVs (patrol / descent / fault /
      unstable / navigation_fault / monitor / corridor_observer); lifecycle
      rows carry the full-duration takeoff/land block, observer rows carry
      observer_lifecycle, corridor_observer rows carry neither.
    3. uav_inspect      - the per-profile contract inspect UAV (role
      U_inspect, _add_contract_inspect_uav at
      regenerate_boundary_scenarios.py:5140): four-element state_sequence
      ["takeoff","inspect_racetrack","orbit","land"], the 17-key
      contract_inspect_uav block, uav_corridor_role inspect_observer,
      inspect_altitude_code / min_path_length_m / full_episode_presence.
    4. vehicle          - vehicle.* rows (mission actors and background
      flow), optionally with spawn_policy / event_actor_motion_contract,
      ground_flow_contract / background_vehicle, vehicle_state.
    5. asset_anchor     - semantic.asset_anchor facade anchors
      (facade_entity, regenerate_boundary_scenarios.py:2555).
    6. facility         - landing pads / chargers / radio towers / barriers:
      scene facility entities (home pads pad_entity:2602, contract
      facilities _add_contract_facility:5192 and its facility_cycle index
      >= 2 assets). semantic_scope is the exact facility_scope_contract
      shape per subtype; contract_facility is present only on contract
      facility entities.
    7. spawn_zone       - semantic.spawn_zone crowd anchors (L4-8
      crowd_anchor_{sid}); four-key initial-state projection, no
      ground_reference and no scene_occupancy fields.
    Branches are mutually exclusive by producer-identity keys
    (label_class + role + asset_id); each branch limits the optional
    preserved keys its producer can write. Runtime family keys remain
    available on every branch because set_runtime_state targets any entity.

Out of scope for L4 rows (published family-wide field paths that other
layers carry; not declared in this grammar): observation markers (observed /
observation_status / present / execution_active / unavailable_runtime_families
- Dataset/world_model/graph/arm_observation.py::record_observed reports every
L4 row observed and the rows carry no marker keys), frame_id / entity_type /
entity_category / deactivation_tick / path_deviation_contract / task_kind /
task_state / contract_logical_sidecar / values.* projection paths.
motion_contract / scene_occupancy / placement / boundary_crossing_injected
are scene-side keys outside PRESERVED_ENTITY_FIELDS, so they never reach a
row. Ground-station identity subtypes are outside the L4 facility asset
cycle and are not declared here.

Checked scope: the grammar compiled and the first 12 rows of six seed00
factual states files (L4-1_v1, L4-4_v1, L4-5_v1, L4-8_v1, L4-10_v1,
L4-11_v1) passed record validation. The sample contains UAV, vehicle and
pedestrian rows, including consumed runtime families. Facility, inspect
UAV, asset_anchor and spawn_zone rows, observer_lifecycle and many accepted
runtime field values have no instance in that bounded check (尚无实例验证).
The check does not establish a whole-file or whole-family result.
"""

from __future__ import annotations

import copy

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
from Dataset.world_model.graph.structure_fragments import (
    nonblank_python_string_rule,
    runtime_enum_string_rule,
    runtime_state_source_guard_rule,
)

__all__ = ["arm_states_schema"]


_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# label_class values written by L4 producers (typed_entity_labels from
# Config/LowAltitude/asset_catalog.json). asset_anchor / uav / vehicle /
# pedestrian occur in the bounded sample; landing_pad / charging_pile /
# radio_tower / barrier label the facility entities the
# facility_cycle can stage; spawn_zone labels the L4-8 crowd anchor.
_LABEL_CLASSES = (
    "pedestrian", "uav", "vehicle", "asset_anchor",
    "landing_pad", "charging_pile", "radio_tower", "barrier", "spawn_zone",
)
# label_class -> category written into the row (batch_generate _row_for_entity
# copies the preserved scene category).
_CATEGORIES = (
    "pedestrian", "uav", "vehicle", "facade_anchor", "facility",
    "crowd_anchor",
)
# label_class -> role (world_entity default semantic.role / scene role).
_ROLES = (
    "semantic_pedestrian", "semantic_background_pedestrian", "mission_uav",
    "U_inspect", "semantic_vehicle", "semantic_background_vehicle",
    "semantic_context", "semantic_facility",
)
# asset_id values on L4 rows; the pair (label_class, asset_id) is fixed by
# Config/LowAltitude/asset_catalog.json + typed_entity_labels. The four
# facility.* assets are the facility_cycle of _add_contract_facility (5192)
# plus the home pads (landing_pad only).
_ASSETS = (
    "uav.inspect.quad.v1",
    "uav.airsim.cv_pawn.v1",
    "uav.airsim.flying_pawn.v1",
    "vehicle.ground.boxcar.v1",
    "vehicle.service.box.v1",
    "vehicle.emergency.ambulance.v1",
    "vehicle.emergency.suv.v1",
    "pedestrian.cityops.basic.v1",
    "semantic.asset_anchor",
    "semantic.spawn_zone",
    "facility.landing_pad.visible.v1",
    "facility.charger.cityops.v1",
    "facility.radio.base_tower.v1",
    "facility.barrier.basic",
)

# state_sequence is written as [mode] by world_entity
# (regenerate_boundary_scenarios.py:2220) or copied verbatim from the
# scene's initial_state. Single-element values: mission UAV modes, vehicle /
# pedestrian modes, facility "available" (the facility_cycle radio tower
# variant uses "online"), spawn-zone "idle". Two-element values: [mode, mode] pairs of
# background actors stamped by the profile builders. Four-element value: the
# U_inspect lifecycle. state_sequence[0] == row["semantic_role"] on every
# corpus row.
_SINGLE_ELEMENT_SEQUENCE = (
    "autonomous", "available", "chatting", "corridor_observer", "descent",
    "fault", "idle", "monitor", "moving", "navigation_fault", "online",
    "patrol", "response", "standing", "unstable", "waiting",
)
_DOUBLE_ELEMENT_SEQUENCE = (
    "cautious_flow", "responder", "traffic_flow", "traffic_slow", "walking",
)
_INSPECT_SEQUENCE = ("takeoff", "inspect_racetrack", "orbit", "land")

# uav_corridor.altitude_layers_m comes from UAV_ALTITUDE_LAYERS_M in
# Dataset/tools/uav_corridor_planner.py:27; every corridor row observed the
# same three layers. The literal triple is a producer constant, so the layers
# sequence is closed with the observed order as prefix and no extras.
_ALTITUDE_LAYERS_M = (30.0, 50.0, 80.0)

# uav_corridor_role: scene initial_state / top-level value preserved onto the
# row. mission_uav (22 scenes), observer (26 observer UAVs),
# inspect_observer (24 contract inspect UAVs).
_UAV_CORRIDOR_ROLES = ("mission_uav", "observer", "inspect_observer")

_FAMILY_COMMAND_ENUMS = {
    "control_state": _CONTROL_ENUMS,
    "mission_state": _MISSION_ENUMS,
    "security_state": _SECURITY_ENUMS,
}


def _enum(values) -> dict:
    return {"const": values[0]} if len(values) == 1 else {"enum": list(values)}


def _string_enum(values) -> dict:
    return {"type": "string", **_enum(values)}


def _object(properties: dict, required=None) -> dict:
    """Closed object: every key declared, nothing else allowed."""
    schema = {
        "type": "object",
        "properties": copy.deepcopy(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def _number3() -> dict:
    """Fixed-length 3-vector of numbers (compiler strict-checks int|float)."""
    return {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "number"},
                        {"type": "number"}],
        "minItems": 3,
        "maxItems": 3,
        # compiler requires a typed tail; maxItems makes it unreachable
        "items": False,
    }


def _engine_float3() -> dict:
    # EpisodeStateEngine._row_for_entity writes round(float(value), 6) for
    # each position and velocity component; the source JSON values are float.
    return {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "number", "x-source-value-types": ["float"]},
    }


_RUNTIME_STATE_NUMBER = {
    # Producer validators reject bool, require a finite int/float, and
    # enforce the field's lower bound (runtime_state_contract.py:583-624).
    "type": "number",
    "minimum": 0,
}
def _runtime_state_family(family: str) -> dict:
    """Closed, nonempty runtime-state family object with governed domains.

    A validated scene carrier or set_runtime_state patch can put a family
    on the entity. The producer rejects an empty family mapping, so
    minProperties >= 1. Patches merge by deep_merge_mapping, so consumed
    members stay optional instead of enumerating observed keysets.

    Member domains mirror invalid_runtime_state_value_paths
    (runtime_state_contract.py:527-625):
      - booleans (RUNTIME_STATE_BOOLEAN_FIELDS) must be bool;
      - enum fields (RUNTIME_STATE_ENUM_FIELDS keyed (family, field)) use
        the audited raw-string fragment for the producer's
        value.strip().lower() membership check;
      - command fields (control/mission/security active_command /
        active_commands / commands, RUNTIME_STATE_COMMAND_FIELDS) are a
        nonempty string or a nonempty array of family-enum members;
      - facility capacity / request_count are numbers >= 0, requester_ids a
        string array with nonempty items, priority_granted_to /
        arbitration_winner_id nonempty strings;
      - communication availability / link_availability are numbers 0..1
        (schema maximum 1; the producer also rejects bools first);
      - incident hazard_concentration_ppm / hazard_radius_m numbers >= 0;
      - the metadata members rule_id / mechanism_id are nonblank strings
        (validate_runtime_state_patch admits them next to consumed members
        and _apply_pending_runtime_state_patches deep-merges them onto the
        entity). Fields without a bounded sample instance keep the same
        executable contract.
    """
    properties: dict = {}
    booleans = RUNTIME_STATE_BOOLEAN_FIELDS.get(family, frozenset())
    for field in RUNTIME_STATE_CONSUMED_FIELDS.get(family, ()):
        allowed = RUNTIME_STATE_ENUM_FIELDS.get((family, field))
        if allowed is not None:
            properties[field] = runtime_enum_string_rule(allowed)
        elif field in booleans:
            properties[field] = {"type": "boolean"}
        elif (family, field) in RUNTIME_STATE_COMMAND_FIELDS:
            member = runtime_enum_string_rule(_FAMILY_COMMAND_ENUMS[family])
            properties[field] = {"oneOf": [
                dict(member),
                {"type": "array", "minItems": 1, "items": member},
            ]}
        elif family == "facility_state" and field in (
            "capacity", "request_count",
        ):
            properties[field] = dict(_RUNTIME_STATE_NUMBER)
        elif family == "facility_state" and field == "requester_ids":
            properties[field] = {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            }
        elif family == "facility_state" and field in (
            "priority_granted_to", "arbitration_winner_id",
        ):
            properties[field] = nonblank_python_string_rule()
        elif family == "communication_state" and field in (
            "availability", "link_availability",
        ):
            properties[field] = {**_RUNTIME_STATE_NUMBER, "maximum": 1}
        elif family == "incident_state" and field in (
            "hazard_concentration_ppm", "hazard_radius_m",
        ):
            properties[field] = dict(_RUNTIME_STATE_NUMBER)
        else:
            raise ValueError(
                f"runtime-state field {family}.{field} has no governed "
                "producer domain"
            )
    for field in sorted(RUNTIME_STATE_METADATA_FIELDS):
        properties[field] = nonblank_python_string_rule()
    return {
        **_object(properties),
        "minProperties": 1,
        **runtime_state_source_guard_rule("engine_whole"),
    }


def _event_actor_motion_contract() -> dict:
    """event_actor_motion_contract: oneOf the two producer-stamped variants.

    Dataset/tools/regenerate_boundary_scenarios.py stage_event_actor (1410)
    stamps the 6-key base contract on every event actor; the continuous-lane
    policy (spawn_event_actor_at_dispatch -> offstage_then_continuous_lane_
    entry, 1476) extends it with 8 entry keys. schema const is the producer
    constant event_actor_motion_contract_v1.
    """
    base = {
        "schema": {"const": "event_actor_motion_contract_v1"},
        "policy": {"type": "string"},
        "semantic_event_id": {"type": "string"},
        "pre_activation_presence": {"const": "absent"},
        "stationary_preplacement_allowed": {"type": "boolean"},
        "yaw_authority": {"type": "string"},
    }
    entry = {
        "entry_event_id": {"type": "string"},
        "entry_tick": {"type": "integer"},
        "arrival_tick": {"type": "integer"},
        "route_length_m": {"type": "number"},
        "entry_speed_mps": {"type": "number"},
        "duration_model": {
            "const": "sum_segment_ceil_distance_over_speed_ticks"},
        "minimum_visible_motion_ratio_before_event": {"type": "number"},
        "entry_capture_boundary_clearance_m": {"type": "number"},
    }
    return {
        "oneOf": [
            _object(base, required=sorted(base)),
            _object({**base, **entry}, required=sorted({**base, **entry})),
        ]
    }


def _waypoint() -> dict:
    return {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "number"},
                        {"type": "number"}],
        "minItems": 3,
        "maxItems": 3,
        "items": False,
    }


def _route_waypoints_enu_m() -> dict:
    """route_waypoints_enu_m: exactly [] or >=1 three-number waypoints.

    Empty routes occur before dispatch and activation; nonempty routes
    contain three-number waypoints. The nonempty<->planned co-occurrence is enforced at the record
    root (allOf if/then on the row), not here: a schema condition on the
    array can only constrain the array itself, not sibling keys.
    """
    return {
        "oneOf": [
            # exactly-empty route; the compiler expands container consts into
            # this shape, written out here for the grammar to stay explicit
            {"type": "array", "minItems": 0, "maxItems": 0, "items": False},
            {
                "type": "array",
                "minItems": 1,
                "items": _waypoint(),
            },
        ]
    }


def _planned_route_waypoints_enu_m() -> dict:
    return {
        "type": "array",
        "minItems": 1,
        "items": _waypoint(),
    }


def _lifecycle() -> dict:
    """lifecycle: producer-stamped block; +terminal_flight_state only on
    L4-4 rows (_stage_terminal_event, regenerate_boundary_scenarios.py:
    9914). The base 6-key dict is written at 7231, then the altitude pass
    (8021-8027) merges assigned_altitude_m and waypoint_repair_applied into
    it, so a lifecycle row carries the 8 base keys and may carry
    terminal_flight_state on L4-4 rows. requires_takeoff /
    requires_landing_or_terminal_resolution / auto_lifecycle_applied /
    waypoint_repair_applied / terminal_flight_state are written as literal
    True; home_pad_entity_id is a scene entity id string; home_hover_enu_m /
    mission_start_enu_m are the 3-number pad / mission-start positions.
    """
    return _object(
        {
            "auto_lifecycle_applied": {"const": True},
            "home_pad_entity_id": {"type": "string"},
            "home_hover_enu_m": _number3(),
            "mission_start_enu_m": _number3(),
            "requires_takeoff": {"const": True},
            "requires_landing_or_terminal_resolution": {"const": True},
            "waypoint_repair_applied": {"const": True},
            "assigned_altitude_m": {"type": "number"},
            "terminal_flight_state": {"const": True},
        },
        required=(
            "assigned_altitude_m",
            "auto_lifecycle_applied",
            "home_hover_enu_m",
            "home_pad_entity_id",
            "mission_start_enu_m",
            "requires_landing_or_terminal_resolution",
            "requires_takeoff",
            "waypoint_repair_applied",
        ),
    )


def _observer_lifecycle() -> dict:
    """observer_lifecycle: stamped by the capture-boundary observer pass
    (regenerate_boundary_scenarios.py:4737) on uav_corridor_role=observer
    UAVs. presence / terminal_behavior are producer constants, the distance
    is the 18.0 m hold limit, and capture_boundary_id references the scene
    capture boundary. No L4 window subject is an observer today, so this
    block has no row instance (declared from the executable contract).
    """
    return _object(
        {
            "presence": {"const": "episode_full_duration"},
            "terminal_behavior": {
                "const": "airborne_hold_inside_runtime_boundary"},
            "maximum_terminal_boundary_distance_m": {"type": "number"},
            "capture_boundary_id": {"type": "string"},
        },
        required=(
            "capture_boundary_id",
            "maximum_terminal_boundary_distance_m",
            "presence",
            "terminal_behavior",
        ),
    )


def _uav_corridor() -> dict:
    """uav_corridor: _set_uav_altitude_metadata stamp (7973)."""
    return _object(
        {
            "assigned_altitude_m": {"type": "number"},
            "altitude_layers_m": {
                "type": "array",
                "prefixItems": [{"const": v} for v in _ALTITUDE_LAYERS_M],
                "minItems": len(_ALTITUDE_LAYERS_M),
                "maxItems": len(_ALTITUDE_LAYERS_M),
                "uniqueItems": True,
                "items": False,
            },
            "used_lateral_bypass": {"type": "boolean"},
        },
        required=("assigned_altitude_m", "altitude_layers_m",
                  "used_lateral_bypass"),
    )


def _ground_flow_contract(kind: str | None = None) -> dict:
    """ground_flow_contract: stamped by ensure_background_ground_flow_routes
    on L4 background actors. vehicle routes carry physical_lane_lateral_m
    (sumo_net_directed_lane_geometry, min_xy_span 24 m, speed 6.0 m/s),
    pedestrian routes omit the key (traffic_bundle_sidewalk, 9 m,
    1.25 m/s). planned_path_length_m is the per-actor route length.
    """
    branches = [
            _object(
                {
                    "actor_kind": {"const": "vehicle"},
                    "loop_policy": {
                        "const": "single_pass_route_waypoints"},
                    "min_visible_motion_ratio": {"const": 0.85},
                    "min_xy_span_m": {"const": 24.0},
                    "physical_lane_lateral_m": {"type": "number"},
                    "planned_path_length_m": {"type": "number"},
                    "policy": {"const": "continuous_capture_ground_flow_v1"},
                    "required": {"const": True},
                    "route_duration_ticks": {"const": 900},
                    "route_source": {
                        "const": "sumo_net_directed_lane_geometry"},
                    "speed_mps": {"const": 6.0},
                },
                required=(
                    "actor_kind", "loop_policy", "min_visible_motion_ratio",
                    "min_xy_span_m", "physical_lane_lateral_m",
                    "planned_path_length_m", "policy", "required",
                    "route_duration_ticks", "route_source", "speed_mps",
                ),
            ),
            _object(
                {
                    "actor_kind": {"const": "pedestrian"},
                    "loop_policy": {
                        "const": "single_pass_route_waypoints"},
                    "min_visible_motion_ratio": {"const": 0.85},
                    "min_xy_span_m": {"const": 9.0},
                    "planned_path_length_m": {"type": "number"},
                    "policy": {"const": "continuous_capture_ground_flow_v1"},
                    "required": {"const": True},
                    "route_duration_ticks": {"const": 900},
                    "route_source": {"const": "traffic_bundle_sidewalk"},
                    "speed_mps": {"const": 1.25},
                },
                required=(
                    "actor_kind", "loop_policy", "min_visible_motion_ratio",
                    "min_xy_span_m", "planned_path_length_m", "policy",
                    "required", "route_duration_ticks", "route_source",
                    "speed_mps",
                ),
            ),
    ]
    if kind is None:
        return {"oneOf": branches}
    if kind == "vehicle":
        return branches[0]
    if kind == "pedestrian":
        return branches[1]
    raise ValueError(f"unsupported ground-flow actor kind: {kind}")


def _contract_participant(kind: str) -> dict:
    """background_vehicle / background_pedestrian: the semantic-event-
    contract marker dict written by ensure_contract_vehicle /
    ensure_contract_pedestrian (contract_scenario_id + policy +
    semantic_role) plus the physical lane offset on vehicles. Pedestrian
    rows carry background_vehicle=None? No - background_vehicle is only
    stamped on vehicles; background_pedestrian (same shape, no lateral key)
    is stamped on pedestrians. Both are optional row keys, but their
    physical-lane fields are different producer shapes.
    """
    participant = {
        "contract_scenario_id": {"type": "string"},
        "policy": {"const": "semantic_event_contract_v1"},
        "semantic_role": {"type": "string"},
    }
    if kind == "vehicle":
        participant["physical_lane_lateral_m"] = {"type": "number"}
    elif kind != "pedestrian":
        raise ValueError(f"unsupported background participant kind: {kind}")
    return _object(participant, required=sorted(participant))


def _contract_inspect_uav() -> dict:
    """contract_inspect_uav: the 17-key block _add_contract_inspect_uav
    (regenerate_boundary_scenarios.py:5140) stamps on the inspect UAV scene
    entity; preserved verbatim. The three route arrays share the loop
    geometry (planned -> loop -> repaired), repaired_path_length_m the
    repaired length. sensor_profile is the single uav_sensor_profile.json
    projection (bottom_center camera, editor_hook capture).
    """
    route = _planned_route_waypoints_enu_m()
    return _object(
        {
            "policy": {"const": "semantic_event_contract_v1"},
            "corridor_role": {"const": "inspect_observer"},
            "corridor_policy": {"const": "inspect_capture_corridor"},
            "motion_policy": {
                "const": "fixed_altitude_loop_observes_capture_boundary"},
            "inspect_altitude_m": {"type": "number"},
            "inspect_altitude_code": {
                "type": "string",
                "pattern": "^I(10|18|20|28|30|50|55|85|105)$",
            },
            "sensor_fov_deg": {"const": 85.0},
            "sensor_fov_source": {
                "const": "Config/LowAltitude/uav_sensor_profile.json#hfov_deg"},
            "sensor_profile": _object(
                {
                    "source": {
                        "const": "Config/LowAltitude/uav_sensor_profile.json"},
                    "camera_name": {"const": "bottom_center"},
                    "capture_backend": {"const": "editor_hook"},
                    "width": {"const": 1280},
                    "height": {"const": 720},
                    "hfov_deg": {"const": 85.0},
                    "fixed_rotation_offset_deg": _object(
                        {
                            "pitch_deg": {"const": -90.0},
                            "yaw_deg": {"const": 0.0},
                            "roll_deg": {"const": 0.0},
                        },
                        required=("pitch_deg", "roll_deg", "yaw_deg"),
                    ),
                },
                required=(
                    "camera_name", "capture_backend", "fixed_rotation_offset_deg",
                    "height", "hfov_deg", "source", "width",
                ),
            ),
            "min_path_length_m": {"const": 80.0},
            "fixed_altitude_loop": {"const": True},
            "inspect_fov_coverage_required": {"const": True},
            "planned_route_enu_m": route,
            "loop_route_enu_m": route,
            "repaired_route_enu_m": route,
            "repaired_path_length_m": {"type": "number"},
            "observed_capture_boundary_id": {"type": "string"},
        },
        required=(
            "corridor_policy", "corridor_role", "fixed_altitude_loop",
            "inspect_altitude_code", "inspect_altitude_m",
            "inspect_fov_coverage_required", "loop_route_enu_m",
            "min_path_length_m", "motion_policy",
            "observed_capture_boundary_id", "planned_route_enu_m", "policy",
            "repaired_path_length_m", "repaired_route_enu_m", "sensor_fov_deg",
            "sensor_fov_source", "sensor_profile",
        ),
    )


def _facility_semantic_scope(subtypes: tuple) -> dict:
    """Bind each L4 facility subtype to its frozen scope contract tuple."""
    specs = {
        "landing_pad": ("world:LandingPad", 1),
        "charging_station": ("world:ChargingStation", 1),
        "communication_base_station": ("world:CommunicationStation", None),
        "barrier": ("world:Barrier", None),
    }
    cases = []
    for subtype in subtypes:
        ontology_class, service_capacity = specs[subtype]
        fields = {
            "authority_id": {"const": "facility_scope_contract.v1"},
            "scope_type": {"const": "facility"},
            "scope_subtype": {"const": subtype},
            "ontology_class_id": {"const": ontology_class},
        }
        if service_capacity is not None:
            fields["service_capacity"] = {"const": service_capacity}
        cases.append(_object(fields, required=sorted(fields)))
    return cases[0] if len(cases) == 1 else {"oneOf": cases}


def _contract_facility_block() -> dict:
    """contract_facility: the marker _add_contract_facility (5192) stamps on
    contract facility entities (not on home pads)."""
    return _object(
        {
            "policy": {"const": "semantic_event_contract_v1"},
            "contract_scenario_id": {"type": "string"},
        },
        required=("contract_scenario_id", "policy"),
    )


def arm_states_schema() -> dict:
    """Whole-record JSON Schema for one L4 raw/states.jsonl row.

    Pure schema constructor: no I/O, no registry or compiler interaction.
    The record root is entity-scoped (field_paths.ENTITY_RECORDS contains
    arm_states; root entity_id resolves identity) and time-anchored
    (arm_states is a DIRECT_TICK_FAMILIES member; root tick is the window
    time_point).
    """
    row_core = {
        "tick": {"type": "integer", "minimum": 0, "maximum": 900},
        "entity_id": {"type": "string", "minLength": 1},
        "label_class": _string_enum(_LABEL_CLASSES),
        "asset_id": _string_enum(_ASSETS),
        "pos_enu": _engine_float3(),
        "vel_mps": _engine_float3(),
        "yaw_deg": {"type": "number", "x-source-value-types": ["float"]},
        "state": {"type": "string"},
        "activity_type": {"type": "string"},
        "animation_hint": {"type": "string"},
        "posture": {"type": "string"},
        "social_state": {"type": "string"},
        "task_id": {"type": "string"},
        "role": _string_enum(_ROLES),
        "state_sequence": {
            "oneOf": [
                {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": _string_enum(_SINGLE_ELEMENT_SEQUENCE),
                },
                {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": _string_enum(_DOUBLE_ELEMENT_SEQUENCE),
                },
                {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "prefixItems": [{"const": v} for v in _INSPECT_SEQUENCE],
                    "items": False,
                },
            ]
        },
        "semantic_role": {"type": "string"},
        "activation_tick": {"type": "integer"},
        "category": _string_enum(_CATEGORIES),
        "route_waypoints_enu_m": _route_waypoints_enu_m(),
    }

    # preserved_fields_from(PRESERVED_ENTITY_FIELDS, batch_generate.py:215):
    # per-key presence follows the engine entity, so keys are optional and
    # never null. row.update(preserved_fields_from(entity)) applies them
    # after row_activity_payload.
    preserved = {
        "assigned_altitude_m": {"type": "number"},
        "ground_reference_z_m": {"type": "number"},
        "ground_reference_source": {"type": "string"},
        "uav_corridor_role": _string_enum(_UAV_CORRIDOR_ROLES),
        "planned_route_waypoints_enu_m": _planned_route_waypoints_enu_m(),
        "lifecycle": _lifecycle(),
        "observer_lifecycle": _observer_lifecycle(),
        "uav_corridor": _uav_corridor(),
        "spawn_policy": {"const": "event_script_only"},
        "event_actor_motion_contract": _event_actor_motion_contract(),
        "ground_flow_contract": _ground_flow_contract(),
        "background_vehicle": _contract_participant("vehicle"),
        "background_pedestrian": _contract_participant("pedestrian"),
        "background_role": {"const": "semantic_context"},
        "contract_facility": _contract_facility_block(),
        "contract_inspect_uav": _contract_inspect_uav(),
        "inspect_altitude_code": _string_enum(
            ("I10", "I18", "I20", "I28", "I30", "I50", "I55", "I85", "I105")),
        "inspect_altitude_m": {"type": "number"},
        "min_path_length_m": {"const": 80.0},
        "full_episode_presence": {"const": True},
        "entity_kind": _string_enum(
            ("facility.landing_pad", "facility.charger",
             "facility.base_station", "facility.barrier")),
        "semantic_scope": _facility_semantic_scope(
            ("landing_pad", "charging_station",
             "communication_base_station", "barrier")),
    }

    families = {
        family: _runtime_state_family(family) for family in RUNTIME_STATE_FIELDS
    }

    # Scene builders preserve different optional fields by entity type.
    # Runtime-state actions may patch any declared family on any entity.
    common_keys = set(row_core) | set(families)

    def source_keys(*extras: str) -> dict:
        return {"enum": sorted(common_keys | set(extras))}

    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_states_l4.record.schema.json",
        "title": "arm_states L4 whole-record grammar (raw/states.jsonl row)",
        "type": "object",
        "properties": {
            **row_core,
            **preserved,
            **families,
        },
        "required": sorted(row_core),
        "additionalProperties": False,
        # route_waypoints_enu_m is nonempty iff planned_route_waypoints_enu_m
        # is present (both written from the same entity route; the planned
        # copy is stamped for UAVs at regenerate_boundary_scenarios.py:7576).
        # These conditions live at the record root because a subschema
        # condition on the array can only constrain the array itself.
        "allOf": [
            {
                "if": {
                    "properties": {
                        "route_waypoints_enu_m": {"minItems": 1},
                    },
                    "required": ["route_waypoints_enu_m"],
                },
                "then": {"required": ["planned_route_waypoints_enu_m"]},
            },
            {
                "if": {"required": ["planned_route_waypoints_enu_m"]},
                "then": {
                    "properties": {
                        "route_waypoints_enu_m": {"minItems": 1},
                    },
                },
            },
        ],
        # Producer branch the row belongs to. Branches are mutually
        # exclusive by producer-identity keys (label_class const + role
        # const + asset_id const/enum), so a row satisfies exactly one:
        # the two top branches split by the pedestrian payload constants,
        # the non-pedestrian branch binds each label<->role<->category<->
        # asset_id tuple the producer derives from
        # Config/LowAltitude/asset_catalog.json. Branch-specific preserved
        # constellations ride with each sub-branch (inspect block only on
        # U_inspect, facility fields only on facility labels, ground-flow /
        # participant dicts only on the background roles they are stamped
        # on). Sub-branch preserved keys are optional (preserved_fields_
        # from writes a key only when the entity carried it), so the
        # bindings constrain when a key is allowed, not that it is present.
        "oneOf": [
            {
                # pedestrian rows: activity catalog payload (row_activity_
                # payload uses get_activity, so posture/social_state vary)
                "propertyNames": source_keys(
                    "planned_route_waypoints_enu_m", "spawn_policy",
                    "event_actor_motion_contract", "ground_flow_contract",
                    "background_pedestrian", "background_role"),
                "properties": {
                    "label_class": {"const": "pedestrian"},
                    "role": {"enum": ["semantic_pedestrian",
                                      "semantic_background_pedestrian"]},
                    "category": {"const": "pedestrian"},
                    "asset_id": {"const": "pedestrian.cityops.basic.v1"},
                    "posture": _string_enum(("fallen", "standing")),
                    "social_state": _string_enum(("group", "solo")),
                    "ground_flow_contract": _ground_flow_contract("pedestrian"),
                },
                "allOf": [
                    {
                        "if": {"required": [key]},
                        "then": {"properties": {
                            "role": {"const": "semantic_background_pedestrian"}}},
                    }
                    for key in ("ground_flow_contract", "background_pedestrian",
                                "background_role")
                ],
            },
            {
                # uav / vehicle / asset_anchor / facility / spawn_zone rows:
                # posture standing / social_state solo constants
                "properties": {
                    "posture": {"const": "standing"},
                    "social_state": {"const": "solo"},
                },
                "oneOf": [
                    {
                        # mission UAV (lifecycle / observer / corridor_
                        # observer variants; branch bindings only)
                        "propertyNames": source_keys(
                            "assigned_altitude_m", "ground_reference_z_m",
                            "ground_reference_source", "uav_corridor_role",
                            "planned_route_waypoints_enu_m", "lifecycle",
                            "observer_lifecycle", "uav_corridor",
                            "spawn_policy", "event_actor_motion_contract"),
                        "properties": {
                            "label_class": {"const": "uav"},
                            "role": {"const": "mission_uav"},
                            "category": {"const": "uav"},
                            "asset_id": {"enum": ["uav.airsim.cv_pawn.v1",
                                                  "uav.airsim.flying_pawn.v1",
                                                  "uav.inspect.quad.v1"]},
                            "uav_corridor_role": {
                                "enum": ["mission_uav", "observer"]},
                        },
                    },
                    {
                        # contract inspect UAV (_add_contract_inspect_uav):
                        # fixed four-element state_sequence, the 17-key
                        # contract block and the inspect field constellation
                        "propertyNames": source_keys(
                            "assigned_altitude_m", "ground_reference_z_m",
                            "ground_reference_source", "uav_corridor_role",
                            "planned_route_waypoints_enu_m", "lifecycle",
                            "observer_lifecycle", "uav_corridor",
                            "contract_inspect_uav", "inspect_altitude_code",
                            "inspect_altitude_m", "min_path_length_m",
                            "full_episode_presence"),
                        "properties": {
                            "label_class": {"const": "uav"},
                            "role": {"const": "U_inspect"},
                            "category": {"const": "uav"},
                            "asset_id": {"const": "uav.inspect.quad.v1"},
                            "uav_corridor_role": {
                                "const": "inspect_observer"},
                            "state_sequence": {
                                # positional takeoff -> inspect_racetrack ->
                                # orbit -> land (semantic_state stamp,
                                # regenerate_boundary_scenarios.py:5141)
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "prefixItems": [
                                    {"const": v} for v in _INSPECT_SEQUENCE
                                ],
                                "items": False,
                            },
                            "inspect_altitude_code": {"type": "string"},
                            "min_path_length_m": {"const": 80.0},
                            "full_episode_presence": {"const": True},
                            "ground_reference_z_m": {"const": 0.0},
                            "ground_reference_source": {
                                "const": "map_ground_reference_v1"},
                        },
                        "required": [
                            "contract_inspect_uav",
                            "full_episode_presence",
                            "inspect_altitude_code",
                            "min_path_length_m",
                            "planned_route_waypoints_enu_m",
                            "uav_corridor",
                            "uav_corridor_role",
                        ],
                    },
                    {
                        # vehicles: mission actors (event-actor stamp) and
                        # background flow (ground-flow / participant mark)
                        "propertyNames": source_keys(
                            "planned_route_waypoints_enu_m", "spawn_policy",
                            "event_actor_motion_contract", "ground_flow_contract",
                            "background_vehicle", "background_role"),
                        "properties": {
                            "label_class": {"const": "vehicle"},
                            "role": {"enum": ["semantic_vehicle",
                                              "semantic_background_vehicle"]},
                            "category": {"const": "vehicle"},
                            "asset_id": {"enum": [
                                "vehicle.emergency.ambulance.v1",
                                "vehicle.emergency.suv.v1",
                                "vehicle.ground.boxcar.v1",
                                "vehicle.service.box.v1"]},
                            "ground_flow_contract": _ground_flow_contract(
                                "vehicle"),
                        },
                        "allOf": [
                            {
                                "if": {"required": [key]},
                                "then": {"properties": {
                                    "role": {"const": "semantic_background_vehicle"}}},
                            }
                            for key in ("ground_flow_contract", "background_vehicle",
                                        "background_role")
                        ],
                    },
                    {
                        # facade anchors (facade_entity)
                        "propertyNames": source_keys(),
                        "properties": {
                            "label_class": {"const": "asset_anchor"},
                            "role": {"const": "semantic_context"},
                            "category": {"const": "facade_anchor"},
                            "asset_id": {"const": "semantic.asset_anchor"},
                        },
                    },
                    {
                        # facility entities: home pads (landing_pad) and
                        # contract facilities (facility_cycle: landing pad,
                        # charger, radio tower, barrier). entity_kind is the
                        # canonical single kind of the row's semantic_scope
                        # subtype (canonical_entity_kind_for_subtype,
                        # facility_scope.py:137-147), and the subtype is the
                        # asset_subtype_index resolution of asset_id
                        # (semantic_scope_for_source), so each row is one
                        # frozen-contract tuple label_class <-> asset_id <->
                        # entity_kind <-> scope_subtype. The pairing is
                        # expressed as one const-bound arm per tuple - the
                        # compiler whitelist has no `not` and a subschema
                        # cannot see sibling keys, so enum arms would leave
                        # cross-products open. Explicit ground_station.*
                        # kinds belong to a different source branch and
                        # have no L4 facility asset in this producer.
                        "propertyNames": source_keys(
                            "contract_facility", "entity_kind",
                            "semantic_scope"),
                        "properties": {
                            "role": {"const": "semantic_facility"},
                            "category": {"const": "facility"},
                        },
                        "oneOf": [
                            {
                                "properties": {
                                    "label_class": {"const": "landing_pad"},
                                    "asset_id": {
                                        "const": "facility.landing_pad.visible.v1"},
                                    "entity_kind": {
                                        "const": "facility.landing_pad"},
                                    "semantic_scope": _facility_semantic_scope(
                                        ("landing_pad",)),
                                    "state_sequence": {
                                        "type": "array", "minItems": 1,
                                        "maxItems": 1,
                                        "items": {"const": "available"}},
                                },
                                "required": ["entity_kind", "semantic_scope"],
                            },
                            {
                                "properties": {
                                    "label_class": {"const": "charging_pile"},
                                    "asset_id": {
                                        "const": "facility.charger.cityops.v1"},
                                    "entity_kind": {
                                        "const": "facility.charger"},
                                    "semantic_scope": _facility_semantic_scope(
                                        ("charging_station",)),
                                    "state_sequence": {
                                        "type": "array", "minItems": 1,
                                        "maxItems": 1,
                                        "items": {"const": "available"}},
                                },
                                "required": ["entity_kind", "semantic_scope"],
                            },
                            {
                                "properties": {
                                    "label_class": {"const": "radio_tower"},
                                    "asset_id": {
                                        "const": "facility.radio.base_tower.v1"},
                                    "entity_kind": {
                                        "const": "facility.base_station"},
                                    "semantic_scope": _facility_semantic_scope(
                                        ("communication_base_station",)),
                                    "state_sequence": {
                                        "type": "array", "minItems": 1,
                                        "maxItems": 1,
                                        "items": {"const": "online"}},
                                },
                                "required": ["entity_kind", "semantic_scope"],
                            },
                            {
                                "properties": {
                                    "label_class": {"const": "barrier"},
                                    "asset_id": {"const": "facility.barrier.basic"},
                                    "entity_kind": {"const": "facility.barrier"},
                                    "semantic_scope": _facility_semantic_scope(
                                        ("barrier",)),
                                    "state_sequence": {
                                        "type": "array", "minItems": 1,
                                        "maxItems": 1,
                                        "items": {"const": "available"}},
                                },
                                "required": ["entity_kind", "semantic_scope"],
                            },
                        ],
                    },
                    {
                        # spawn-zone crowd anchors (L4-8 crowd_anchor_{sid}):
                        # four-key initial-state projection. The current L4
                        # builder writes no other preserved field on these
                        # anchors; set_runtime_state can still add a runtime
                        # family to any engine entity. Restrict the root key
                        # names here instead of generating dead if/then arms
                        # for every forbidden preserved key.
                        "propertyNames": source_keys(),
                        "properties": {
                            "label_class": {"const": "spawn_zone"},
                            "role": {"const": "semantic_context"},
                            "category": {"const": "crowd_anchor"},
                            "asset_id": {"const": "semantic.spawn_zone"},
                            "state_sequence": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 1,
                                "items": {"const": "idle"},
                            },
                        },
                    },
                ],
            },
        ],
    }
