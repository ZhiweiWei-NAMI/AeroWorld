"""Walk source values with one field template and a separate source location."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Iterator, Mapping

from Dataset.tools.runtime_state_contract import RUNTIME_STATE_FIELDS
from Dataset.world_model.graph.entity_identity import action_result_target, action_target_scope
from Dataset.world_model.graph.structure_schema import is_publication_excluded_path


SNAPSHOT_DICTIONARIES = frozenset({
    "render_truth_snapshots_by_tick",
    "source_truth_snapshots_by_tick",
})
SNAPSHOT_ENTITY_PATHS = frozenset(f"{name}.{{tick}}" for name in SNAPSHOT_DICTIONARIES)

# These objects are keyed by a source entity ID. Each entry is a value for one
# entity, not another field definition. The path is matched structurally; an
# identifier elsewhere in a record remains ordinary source data.
ENTITY_KEYED_OBJECTS: dict[str, frozenset[str]] = {
    "formal_truth_frame": frozenset({"entity_motion_state", "sumo_traffic_light_states"}),
    "arm_local_business_truth": frozenset({
        "metrics.destination_distance_m", "metrics.destinations_enu_m",
    }),
    "arm_predicate_truth": frozenset({"metrics.speeds_mps"}),
    "arm_predicate_truth_ticks": frozenset({"metrics.speeds_mps"}),
    "domain_state": frozenset({
        "values.pedestrian_displacement_m", "values.red_duration_ticks_by_controller",
        "values.queue_sampling_window.stopped_vehicle_count_by_lane",
        "values.target_pedestrian_distances_m", "values.target_pedestrian_safe_dwell_ticks",
    }),
    "arm_script_plan": frozenset({
        "parameters.fixed_uav_assigned_altitudes_m",
        "parameters.l1_4_corridor_congestion_contract.inside_sample_positions_enu_m",
        "parameters.uav_assigned_altitudes_m",
        "parameters.uav_lateral_bypass_used",
        "parameters.x1_physical_chain.crowd_clearance_targets_enu_m",
        "parameters.x6_physical_chain.safe_targets_enu_m",
    }),
}

ENTITY_LISTS: dict[str, frozenset[str]] = {
    "formal_truth_frame": frozenset({"entities"}),
    "formal_event_realization": frozenset({"action_realizations"}),
    "formal_roster": frozenset({""}),
    "arm_branch_roster": frozenset({""}),
    "arm_scene_setup": frozenset({"", "entities"}),
    "formal_world_truth_graph_base": frozenset({"scope_entities"}),
    "arm_script_plan": frozenset({
        "events[].actions",
        "parameters.uav_corridor_segments",
        "parameters.uav_corridor_segment_details",
        "parameters.utm_service_plan.uav_plans",
    }),
    "predicate_truth_matrix": frozenset({"scope_values"}),
}

ENTITY_RECORDS = frozenset({"arm_states", "arm_trajectories"})

# Runtime-state family names under a visual_state / runtime-state carrier
# (Dataset/tools/runtime_state_contract.py). Those keys, and the other fixed
# business keys below, are named fields and must stay literal; only the open
# pattern keys of the same maps bind a dimension.
_VISUAL_FIXED_KEYS = frozenset({
    "mode", "lights_on", "initial_state", "visual_state", "runtime_state",
}) | frozenset(RUNTIME_STATE_FIELDS)
_CARRIER_FIXED_KEYS = frozenset({"runtime_state"}) | frozenset(RUNTIME_STATE_FIELDS)
# formal_episode_manifest traffic_profile.seed_semantics names its three real
# seeds as fixed properties in L6/X; L1-L5 express the same keys as the open
# pattern ^seed[0-9]{2}$ with required [seed00, seed01, seed02].  Both shapes
# describe the same three business keys, so they stay literal and only other
# seed keys bind the dimension.
_SEED_SEMANTIC_KEYS = frozenset({"seed00", "seed01", "seed02"})

# Mixed maps carry named business fields and open pattern keys side by side
# (properties + patternProperties). Runtime iter_field_values and the grammar
# _field_path projection share this declaration so both keep the fixed keys
# literal and template only the pattern keys.
MIXED_KEYED_OBJECTS: dict[str, dict[str, tuple[str, frozenset[str]]]] = {
    "arm_actions": {
        "action.visual_state": ("builder_field", _VISUAL_FIXED_KEYS),
        "action.visual_state.initial_state": ("builder_field", _CARRIER_FIXED_KEYS),
        "action.visual_state.visual_state": ("builder_field", _CARRIER_FIXED_KEYS),
    },
    "formal_truth_frame": {
        "entities[].sumo_vehicle.semantic_vehicle_state.visual_state": (
            "builder_field", frozenset({"lights_on", "mode"})),
    },
    "formal_episode_manifest": {
        # Traffic-profile seed semantics carry the three named seeds as business
        # keys (fixed properties in L6/X, required keys of the seed pattern in
        # L1-L5).  Only other seed keys bind the seed_profile dimension.
        "source_vehicle_authority.explicit_vehicle_plan.traffic_profile.seed_semantics":
            ("seed_profile", _SEED_SEMANTIC_KEYS),
        "sumo_explicit_vehicle_plan.traffic_profile.seed_semantics":
            ("seed_profile", _SEED_SEMANTIC_KEYS),
        "sumo_traffic.explicit_vehicle_plan.traffic_profile.seed_semantics":
            ("seed_profile", _SEED_SEMANTIC_KEYS),
    },
}

# Per-runtime-family measurement maps hold open state-field keys
# (arm_predicate_ticks_x_structure._runtime_row / _pedestrian_row).
_MEASUREMENT_STATE_MAPS = {
    f"measurements.{family}": "state_field"
    for family in (
        "communication_state", "control_state", "facility_state",
        "incident_state", "mission_state", "navigation_state",
        "pedestrian_state",
    )
}

# X's pedestrian emitter preserves None when this entity has no pedestrian
# state (x_arm_pipeline.calculate and arm_predicate_ticks_x_structure).
NULLABLE_DYNAMIC_OBJECTS = {
    family: frozenset({"measurements.pedestrian_state"})
    for family in ("arm_predicate_truth", "arm_predicate_truth_ticks")
}

DYNAMIC_KEYED_OBJECTS: dict[str, dict[str, str]] = {
    "arm_script_plan": {
        "parameters.deterministic_sumo_traffic_template.seed_profiles": "seed_profile",
    },
    "arm_manifest": {
        # X ARM manifests key source_refs by source path; L3 window status keys
        # event_family_counts by event family.  Both are real data kept as
        # dynamic maps.  source_refs is re-derived per record below so L5/L6
        # fixed-key source_refs stay literal.
        "source_refs": "source_path",
        "window_status.event_family_counts": "event_family",
    },
    "arm_predicate_truth": dict(_MEASUREMENT_STATE_MAPS),
    "arm_predicate_truth_ticks": dict(_MEASUREMENT_STATE_MAPS),
    "formal_world_truth_graph_base": {
        "summary.initial_candidate_counts_by_predicate": "predicate",
    },
    "formal_truth_frame": {
        # roster_summary.by_category counts entities per entity_category, so the
        # category-token keys are a real dynamic map.  Bound in both runtime
        # iter_field_values and grammar _field_path so their field paths agree.
        "roster_summary.by_category": "category",
    },
    "predicate_truth_matrix": {
        "scope_values[].predicate_values": "predicate",
    },
    "formal_episode_manifest": {
        # Blocking-asset radius maps are keyed by blocking asset id.
        "capture_truth_sync.stats.road_semantics.road_blocking_asset_radius_m": "asset_id",
        "capture_visible_truth_filter.stats.road_semantics.road_blocking_asset_radius_m": "asset_id",
        # Replaced-source counts are keyed by vehicle role.
        "source_vehicle_authority.replaced_source_vehicle_counts_by_role": "role",
        # SUMO traffic maps are keyed by SUMO vehicle id.
        "sumo_traffic.canonical_asset_records": "vehicle_id",
        "sumo_traffic.required_vehicle_lifecycle_policy.source_lifecycle_by_vehicle_id": "vehicle_id",
        "sumo_traffic.required_vehicle_lifecycle_policy.truth_lifecycle_by_vehicle_id": "vehicle_id",
        "sumo_traffic.selection.entity_ids": "vehicle_id",
        "sumo_traffic.selection.frames_seen_by_vehicle_id": "vehicle_id",
        "sumo_traffic.selection.max_speed_mps_by_vehicle_id": "vehicle_id",
        "sumo_traffic.selection.min_distance_m_by_vehicle_id": "vehicle_id",
        "sumo_traffic.selection.motion_span_m_by_vehicle_id": "vehicle_id",
        # UAV global-flow maps are keyed by UAV id (entity identity kept out of
        # the entity dimension on purpose).
        "uav_global_flow.selection.entity_ids": "uav_id",
        "uav_global_flow.selection.frames_seen_by_uav_id": "uav_id",
        "uav_global_flow.selection.max_speed_mps_by_uav_id": "uav_id",
        "uav_global_flow.selection.min_distance_m_by_uav_id": "uav_id",
        "uav_global_flow.selection.mission_type_by_uav_id": "uav_id",
        "uav_global_flow.selection.motion_span_m_by_uav_id": "uav_id",
        "uav_global_flow.selection.task_ids": "uav_id",
        "uav_global_flow.source.task_count_by_type": "task_type",
    },
    "formal_objective_manifest": {
        "runtime_state_materialization.event_fire_ticks": "event_id",
        "world_truth_summary.evaluated_candidate_counts_by_predicate": "predicate",
        "world_truth_summary.value_counts_by_predicate": "predicate",
    },
}
DOMAIN_BRANCH_KEY = "observation_family"
FACILITY_SUBTYPE_KEY = "values.facility_subtype"
_BRANCH_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")

# ``arm_window_semantics`` serializes predicate evidence as tagged union items:
# ``{"path": <state-contract field>, "value": <scalar>}``.  These are the
# reviewed numeric branches used by the L3 predicate windows.  Units and valid
# predicate/path pairs are compiled below from the executable core predicate
# contracts; this roster only fixes the authorized materialization surface.
WORLD_OBSERVATION_LIST_PATH = (
    "evidence.observations[].value.world_observations"
)
WORLD_OBSERVATION_VALUE_PATH = WORLD_OBSERVATION_LIST_PATH + "[].value"
WORLD_OBSERVATION_NUMERIC_PATHS = frozenset({
    "derived.nearest_aircraft_distance_m",
    "derived.nearest_building_distance_m",
    "derived.nearest_ground_vehicle_distance_m",
    "derived.nearest_pedestrian_distance_m",
    "domain.pad_facility.capacity",
    "domain.pad_facility.requester_count",
    "geometry.assigned_altitude_m",
    "geometry.local_ground_reference_z_m",
    "geometry.minimum_restricted_boundary_distance_m",
    "geometry.position_z_m",
    "geometry.speed_mps",
    "geometry.xy_distance_to_assigned_landing_zone_m",
    "geometry.xy_distance_to_home_pad_m",
    "geometry.z_agl_m",
    "scene.maximum_corridor_capacity",
    "scene.maximum_corridor_occupancy_count",
    "weather.fog_density",
    "weather.illumination_lux",
    "weather.rain",
    "weather.temperature_c",
})
_WORLD_OBSERVATION_PATH = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z")
_WORLD_OBSERVATION_VARIANT_PREFIX = WORLD_OBSERVATION_VALUE_PATH + "[path="


@lru_cache(maxsize=1)
def world_observation_numeric_contracts() -> dict[str, dict[str, Any]]:
    """Compile units and valid predicates for the reviewed numeric branches."""
    from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates

    found: dict[str, dict[str, set[str]]] = {}
    for template in get_core_predicate_templates():
        if template.get("implementation_status") != "executable_l1":
            continue
        predicate_id = template.get("id")
        state_contract = template.get("state_contract")
        fields = state_contract.get("required_fields", ()) if isinstance(state_contract, Mapping) else ()
        if not isinstance(predicate_id, str) or not isinstance(fields, list):
            raise ValueError("executable predicate has an invalid state contract")
        for field in fields:
            if not isinstance(field, Mapping) or field.get("field") not in WORLD_OBSERVATION_NUMERIC_PATHS:
                continue
            path = field["field"]
            unit = field.get("unit")
            source = field.get("source")
            if (not isinstance(unit, str) or not unit or
                    not isinstance(source, str) or not source):
                raise ValueError(f"numeric world observation lacks unit authority: {predicate_id}:{path}")
            item = found.setdefault(path, {"units": set(), "predicate_ids": set(), "sources": set()})
            item["units"].add(unit)
            item["predicate_ids"].add(predicate_id)
            item["sources"].add(source)
    if set(found) != set(WORLD_OBSERVATION_NUMERIC_PATHS):
        missing = sorted(set(WORLD_OBSERVATION_NUMERIC_PATHS) - set(found))
        extra = sorted(set(found) - set(WORLD_OBSERVATION_NUMERIC_PATHS))
        raise ValueError(f"numeric world-observation contract roster differs: missing={missing}, extra={extra}")
    result: dict[str, dict[str, Any]] = {}
    for path, item in sorted(found.items()):
        if len(item["units"]) != 1:
            raise ValueError(f"numeric world observation has conflicting units: {path}:{sorted(item['units'])}")
        result[path] = {
            "unit": next(iter(item["units"])),
            "predicate_ids": tuple(sorted(item["predicate_ids"])),
            "sources": tuple(sorted(item["sources"])),
        }
    return result


def world_observation_variant_field_path(source_family: str, field_path: str,
                                         source_path_key: str) -> str:
    """Name one semantic branch while leaving its raw source location intact."""
    if source_family != "arm_predicate_truth" or field_path != WORLD_OBSERVATION_VALUE_PATH:
        raise ValueError(f"world-observation discriminator is inapplicable: {source_family}:{field_path}")
    if source_path_key not in world_observation_numeric_contracts():
        raise ValueError(f"numeric world observation has no declared branch: {source_path_key!r}")
    return _WORLD_OBSERVATION_VARIANT_PREFIX + source_path_key + "]"


def world_observation_path_key(source_family: str, field_path: str) -> str | None:
    """Decode a semantic catalog branch created by the tagged-union projection."""
    if source_family != "arm_predicate_truth" or not field_path.startswith(
            _WORLD_OBSERVATION_VARIANT_PREFIX) or not field_path.endswith("]"):
        return None
    key = field_path[len(_WORLD_OBSERVATION_VARIANT_PREFIX):-1]
    if (_WORLD_OBSERVATION_PATH.fullmatch(key) is None or
            world_observation_variant_field_path(source_family, WORLD_OBSERVATION_VALUE_PATH, key)
            != field_path):
        raise ValueError(f"world-observation branch path is invalid: {field_path}")
    return key


def source_branch_for_record(source_family: str, record: Mapping[str, Any]) -> str | None:
    """Read the producer's payload branch before walking a record or subtree."""
    if source_family != "domain_state":
        return None
    branch = record.get(DOMAIN_BRANCH_KEY)
    if not isinstance(branch, str) or _BRANCH_NAME.fullmatch(branch) is None:
        raise ValueError(f"domain-state record lacks a valid {DOMAIN_BRANCH_KEY}: {branch!r}")
    return branch


def source_subtype_for_record(source_family: str, record: Mapping[str, Any]) -> str | None:
    """Keep facility production subtypes separate under pad_facility."""
    if source_branch_for_record(source_family, record) != "pad_facility":
        return None
    from Dataset.semantic_truth.facility_scope import load_facility_scope_contract

    values = record.get("values")
    subtype = values.get("facility_subtype") if isinstance(values, Mapping) else None
    if (not isinstance(subtype, str) or
            subtype not in load_facility_scope_contract()["subtypes"]):
        raise ValueError(f"pad-facility record lacks a declared {FACILITY_SUBTYPE_KEY}: {subtype!r}")
    return subtype


def source_field_id(source_family: str, field_path: str,
                    source_branch: str | None = None,
                    source_subtype: str | None = None) -> str:
    """Build the same branch-qualified source ID for catalogs and value readers."""
    if source_family == "domain_state":
        if not isinstance(source_branch, str) or _BRANCH_NAME.fullmatch(source_branch) is None:
            raise ValueError(f"domain-state field lacks a valid {DOMAIN_BRANCH_KEY}: {field_path}")
        if source_branch == "pad_facility":
            from Dataset.semantic_truth.facility_scope import load_facility_scope_contract

            if source_subtype not in load_facility_scope_contract()["subtypes"]:
                raise ValueError(f"pad-facility field lacks a declared {FACILITY_SUBTYPE_KEY}: {field_path}")
            branch = f"{DOMAIN_BRANCH_KEY}={source_branch},{FACILITY_SUBTYPE_KEY}={source_subtype}"
        else:
            if source_subtype is not None:
                raise ValueError(f"unexpected facility subtype on {source_branch}:{field_path}")
            branch = f"{DOMAIN_BRANCH_KEY}={source_branch}"
        return f"{source_family}[{branch}]:{field_path}"
    if source_branch is not None or source_subtype is not None:
        raise ValueError(f"unexpected source branch for {source_family}:{field_path}")
    return f"{source_family}:{field_path}"


def catalog_source_branch(row: Mapping[str, Any]) -> str | None:
    """Validate a catalog row's source predicate before using its field ID."""
    family = row["source_family"]
    observation_key = world_observation_path_key(family, row["field_path"])
    if observation_key is not None:
        contract = world_observation_numeric_contracts()[observation_key]
        if (row.get("source_path_key") != observation_key or
                row.get("predicate_ids") != list(contract["predicate_ids"])):
            raise ValueError(f"world-observation field lacks its compiled branch authority: {row['id']}")
    elif family == "arm_predicate_truth" and row["field_path"] == WORLD_OBSERVATION_VALUE_PATH:
        if row.get("source_path_key") is not None or row.get("predicate_ids") is not None:
            raise ValueError(f"unqualified world-observation field carries a branch predicate: {row['id']}")
    predicate = row.get("source_branch")
    if family == "domain_state":
        if not isinstance(predicate, dict) or DOMAIN_BRANCH_KEY not in predicate:
            raise ValueError(f"domain-state field lacks its branch predicate: {row['id']}")
        branch = predicate[DOMAIN_BRANCH_KEY]
        expected_keys = ({DOMAIN_BRANCH_KEY, FACILITY_SUBTYPE_KEY}
                         if branch == "pad_facility" else {DOMAIN_BRANCH_KEY})
        if set(predicate) != expected_keys:
            raise ValueError(f"source branch predicate differs from producer structure: {row['id']}")
        subtype = predicate.get(FACILITY_SUBTYPE_KEY)
    else:
        if predicate is not None:
            raise ValueError(f"unexpected source branch on field: {row['id']}")
        branch, subtype = None, None
    expected = source_field_id(family, row["field_path"], branch, subtype)
    if row["id"] != expected:
        raise ValueError(f"source field ID differs from its path and branch: {row['id']}")
    return branch


def catalog_source_subtype(row: Mapping[str, Any]) -> str | None:
    catalog_source_branch(row)
    return (row.get("source_branch") or {}).get(FACILITY_SUBTYPE_KEY)


@dataclass(frozen=True)
class FieldValue:
    """source_tokens address source_file/source_line; input_tokens address the walked value."""
    field_path: str
    snapshot_tick: int | None
    raw_path: str
    value: Any
    element_positions: tuple[int, ...]
    source_tokens: tuple[str | int, ...]
    input_tokens: tuple[str | int, ...]
    entity_id: str | None = None
    identity_status: str = "not_applicable"
    identity_reason: str | None = None
    declared_entity_id: str | None = None
    entity_namespace: str | None = None
    target_scope: str | None = None
    value_tick: int | None = None
    time_role: str | None = None
    endpoint_role: str | None = None
    source_branch: str | None = None
    source_subtype: str | None = None
    predicate_id: str | None = None
    source_path_key: str | None = None


def source_value_at_tokens(record: Any, tokens: tuple[str | int, ...]) -> Any:
    """Read tokens from one complete source-file record; string keys stay whole."""
    current = record
    for token in tokens:
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise TypeError(f"source location expects an object at {token!r}")
        elif type(token) is int:
            if not isinstance(current, list):
                raise TypeError(f"source location expects an array at {token}")
        else:
            raise TypeError(f"source location token is not a key or array index: {token!r}")
        current = current[token]
    return current


def source_path_from_tokens(tokens: tuple[str | int, ...]) -> str:
    path = ""
    for token in tokens:
        if type(token) is int:
            path += f"[{token}]"
        else:
            path += ("." if path else "") + token
    return path


def manifest_source_ref_dimension(record: Any) -> str | None:
    if not isinstance(record, dict) or "source_refs" not in record:
        return None
    schema = record.get("schema")
    if schema in {"aero_l6_v2_arm_manifest", "aeroworld_l5_candidate_window"}:
        return None
    if ("schema" not in record and
            {"sources", "branch_changes", "removed_event_ids", "semantics"} <= record.keys()):
        refs = record["source_refs"]
        if (not isinstance(refs, dict) or not refs or
                any(not isinstance(key, str) or not key or value != key
                    for key, value in refs.items())):
            raise ValueError("X ARM manifest source_refs differs from path-key producer")
        return "source_path"
    raise ValueError(f"ARM manifest source_refs has no declared producer/schema: {schema!r}")


def iter_field_values(value: Any, *, source_family: str,
                      field_prefix: str = "",
                      source_prefix_tokens: tuple[str | int, ...] = (),
                      entity_key_namespaces: Mapping[str, str] | None = None,
                      source_branch: str | None = None,
                      source_subtype: str | None = None) -> Iterator[FieldValue]:
    """Yield every leaf, including empty containers, in source traversal order.

    Only declared keyed objects become template dimensions. source_tokens are
    relative to the complete source-file record; input_tokens are relative to
    value passed here. Array positions include indices in the source prefix.
    """

    if field_prefix and not source_prefix_tokens:
        raise ValueError(f"source subtree lacks its source-record location: {source_family}:{field_prefix}")

    if source_family == "domain_state":
        if not field_prefix:
            if not isinstance(value, Mapping):
                raise TypeError("domain-state record is not an object")
            actual_branch = source_branch_for_record(source_family, value)
            actual_subtype = source_subtype_for_record(source_family, value)
            if source_branch is not None and source_branch != actual_branch:
                raise ValueError("domain-state source branch differs from record")
            if source_subtype is not None and source_subtype != actual_subtype:
                raise ValueError("domain-state facility subtype differs from record")
            source_branch = actual_branch
            source_subtype = actual_subtype
        source_field_id(source_family, field_prefix, source_branch, source_subtype)
    elif source_branch is not None or source_subtype is not None:
        raise ValueError(f"unexpected source branch for {source_family}")

    initial_predicate_id: str | None = None
    if source_family == "arm_predicate_truth" and not field_prefix:
        if not isinstance(value, Mapping):
            raise TypeError("ARM predicate truth row is not an object")
        candidate = value.get("predicate_id")
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("ARM predicate truth row lacks a nonempty predicate_id")
        initial_predicate_id = candidate

    dynamic_objects = dict(DYNAMIC_KEYED_OBJECTS.get(source_family, {}))
    mixed_objects = MIXED_KEYED_OBJECTS.get(source_family, {})
    if source_family == "arm_manifest":
        dimension = manifest_source_ref_dimension(value)
        if dimension is not None:
            dynamic_objects["source_refs"] = dimension
        else:
            # L5/L6 fixed-key source_refs are not a path-keyed producer, so the
            # static arm_manifest source_refs template must not bind their keys.
            dynamic_objects.pop("source_refs", None)

    def entity_ref(entity: Any, declared: Any = None,
                   namespace: str = "sample") -> tuple[str | None, str, str | None, str | None, str]:
        if not isinstance(entity, str) or not entity:
            return None, "unresolved", "entity ID is missing or not a nonempty string", None, namespace
        if declared is not None and declared != entity:
            return entity, "unresolved", "object key and inner entity_id differ", str(declared), namespace
        return entity, "source_reference", None, declared if isinstance(declared, str) else None, namespace

    initial_scope = (action_target_scope(value) if isinstance(value, dict) and
                     ((source_family == "formal_event_realization" and
                       field_prefix == "action_realizations[].") or
                      (source_family == "arm_script_plan" and
                       field_prefix == "events[].actions[].")) else None)
    initial_entity = ((None, "not_applicable", None, None, None) if initial_scope else
                      entity_ref(value.get("node_id"), namespace="compute_node")
                      if source_family == "compute_state" and isinstance(value, dict) else
                      entity_ref(value.get("entity_id"), namespace=(
                          "compute_node" if source_family == "formal_world_truth_graph_base"
                          and field_prefix == "scope_entities[]." and
                          value.get("scope_type") == "compute_node" else "sample"))
                      if isinstance(value, dict) and
                      (source_family in ENTITY_RECORDS or
                       (source_family == "formal_truth_frame" and field_prefix == "entities[].") or
                       (source_family == "formal_world_truth_graph_base" and
                        field_prefix == "scope_entities[].") or
                       (source_family == "formal_event_realization" and
                        field_prefix == "action_realizations[].") or
                       (source_family == "arm_script_plan" and
                        field_prefix == "events[].actions[].") or
                       (source_family in {"formal_roster", "arm_branch_roster", "arm_scene_setup"}
                        and len(source_prefix_tokens) == 2
                        and source_prefix_tokens[0] == "entities"
                        and type(source_prefix_tokens[1]) is int))
                      else (None, "not_applicable", None, None, None))

    def value_time(path: str, snapshot_tick: int | None, item: Any) -> tuple[int | None, str | None]:
        if snapshot_tick is not None:
            return snapshot_tick, "snapshot_tick"
        if source_family == "arm_predicate_transitions" and path in {"from_tick", "to_tick"}:
            if (not isinstance(value, dict) or type(item) is not int or
                    type(value.get("to_tick")) is not int or
                    type(value.get("from_tick")) is not int or
                    value["from_tick"] >= value["to_tick"]):
                raise TypeError(f"transition endpoints are not ordered integer ticks: {source_path_from_tokens(source_prefix_tokens)}")
            return value["to_tick"], "transition_record_tick"
        if source_family == "arm_predicate_truth_ticks" and path == "metrics.approach_dispatch_tick":
            if item is not None and type(item) is not int:
                raise TypeError(f"approach dispatch tick is not an integer: {item!r}")
            return item, "action_dispatch_tick"
        if source_family == "arm_local_business_truth":
            if path in {
                "metrics.arrival_dispatch_tick", "metrics.handoff_dispatch_tick",
                "metrics.landing_dispatch_tick", "metrics.dispatch_action_tick",
            }:
                if item is not None and type(item) is not int:
                    raise TypeError(f"action dispatch tick is not an integer: {path}: {item!r}")
                return item, "action_dispatch_tick"
            if (path.startswith("metrics.destinations_enu_m.") or path in {
                "metrics.distance_limit_m", "metrics.horizontal_limit_m",
                "metrics.vertical_limit_m", "metrics.handoff_delay_ticks",
                "metrics.required_dwell_ticks",
            }):
                return None, "planned_parameter"
        if source_family == "formal_roster" and path in {
            "initial_position_enu_m[]", "initial_yaw_deg",
        }:
            return None, "source_time_unresolved"
        return None, None

    def leaf(path: str, snapshot_tick: int | None, raw_path: str, item: Any,
             input_tokens: tuple[str | int, ...],
             entity: tuple[str | None, str, str | None, str | None, str | None],
             target_scope: str | None, predicate_id: str | None,
             source_path_key: str | None) -> FieldValue:
        value_tick, time_role = value_time(path, snapshot_tick, item)
        endpoint_role = ("transition_from_endpoint" if path == "from_tick" else
                         "transition_to_endpoint" if path == "to_tick" else None
                         ) if source_family == "arm_predicate_transitions" else None
        source_tokens = source_prefix_tokens + input_tokens
        positions = tuple(token for token in source_tokens if type(token) is int)
        return FieldValue(field_prefix + path, snapshot_tick, raw_path, item,
                          positions, source_tokens, input_tokens, *entity,
                          target_scope, value_tick, time_role, endpoint_role,
                          source_branch, source_subtype, predicate_id, source_path_key)

    def walk(item: Any, field_path: str, raw_path: str,
             input_tokens: tuple[str | int, ...], snapshot_tick: int | None,
             entity: tuple[str | None, str, str | None, str | None, str | None],
             tick_keys: bool = False, entity_keys: bool = False,
             target_scope: str | None = None, dynamic_key: str | None = None,
             predicate_id: str | None = None,
             source_path_key: str | None = None,
             mixed_fixed: frozenset[str] | None = None) -> Iterator[FieldValue]:
        if isinstance(item, dict):
            if not item:
                yield leaf(field_path, snapshot_tick, raw_path, item, input_tokens,
                           entity, target_scope, predicate_id, source_path_key)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"source object key is not a string: {raw_path}: {key!r}")
                if is_publication_excluded_path(input_tokens + (key,)):
                    continue
                child_raw = f"{raw_path}.{key}" if raw_path else key
                if (source_family == "formal_event_realization" and entity_keys
                        and snapshot_tick is not None and field_path in SNAPSHOT_ENTITY_PATHS):
                    if not isinstance(child, dict):
                        raise TypeError(f"entity snapshot is not an object: {child_raw}")
                    if child.get("present") is True and "tick" not in child:
                        raise ValueError(f"present entity snapshot lacks its tick value: {child_raw}")
                    if "tick" in child and (type(child["tick"]) is not int
                                            or child["tick"] != snapshot_tick):
                        raise ValueError(f"entity snapshot tick differs from dictionary key: {child_raw}")
                if tick_keys:
                    if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                        raise ValueError(f"snapshot tick key is not a canonical integer: {child_raw}")
                    if not isinstance(child, dict):
                        raise TypeError(f"snapshot tick value is not an object: {child_raw}")
                    child_tick = int(key)
                    child_field = f"{field_path}.{{tick}}"
                    child_entity_keys = source_family == "formal_event_realization"
                else:
                    child_tick = snapshot_tick
                    child_entity_keys = False
                    child_field = f"{field_path}.{{entity}}" if entity_keys else (
                        f"{field_path}.{key}" if (dynamic_key and mixed_fixed is not None
                                                 and key in mixed_fixed) else
                        f"{field_path}.{{{dynamic_key}}}" if dynamic_key else
                        f"{field_path}.{key}" if field_path else key)
                child_predicate_id = key if dynamic_key == "predicate" else predicate_id
                child_source_path_key = key if dynamic_key == "source_path" else source_path_key
                if source_family == "domain_state" and field_prefix + field_path == "values.red_duration_ticks_by_controller":
                    if entity_key_namespaces is None or key not in entity_key_namespaces:
                        raise ValueError(f"controller key lacks current-frame namespace evidence: {child_raw}")
                    namespace = entity_key_namespaces[key]
                elif (source_family == "domain_state" and field_prefix + field_path ==
                      "values.queue_sampling_window.stopped_vehicle_count_by_lane"):
                    namespace = "sumo_lane"
                else:
                    namespace = ("sumo_tls_controller" if source_family == "formal_truth_frame"
                                 and field_prefix + field_path == "sumo_traffic_light_states" else "sample")
                child_entity = (entity_ref(key, child.get("entity_id") if isinstance(child, dict) else None,
                                           namespace) if entity_keys else entity)
                if (source_family == "domain_state" and not field_prefix and
                        not field_path and key == "values" and isinstance(value, dict) and
                        value.get("observation_family") == "gnss_navigation"):
                    child_entity = entity_ref(value.get("subject_id"))
                    if value.get("subject_category") != "uav":
                        child_entity = (child_entity[0], "unresolved",
                                        "GNSS observation subject is not declared as a UAV",
                                        child_entity[3], child_entity[4])
                child_scope = target_scope
                if (source_family == "arm_actions" and not field_prefix and not field_path
                        and key in {"action", "result"} and isinstance(child, dict)):
                    action = value.get("action", {}) if isinstance(value, dict) else {}
                    child_scope = action_target_scope(action)
                    child_entity = ((None, "not_applicable", None, None, None)
                                    if child_scope else entity_ref(
                                        child.get("entity_id") or action_result_target(action, child)
                                        if key == "result" else child.get("entity_id")))
                is_snapshot_dictionary = (source_family == "formal_event_realization"
                                          and not field_prefix and not field_path
                                          and key in SNAPSHOT_DICTIONARIES)
                if is_snapshot_dictionary and not isinstance(child, dict):
                    raise TypeError(f"snapshot dictionary is not an object: {child_raw}")
                keyed = (field_prefix + child_field in ENTITY_KEYED_OBJECTS.get(source_family, ())
                         and not tick_keys)
                if keyed and not isinstance(child, dict):
                    raise TypeError(f"entity-keyed source object is not an object: {child_raw}")
                child_template_path = field_prefix + child_field
                if child_template_path in dynamic_objects:
                    next_dynamic, next_mixed = dynamic_objects[child_template_path], None
                else:
                    next_mixed = mixed_objects.get(child_template_path)
                    next_dynamic = next_mixed[0] if next_mixed is not None else None
                if next_dynamic is not None and not isinstance(child, dict):
                    if (child is None and child_template_path in
                            NULLABLE_DYNAMIC_OBJECTS.get(source_family, ())):
                        next_dynamic = None
                    else:
                        raise TypeError(f"dynamic-keyed source object is not an object: {child_raw}")
                yield from walk(child, child_field, child_raw, input_tokens + (key,),
                                child_tick, child_entity, is_snapshot_dictionary,
                                child_entity_keys or keyed, child_scope, next_dynamic,
                                child_predicate_id, child_source_path_key,
                                next_mixed[1] if next_mixed is not None else None)
        elif isinstance(item, list):
            if not item:
                yield leaf(field_path, snapshot_tick, raw_path, item, input_tokens,
                           entity, target_scope, predicate_id, source_path_key)
            for index, child in enumerate(item):
                child_entity = entity
                action_scope = None
                child_source_path_key = source_path_key
                if (source_family == "arm_predicate_truth" and
                        field_prefix + field_path == WORLD_OBSERVATION_LIST_PATH):
                    if not isinstance(child, Mapping):
                        raise TypeError(f"world observation is not an object: {raw_path}[{index}]")
                    observation_path = child.get("path")
                    if (not isinstance(observation_path, str) or
                            _WORLD_OBSERVATION_PATH.fullmatch(observation_path) is None or
                            "value" not in child):
                        raise ValueError(f"world observation lacks an exact path/value carrier: {raw_path}[{index}]")
                    observation_value = child["value"]
                    contract = world_observation_numeric_contracts().get(observation_path)
                    if type(observation_value) in {int, float}:
                        if contract is None:
                            raise ValueError(
                                f"numeric world observation has no declared branch: "
                                f"{predicate_id}:{observation_path}")
                        if predicate_id not in contract["predicate_ids"]:
                            raise ValueError(
                                f"numeric world observation differs from its predicate contract: "
                                f"{predicate_id}:{observation_path}")
                    elif contract is not None:
                        raise TypeError(
                            f"declared numeric world observation is not numeric: "
                            f"{predicate_id}:{observation_path}:{type(observation_value).__name__}")
                    child_source_path_key = observation_path
                if (field_prefix + field_path in ENTITY_LISTS.get(source_family, ())
                        and isinstance(child, dict)):
                    if (source_family == "arm_script_plan" and
                            field_prefix + field_path == "parameters.utm_service_plan.uav_plans"):
                        child_entity = entity_ref(child.get("uav_id"))
                    elif source_family == "predicate_truth_matrix" and field_prefix + field_path == "scope_values":
                        scope = child.get("scope_type")
                        namespace = "sample" if scope == "uav" else "compute_node" if scope == "compute_node" else "unknown_scope"
                        child_entity = entity_ref(child.get("scope_entity_id"), namespace=namespace)
                        if namespace == "unknown_scope":
                            child_entity = (child_entity[0], "unresolved",
                                            f"scope type has no entity namespace: {scope!r}",
                                            child_entity[3], namespace)
                    elif (source_family == "formal_world_truth_graph_base" and
                          field_prefix + field_path == "scope_entities"):
                        namespace = "compute_node" if child.get("scope_type") == "compute_node" else "sample"
                        child_entity = entity_ref(child.get("entity_id"), namespace=namespace)
                    else:
                        action_scope = (action_target_scope(child) if
                                        ((source_family == "formal_event_realization" and
                                          field_prefix + field_path == "action_realizations") or
                                         (source_family == "arm_script_plan" and
                                          field_prefix + field_path == "events[].actions")) else None)
                        child_entity = ((None, "not_applicable", None, None, None)
                                        if action_scope else entity_ref(child.get("entity_id")))
                yield from walk(child, field_path + "[]", f"{raw_path}[{index}]",
                                input_tokens + (index,), snapshot_tick, child_entity,
                                target_scope=action_scope or target_scope,
                                predicate_id=predicate_id,
                                source_path_key=child_source_path_key)
        else:
            yield leaf(field_path, snapshot_tick, raw_path, item, input_tokens,
                       entity, target_scope, predicate_id, source_path_key)

    initial_entity_keys = field_prefix in ENTITY_KEYED_OBJECTS.get(source_family, ())
    yield from walk(value, "", source_path_from_tokens(source_prefix_tokens), (), None,
                    initial_entity, entity_keys=initial_entity_keys,
                    target_scope=initial_scope, predicate_id=initial_predicate_id)
