"""Exact source container slots, separate from field meaning and units.

The declarations below describe producer record structure.  A source leaf keeps
its original field ID and tokens; one structural slot can have several source
leaf paths, and one source path can select different tuple members by index.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterator, Mapping

from Dataset.world_model.graph.field_paths import source_field_id, source_value_at_tokens
from Dataset.world_model.graph.structure_reuse import resolve_reuse


# Each path is a complete source-family path.  The outer list length varies;
# the listed element type and array rank come from the current producer/records.
_STRING_LIST_PATHS: dict[str, tuple[str, ...]] = {
    "arm_actions": ("action.state_patch.facility_state.requester_ids",),
    "arm_branch_roster": ("sumo_vehicle.semantic_metadata.required_edges",),
    "arm_chain_nodes:observed_action_dispatch": (
        "action.state_patch.facility_state.requester_ids",),
    "arm_domain_state_ticks": (
        "missing_source_record", "values.containing_corridor_ids",
        "values.occupying_aircraft_ids",),
    "arm_event_occurrences": ("supporting_parameter_refs",),
    "arm_event_outcomes": ("supporting_truth_ids",),
    "arm_manifest": (
        "branch_changes.added_action_ids", "branch_changes.added_event_ids",
        "branch_changes.changed_event_ids", "branch_changes.changed_trigger_ids",
        "branch_changes.removed_action_ids", "branch_changes.removed_event_ids",
        "branch_changes.replaced_action_ids", "declared_mechanism.cancel_event_ids",
        "declared_mechanism.omit_action_ids", "execution.actions",
        "execution.omit_action_ids", "injected_action_ids", "omit_action_ids",
        "removed_event_ids", "scope.gcs_ids", "scope.station_ids",
        "window_status.inherited_event_ids", "window_status.new_event_ids",
        "window_status.pending_event_ids",),
    "arm_predicate_truth": (
        "evidence.missing_requirements",
        "evidence.observations[].value.domain.pad_facility.missing",
        "evidence.observations[].value.domain.pad_facility.requester_ids",
        "evidence.observations[].value.missing_source_record",
        "measurements.requester_ids", "source_refs",),
    "arm_predicate_truth_ticks": (
        "context.domain.pad_facility.missing",
        "context.domain.pad_facility.requester_ids", "measurements.requester_ids",
        "missing_inputs", "missing_source_record", "occupied_aircraft_ids",
        "source_refs",),
    "arm_scene_setup": (
        "validation_rules[].contract.pad_boundary_policy.inside_required_for",),
    "arm_script_plan": (
        "events[].actions[].state_patch.facility_state.requester_ids",
        "events[].log_event.target_ids",
        "parameters.semantic_event_contract.pad_boundary_policy.inside_required_for",),
    "arm_states": ("facility_state.requester_ids",),
    "arm_trajectories": ("facility_state.requester_ids",),
    "communication_state": ("station_operational_state.state_reasons",),
    "formal_event_occurrences": ("supporting_parameter_refs",),
    "formal_event_outcomes": ("supporting_truth_ids",),
    "formal_predicate_truth": (
        "evidence.missing_requirements",
        "evidence.observations[].value.missing_source_record",),
    "formal_roster": (
        "facility_state.requester_ids", "sumo_vehicle.semantic_metadata.required_edges",),
    "formal_truth_frame": (
        "entities[].facility_state.requester_ids",
        "entities[].sumo_vehicle.semantic_metadata.required_edges",
        "entities[].uav_global_flow.active_event_ids",
        "pad_boundary_policy.inside_required_for",
        "sumo_active_incidents[].affected_vehicle_ids",
        "sumo_semantics.active_incidents[].affected_vehicle_ids",),
    "formal_world_truth_graph_base": (
        "initial_assertions[].missing_source_record",),
    "formal_world_truth_graph_deltas": (
        "operations[].missing_source_record",),
}

_INT_LIST_PATHS: dict[str, tuple[str, ...]] = {
    "arm_manifest": ("factual_source_audit.prefix_state_mismatch_ticks",),
}

# Domain-state records qualify identical value paths by observation family and,
# for facilities, by producer subtype.  The keyed object is not a name heuristic.
_DOMAIN_STRING_LIST_PATHS: dict[tuple[str, str | None], tuple[str, ...]] = {
    ("ambulance_priority", None): (
        "values.ambulance_vehicle_ids", "values.nearby_civilian_vehicle_ids",
        "values.yielding_vehicle_ids",),
    ("av_safe_stop", None): (
        "values.affected_vehicle_ids", "values.stopped_vehicle_ids",),
    ("crowd_evacuation", None): ("values.cohort_member_ids",),
    ("forced_landing_state", None): ("values.ground_zone_occupant_ids",),
    ("gnss_navigation", None): ("values.missing_inputs",),
    ("pad_facility", "charging_station"): ("values.requester_ids",),
    ("pad_facility", "landing_pad"): (
        "values.occupier_ids", "values.requester_ids",),
    ("payload_energy", None): ("values.missing_inputs",),
    ("road_closure", None): (
        "values.affected_vehicle_ids", "values.barrier_prop_ids",
        "values.closed_road_segment_ids", "values.closed_segment_refs",),
    ("security_command", None): ("values.missing_inputs",),
    ("signal_queue", None): (
        "values.queue_sampling_window.queued_vehicle_ids",
        "values.queued_vehicle_ids",),
    ("signal_queue_lane_state", None): ("values.queued_vehicle_ids",),
}

_ROUTE_PATHS: dict[str, str] = {
    "arm_branch_roster": "route_waypoints_enu_m",
    "arm_scene_setup": "route_waypoints_enu_m",
    "arm_states": "route_waypoints_enu_m",
    "arm_trajectories": "route_waypoints_enu_m",
    "formal_roster": "route_waypoints_enu_m",
    "formal_truth_frame": "entities[].route_waypoints_enu_m",
}

_NULL_VECTOR_PATHS = (
    "values.assigned_landing_zone_pose_enu_m",
    "values.home_pad_pose_enu_m",
)

_DOMAIN_UNKNOWN_VECTOR_PATHS: dict[str, tuple[str, ...]] = {
    "crowd_evacuation": (
        "values.crowd_centroid_enu_m", "values.hazard_reference_enu_m",),
    "forced_landing_state": (
        "values.landing_zone_approach_enu_m", "values.landing_zone_center_enu_m",),
    "gnss_navigation": (
        "values.drift_offset_enu_m", "values.reported_position_enu_m",),
}


@dataclass(frozen=True)
class StructureRule:
    source_family: str
    base_path: str
    source_branch: str | None
    source_subtype: str | None
    axes: tuple[int | None, ...]
    element_types: tuple[str, ...]
    base_statuses: tuple[str, ...]
    binding_slot: str
    normalized_id: str
    declaration_basis: str
    record_family: str | None = None
    record_schema: str | None = None


@dataclass(frozen=True)
class StructureBinding:
    source_family: str
    source_field_path: str
    source_branch: str | None
    source_subtype: str | None
    source_field_id: str
    normalized_id: str
    binding_slot: str
    selector: str
    source_value_types: tuple[str, ...]
    value_types: tuple[str, ...]
    value_statuses: tuple[str, ...]
    structure_statuses: tuple[str, ...]
    declared_shape: tuple[int | None, ...]
    declaration_basis: str
    record_family: str | None = None
    record_schema: str | None = None


@dataclass(frozen=True)
class StructuralProjection:
    normalized_id: str
    binding_slot: str
    value: Any
    value_type: str | None
    value_status: str
    structure_status: str
    declared_shape: tuple[int | None, ...]
    observed_shape: tuple[int | None, ...] | None
    normalized_shape: tuple[int | None, ...] | None
    envelope_tokens: tuple[str | int, ...]
    payload_tokens: tuple[str | int, ...]
    payload_positions: tuple[int, ...]


def _rule(source_family: str, base_path: str, axes: tuple[int | None, ...],
          element_types: tuple[str, ...], base_statuses: tuple[str, ...],
          *, branch: str | None = None, subtype: str | None = None,
          basis: str, record_family: str | None = None,
          record_schema: str | None = None) -> StructureRule:
    canonical_path = base_path + "[]" * len(axes)
    return StructureRule(source_family, base_path, branch, subtype, axes,
                         element_types, base_statuses, base_path,
                         "structural:" + source_field_id(
                             source_family, canonical_path, branch, subtype),
                         basis, record_family, record_schema)


def _rules() -> tuple[StructureRule, ...]:
    rows: list[StructureRule] = []
    for family, paths in _STRING_LIST_PATHS.items():
        for path in paths:
            rows.append(_rule(family, path, (None,), ("string",),
                              ("empty_list",), basis="current producer list and source records"))
    for family, paths in _INT_LIST_PATHS.items():
        for path in paths:
            rows.append(_rule(family, path, (None,), ("int",),
                              ("empty_list",), basis="current ARM manifest list and source records"))
    for (branch, subtype), paths in _DOMAIN_STRING_LIST_PATHS.items():
        for path in paths:
            statuses = (("empty_list", "unknown") if branch == "pad_facility"
                        and subtype == "landing_pad" and path == "values.occupier_ids"
                        else ("empty_list",))
            rows.append(_rule("domain_state", path, (None,), ("string",), statuses,
                              branch=branch, subtype=subtype,
                              basis="Dataset/semantic_simulation/domain_state.py"))
    for family, path in _ROUTE_PATHS.items():
        rows.append(_rule(family, path, (None, 3), ("int", "float"),
                          ("empty_list",), basis=(
                              "Dataset/tools/batch_generate.py:preserved_fields_from;"
                              "Dataset/scenarios/L2_infrastructure/failure/L2-1_v1/scene_setup.json")))
    for path in _NULL_VECTOR_PATHS:
        rows.append(_rule("arm_domain_state_ticks", path, (3,),
                          ("int", "float"), ("null",),
                          basis="Dataset/semantic_simulation/predicate_state_computers.py",
                          record_family="predicate_contract_aircraft_geometry"))
    for branch, paths in _DOMAIN_UNKNOWN_VECTOR_PATHS.items():
        for path in paths:
            rows.append(_rule("domain_state", path, (3,), ("int", "float"),
                              ("unknown",), branch=branch,
                              basis="Dataset/semantic_simulation/domain_state.py"))
    rows.append(_rule("arm_manifest", "provenance.implementation_sources",
                      (None,), ("string",), ("empty_list", "empty_object"),
                      basis="Dataset/tools/l6_v2/arm_runtime.py:_implementation_sources",
                      record_schema="aero_l6_v2_arm_manifest"))
    keys = [(row.source_family, row.source_branch, row.source_subtype,
             row.base_path) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate exact source container declaration")
    return tuple(rows)


STRUCTURE_RULES = _rules()
IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID = next(
    rule.normalized_id for rule in STRUCTURE_RULES
    if rule.source_family == "arm_manifest"
    and rule.base_path == "provenance.implementation_sources"
)
IMPLEMENTATION_SOURCES_GAP_SOURCE_ID = source_field_id(
    "arm_manifest", "provenance.implementation_sources",
)


def is_declared_implementation_sources_gap(
    normalized_id: str, value_type: str | None,
    value_status: str, structure_status: str,
) -> bool:
    """Recognize only the producer-declared L6 empty-object metadata gap."""
    return (
        normalized_id == IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID
        and value_type is None
        and value_status == "empty_object"
        and structure_status == "declared_only"
    )


def is_declared_implementation_sources_gap_projection(projected: Any) -> bool:
    """Require the exact raw empty object and absent normalized list shape."""
    return (
        is_declared_implementation_sources_gap(
            projected.normalized_id, projected.value_type,
            projected.value_status, projected.structure_status,
        )
        and projected.value is None
        and projected.declared_shape == (None,)
        and projected.observed_shape is None
        and projected.normalized_shape is None
        and len(projected.sources) == 1
        and projected.sources[0].source_field_id
        == IMPLEMENTATION_SOURCES_GAP_SOURCE_ID
        and projected.sources[0].field_value.value == {}
        and type(projected.sources[0].field_value.value) is dict
    )


_RULE_BY_PATH = {
    (rule.source_family, rule.source_branch, rule.source_subtype, rule.base_path): rule
    for rule in STRUCTURE_RULES
}

COMPUTE_QUEUE_SOURCE_IDS = (
    "compute_state:queue_depth",
    "compute_state:resource_balance.queue_depth",
)


def extra_structure_bindings(source_id: str) -> tuple[dict[str, Any], ...]:
    """Exact queue bindings shared by the structure and graph catalogs."""
    if source_id not in COMPUTE_QUEUE_SOURCE_IDS:
        return ()
    raw_types = ["int"] if source_id == COMPUTE_QUEUE_SOURCE_IDS[0] else ["float"]
    return ({"normalized_id": "structural:compute_state:queue_depth",
             "binding_slot": "queue_complete",
             "selector": "failure.requirement_status=not_required;equal_nonnegative_integer_values",
             "source_value_types": raw_types,
             "value_types": ["int"], "value_statuses": ["present"],
             "structure_statuses": ["resolved"], "declared_shape": [],
             "declaration_basis": "Dataset/world_model/graph/source_projection.py:_compute_queue_value"},
            {"normalized_id": "structural:source:" + source_id,
             "binding_slot": "queue_missing",
             "selector": "failure.requirement_status=missing_source_record",
             "source_value_types": ["string"] if source_id == COMPUTE_QUEUE_SOURCE_IDS[0] else ["float"],
             "value_types": ["string"] if source_id == COMPUTE_QUEUE_SOURCE_IDS[0] else ["float"],
             "value_statuses": ["unknown"],
             "structure_statuses": ["declared_only"], "declared_shape": [],
             "declaration_basis": "Dataset/world_model/graph/source_projection.py:_compute_queue_value"})


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "empty_list"
    if type(value) is dict:
        return "empty_object"
    raise TypeError(f"unsupported source leaf type: {type(value).__name__}")


def _reuse_existing_binding(binding: StructureBinding) -> StructureBinding:
    target = resolve_reuse(
        source_family=binding.source_family,
        field_path=binding.source_field_path,
        source_branch=binding.source_branch,
        source_subtype=binding.source_subtype,
        current_normalized_id=binding.normalized_id,
        current_binding_slot=binding.binding_slot,
    )
    if target is None:
        return binding
    if target.rule.structure_mode != "existing_structure":
        raise ValueError(f"source-only reuse overlaps an existing structure: {binding.source_field_id}")
    return replace(binding, normalized_id=target.normalized_id,
                   binding_slot=target.binding_slot)


def iter_structure_bindings() -> Iterator[StructureBinding]:
    """Enumerate source-path bindings without turning zero instances into observations."""
    for rule in STRUCTURE_RULES:
        base_id = source_field_id(rule.source_family, rule.base_path,
                                  rule.source_branch, rule.source_subtype)
        leaf_path = rule.base_path + "[]" * len(rule.axes)
        leaf_id = source_field_id(rule.source_family, leaf_path,
                                  rule.source_branch, rule.source_subtype)
        if rule.base_statuses:
            raw_types = tuple("empty_list" if status == "empty_list" else
                              "empty_object" if status == "empty_object" else
                              "null" if status == "null" else "string"
                              for status in rule.base_statuses)
            historical_metadata_gap = (rule.source_family == "arm_manifest" and
                                       rule.base_path == "provenance.implementation_sources")
            structure_statuses = (("resolved", "declared_only")
                                  if historical_metadata_gap else
                                  ("resolved", "structure_conflict")
                                  if "empty_object" in rule.base_statuses else
                                  ("resolved", "declared_only"))
            value_types = tuple(value_type for value_type in raw_types
                                if value_type != "empty_object")
            yield _reuse_existing_binding(StructureBinding(rule.source_family, rule.base_path,
                rule.source_branch, rule.source_subtype, base_id,
                rule.normalized_id, rule.binding_slot, "base_status",
                raw_types, value_types, rule.base_statuses, structure_statuses,
                rule.axes, rule.declaration_basis, rule.record_family,
                rule.record_schema))
        yield _reuse_existing_binding(StructureBinding(rule.source_family, leaf_path,
            rule.source_branch, rule.source_subtype, leaf_id,
            rule.normalized_id, rule.binding_slot, "array_element",
            rule.element_types, rule.element_types, ("present",), ("resolved",),
            rule.axes, rule.declaration_basis, rule.record_family,
            rule.record_schema))
    # The L5 engine stores each keyframe as [time, position[3], state].
    # Time and state share one source field ID and differ by tuple member index.
    family = "arm_actions"
    raw_scalar = "effect.keyframes_after[][]"
    raw_vector = "effect.keyframes_after[][][]"
    for member, label, path, source_types, axes in (
        (0, "time", raw_scalar, ("int",), ()),
        (1, "position", raw_vector, ("float",), (3,)),
        (2, "state", raw_scalar, ("string",), ()),
    ):
        yield StructureBinding(family, path, None, None,
            source_field_id(family, path),
            "structural:arm_actions:effect.keyframes_after[]." + label +
                ("[]" if member == 1 else ""),
            "effect.keyframes_after[]." + label,
            f"tuple_member={member}", source_types, source_types,
            ("present",), ("resolved",), axes,
            "Dataset/tools/batch_generate.py:EpisodeStateEngine.keyframes;"
            "Dataset/tools/l5_arm_engine.py:effect.keyframes_after")


def _matched_rule(source_family: str, field_path: str,
                  branch: str | None, subtype: str | None,
                  record: Any) -> StructureRule | None:
    base_path = field_path.rstrip("[]")
    rule = _RULE_BY_PATH.get((source_family, branch, subtype, base_path))
    if rule is None:
        return None
    if rule.record_family is not None and (
            not isinstance(record, Mapping) or
            record.get("observation_family") != rule.record_family):
        raise ValueError(f"source container producer family differs: {source_family}:{field_path}")
    if rule.record_schema is not None and (
            not isinstance(record, Mapping) or
            record.get("schema") != rule.record_schema):
        raise ValueError(f"source container schema differs: {source_family}:{field_path}")
    return rule


def _location(record: Any, field_path: str,
              source_tokens: tuple[str | int, ...],
              input_tokens: tuple[str | int, ...],
              payload_axes: tuple[int | None, ...]
              ) -> tuple[tuple[str | int, ...], tuple[str | int, ...],
                         tuple[int, ...], tuple[int | None, ...]]:
    if input_tokens and source_tokens[-len(input_tokens):] != input_tokens:
        raise ValueError(f"source and input locations differ: {source_tokens}:{input_tokens}")
    prefix = source_tokens[:len(source_tokens) - len(input_tokens)]
    trailing = len(field_path) - len(field_path.rstrip("[]"))
    trailing_rank = trailing // 2
    shell_rank = field_path.count("[]") - trailing_rank
    local_indices = [i for i, token in enumerate(input_tokens) if type(token) is int]
    prefix_rank = sum(type(token) is int for token in prefix)
    local_shell_rank = max(0, shell_rank - prefix_rank)
    if local_shell_rank > len(local_indices):
        raise ValueError(f"source shell array index is absent: {field_path}:{source_tokens}")
    split = local_indices[local_shell_rank - 1] + 1 if local_shell_rank else 0
    envelope = prefix + input_tokens[:split]
    payload = input_tokens[split:]
    positions = tuple(token for token in payload if type(token) is int)
    current = source_value_at_tokens(record, input_tokens[:split])
    observed: list[int | None] = []
    for token in payload:
        if type(token) is int:
            if type(current) is not list:
                raise TypeError(f"source payload axis is not an array: {field_path}:{source_tokens}")
            observed.append(len(current))
        current = current[token]
    if type(current) is list and not current:
        observed.append(0)
    if len(observed) > len(payload_axes):
        raise ValueError(f"source payload rank exceeds declaration: {field_path}:{source_tokens}")
    observed.extend(None for _ in range(len(payload_axes) - len(observed)))
    for index, fixed in enumerate(payload_axes):
        if fixed is not None and observed[index] is not None and observed[index] != fixed:
            raise ValueError(f"source payload fixed axis differs: {field_path}:{source_tokens}")
    return envelope, payload, positions, tuple(observed)


def project_structural_leaf(*, source_family: str, field_path: str,
                            branch: str | None, subtype: str | None,
                            source_tokens: tuple[str | int, ...],
                            input_tokens: tuple[str | int, ...],
                            value: Any, record: Any) -> StructuralProjection | None:
    """Resolve only exact producer slots; never infer structure from a name suffix."""
    if (source_family == "arm_actions" and
            field_path.startswith("effect.keyframes_after[]")):
        return _project_keyframe(field_path, source_tokens, input_tokens, value, record)
    rule = _matched_rule(source_family, field_path, branch, subtype, record)
    if rule is None:
        return None
    canonical = rule.base_path + "[]" * len(rule.axes)
    if field_path not in (rule.base_path, canonical):
        raise ValueError(f"source container has a partial or nested leaf: {source_family}:{field_path}")
    envelope, payload, positions, observed = _location(
        record, field_path, source_tokens, input_tokens, rule.axes)
    if field_path == rule.base_path:
        status = ("empty_list" if type(value) is list and not value else
                  "empty_object" if type(value) is dict and not value else
                  "null" if value is None else
                  "unknown" if value == "unknown" and type(value) is str else None)
        if status not in rule.base_statuses:
            raise ValueError(f"source container status differs from declaration: {source_family}:{field_path}:{value!r}")
        if status == "empty_object":
            historical_metadata_gap = (rule.source_family == "arm_manifest" and
                                       rule.base_path == "provenance.implementation_sources")
            return StructuralProjection(rule.normalized_id, rule.binding_slot,
                None, None, status,
                "declared_only" if historical_metadata_gap else "structure_conflict",
                rule.axes, None, None,
                envelope, payload, (),)
        if status in ("null", "unknown"):
            observed = tuple(None for _ in rule.axes)
            structure_status = "declared_only"
        else:
            structure_status = "resolved"
        normalized_shape = tuple(fixed if fixed is not None else actual
                                 for fixed, actual in zip(rule.axes, observed))
        return StructuralProjection(rule.normalized_id, rule.binding_slot,
            value, _value_type(value), status, structure_status, rule.axes,
            observed, normalized_shape, envelope, payload, positions)
    value_type = _value_type(value)
    if value_type not in rule.element_types or (value_type in ("int", "float")
                                               and not math.isfinite(value)):
        raise ValueError(f"source container element type differs: {source_family}:{field_path}:{value!r}")
    if any(axis is None for axis in observed):
        raise ValueError(f"nonempty source element lacks an observed axis: {source_family}:{field_path}")
    normalized_shape = tuple(fixed if fixed is not None else actual
                             for fixed, actual in zip(rule.axes, observed))
    return StructuralProjection(rule.normalized_id, rule.binding_slot,
        value, value_type, "present", "resolved", rule.axes, observed,
        normalized_shape, envelope, payload, positions)


def _project_keyframe(field_path: str,
                      source_tokens: tuple[str | int, ...],
                      input_tokens: tuple[str | int, ...],
                      value: Any, record: Any) -> StructuralProjection:
    base = "effect.keyframes_after"
    if field_path not in (base + "[][]", base + "[][][]"):
        raise ValueError(f"keyframe source path differs from tuple producer: {field_path}")
    if input_tokens and source_tokens[-len(input_tokens):] != input_tokens:
        raise ValueError(f"keyframe source and input locations differ: {source_tokens}:{input_tokens}")
    if len(input_tokens) < 4 or input_tokens[:2] != ("effect", "keyframes_after"):
        raise ValueError(f"keyframe source location differs: {source_tokens}")
    outer_index, member = input_tokens[2:4]
    if type(outer_index) is not int or type(member) is not int:
        raise ValueError(f"keyframe tuple indices are missing: {source_tokens}")
    keyframes = source_value_at_tokens(record, ("effect", "keyframes_after"))
    if type(keyframes) is not list or outer_index >= len(keyframes):
        raise ValueError(f"keyframe outer array differs: {source_tokens}")
    item = keyframes[outer_index]
    if (type(item) is not list or len(item) != 3 or
            type(item[0]) is not int or type(item[1]) is not list or
            len(item[1]) != 3 or any(type(x) is not float or not math.isfinite(x)
                                     for x in item[1]) or type(item[2]) is not str):
        raise ValueError(f"keyframe tuple differs from producer: {source_tokens}")
    if member == 0 and field_path == base + "[][]" and len(input_tokens) == 4:
        label, axes, positions = "time", (), ()
    elif member == 1 and field_path == base + "[][][]" and len(input_tokens) == 5:
        if type(input_tokens[4]) is not int or not 0 <= input_tokens[4] < 3:
            raise ValueError(f"keyframe position index differs: {source_tokens}")
        label, axes, positions = "position", (3,), (input_tokens[4],)
    elif member == 2 and field_path == base + "[][]" and len(input_tokens) == 4:
        label, axes, positions = "state", (), ()
    else:
        raise ValueError(f"keyframe member path differs: {source_tokens}")
    envelope = source_tokens[:len(source_tokens) - len(input_tokens)] + input_tokens[:3]
    payload = input_tokens[3:]
    normalized_id = ("structural:arm_actions:effect.keyframes_after[]." + label +
                     ("[]" if label == "position" else ""))
    return StructuralProjection(normalized_id,
        "effect.keyframes_after[]." + label, value, _value_type(value),
        "present", "resolved", axes, axes, axes,
        envelope, payload, positions)
