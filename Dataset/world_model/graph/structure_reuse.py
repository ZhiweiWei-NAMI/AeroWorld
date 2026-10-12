"""Exact producer-backed aliases for structural field definitions.

An alias changes a normalized ID and its source binding slot. It never changes
the projected value, its missing status, shape, source location, or instance.
Callers use these same rules for source projection, catalog declarations, and
the migration of existing structural Parquet rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from Dataset.world_model.graph.field_paths import source_field_id


_UNSET = object()
_ENTITY_CONTEXT_PATHS = {
    "entity_category": "entities[].entity_category",
    "source": "entities[].source",
    "background_role": "entities[].background_role",
}


@dataclass(frozen=True)
class ReuseRule:
    source_family: str
    source_path: str
    canonical_path: str
    normalized_id: str
    scope: str
    producer_basis: str
    condition: str = "always"
    structure_mode: str = "source_only"
    existing_normalized_id: str | None = None
    existing_binding_slot: str | None = None
    value_relation: str = "definition_only"
    source_branch: str | None = None
    source_subtype: str | None = None
    # None means that this rule does not close the source type/status schema.
    # A catalog must obtain any such declaration from a producer contract.
    source_value_types: tuple[str, ...] | None = None
    value_types: tuple[str, ...] | None = None
    value_statuses: tuple[str, ...] | None = None

    @property
    def source_id(self) -> str:
        return source_field_id(self.source_family, self.source_path,
                               self.source_branch, self.source_subtype)

    @property
    def canonical_source_id(self) -> str:
        return source_field_id(self.source_family, self.canonical_path,
                               self.source_branch, self.source_subtype)

    @property
    def old_normalized_id(self) -> str:
        return (self.existing_normalized_id or
                "structural:source:" + self.source_id)

    @property
    def target_binding_slot(self) -> str:
        if self.structure_mode == "existing_structure":
            if self.existing_binding_slot is None:
                raise ValueError(f"structural reuse rule lacks its slot: {self.source_id}")
            return self.existing_binding_slot
        if self.structure_mode == "source_only":
            return "source:" + self.source_id
        raise ValueError(f"unknown structural reuse mode: {self.structure_mode}")

    @property
    def required_context_source_ids(self) -> tuple[tuple[str, str], ...]:
        if self.condition == "always":
            return ()
        keys = (("entity_category", "source") if self.condition in
                {"sumo_vehicle", "sumo_nonempty_source_entity_id"} else
                ("entity_category", "source", "background_role") if
                self.condition == "global_uav" else None)
        if keys is None:
            raise ValueError(f"unknown structural reuse condition: {self.condition}")
        return tuple((key, source_field_id(self.source_family,
                      _ENTITY_CONTEXT_PATHS[key])) for key in keys)


@dataclass(frozen=True)
class ReuseTarget:
    normalized_id: str
    binding_slot: str
    rule: ReuseRule


def _scalar(family: str, alias: str, canonical: str, *, scope: str,
            basis: str, condition: str = "always",
            relation: str = "definition_only") -> ReuseRule:
    return ReuseRule(family, alias, canonical,
        "structural:source:" + source_field_id(family, canonical),
        scope, basis, condition=condition, value_relation=relation)


def _rules() -> tuple[ReuseRule, ...]:
    rows: list[ReuseRule] = []

    weather_basis = "Dataset/tools/batch_generate.py:weather_payload"
    for family, prefix in (
        ("arm_actions", "effect.weather_before"),
        ("arm_actions", "effect.weather_after"),
        ("arm_predicate_truth", "evidence.observations[].value.weather"),
        ("arm_predicate_truth_ticks", "context.weather"),
    ):
        for alias, canonical in (("visibility", "visibility_m"),
                                 ("fog", "fog_density")):
            rows.append(_scalar(family, f"{prefix}.{alias}",
                f"{prefix}.{canonical}", scope=prefix, basis=weather_basis,
                relation="exact_copy"))

    compute_basis = "Dataset/semantic_simulation/compute_comm.py:_allocate_compute"
    for alias, canonical in (
        ("resource_balance.cpu_utilization_pct", "cpu_utilization_pct"),
        ("resource_balance.gpu_utilization_pct", "gpu_utilization_pct"),
        ("memory_used_mb", "allocation.memory_mb"),
        ("resource_balance.memory_allocated", "allocation.memory_mb"),
        ("resource_balance.cpu_allocated", "allocation.cpu_cores"),
        ("resource_balance.gpu_allocated", "allocation.gpu_units"),
        ("resource_balance.cpu_requested", "resource_request.cpu_cores"),
        ("resource_balance.gpu_requested", "resource_request.gpu_units"),
        ("resource_balance.memory_requested", "resource_request.memory_mb"),
    ):
        relation = ("conditional_capacity_copy" if alias.endswith("_requested")
                    else "definition_only")
        rows.append(_scalar("compute_state", alias, canonical,
            scope="compute_record", basis=compute_basis, relation=relation))

    communication_basis = "Dataset/semantic_simulation/compute_comm.py:_link_quality"
    rows.append(_scalar("communication_state", "link_quality.distance_m",
        "route.distance_m", scope="communication_record",
        basis=communication_basis))
    for name in ("availability", "capacity_factor", "operational_factor",
                 "range_factor"):
        rows.append(_scalar("communication_state", f"link_quality.{name}",
            f"station_operational_state.{name}",
            scope="communication_record", basis=communication_basis,
            relation="rounded_6"))

    formal_basis = "Dataset/tools/convert_to_render_ready.py:formal frame construction"
    segment_keys = ("seed_index", "seed_label", "segment_start_s",
                    "segment_end_s", "duration_s", "phase_name")
    for alias_prefix, canonical_prefix in (
        ("sumo_semantics.segment", "sumo_segment"),
        ("uav_global_flow.segment", "uav_segment"),
    ):
        for name in segment_keys:
            rows.append(_scalar("formal_truth_frame",
                f"{alias_prefix}.{name}", f"{canonical_prefix}.{name}",
                scope="formal_frame", basis=formal_basis,
                relation="exact_copy"))

    incident_suffixes = (
        "", "[].accident_class", "[].affected_vehicle_ids",
        "[].affected_vehicle_ids[]", "[].anchor.geometry_source",
        "[].anchor.sumo_edge_id", "[].anchor.sumo_lane_id",
        "[].anchor.truth_position_enu_m[]", "[].end_s",
        "[].episode_event_id", "[].episode_scenario_id", "[].incident_id",
        "[].injection_method", "[].intent", "[].intent_stage", "[].start_s",
    )
    for suffix in incident_suffixes:
        alias = "sumo_semantics.active_incidents" + suffix
        canonical = "sumo_active_incidents" + suffix
        if suffix in ("[].affected_vehicle_ids", "[].affected_vehicle_ids[]"):
            rows.append(ReuseRule("formal_truth_frame", alias, canonical,
                "structural:formal_truth_frame:sumo_active_incidents[].affected_vehicle_ids[]",
                "formal_frame_incident", formal_basis,
                structure_mode="existing_structure",
                existing_normalized_id=(
                    "structural:formal_truth_frame:sumo_semantics.active_incidents[].affected_vehicle_ids[]"),
                existing_binding_slot="sumo_semantics.active_incidents[].affected_vehicle_ids",
                value_relation="exact_copy"))
        else:
            rows.append(_scalar("formal_truth_frame", alias, canonical,
                scope="formal_frame_incident", basis=formal_basis,
                relation="exact_copy"))

    capture_basis = "Dataset/tools/regenerate_boundary_scenarios.py:capture_boundary"
    for name in (
        "boundary_id", "center_enu_m[]", "geometry_authority",
        "inspect_entity_role", "inspect_fov_coverage_required",
        "pad_boundary_policy", "polygon_enu_m[][]",
        "post_conflict_landing_boundary_clearance_m",
        "pre_mission_takeoff_boundary_clearance_m",
        "pre_mission_takeoff_max_route_length_m",
        "pre_mission_takeoff_speed_mps", "source", "source_entity_id",
        "uav_boundary_crossing_required",
    ):
        rows.append(_scalar("arm_script_plan",
            f"parameters.semantic_event_contract.capture_boundary.{name}",
            f"parameters.capture_boundary.{name}",
            scope="script_parameters", basis=capture_basis,
            relation="exact_copy"))

    entity_basis = "Dataset/tools/convert_to_render_ready.py:formal entity construction"
    family = "formal_truth_frame"
    entity = "entities[]."
    rows.append(_scalar(family,
        entity + "annotations.state_facets.activity.activity_type",
        entity + "annotations.activity_type", scope="formal_entity",
        basis=entity_basis + ";Dataset/tools/pedestrian_activity_catalog.py",
        relation="exact_copy"))

    for alias, canonical in (
        ("logical_asset_id", "sumo_vehicle.semantic_metadata.logical_asset_id"),
        ("source", "background_vehicle.source"),
        ("background_vehicle.semantic_vehicle", "sumo_vehicle.semantic_vehicle"),
        ("background_vehicle.source_presence_required",
         "sumo_vehicle.source_presence_required"),
        ("sumo_vehicle.semantic_metadata.logical_asset_authority",
         "sumo_vehicle.logical_asset_authority"),
    ):
        rows.append(_scalar(family, entity + alias, entity + canonical,
            scope="formal_sumo_entity", basis=entity_basis,
            condition="sumo_vehicle", relation="exact_copy"))
    rows.append(_scalar(family,
        entity + "background_vehicle.source_entity_id",
        entity + "sumo_vehicle.source_entity_id",
        scope="formal_sumo_entity", basis=entity_basis,
        condition="sumo_nonempty_source_entity_id", relation="exact_copy"))
    for name in segment_keys:
        rows.append(_scalar(family,
            entity + f"ground_flow_contract.segment.{name}",
            entity + f"sumo_segment.{name}",
            scope="formal_sumo_entity", basis=entity_basis,
            condition="sumo_vehicle", relation="exact_copy"))

    for name in segment_keys:
        rows.append(_scalar(family,
            entity + f"motion_contract.segment.{name}",
            entity + f"uav_segment.{name}",
            scope="formal_global_uav_entity", basis=entity_basis,
            condition="global_uav", relation="exact_copy"))
    rows.append(_scalar(family,
        entity + "motion_contract.altitude_layer_m",
        entity + "uav_global_flow.altitude_layer_m",
        scope="formal_global_uav_entity", basis=entity_basis,
        condition="global_uav", relation="exact_copy"))

    keys = [(row.source_family, row.source_path, row.source_branch,
             row.source_subtype) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate exact structural reuse rule")
    for rule in rows:
        if rule.source_family != "domain_state" and (
                rule.source_branch is not None or rule.source_subtype is not None):
            raise ValueError(f"non-domain reuse rule has a source branch: {rule.source_id}")
        if rule.source_id == rule.canonical_source_id or not rule.normalized_id.startswith("structural:"):
            raise ValueError(f"structural reuse rule has no distinct canonical field: {rule.source_id}")
        if rule.structure_mode not in {"source_only", "existing_structure"}:
            raise ValueError(f"invalid structural reuse mode: {rule.source_id}")
        if rule.condition not in {"always", "sumo_vehicle",
                                  "sumo_nonempty_source_entity_id", "global_uav"}:
            raise ValueError(f"unknown structural reuse condition: {rule.source_id}")
        if rule.source_value_types is not None or rule.value_types is not None or rule.value_statuses is not None:
            raise ValueError(f"reuse rule cannot infer a source schema: {rule.source_id}")
    return tuple(rows)


REUSE_RULES = _rules()
_RULE_BY_KEY = {
    (rule.source_family, rule.source_path, rule.source_branch,
     rule.source_subtype): rule for rule in REUSE_RULES
}
_RULES_BY_SOURCE = {rule.source_id: (rule,) for rule in REUSE_RULES}


def iter_reuse_rules() -> tuple[ReuseRule, ...]:
    return REUSE_RULES


def rules_for_source(source_id: str) -> tuple[ReuseRule, ...]:
    return _RULES_BY_SOURCE.get(source_id, ())


def candidate_source_ids() -> frozenset[str]:
    """Select only affected source leaves from a stored Parquet batch."""
    return frozenset(_RULES_BY_SOURCE)


def _selected(rule: ReuseRule, context: Mapping[str, Any] | None,
              candidate_value: Any) -> bool:
    if rule.condition == "always":
        return True
    if context is None:
        raise ValueError(f"conditional structural reuse lacks entity context: {rule.source_id}")
    category = context.get("entity_category", _UNSET)
    if type(category) is not str:
        raise ValueError(f"conditional structural reuse lacks entity_category: {rule.source_id}")
    if rule.condition in {"sumo_vehicle", "sumo_nonempty_source_entity_id"}:
        if category != "vehicle":
            return False
        source = context.get("source", _UNSET)
        if source is _UNSET:
            return False
        if type(source) is not str:
            raise ValueError(f"SUMO entity source is malformed: {rule.source_id}")
        matches = source == "sumo_traci"
        if rule.condition == "sumo_nonempty_source_entity_id" and matches:
            if candidate_value is _UNSET or type(candidate_value) is not str:
                raise ValueError(f"SUMO source entity ID is absent or malformed: {rule.source_id}")
            matches = bool(candidate_value)
        return matches
    if rule.condition == "global_uav":
        if category != "uav":
            return False
        source = context.get("source", _UNSET)
        if source is _UNSET:
            return False
        if type(source) is not str:
            raise ValueError(f"UAV entity source is malformed: {rule.source_id}")
        if source != "uav_global_flow":
            return False
        role = context.get("background_role", _UNSET)
        if type(role) is not str:
            raise ValueError(f"global UAV lacks background_role: {rule.source_id}")
        return role == "donghu_global_uav_flow"
    raise ValueError(f"unknown structural reuse condition: {rule.condition}")


def resolve_reuse(*, source_family: str, field_path: str,
                  current_normalized_id: str, current_binding_slot: str | None,
                  source_branch: str | None = None,
                  source_subtype: str | None = None,
                  context: Mapping[str, Any] | None = None,
                  candidate_value: Any = _UNSET) -> ReuseTarget | None:
    """Select one existing canonical definition without altering source data.

    Conditional callers supply the named marker leaves from the same entity.
    A false producer condition leaves the current source-specific binding intact.
    """
    rule = _RULE_BY_KEY.get((source_family, field_path,
                            source_branch, source_subtype))
    if rule is None:
        return None
    if (current_normalized_id != rule.old_normalized_id or
            current_binding_slot != rule.existing_binding_slot):
        raise ValueError(f"structural reuse source binding differs: {rule.source_id}:"
                         f"{current_normalized_id}:{current_binding_slot}")
    if not _selected(rule, context, candidate_value):
        return None
    return ReuseTarget(rule.normalized_id, rule.target_binding_slot, rule)
