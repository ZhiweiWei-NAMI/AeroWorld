"""Materialize one ARM branch without mixing branches or source layers."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from Dataset.world_model.graph.arm_observation import record_observed
from Dataset.world_model.graph.contract import (
    ENTITY_CATEGORIES, REPO, canonical_predicate_id, iter_jsonl, read_json, source_path, strip_source_digests,
)
from Dataset.world_model.graph.controller_scope import controller_evidence
from Dataset.world_model.graph.entity_identity import (
    BUSINESS_SOURCE_OBJECTS, ENTITY_TARGET_ACTIONS, IdentityResolver,
    action_target_scope, binding_identity_namespace, identity_node_id, read_roster,
)
from Dataset.world_model.graph.records import DEFAULT_ROOT, GraphWriter
from Dataset.world_model.graph.sample_identity import (
    arm_scope, arm_source_family, manifest_reference, validate_record_identity,
)
from Dataset.world_model.graph.arm_history import load_l6_history, qualify_history_refs
from Dataset.world_model.graph.validator import ContractError
from Dataset.semantic_truth.entity_scope import participant_binding_allowed, roster_index


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _declared_parent_roster(manifest: dict[str, Any]) -> Path | None:
    """Resolve only an explicit parent roster or capture-root declaration."""
    roster_ref = manifest_reference(manifest, "roster")
    if roster_ref:
        return REPO / roster_ref
    capture_ref = manifest_reference(manifest, "capture")
    return REPO / capture_ref / "global_entity_roster.json" if capture_ref else None


def _scene_landing_area_authorities(
    scene: dict[str, Any], roster_payloads: list[tuple[Path, dict | None]],
) -> dict[str, str]:
    """Resolve terminal-area identities using the same governed producer rule."""
    from Dataset.semantic_simulation.predicate_state_computers import (
        _assigned_landing_zone, _entity_position,
    )
    roster_entities: dict[str, dict] = {}
    for _, roster in roster_payloads:
        if roster is not None:
            for entity in roster["entities"]:
                if entity["entity_id"] not in roster_entities:
                    roster_entities[entity["entity_id"]] = entity
    result: dict[str, str] = {}
    for index, vehicle in enumerate(scene["entities"]):
        if vehicle.get("category") != "uav":
            continue
        lifecycle = vehicle.get("lifecycle") or {}
        home_id = lifecycle.get("home_pad_entity_id")
        pad = roster_entities.get(home_id)
        pad_scope = (pad or {}).get("semantic_scope") or {}
        if (pad is None or pad_scope.get("scope_type") != "facility"
                or pad_scope.get("scope_subtype") != "landing_pad"):
            continue
        home_pose = _entity_position(pad)
        area_id, _ = _assigned_landing_zone(
            vehicle["entity_id"], vehicle, home_id, home_pose,
            lifecycle.get("home_hover_enu_m"),
        )
        if area_id is not None and area_id not in roster_entities:
            result[area_id] = f"entities[{index}].route_waypoints_enu_m"
    return result


@lru_cache(maxsize=4)
def _sumo_road_edge_authority(map_id: str) -> tuple[Path, frozenset[str]]:
    """Read the map package's declared SUMO edges without inferring IDs."""
    package_path = REPO / "Config/LowAltitude/Maps" / map_id / "map_package.json"
    package = read_json(package_path)
    if package.get("map_id") != map_id:
        raise ValueError(f"map package identity differs: {package_path}")
    inputs = package.get("plugin_owned_source_inputs") or {}
    net_ref = inputs.get("sumo_net_xml")
    if not isinstance(net_ref, str) or not net_ref:
        raise ValueError(f"map package lacks SUMO network authority: {package_path}")
    net_path = REPO / net_ref
    edges: set[str] = set()
    for _, element in ET.iterparse(net_path, events=("start",)):
        if element.tag == "edge":
            edge_id = element.get("id")
            if not isinstance(edge_id, str) or not edge_id or edge_id in edges:
                raise ValueError(f"SUMO edge identity missing or duplicated: {net_path}:{edge_id}")
            edges.add(edge_id)
        element.clear()
    return net_path, frozenset(edges)


@lru_cache(maxsize=1)
def _configured_domain_object_authorities() -> dict[str, tuple[str, Path, str]]:
    """Return the two stable service objects declared by the domain profile."""
    path = REPO / "Dataset/semantic_rules/profiles/domain_state_supplement_profile.json"
    profile = read_json(path)
    gnss = (profile.get("domain_models") or {}).get("gnss") or {}
    result: dict[str, tuple[str, Path, str]] = {}
    for key in ("service_instance", "spoofing_signal_instance"):
        item = gnss.get(key) or {}
        object_id, ontology = item.get("service_id"), item.get("ontology_class_id")
        if (not isinstance(object_id, str) or not object_id
                or not isinstance(ontology, str) or not ontology
                or item.get("authority") != "domain_state_supplement_profile"):
            raise ValueError(f"configured GNSS identity differs: {path}:{key}")
        if object_id in result:
            raise ValueError(f"configured GNSS identity is duplicated: {path}:{object_id}")
        result[object_id] = ontology, path, f"domain_models.gnss.{key}.service_id"
    return result


@lru_cache(maxsize=4)
def _stable_business_object_authorities(
    episode_id: str, wanted: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], tuple[Path, int, str]]:
    """Locate requested logical objects in the actual current state records."""
    root = REPO / "aw_data/objective_semantic_truth" / episode_id
    result: dict[tuple[str, str], tuple[Path, int, str]] = {}
    families = {}
    for ontology, entity in wanted:
        declaration = BUSINESS_SOURCE_OBJECTS.get(ontology)
        if declaration is None:
            continue
        family, field, _ = declaration
        families.setdefault(family, {}).setdefault((ontology, field), set()).add(entity)
    for family, declarations in families.items():
        path = root / f"{family}.jsonl"
        if not path.is_file():
            continue
        for line, row in iter_jsonl(path):
            if (row.get("schema_name") != family or row.get("episode_id") != episode_id):
                raise ValueError(f"business identity source differs: {path}:{line}")
            for (ontology, field), requested in declarations.items():
                object_id = row.get(field)
                if object_id in requested and (ontology, object_id) not in result:
                    result[(ontology, object_id)] = (path, line, field)
            if all((ontology, entity) in result for (ontology, _), entities in declarations.items()
                   for entity in entities):
                break
    return result


def _l4_logical_bindings(predicate_id: str, source_bindings: dict, entities: dict[str, dict]) -> dict[str, str]:
    name = predicate_id.removeprefix("l4_business.")
    ids = source_bindings.get("entities", source_bindings.get("vehicles", []))
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"L4 business predicate lacks mechanism scope: {predicate_id}")
    def of_kind(kind: str) -> list[str]:
        return sorted(entity for entity in ids if entities.get(entity, {}).get("category") == kind)
    uavs, vehicles, pedestrians = of_kind("uav"), of_kind("vehicle"), of_kind("pedestrian")
    if name in {"near_pair", "braking_active", "yielding_active", "braking_stopped", "yield_response"}:
        if len(vehicles) != 2 or vehicles[0] == vehicles[1]:
            raise ValueError("L4 unordered vehicle pair is not two distinct vehicles")
        return {"first": vehicles[0], "second": vehicles[1]}
    if name in {"uav_vehicle_proximity", "uav_vehicle_contact"}:
        return {"aircraft": uavs[0], "vehicle": vehicles[0]}
    if name == "aircraft_near_pedestrian":
        return {"aircraft": uavs[0], "pedestrian": pedestrians[0]}
    if name in {"landing_zone_occupied", "pedestrian_clear_of_landing_zone"}:
        return {"pedestrian": pedestrians[0]}
    if name in {"vehicle_emergency_stop_active", "vehicle_physically_stopped", "sensor_fault",
                "minimal_risk_maneuver", "declared_stopped_safe", "physically_stopped_safe"}:
        return {"vehicle": vehicles[0]}
    if name in {"ambulance_approaching_vehicle", "yield_clearance_satisfied", "ambulance_geometry_passed",
                "priority_passage_completed", "priority_request_active", "priority_request_acknowledged", "civilian_yielding"}:
        ambulances = [entity for entity in vehicles if "ambulance" in str(entities[entity].get("asset_id", ""))]
        if len(ambulances) != 1 or len(vehicles) != 2:
            raise ValueError("L4 ambulance and counterpart cannot be resolved from source assets")
        ambulance = ambulances[0]
        counterpart = next(entity for entity in vehicles if entity != ambulance)
        if name in {"priority_request_active", "priority_request_acknowledged"}:
            return {"ambulance": ambulance}
        if name == "civilian_yielding":
            return {"vehicle": counterpart}
        return {"ambulance": ambulance, "vehicle": counterpart}
    if name in {"uav_low_altitude", "uav_hold_active", "uav_landed", "uav_damaged", "uav_airborne"}:
        return {"aircraft": uavs[0]}
    raise ValueError(f"L4 predicate signature not fixed: {name}")


def _event_category(row: dict) -> str:
    if row.get("event_type_id"):
        return row["event_type_id"]
    family = row.get("event_family_id")
    if isinstance(family, str) and family.startswith("l4_"):
        return "l4." + family.removeprefix("l4_")
    raise ValueError("objective event has no declared category")


def _logical_bindings(predicate_id: str, source_bindings: dict,
                      entities: dict[str, dict], signatures: dict) -> dict[str, str]:
    bindings = (_l4_logical_bindings(predicate_id, source_bindings, entities)
                if predicate_id.startswith("l4_business.") else dict(source_bindings))
    signature = signatures.get(canonical_predicate_id(predicate_id))
    if signature is not None:
        for roles in signature["unordered_role_sets"]:
            values = [bindings[role] for role in roles]
            if any(type(value) is not str for value in values):
                raise ValueError(f"unordered predicate binding is not an entity ID: {predicate_id}")
            bindings.update(zip(roles, sorted(values)))
    return bindings


def materialize_arm(branch_root: Path, output_root: Path = DEFAULT_ROOT, *,
                    replace_generated: bool = False) -> dict[str, int]:
    branch_root = (REPO / branch_root).resolve() if not branch_root.is_absolute() else branch_root.resolve()
    if not branch_root.is_relative_to(REPO / "arms"):
        raise ValueError("ARM branch must be under arms/")
    relative = branch_root.relative_to(REPO / "arms")
    if len(relative.parts) != 4:
        raise ValueError("expected arms/<profile>/<seed>/<group>/<branch>")
    profile, seed, group, branch = relative.parts
    manifest_path = branch_root / "manifest.json"
    manifest = read_json(manifest_path)
    scope = arm_scope(branch_root, manifest)

    def source_rows(path: Path):
        for line, row in iter_jsonl(path):
            validate_record_identity(row, scope,
                source_family=arm_source_family(branch_root, path, row),
                source_file=source_path(path), source_line=line)
            yield line, row

    resolver = IdentityResolver(scope["sample_key"])
    parent_roster_path = _declared_parent_roster(manifest)
    roster_paths = [branch_root / "ue/global_entity_roster.json"]
    if parent_roster_path is not None:
        roster_paths.append(parent_roster_path)
    capture_ref = manifest_reference(manifest, "capture")
    parent_capture_root = REPO / capture_ref if capture_ref else (
        parent_roster_path.parent if parent_roster_path is not None
        and parent_roster_path.name == "global_entity_roster.json"
        and parent_roster_path.is_relative_to(REPO / "aw_data/render_ready_episodes_capture_filtered")
        else None
    )
    parent_episode_manifest: dict[str, Any] | None = None
    if parent_capture_root is not None:
        parent_manifest_path = parent_capture_root / "episode_manifest.json"
        parent_episode_manifest = read_json(parent_manifest_path)
        if parent_episode_manifest.get("episode_id") != scope["episode_id"]:
            raise ValueError(f"ARM parent capture episode differs: {parent_manifest_path}")
    capture_parent_only = (
        capture_ref is not None and manifest_reference(manifest, "roster") is None
    )
    parent_roster_records: dict[str, tuple[int, dict[str, Any]]] = {}
    roster_payloads: list[tuple[Path, dict | None]] = []
    for index, roster_path in enumerate(roster_paths):
        if not roster_path.exists():
            roster_payloads.append((roster_path, None))
            continue
        roster = read_roster(roster_path)
        roster_payloads.append((roster_path, roster))
        for entity_index, item in enumerate(roster["entities"]):
            if roster_path == parent_roster_path:
                parent_roster_records[item["entity_id"]] = entity_index, item
            resolver.declare(item, source_file=source_path(roster_path),
                source_path=f"entities[{entity_index}]",
                authority="branch_roster" if index == 0 else "parent_roster",
                inherited=index > 0)
    scene_ref = manifest_reference(manifest, "scene")
    scene_path = REPO / scene_ref if scene_ref else None
    scene_payload = read_roster(scene_path) if scene_path is not None and scene_path.exists() else None
    if scene_payload is not None:
        for entity_index, item in enumerate(scene_payload["entities"]):
            resolver.declare(item, source_file=source_path(scene_path),
                source_path=f"entities[{entity_index}]", authority="manifest_scene",
                inherited=True)
    history_support = load_l6_history(
        branch_root, manifest, profile=profile, seed=seed, group=group, branch=branch
    ) if profile.startswith("L6-") else None
    graph = GraphWriter(scope, output_root, replace_generated=replace_generated)
    for conflict in resolver.conflicts:
        graph.unresolved_relation(source_file=conflict["sources"][-1], source_line=None,
            relation=conflict, reason="entity identity declaration conflict")
    sampled_path = branch_root / "objective/predicate_truth.jsonl"
    dense_path = branch_root / "objective/predicate_truth_ticks.jsonl"
    sampled = [(line, row) for line, row in source_rows(sampled_path)]
    dense = [(line, row) for line, row in source_rows(dense_path)]
    label_class_claims: dict[str, tuple[str, str, int | None]] = {}
    class_conflicts: set[str] = set()
    parent_base_for_classes = REPO / "aw_data/objective_semantic_truth" / scope["episode_id"] / "world_truth_graph_base.json"
    if parent_base_for_classes.exists():
        parent_base_row = read_json(parent_base_for_classes)
        if parent_base_row.get("schema_version") != "3.1.0":
            raise ValueError(f"ARM parent world truth version mismatch: {parent_base_for_classes}")
        parent_scope = parent_base_row.get("scope_entities", [])
        roster_index(parent_scope)
        for item in parent_scope:
            if item.get("ontology_class_id"):
                label_class_claims[item["entity_id"]] = (item["ontology_class_id"], source_path(parent_base_for_classes), None)
    for label_path, label_rows in ((sampled_path, sampled), (dense_path, dense)):
        for label_line, label_row in label_rows:
            bindings = label_row.get("bindings") or {}
            if not isinstance(bindings, dict) or "predicate_id" not in label_row:
                continue
            classes = dict(label_row.get("binding_ontology_classes") or {})
            for observation in (label_row.get("evidence") or {}).get("observations", []):
                for role, class_id in ((observation.get("value") or {}).get("binding_ontology_classes") or {}).items():
                    if role in classes and classes[role] != class_id:
                        raise ValueError(f"binding ontology conflict at {label_path}:{label_line} role {role}")
                    classes[role] = class_id
            for role, entity in bindings.items():
                if not isinstance(entity, str) or role not in classes:
                    continue
                previous = label_class_claims.get(entity)
                if previous and previous[0] != classes[role]:
                    from Dataset.semantic_truth.facility_scope import ontology_class_is_a
                    if ontology_class_is_a(previous[0], classes[role]):
                        continue
                    if ontology_class_is_a(classes[role], previous[0]):
                        label_class_claims[entity] = (classes[role], source_path(label_path), label_line)
                        continue
                    if entity not in class_conflicts:
                        class_conflicts.add(entity)
                        graph.unresolved_relation(source_file=source_path(label_path), source_line=label_line,
                            relation={"entity_id": entity, "class_a": previous[0], "class_b": classes[role]},
                            reason="source ontology class claims differ")
                elif previous is None and entity not in class_conflicts:
                    label_class_claims[entity] = (classes[role], source_path(label_path), label_line)
    from Dataset.world_model.graph.entity_identity import crowd_cohort_authorities
    crowd_authorities = crowd_cohort_authorities(scope["episode_id"],
        {entity for entity, (ontology, _, _) in label_class_claims.items() if ontology == "world:Crowd"})
    for entity, declaration in crowd_authorities.items():
        resolver.declare({"entity_id": entity, "ontology_class_id": "world:Crowd"},
            source_file=source_path(declaration["path"]), source_path=f"line[{declaration['line']}].values.crowd_id",
            authority="registered_crowd_cohort")
    if profile.startswith("X"):
        weather_domain = profile + ".authored_weather_domain"
        weather_labels = [row for _, row in sampled
                          if row["predicate_id"].startswith("environment.")
                          and row.get("bindings", {}).get("region") == weather_domain]
        if weather_labels:
            producer = REPO / "Dataset/tools/x_arm_pipeline.py"
            label_class_claims[weather_domain] = ("world:OperationalRegion", source_path(producer), None)
            resolver.declare({"entity_id": weather_domain, "ontology_class_id": "world:OperationalRegion"},
                source_file=source_path(producer), source_path="semantic_rows.authored_weather_domain",
                authority="x_arm_authored_weather_domain")
    declared_sample_ids = {
        item["entity_id"]
        for _, roster in roster_payloads if roster is not None
        for item in roster["entities"]
    }
    if scene_payload is not None:
        declared_sample_ids.update(item["entity_id"] for item in scene_payload["entities"])
    claimed_ontologies = {entity: claim[0] for entity, claim in label_class_claims.items()
                          if entity not in class_conflicts}
    authority_conflict_start = len(resolver.conflicts)
    if parent_capture_root is not None:
        controllers = controller_evidence(scope["episode_id"])
        expected_controller_source = source_path(parent_capture_root / "truth_frames.jsonl")
        if controllers.source_file != expected_controller_source:
            raise ValueError(
                f"ARM controller authority differs from declared capture: {controllers.source_file}"
            )
        for controller_id, line in sorted(controllers.sumo_first_line.items()):
            if claimed_ontologies.get(controller_id) != "world:TrafficSignal":
                continue
            resolver.declare(
                {"entity_id": controller_id, "scope_type": "sumo_tls_controller",
                 "ontology_class_id": "world:TrafficSignal"},
                source_file=controllers.source_file,
                source_path=f"truth_frames.jsonl:{line}:sumo_traffic_light_states.{controller_id}",
                authority="truth_frame_sumo_tls", namespace="sumo_tls_controller",
            )
        map_id = parent_episode_manifest.get("map_id") if parent_episode_manifest else None
        if not isinstance(map_id, str) or not map_id:
            raise ValueError(f"ARM parent capture has no map ID: {parent_capture_root}")
        sumo_net_path, road_edges = _sumo_road_edge_authority(map_id)
        for entity, ontology in sorted(claimed_ontologies.items()):
            if (ontology == "world:RoadSegment" and entity in road_edges
                    and entity not in declared_sample_ids):
                resolver.declare(
                    {"entity_id": entity, "ontology_class_id": ontology},
                    source_file=source_path(sumo_net_path),
                    source_path=f"edge[@id={json.dumps(entity)}]",
                    authority="sumo_network_edge",
                )
    for entity, (ontology, path, field_path) in _configured_domain_object_authorities().items():
        if claimed_ontologies.get(entity) == ontology and entity not in declared_sample_ids:
            resolver.declare(
                {"entity_id": entity, "ontology_class_id": ontology},
                source_file=source_path(path), source_path=field_path,
                authority="domain_state_supplement_profile",
            )
    business_authorities = _stable_business_object_authorities(
        scope["episode_id"], tuple(sorted((ontology, entity) for entity, ontology in claimed_ontologies.items()
                                         if ontology in BUSINESS_SOURCE_OBJECTS))
    )
    for (ontology, entity), (path, line, field) in sorted(business_authorities.items()):
        if claimed_ontologies.get(entity) == ontology and entity not in declared_sample_ids:
            resolver.declare(
                {"entity_id": entity, "ontology_class_id": ontology},
                source_file=source_path(path), source_path=f"line[{line}].{field}",
                authority="compute_comm_supplement",
            )
    if parent_episode_manifest is not None:
        capture_boundary = parent_episode_manifest.get("capture_boundary_id")
        if (isinstance(capture_boundary, str) and
                claimed_ontologies.get(capture_boundary.strip()) == "world:OperationalRegion"
                and capture_boundary.strip() not in declared_sample_ids):
            resolver.declare(
                {"entity_id": capture_boundary.strip(),
                 "ontology_class_id": "world:OperationalRegion"},
                source_file=source_path(parent_capture_root / "episode_manifest.json"),
                source_path="capture_boundary_id", authority="episode_manifest_capture_boundary",
            )
    if scene_payload is not None and "world:LandingArea" in claimed_ontologies.values():
        for area_id, field_path in _scene_landing_area_authorities(
                scene_payload, roster_payloads).items():
            if (claimed_ontologies.get(area_id) == "world:LandingArea"
                    and area_id not in declared_sample_ids):
                resolver.declare(
                    {"entity_id": area_id, "ontology_class_id": "world:LandingArea"},
                    source_file=source_path(scene_path), source_path=field_path,
                    authority="predicate_state_computers._assigned_landing_zone",
                )
    from Dataset.world_model.graph.entity_identity import arm_grounded_object_bindings
    arm_grounded_object_bindings(resolver, sampled,
        source_path(branch_root / "objective/predicate_truth.jsonl"))
    for conflict in resolver.conflicts[authority_conflict_start:]:
        graph.unresolved_relation(source_file=conflict["sources"][-1], source_line=None,
            relation=conflict, reason="entity identity declaration conflict")
    prefix = scope["sample_key"] + "|"
    ident = lambda entity, namespace="sample": identity_node_id(scope["sample_key"], namespace, entity)
    time_id = lambda tick: prefix + f"t:{tick}"
    native_index: dict[str, str] = {}
    entity_info: dict[str, dict] = {}
    state_at: dict[tuple[int, str], str] = {}
    last_state: dict[tuple[str, str], tuple[int, str]] = {}
    frozen_parent_state_ids: set[str] = set()

    def relative(tick: int, first: str | None, second: str | None, *, role: str,
                 field: str, value: Any, metric: str | None, path: Path, line: int) -> None:
        if not isinstance(first, str) or not isinstance(second, str) or type(value) not in {int, float}:
            return
        endpoints = (state_at.get((tick, first)), state_at.get((tick, second)))
        if not all(endpoints):
            node = prefix + f"recorded_metric:{line}:{field}"
            graph.add_node(node, "ContextAnchor", "measured_geometry_support", tick,
                {"anchor_kind": "geometry_support", "source_family": "measured_geometry_support",
                 "first_entity": first, "second_entity": second, "field": field,
                 "recorded_value": value, "metric": metric,
                 "state_availability": [endpoint is not None for endpoint in endpoints],
                 "geometry_status": "recorded_metric_without_complete_pose_pair",
                 "recomputed_geometry": False}, source_path(path), line)
            at_time(node, tick, path, line)
            return
        if field.endswith("following_distance_m"):
            unit_basis = "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#regulation.vehicle_violates_following_distance"
        elif profile.startswith("X"):
            unit_basis = "Dataset/tools/x_arm_pipeline.py#meter_coordinate_geometry"
        else:
            unit_basis = "Dataset/semantic_simulation/predicate_state_computers.py#meter_coordinate_geometry"
        attrs = {"field": field, "value": value, "unit": "m", "unit_status": "declared",
                 "unit_basis": unit_basis, "metric": metric, "value_tick": tick}
        if any(endpoint in frozen_parent_state_ids for endpoint in endpoints):
            attrs.update({"background_policy": "frozen_parent_reference_no_feedback_claim",
                          "feedback_claim": False})
        graph.add_edge("RelativeState", endpoints[0], endpoints[1], role=role,
            source_file=source_path(path), source_line=line,
            attrs=attrs)

    def time(tick: int, path: Path, line: int | None = None) -> str:
        node = time_id(tick)
        if node not in graph.node_index:
            graph.add_node(node, "TimePoint", "simulation_tick", tick, {"tick": tick}, source_path(path), line)
        return node

    def at_time(node: str, tick: int | None, path: Path, line: int | None = None,
                role: str = "target_tick") -> None:
        if tick is not None:
            time(tick, path, line)
            graph.add_edge("at_time", node, time_id(tick), role=role,
                           source_file=source_path(path), source_line=line)

    def identity(entity: str, *, source: Path, line: int | None = None,
                 category: str | None = None, ontology: str | None = None,
                 raw: dict | None = None, completeness: str = "source",
                 source_entity_index: int | None = None,
                 namespace: str = "sample") -> str:
        if entity in entity_info:
            existing_namespace = entity_info[entity]["identity_namespace"]
            if existing_namespace != namespace:
                raise ValueError(
                    f"ARM entity source namespace differs: {entity}:{existing_namespace}:{namespace}"
                )
            return ident(entity, namespace)
        resolved = resolver.resolve(entity, source_file=source_path(source),
                                    source_line=line, source_path="entity_id",
                                    namespace=namespace)
        inherited_record = parent_roster_records.get(entity)
        authority_raw = raw or (inherited_record[1] if inherited_record is not None else None)
        if category is None and authority_raw is not None:
            category = authority_raw.get("entity_category") or authority_raw.get("category")
        cat = category if category in ENTITY_CATEGORIES else None
        if not ontology and entity in label_class_claims and entity not in class_conflicts:
            ontology = label_class_claims[entity][0]
        roster_class_basis = None
        if (not ontology and entity not in class_conflicts and authority_raw is not None
                and cat in {"uav", "vehicle", "pedestrian"}):
            from Dataset.semantic_truth.world_truth import _ScopeEntity, _scope_ontology_class
            ontology = _scope_ontology_class(_ScopeEntity(cat, entity))
            roster_class_basis = ["Dataset/semantic_truth/world_truth.py:_scope_ontology_class", None]
        if (not ontology and profile.startswith("X") and cat in {"crowd_anchor", "facade_anchor", "vehicle_anchor", "prop"}
                and authority_raw is not None):
            ontology = "world:Entity"
            roster_class_basis = ["Dataset/tools/x_arm_pipeline.py:calculate", None]
        source_declares_ontology = ((authority_raw or {}).get("semantic_scope") or {}).get("ontology_class_id") == ontology
        ontology_source = ([resolved.authority, None] if source_declares_ontology and raw is None
                           else [source_path(source), line] if source_declares_ontology
                           else list(label_class_claims[entity][1:]) if entity in label_class_claims and entity not in class_conflicts
                           else roster_class_basis)
        roster_authority = completeness in {"branch_roster", "parent_roster"} or (
            inherited_record is not None and resolved.status == "resolved"
        )
        roster_index = {entity: authority_raw} if roster_authority and authority_raw is not None else {}
        event_actor_allowed = participant_binding_allowed(entity, "actor", roster_index)
        event_non_actor_allowed = participant_binding_allowed(entity, "region", roster_index)
        entity_info[entity] = {"category": cat, "ontology_class": ontology,
                               "identity_namespace": namespace,
                               "identity_status": resolved.status,
                               "identity_basis": resolved.authority,
                               "identity_basis_path": resolved.authority_path,
                               "asset_id": (authority_raw or {}).get(
                                   "asset_id", (authority_raw or {}).get("logical_asset_id")
                               ),
                               "source_completeness": completeness,
                               "background_role": (authority_raw or {}).get("background_role"),
                               "uav_role_classification": (authority_raw or {}).get("uav_role_classification"),
                               "event_actor_allowed": event_actor_allowed,
                               "event_non_actor_allowed": event_non_actor_allowed}
        graph.add_node(ident(entity, namespace), "EntityIdentity", cat, None,
            {"entity_id": entity, "entity_category": cat, "source_entity_category": category,
             "identity_namespace": namespace,
             "source_scope_type": "sumo_tls_controller" if namespace == "sumo_tls_controller" else None,
             "ontology_class": ontology,
             "ontology_source": ontology_source if ontology else None,
             "raw_category_status": "resolved" if cat else "unregistered" if category else "missing",
             "ontology_status": "resolved" if ontology else "source_conflict" if entity in class_conflicts else "missing",
             "classification_status": "resolved" if cat and ontology else "unresolved",
             "identity_status": resolved.status, "identity_basis": resolved.authority,
             "identity_basis_path": resolved.authority_path,
             "identity_inherited_from": list(resolved.inherited_from),
             "identity_reason": resolved.reason,
             "source_completeness": completeness,
             "source_entity_index": source_entity_index,
             "event_actor_allowed": event_actor_allowed,
             "event_non_actor_allowed": event_non_actor_allowed,
             "raw": raw}, source_path(source), line)
        return ident(entity, namespace)

    # Branch capture roster is the nearest authority. The manifest's parent roster
    # fills absent identities. Dynamic classes reuse the world's executable
    # scope classifier; facilities retain their explicit semantic-scope classes.
    for roster_path, roster in roster_payloads:
        if roster is None:
            graph.unresolved_relation(source_file=source_path(manifest_path), source_line=None,
                relation={"roster": source_path(roster_path)}, reason="declared roster source missing")
            continue
        if capture_parent_only and roster_path == parent_roster_path:
            continue
        for source_entity_index, item in enumerate(roster["entities"]):
            entity = item["entity_id"]
            if entity in entity_info:
                continue
            identity(entity, source=roster_path,
                category=item.get("entity_category") or item.get("category"),
                ontology=(item.get("semantic_scope") or {}).get("ontology_class_id"),
                raw=item, completeness="branch_roster" if roster_path == roster_paths[0] else "parent_roster",
                source_entity_index=source_entity_index)

    if scene_ref:
        if scene_payload is not None:
            for source_entity_index, item in enumerate(scene_payload["entities"]):
                entity = item["entity_id"]
                category = item.get("category") or item.get("entity_category")
                ontology = item.get("semantic_scope", {}).get("ontology_class_id")
                identity(entity, source=scene_path, category=category, ontology=ontology,
                         raw=item, completeness="scene", source_entity_index=source_entity_index)
        else:
            graph.unresolved_relation(source_file=source_path(manifest_path), source_line=None,
                relation={"scene": scene_ref}, reason="declared scene source missing")

    # Preserve states as authority. Trajectories are aliases where their current-execution
    # fields agree; otherwise they become separate state records with their own provenance.
    states_path = branch_root / "raw/states.jsonl"
    trajectory_path = branch_root / "raw/trajectories.jsonl"
    trajectory_rows: dict[tuple[int, str], tuple[int, dict]] = {}
    if trajectory_path.exists():
        for line, row in source_rows(trajectory_path):
            key = (row["tick"], row["entity_id"])
            if key in trajectory_rows:
                raise ValueError(f"duplicate trajectory key {key}: {trajectory_path}")
            trajectory_rows[key] = line, row
    branch_record_keys = set(trajectory_rows)
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    state_rows: list[tuple[str, int, dict, list[dict]]] = []
    seen_state_keys: set[tuple[int, str]] = set()
    for line, row in source_rows(states_path):
        key = (row["tick"], row["entity_id"])
        if key in seen_state_keys:
            raise ValueError(f"duplicate ARM state key {key}: {states_path}")
        seen_state_keys.add(key)
        branch_record_keys.add(key)
        aliases: list[dict] = []
        trajectory = trajectory_rows.pop(key, None)
        if trajectory:
            trajectory_line, trajectory_row = trajectory
            state_compare = {k: v for k, v in row.items() if k not in {"observed", "record_provenance"}}
            trajectory_compare = {k: v for k, v in trajectory_row.items() if k not in {"observed", "record_provenance"}}
            if state_compare == trajectory_compare:
                aliases.append({"source_file": source_path(trajectory_path), "source_line": trajectory_line})
            else:
                state_rows.append(("trajectory", trajectory_line, trajectory_row, []))
        state_rows.append(("state", line, row, aliases))
    state_rows.extend(("trajectory", line, row, []) for _, (line, row) in trajectory_rows.items())
    state_rows.sort(key=lambda item: (item[2]["tick"], item[2]["entity_id"], item[0]))
    for layer, line, row, aliases in state_rows:
        tick, entity = int(row["tick"]), row["entity_id"]
        if entity not in entity_info:
            category = row.get("entity_category") or row.get("category") or row.get("label_class")
            identity(entity, source=states_path if layer == "state" else trajectory_path, line=line,
                     category=category, raw=row, completeness="state_only")
        path = states_path if layer == "state" else trajectory_path
        observed = record_observed(profile, layer, row, path, line)
        node = prefix + f"s:{layer}:{entity}@{tick}"
        state_category = row.get("entity_category") or row.get("category") or row.get("label_class")
        identity_status = entity_info[entity]["identity_status"]
        if (state_category in ENTITY_CATEGORIES and entity_info[entity]["category"] is not None
                and state_category != entity_info[entity]["category"]):
            identity_status = "unresolved"
            graph.unresolved_relation(source_file=source_path(path), source_line=line,
                relation={"entity_id": entity, "state_category": state_category,
                          "identity_category": entity_info[entity]["category"]},
                reason="state and declared roster categories differ")
        graph.add_node(node, "EntityState", entity_info[entity]["category"], tick,
            {"entity_id": entity, "ontology_class": entity_info[entity]["ontology_class"],
             "identity_status": identity_status,
             "identity_ref": ident(entity) if identity_status == "resolved" else None,
             "identity_basis": entity_info[entity]["identity_basis"],
             "identity_basis_path": entity_info[entity]["identity_basis_path"],
             "entity_category": entity_info[entity]["category"],
             "source_completeness": entity_info[entity]["source_completeness"],
             "background_role": entity_info[entity]["background_role"],
             "uav_role_classification": entity_info[entity]["uav_role_classification"],
             "event_actor_allowed": entity_info[entity]["event_actor_allowed"],
             "event_non_actor_allowed": entity_info[entity]["event_non_actor_allowed"],
             "state_layer": layer, "observed": observed,
             "record_provenance": row.get("record_provenance", "branch_simulation"),
             "raw": row, "source_aliases": aliases,
             "field_context": {"source_file": source_path(path), "source_line": line, "value_tick": tick}},
            source_path(path), line)
        if identity_status == "resolved":
            graph.add_edge("identity_state", ident(entity), node, source_file=source_path(path), source_line=line)
        at_time(node, tick, path, line)
        previous = last_state.get((layer, entity))
        if previous and previous[0] == tick - 1 and identity_status == "resolved" and graph.node_index[previous[1]]["attrs"]["identity_status"] == "resolved":
            graph.add_edge("same_entity_next", previous[1], node, source_file=source_path(path), source_line=line)
        last_state[(layer, entity)] = tick, node
        if layer == "state" and observed:
            state_at[(tick, entity)] = node

    # Dense geometry first consumes this branch's trajectories, then fills entities
    # outside the branch scope from the manifest's formal parent trajectories.  Add
    # only those frozen parent states that are actual endpoints of saved relative
    # measurements.  Scoped lifecycle gaps remain absent and therefore unresolved.
    relative_endpoint_keys: set[tuple[int, str]] = set()
    for _, row in dense:
        tick = row.get("tick")
        if not isinstance(tick, int):
            continue
        values = row.get("values") or {}
        if any(type(values.get(field)) in {int, float} for field in (
                "pair_euclidean_distance_m", "pair_horizontal_distance_m",
                "pair_vertical_distance_m", "pair_distance_m")):
            for entity in (values.get("first_aircraft_id"), values.get("second_aircraft_id")):
                if isinstance(entity, str):
                    relative_endpoint_keys.add((tick, entity))
        if type(values.get("nearest_aircraft_distance_m")) in {int, float}:
            for entity in (row.get("subject_id"), values.get("nearest_aircraft_id")):
                if isinstance(entity, str):
                    relative_endpoint_keys.add((tick, entity))
    for _, row in sampled:
        if canonical_predicate_id(row.get("predicate_id", "")) != "x.geometry.entity_proximity":
            continue
        tick, bindings = row.get("tick"), row.get("bindings") or {}
        measurements = row.get("measurements") or {}
        if (not isinstance(tick, int) or
                not any(type(measurements.get(field)) in {int, float} for field in
                        ("distance_3d_m", "distance_xy_m", "distance_z_m"))):
            continue
        for entity in (bindings.get("first"), bindings.get("second")):
            if isinstance(entity, str):
                relative_endpoint_keys.add((tick, entity))

    manifest_scope = manifest.get("scope")
    parent_keys: set[tuple[int, str]] = set()
    missing_endpoint_keys = {
        key for key in relative_endpoint_keys
        if key not in state_at and key not in branch_record_keys
    }
    if missing_endpoint_keys:
        if (not isinstance(manifest_scope, dict) or
                not isinstance(manifest_scope.get("entity_ids"), list)):
            raise ValueError(f"ARM manifest scope lacks entity_ids: {manifest_path}")
        declared_entities = manifest_scope["entity_ids"]
        if any(not isinstance(entity, str) for entity in declared_entities):
            raise ValueError(f"ARM manifest scope has a non-string entity ID: {manifest_path}")
        scoped_entities = set(declared_entities)
        parent_keys = {
            key for key in missing_endpoint_keys if key[1] not in scoped_entities
        }
    if parent_keys:
        if manifest_scope.get("background_policy") != "frozen_parent_reference_no_feedback_claim":
            raise ValueError(f"relative endpoint lacks declared frozen-parent policy: {manifest_path}")
        parent_trajectory_ref = manifest_reference(manifest, "trajectories")
        if parent_trajectory_ref is None:
            raise ValueError(f"relative endpoint lacks parent trajectory authority: {manifest_path}")
        parent_trajectory_path = (REPO / parent_trajectory_ref).resolve()
        expected_parent = (REPO / "aw_data/render_ready_episodes_capture_filtered" /
                           scope["episode_id"] / "trajectories.jsonl").resolve()
        if parent_trajectory_path != expected_parent:
            raise ValueError(f"relative endpoint parent trajectory differs: {parent_trajectory_path}")
        parent_truth_path = parent_trajectory_path.with_name("truth_frames.jsonl")
        if not parent_trajectory_path.is_file() or not parent_truth_path.is_file():
            raise FileNotFoundError(parent_truth_path)

        parent_trajectory_rows: dict[tuple[int, str], tuple[int, dict]] = {}
        for line, row in iter_jsonl(parent_trajectory_path):
            key = (row.get("tick"), row.get("entity_id"))
            if key not in parent_keys:
                continue
            if key in parent_trajectory_rows:
                raise ValueError(f"duplicate parent trajectory key {key}: {parent_trajectory_path}")
            parent_trajectory_rows[key] = line, row
        parent_truth_rows: dict[tuple[int, str], tuple[int, int, dict]] = {}
        wanted_ticks = {tick for tick, _ in parent_keys}
        for line, frame in iter_jsonl(parent_truth_path):
            tick = frame.get("tick")
            if tick not in wanted_ticks:
                continue
            if frame.get("episode_id") != scope["episode_id"]:
                raise ValueError(f"parent truth frame episode differs: {parent_truth_path}:{line}")
            entities = frame.get("entities")
            if not isinstance(entities, list):
                raise ValueError(f"parent truth frame entities differ: {parent_truth_path}:{line}")
            for source_entity_index, row in enumerate(entities):
                key = (tick, row.get("entity_id"))
                if key not in parent_keys:
                    continue
                if key in parent_truth_rows:
                    raise ValueError(f"duplicate parent truth state {key}: {parent_truth_path}")
                parent_truth_rows[key] = line, source_entity_index, row

        for key in sorted(parent_keys):
            trajectory_source = parent_trajectory_rows.get(key)
            truth_source = parent_truth_rows.get(key)
            if trajectory_source is None or truth_source is None:
                continue
            trajectory_line, trajectory_row = trajectory_source
            truth_line, source_entity_index, row = truth_source
            truth_pose = row.get("truth_pose") or {}
            if trajectory_row.get("pos_enu") != truth_pose.get("position_enu_m"):
                raise ValueError(f"parent trajectory/truth pose differs for {key}")
            tick, entity = key
            if entity not in entity_info:
                raise ValueError(f"frozen parent state lacks roster identity: {entity}")
            state_category = row.get("entity_category") or row.get("category") or row.get("label_class")
            identity_status = entity_info[entity]["identity_status"]
            if (state_category in ENTITY_CATEGORIES and entity_info[entity]["category"] is not None
                    and state_category != entity_info[entity]["category"]):
                identity_status = "unresolved"
                graph.unresolved_relation(source_file=source_path(parent_truth_path),
                    source_line=truth_line,
                    relation={"entity_id": entity, "state_category": state_category,
                              "identity_category": entity_info[entity]["category"]},
                    reason="parent state and declared roster categories differ")
            node = prefix + f"s:frozen_parent:{entity}@{tick}"
            graph.add_node(node, "EntityState", entity_info[entity]["category"], tick,
                {"entity_id": entity, "ontology_class": entity_info[entity]["ontology_class"],
                 "identity_status": identity_status,
                 "identity_ref": ident(entity) if identity_status == "resolved" else None,
                 "identity_basis": entity_info[entity]["identity_basis"],
                 "identity_basis_path": entity_info[entity]["identity_basis_path"],
                 "entity_category": entity_info[entity]["category"],
                 "source_entity_category": state_category,
                 "source_completeness": entity_info[entity]["source_completeness"],
                 "background_role": entity_info[entity]["background_role"],
                 "uav_role_classification": entity_info[entity]["uav_role_classification"],
                 "event_actor_allowed": entity_info[entity]["event_actor_allowed"],
                 "event_non_actor_allowed": entity_info[entity]["event_non_actor_allowed"],
                 "state_layer": "frozen_parent_reference", "observed": True,
                 "record_provenance": "frozen_parent_reference_no_feedback_claim",
                 "raw": row, "source_entity_index": source_entity_index,
                 "source_aliases": [{"source_file": source_path(parent_trajectory_path),
                                      "source_line": trajectory_line}],
                 "field_context": {"source_file": source_path(parent_truth_path),
                                   "source_line": truth_line, "value_tick": tick}},
                source_path(parent_truth_path), truth_line)
            if identity_status == "resolved":
                graph.add_edge("identity_state", ident(entity), node,
                    source_file=source_path(parent_truth_path), source_line=truth_line)
            at_time(node, tick, parent_truth_path, truth_line)
            state_at[key] = node
            frozen_parent_state_ids.add(node)

    # Assemble predicate layers without confusing L4 domain observations with labels.
    transition_path = branch_root / "objective/predicate_transitions.jsonl"
    branch_transition_rows = [(line, row) for line, row in source_rows(transition_path)] if transition_path.exists() else []
    sampled_by_native = {row.get("truth_id"): (line, row) for line, row in sampled if row.get("truth_id")}
    sampled_by_key = {(row["tick"], row["predicate_id"], _json_key(row["bindings"])): (line, row)
                      for line, row in sampled if "predicate_id" in row}
    predicate_rows: list[tuple[str, Path, int, dict, list[dict]]] = []
    dense_is_predicate = bool(dense) and all("predicate_id" in row for _, row in dense)
    dense_keys = set()
    dense_by_key = {(row["tick"], row["predicate_id"], _json_key(row["bindings"])): (line, row)
                    for line, row in dense if "predicate_id" in row}
    dense_alias_by_truth: dict[str, dict] = {}
    dense_alias_by_key: dict[tuple[int, str, str], dict] = {}
    if dense_is_predicate:
        for line, row in dense:
            key = (row["tick"], row["predicate_id"], _json_key(row["bindings"]))
            dense_keys.add(key)
            matching = sampled_by_key.get(key)
            if row.get("truth_id") and matching and row["truth_id"] == matching[1].get("truth_id"):
                dense_alias_by_truth[row["truth_id"]] = {"source_file": source_path(dense_path),
                    "source_line": line, "source_view": row}
                continue
            duplicate = bool(matching and matching[1]["value"] == row["value"])
            if duplicate and not profile.startswith("X"):
                dense_alias_by_key[key] = {"source_file": source_path(dense_path),
                    "source_line": line, "source_view": row}
                continue
            aliases = [{"source_file": source_path(sampled_path), "source_line": matching[0],
                        "source_view": matching[1]}] if duplicate else []
            layer = "dense" if profile.startswith("X") else "projection"
            predicate_rows.append((layer, dense_path, line, row, aliases))
    for line, row in sampled:
        key = (row["tick"], row["predicate_id"], _json_key(row["bindings"]))
        dense_match = dense_by_key.get(key)
        dense_same = bool(dense_match and dense_match[1]["value"] == row["value"] and profile.startswith("X"))
        if dense_same:
            continue
        alias = dense_alias_by_truth.get(row.get("truth_id")) or dense_alias_by_key.get(key)
        predicate_rows.append(("sampled", sampled_path, line, row, [alias] if alias else []))
    predicate_rows.sort(key=lambda item: (item[3]["tick"], item[0], item[2]))
    ticks_by_layer: dict[str, set[int]] = defaultdict(set)
    for layer, _, _, row, _ in predicate_rows:
        ticks_by_layer[layer].add(int(row["tick"]))
    step_by_layer = {layer: min((b - a for a, b in zip(sorted(ticks), sorted(ticks)[1:])), default=0)
                     for layer, ticks in ticks_by_layer.items()}
    assertions_by_tuple: dict[tuple[str, int], str] = {}
    assertions_by_binding: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    truth_ref: dict[str, str] = {}
    previous_predicate: dict[tuple[str, str, str], tuple[int, str]] = {}
    projections: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for layer, path, line, row, aliases in predicate_rows:
        tick = int(row["tick"])
        raw_pid = row["predicate_id"]
        pid = canonical_predicate_id(raw_pid)
        raw_bindings = row["bindings"]
        bindings = _logical_bindings(raw_pid, raw_bindings, entity_info, graph.validator.predicates)
        key = (pid, _json_key(bindings), tick)
        node = prefix + f"p:{layer}:{line}@{tick}"
        ontology_bindings = dict(row.get("binding_ontology_classes") or {})
        for observation in (row.get("evidence") or {}).get("observations", []):
            declared = (observation.get("value") or {}).get("binding_ontology_classes") or {}
            for role, ontology_class in declared.items():
                if role in ontology_bindings and ontology_bindings[role] != ontology_class:
                    raise ValueError(f"conflicting binding ontology classes at {path}:{line} role {role}")
                ontology_bindings[role] = ontology_class
        if not ontology_bindings:
            ontology_bindings = {role: entity_info.get(entity, {}).get("ontology_class")
                                 for role, entity in bindings.items() if entity in entity_info}
        nested_evidence = row.get("evidence")
        source_refs = row.get("source_refs")
        if source_refs is None and isinstance(nested_evidence, dict):
            source_refs = nested_evidence.get("source_refs")
        evidence_update_tick = row.get("evidence_update_tick")
        evidence_source: str | None = None
        evidence_source_line: int | None = None
        if evidence_update_tick is not None:
            evidence_source, evidence_source_line = source_path(path), line
        graph.add_node(node, "PredicateAssertion", pid, tick,
            {"predicate_id": pid, "raw_predicate_id": raw_pid, "tuple_id": row.get("tuple_id"),
             "native_truth_id": row.get("truth_id"), "bindings": bindings,
             "mechanism_scope": raw_bindings if raw_pid.startswith("l4_business.") else None,
             "binding_ontology_classes": ontology_bindings,
             "truth_status": row["value"], "candidate_present": True, "assertion_layer": layer,
             "sample_step": step_by_layer[layer], "evidence_update_tick": evidence_update_tick,
             "truth_state_update_tick": row.get("truth_state_update_tick"),
             "evidence_time_authority": evidence_source,
             "evidence_time_source_line": evidence_source_line,
             "evidence": row.get("evidence", row.get("measurements", row.get("metrics", row.get("operands")))),
             "source_refs": source_refs or [], "source_aliases": aliases,
             "raw": row}, source_path(path), line,
            rule_ref=row.get("authority") or ("Dataset/semantic_rules/profiles/l4_arm_business_semantics.json#" + pid
                    if pid.startswith("l4.") else "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#" + pid
                    if not pid.startswith("x.") else "Dataset/tools/x_arm_pipeline.py#" + pid))
        at_time(node, tick, path, line)
        if row.get("truth_id"):
            truth_ref[row["truth_id"]] = node
        if row.get("tuple_id"):
            assertions_by_tuple[(row["tuple_id"], tick)] = node
        assertions_by_binding[key].append(node)
        projections[key].append(node)
        for role, entity in bindings.items():
            ontology = ontology_bindings.get(role)
            namespace = binding_identity_namespace(ontology)
            if entity not in entity_info:
                identity(entity, source=path, line=line,
                         ontology=ontology, completeness="binding_only", namespace=namespace)
            else:
                namespace = entity_info[entity]["identity_namespace"]
            target = (state_at.get((tick, entity), ident(entity))
                      if namespace == "sample" else ident(entity, namespace))
            graph.add_edge("predicate_argument", node, target, role=role,
                           source_file=source_path(path), source_line=line)
        if pid == "x.geometry.entity_proximity":
            measurements = row.get("measurements") or {}
            trigger = measurements.get("trigger") or {}
            for field, metric in (("distance_3d_m", "euclidean_3d"),
                                  ("distance_xy_m", "horizontal_euclidean"),
                                  ("distance_z_m", "absolute_vertical_delta")):
                relative(tick, bindings.get("first"), bindings.get("second"),
                    role="first_to_second", field="measurements." + field,
                    value=measurements.get(field), metric=metric, path=path, line=line)
        series = (layer, pid, _json_key(bindings))
        previous = previous_predicate.get(series)
        if previous and step_by_layer[layer] and previous[0] == tick - step_by_layer[layer]:
            graph.add_edge("same_predicate_next", previous[1], node,
                           source_file=source_path(path), source_line=line)
        previous_predicate[series] = tick, node
    for key, nodes in projections.items():
        if len(nodes) > 1:
            for earlier, later in zip(nodes, nodes[1:]):
                if graph.node_index[earlier]["attrs"]["truth_status"] == graph.node_index[later]["attrs"]["truth_status"]:
                    graph.add_edge("projection_equivalent", earlier, later,
                                   source_file=graph.node_index[later]["source_file"],
                                   source_line=graph.node_index[later]["source_line"])
                else:
                    graph.unresolved_relation(source_file=graph.node_index[later]["source_file"],
                        source_line=graph.node_index[later]["source_line"], relation={"assertion_nodes": nodes},
                        reason="projection truth value differs from sampled assertion")
    if dense and not dense_is_predicate:
        for line, row in dense:
            node = prefix + f"source_state:predicate_truth_ticks:{line}"
            tick = row.get("tick")
            observation_family = row.get("observation_family")
            graph.add_node(node, "ContextAnchor", observation_family or "domain_state_observation", tick,
                {"anchor_kind": "domain_state_observation",
                 "source_family": observation_family or "domain_state_observation",
                 "observation_family": observation_family,
                 "observation_family_status": "source_value" if observation_family else "missing_source_value",
                 "raw": row, "source_layer": "domain_state_observation"}, source_path(dense_path), line)
            at_time(node, tick, dense_path, line)
            values = row.get("values") or {}
            if isinstance(tick, int):
                first, second = values.get("first_aircraft_id"), values.get("second_aircraft_id")
                for field, metric in (("pair_euclidean_distance_m", "euclidean_3d"),
                    ("pair_horizontal_distance_m", "horizontal_euclidean"),
                    ("pair_vertical_distance_m", "absolute_vertical_delta"),
                    ("pair_distance_m", values.get("pair_distance_metric"))):
                    relative(tick, first, second, role="first_aircraft_to_second_aircraft",
                        field="values." + field, value=values.get(field), metric=metric,
                        path=dense_path, line=line)
                nearest = values.get("nearest_aircraft_id")
                relative(tick, row.get("subject_id"), nearest,
                    role="subject_to_nearest_aircraft", field="values.nearest_aircraft_distance_m",
                    value=values.get("nearest_aircraft_distance_m"),
                    metric=values.get("nearest_aircraft_distance_metric") or "source_rule_metric_unspecified",
                    path=dense_path, line=line)
    local_truth = branch_root / "objective/local_business_truth.jsonl"
    if local_truth.exists():
        for line, row in source_rows(local_truth):
            node = prefix + f"local_business:{line}"
            graph.add_node(node, "ContextAnchor", "local_business_measurement", row.get("tick"),
                {"anchor_kind": "local_business_measurement",
                 "source_family": "local_business_measurement", "raw": row},
                source_path(local_truth), line)
            at_time(node, row.get("tick"), local_truth, line)

    if history_support is not None:
        history_path = history_support["source_path"]
        for index, row in history_support["truths"]:
            tick = row["tick"]
            native = row["truth_id"]
            pid = canonical_predicate_id(row["predicate_id"])
            bindings = _logical_bindings(row["predicate_id"], row["bindings"], entity_info,
                                         graph.validator.predicates)
            raw_evidence = row.get("evidence") or {}
            history_refs = qualify_history_refs(
                raw_evidence.get("source_refs"), history_path=history_path,
                start_tick=scope["start_tick"])
            evidence = {**raw_evidence, "source_refs": history_refs}
            ontology_bindings = dict(row.get("binding_ontology_classes") or {})
            for observation in raw_evidence.get("observations", []):
                for role, class_id in ((observation.get("value") or {}).get("binding_ontology_classes") or {}).items():
                    if role in ontology_bindings and ontology_bindings[role] != class_id:
                        raise ValueError(f"{history_path}: historical ontology class differs: {native}:{role}")
                    ontology_bindings[role] = class_id
            node = prefix + "p:history:" + native
            graph.add_node(node, "PredicateAssertion", pid, tick,
                {"predicate_id": pid, "raw_predicate_id": row["predicate_id"],
                 "tuple_id": row.get("tuple_id"), "native_truth_id": native,
                 "bindings": bindings, "binding_ontology_classes": ontology_bindings,
                 "truth_status": row["value"], "candidate_present": True,
                 "assertion_layer": "inherited_history", "sample_step": None,
                 "truth_state_update_tick": row.get("truth_state_update_tick"),
                 "evidence_update_tick": row.get("evidence_update_tick"),
                 "evidence": evidence,
                 "source_refs": history_refs,
                 "source_pointer": f"/objective/predicate_truth/{index}",
                 "source_history_index": index, "source_record_family": "arm_semantic_history", "raw": row},
                source_path(history_path), None,
                rule_ref="Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#" + pid)
            at_time(node, tick, history_path)
            truth_ref[native] = node
            if row.get("tuple_id"):
                assertions_by_tuple[(row["tuple_id"], tick)] = node
            assertions_by_binding[(pid, _json_key(bindings), tick)].append(node)
            for role, entity in bindings.items():
                ontology = ontology_bindings.get(role)
                namespace = binding_identity_namespace(ontology)
                if entity not in entity_info:
                    identity(entity, source=history_path, ontology=ontology,
                             completeness="binding_only", namespace=namespace)
                else:
                    namespace = entity_info[entity]["identity_namespace"]
                target = (state_at.get((tick, entity), ident(entity))
                          if namespace == "sample" else ident(entity, namespace))
                graph.add_edge("predicate_argument", node, target, role=role,
                               source_file=source_path(history_path))

    transition_ref: dict[str, str] = {}
    transition_inputs = (
        [(history_support["source_path"], None, row, index) for index, row in history_support["transitions"]]
        if history_support is not None else []
    ) + [(transition_path, line, row, None) for line, row in branch_transition_rows]
    event_path = branch_root / "objective/event_occurrences.jsonl"
    outcome_path = branch_root / "objective/event_outcomes.jsonl"
    event_rows = list(source_rows(event_path)) if event_path.exists() else []
    outcome_rows = list(source_rows(outcome_path)) if outcome_path.exists() else []
    from Dataset.world_model.graph.window_evidence import boundary_references
    boundary_refs = boundary_references((row for _, row in sampled + dense),
        [row for _, row in event_rows], [row for _, row in outcome_rows],
        scope["start_tick"], scope["end_tick"])
    for transition_source, line, row, history_index in transition_inputs:
        native = row["transition_id"]
        pid = canonical_predicate_id(row["predicate_id"])
        raw_bindings = row["bindings"]
        bindings = _logical_bindings(row["predicate_id"], raw_bindings, entity_info,
                                     graph.validator.predicates)
        node = prefix + "tr:" + native
        transition_refs = (qualify_history_refs(row.get("source_refs"), history_path=transition_source,
                           start_tick=scope["start_tick"]) if history_index is not None else row.get("source_refs", []))
        graph.add_node(node, "PredicateTransition", pid, row["to_tick"],
            {"native_transition_id": native, "predicate_id": pid, "raw_predicate_id": row["predicate_id"],
             "bindings": bindings, "from_tick": row["from_tick"], "to_tick": row["to_tick"],
             "from_value": row.get("from_value", row.get("before")),
             "to_value": row.get("to_value", row.get("after")),
             "evidence_truth_ids": row.get("evidence_truth_ids", []),
             "source_refs": transition_refs,
             "source_pointer": f"/objective/predicate_transitions/{history_index}" if history_index is not None else None,
             "source_history_index": history_index,
             "source_record_family": "arm_semantic_history" if history_index is not None else "arm_predicate_transitions",
             "boundary_endpoints": [{"role": kind, "tick": tick,
                 "availability": "before_saved_window" if tick < scope["start_tick"] else "after_saved_window"}
                 for kind, tick in (("transition_from", row["from_tick"]), ("transition_to", row["to_tick"]))
                 if not scope["start_tick"] <= tick <= scope["end_tick"]],
             "raw": row},
            source_path(transition_source), line)
        at_time(node, row["to_tick"], transition_source, line)
        transition_ref[native] = node
        native_index[native] = node
        for endpoint_index, (kind, tick) in enumerate((("transition_from", row["from_tick"]),
                                                       ("transition_to", row["to_tick"]))):
            if history_index is not None:
                endpoint = truth_ref[row["evidence_truth_ids"][endpoint_index]]
            else:
                endpoint = assertions_by_tuple.get((row.get("tuple_id"), tick)) if row.get("tuple_id") else None
            if endpoint is None and history_index is None:
                matches = assertions_by_binding.get((pid, _json_key(bindings), tick), [])
                endpoint = next((item for item in matches if graph.node_index[item]["attrs"].get("truth_status") ==
                                 (row.get("from_value", row.get("before")) if kind == "transition_from" else row.get("to_value", row.get("after")))), None)
            if endpoint:
                graph.add_edge(kind, endpoint, node, source_file=source_path(transition_source), source_line=line) if kind == "transition_from" else graph.add_edge(kind, node, endpoint, source_file=source_path(transition_source), source_line=line)
            elif scope["start_tick"] <= tick <= scope["end_tick"]:
                graph.unresolved_relation(source_file=source_path(transition_source), source_line=line,
                    relation={"transition_id": native, "endpoint": kind, "tick": tick},
                    reason="transition assertion outside saved branch label range")

    event_ref: dict[str, str] = {}
    if event_path.exists():
        for line, row in event_rows:
            native = row["event_id"]
            if row.get("kind") == "measured_business_predicate_onset" and row.get("ontology_event") is False:
                tick = row["tick"]
                pid = canonical_predicate_id(row["predicate_id"])
                bindings = _logical_bindings(row["predicate_id"], row["bindings"], entity_info,
                                             graph.validator.predicates)
                onset = prefix + "onset:" + native
                graph.add_node(onset, "ContextAnchor", "measured_business_predicate_onset", tick,
                    {"anchor_kind": "predicate_onset",
                     "source_family": "measured_business_predicate_onset",
                     "native_onset_id": native,
                     "predicate_id": pid, "bindings": bindings,
                     "source_refs": row.get("source_refs", []), "raw": row}, source_path(event_path), line)
                at_time(onset, tick, event_path, line)
                event_ref[native] = onset
                native_index[native] = onset
                matches = assertions_by_binding.get((pid, _json_key(bindings), tick), [])
                target = next((item for item in matches if graph.node_index[item]["attrs"].get("truth_status") == "true"), None)
                if target:
                    graph.add_edge("predicate_onset_assertion", onset, target,
                                   source_file=source_path(event_path), source_line=line)
                else:
                    graph.unresolved_relation(source_file=source_path(event_path), source_line=line,
                        relation={"onset_id": native}, reason="X onset has no true predicate assertion")
                continue
            category = _event_category(row)
            tick = row.get("detection_tick", row.get("trigger_tick"))
            node = prefix + "ev:" + native
            bindings = row.get("bindings", {})
            if row.get("schema_name") == "l4_business_event_occurrence":
                from Dataset.tools.l4_business_semantics import profile_rule
                onset = profile_rule(profile)["onset_predicate"]
                bindings = _logical_bindings(onset, bindings, entity_info, graph.validator.predicates)
            graph.add_node(node, "EventOccurrence", category, tick,
                {"native_event_id": native, "event_family_id": row.get("event_family_id"),
                 "bindings": bindings, "rule_id": row.get("rule_id"),
                 "onset_evidence_status": "left_censored" if tick < scope["start_tick"] else
                     "boundary_limited" if any(ref in boundary_refs for ref in row.get("supporting_truth_ids", [])) else "within_saved_window",
                 "unobserved_support_references": [boundary_refs[ref] for ref in
                     row.get("supporting_truth_ids", []) + row.get("supporting_transition_ids", [])
                     if ref in boundary_refs],
                 "start_tick": row.get("start_tick"), "end_tick": row.get("end_tick"), "raw": row},
                source_path(event_path), line, rule_ref=row.get("rule_id"))
            event_ref[native] = node
            native_index[native] = node
            at_time(node, tick, event_path, line, "detection_tick")
            event_roles = {item["key"]: item["class"]
                           for item in graph.validator.events[category]["roles"]}
            for role, entity in bindings.items():
                if isinstance(entity, list):
                    continue
                ontology = event_roles.get(role)
                if ontology and ":" not in ontology:
                    ontology = "world:" + ontology
                namespace = binding_identity_namespace(ontology)
                if entity not in entity_info:
                    identity(entity, source=event_path, line=line, ontology=ontology,
                             completeness="binding_only", namespace=namespace)
                else:
                    namespace = entity_info[entity]["identity_namespace"]
                graph.add_edge("event_participant", node, ident(entity, namespace), role=role,
                               source_file=source_path(event_path), source_line=line)
            for native_transition in row.get("supporting_transition_ids", []):
                if native_transition in transition_ref:
                    graph.add_edge("transition_supports_event", transition_ref[native_transition], node,
                                   source_file=source_path(event_path), source_line=line, rule_ref=row.get("rule_id"))
                elif native_transition not in boundary_refs:
                    graph.unresolved_relation(source_file=source_path(event_path), source_line=line,
                        relation={"event_id": native, "transition_id": native_transition},
                        reason="objective support transition outside saved branch label range")
            for native_truth in row.get("supporting_truth_ids", []):
                if native_truth in truth_ref:
                    graph.add_edge("assertion_supports_event", truth_ref[native_truth], node,
                                   source_file=source_path(event_path), source_line=line, rule_ref=row.get("rule_id"))
                elif native_truth not in boundary_refs:
                    graph.unresolved_relation(source_file=source_path(event_path), source_line=line,
                        relation={"event_id": native, "truth_id": native_truth},
                        reason="objective support assertion outside saved branch label range")
            for index, phase in enumerate(row.get("event_phases", [])):
                phase_tick = phase.get("tick")
                phase_node = prefix + f"phase:{native}:{index}"
                graph.add_node(phase_node, "EventPhase", phase["phase_kind"], phase_tick,
                    {"native_event_id": native, "phase_index": index, "raw": phase}, source_path(event_path), line)
                at_time(phase_node, phase_tick, event_path, line, "phase_tick")
                graph.add_edge("event_phase", node, phase_node, role=phase.get("phase_kind"),
                               source_file=source_path(event_path), source_line=line)
                tid = phase.get("transition_id")
                if tid in transition_ref:
                    graph.add_edge("transition_supports_phase", transition_ref[tid], phase_node,
                                   source_file=source_path(event_path), source_line=line)
                    if phase["phase_kind"] == "onset" and row.get("schema_name") == "l4_business_event_occurrence":
                        graph.add_edge("transition_supports_event", transition_ref[tid], node,
                                       source_file=source_path(event_path), source_line=line,
                                       rule_ref=row["rule_id"])
    if outcome_path.exists():
        for line, row in outcome_rows:
            native = row["event_id"]
            terminal = row.get("terminal_tick")
            declared_terminal = terminal
            right_censored = terminal is not None and terminal > scope["end_tick"]
            terminal_boundary = [boundary_refs[ref] for phase in row.get("lifecycle_phase_evidence", [])
                for ref in phase.get("supporting_truth_ids", []) + ([phase["entry_transition_id"]] if phase.get("entry_transition_id") else [])
                if ref in boundary_refs]
            status = "right_censored" if right_censored else "left_censored" if terminal_boundary else row["status"]
            if right_censored:
                terminal = None
            if native in event_ref and graph.node_index[event_ref[native]]["node_type"] == "ContextAnchor":
                node = prefix + f"pred_out:{native}:{line}"
                graph.add_node(node, "PredicateEpisodeOutcome", "x.predicate_episode_outcome", terminal,
                    {"native_event_id": native, "status": status,
                     "declared_terminal_tick": declared_terminal, "censored": right_censored or bool(terminal_boundary),
                     "observation_end_tick": scope["end_tick"], "raw": row},
                    source_path(outcome_path), line)
                graph.add_edge("predicate_episode_outcome", event_ref[native], node,
                               source_file=source_path(outcome_path), source_line=line)
            else:
                category = _event_category(row) if row.get("event_type_id") or row.get("event_family_id") else graph.node_index[event_ref[native]]["category_id"] if native in event_ref else "unresolved_event"
                node = prefix + "out:" + str(row.get("outcome_id", native + ":" + str(line)))
                graph.add_node(node, "EventOutcome", category, terminal,
                    {"native_outcome_id": row.get("outcome_id"), "native_event_id": native,
                     "status": status, "declared_terminal_tick": declared_terminal,
                     "censored": right_censored or bool(terminal_boundary),
                     "censoring_side": "right" if right_censored else "left" if terminal_boundary else None,
                     "terminal_evidence_status": "outside_saved_window" if right_censored else
                         "boundary_limited" if terminal_boundary else "within_saved_window",
                     "observation_end_tick": scope["end_tick"],
                     "unobserved_support_references": [boundary_refs[ref] for ref in row.get("supporting_truth_ids", []) if ref in boundary_refs],
                     "raw": row}, source_path(outcome_path), line)
                if native in event_ref:
                    graph.add_edge("event_outcome", event_ref[native], node,
                                   source_file=source_path(outcome_path), source_line=line)
                else:
                    graph.unresolved_relation(source_file=source_path(outcome_path), source_line=line,
                        relation={"event_id": native}, reason="outcome event endpoint absent")
            native_index[row.get("outcome_id", native + ":outcome")] = node
            at_time(node, terminal, outcome_path, line, "terminal_tick")
            for truth_id in row.get("supporting_truth_ids", []):
                endpoint = truth_ref.get(truth_id)
                if endpoint:
                    graph.add_edge("assertion_supports_outcome", endpoint, node, role="outcome_support",
                        source_file=source_path(outcome_path), source_line=line)
                elif truth_id not in boundary_refs:
                    graph.unresolved_relation(source_file=source_path(outcome_path), source_line=line,
                        relation={"outcome_id": row.get("outcome_id"), "truth_id": truth_id},
                        reason="outcome support truth ID absent from branch labels")
            for phase_index, phase_evidence in enumerate(row.get("lifecycle_phase_evidence", [])):
                for truth_id in phase_evidence.get("supporting_truth_ids", []):
                    endpoint = truth_ref.get(truth_id)
                    if endpoint:
                        graph.add_edge("assertion_supports_outcome", endpoint, node,
                            role=f"lifecycle_phase:{phase_index}",
                            source_file=source_path(outcome_path), source_line=line)
                    elif truth_id not in boundary_refs:
                        graph.unresolved_relation(source_file=source_path(outcome_path), source_line=line,
                            relation={"outcome_id": row.get("outcome_id"), "truth_id": truth_id,
                                      "phase_index": phase_index},
                            reason="lifecycle support truth ID absent from branch labels")
                transition_id = phase_evidence.get("entry_transition_id")
                if transition_id:
                    endpoint = transition_ref.get(transition_id)
                    if endpoint:
                        graph.add_edge("transition_supports_outcome", endpoint, node,
                            role=f"lifecycle_phase:{phase_index}",
                            source_file=source_path(outcome_path), source_line=line)
                    elif transition_id not in boundary_refs:
                        graph.unresolved_relation(source_file=source_path(outcome_path), source_line=line,
                            relation={"outcome_id": row.get("outcome_id"),
                                      "entry_transition_id": transition_id},
                            reason="lifecycle transition ID absent from branch labels")

    action_path = branch_root / "raw/actions.jsonl"
    action_audits: dict[str, tuple[str, int, str, str, str, dict]] = {}
    script_dispatches: dict[str, str] = {}
    if action_path.exists():
        for line, row in source_rows(action_path):
            action = row["action"]
            action_id = action.get("action_id")
            if not isinstance(action_id, str) or not action_id or action_id in action_audits:
                raise ValueError(f"ARM action audit has missing or repeated action ID: {action_path}:{line}")
            status = row["result"]["status"]
            origin = row.get("origin")
            if origin not in {"source_script", "intervention"}:
                raise ValueError(f"ARM action audit has unknown origin: {action_path}:{line}:{origin}")
            if status not in {"ok", "omitted_by_explicit_intervention", "omitted_by_configuration"}:
                raise ValueError(f"ARM action audit has unknown status: {action_path}:{line}:{status}")
            if status != "ok" and origin != "source_script":
                raise ValueError(f"omitted ARM command is not a source script action: {action_path}:{line}")
            audit_node = prefix + f"action_audit:{line}"
            graph.add_node(audit_node, "ContextAnchor", "arm_action_audit", row["tick"],
                {"anchor_kind": "command_audit", "source_family": "arm_action_audit",
                 "native_action_id": action_id, "dispatch_status": status,
                 "origin": origin, "raw": row},
                source_path(action_path), line)
            at_time(audit_node, row["tick"], action_path, line)
            action_audits[action_id] = audit_node, line, status, action["type"], origin, action
            if status != "ok":
                continue
            node = prefix + f"act:{line}"
            tick = row["tick"]
            graph.add_node(node, "CommandDispatch", action["type"], tick,
                {"native_action_id": action_id, "target_scope": action_target_scope(action),
                 "target_entity_id": action.get("entity_id"),
                 "source_action": action, "source_result": row["result"],
                 "dispatch_status": status, "origin": origin,
                 "semantic_use": "capture_request" if action["type"] == "capture_screenshot" else "simulation_control"},
                source_path(action_path), line)
            at_time(node, tick, action_path, line, "dispatch_tick")
            native_index[action_id] = node
            native_index["action:" + action_id] = node
            if origin == "source_script":
                script_dispatches[action_id] = node
            target = action.get("entity_id")
            if action["type"] in ENTITY_TARGET_ACTIONS and not target:
                graph.unresolved_relation(source_file=source_path(action_path), source_line=line,
                    relation={"action_id": action_id},
                    reason="dispatched entity-target command lacks target ID")
            if target:
                if target not in entity_info:
                    identity(target, source=action_path, line=line, completeness="action_only")
                graph.add_edge("action_target", node, ident(target), role="target",
                               source_file=source_path(action_path), source_line=line)

    # Script plan and actual dispatch are separate even when coordinates match.
    chain_nodes = branch_root / "chain/nodes.jsonl"
    if not chain_nodes.exists():
        chain_nodes = branch_root / "chain/events.jsonl"
    chain_emissions: dict[str, tuple[int | None, Path, int, dict]] = {}
    if chain_nodes.exists():
        for chain_line, chain_row in source_rows(chain_nodes):
            if chain_row.get("kind") == "script_emission" and chain_row.get("event_id"):
                native_event = chain_row["event_id"]
                if native_event in chain_emissions:
                    raise ValueError(f"duplicate chain script emission: {chain_nodes}:{native_event}")
                chain_emissions[native_event] = chain_row.get("tick"), chain_nodes, chain_line, chain_row
    parent_script_ref = manifest_reference(manifest, "script")
    script_path = branch_root / "ue/event_script.json"
    matched_action_audits: set[str] = set()
    if not script_path.is_file():
        raise FileNotFoundError(f"ARM exported execution script missing: {script_path}")
    script = read_json(script_path)
    seen_plan_action_ids: set[str] = set()
    if "parameters" in script:
        graph.add_node(prefix + "script_parameters", "ContextAnchor", "arm_script_plan", None,
            {"anchor_kind": "script_parameters", "source_family": "arm_script_plan",
             "raw": {"parameters": script["parameters"]}}, source_path(script_path))
    trigger_by_id = {item["trigger_id"]: item for item in script.get("triggers", [])}
    for event_index, event in enumerate(script.get("events", [])):
        native = event["event_id"]
        trigger = trigger_by_id[event["trigger_ref"]]
        scheduled_tick = trigger.get("tick") if trigger.get("type") == "tick" else None
        emission = chain_emissions.get(native)
        tick = emission[0] if emission is not None else scheduled_tick
        script_node = native_index.get("script:" + native)
        if script_node is None:
            script_node = prefix + "script:" + native
            intent = event.get("intent")
            graph.add_node(script_node, "ScriptEvent", intent, tick,
                {"native_event_id": native, "intent": intent,
                 "category_status": "source_intent" if intent else "unresolved_source_intent",
                 "intent_stage": event.get("intent_stage"),
                 "log_category": (event.get("log_event") or {}).get("category"),
                 "overlay": (event.get("log_event") or {}).get("overlay"),
                 "overlay_role": "render_hint",
                 "causal_chain_id": event.get("causal_chain_id"),
                 "trigger_ref": event["trigger_ref"], "priority": event.get("priority"),
                 "max_fire_count": event.get("max_fire_count"),
                 "target_roles": event.get("target_roles"), "actions": event.get("actions", []),
                 "script_ontology_event_id": event.get("ontology_event"),
                 "script_emission_ontology_flag": emission[3].get("ontology_event") if emission is not None else None,
                 "raw": event, "source_event_index": event_index,
                 "execution_script_role": "saved_branch_execution_script",
                 "parent_script_source_file": parent_script_ref,
                 "script_scheduled": trigger.get("type") == "tick",
                 "scheduled_tick": scheduled_tick,
                 "emitted_tick": emission[0] if emission is not None else None,
                 "time_source_file": source_path(emission[1]) if emission is not None else source_path(script_path),
                 "time_source_line": emission[2] if emission is not None else None}, source_path(script_path))
            native_index["script:" + native] = script_node
            if tick is not None:
                time_path = emission[1] if emission is not None else script_path
                time_line = emission[2] if emission is not None else None
                at_time(script_node, tick, time_path, time_line,
                        "dispatch_tick" if emission is not None else "scheduled_tick")
        for index, action in enumerate(event.get("actions", [])):
            action_id = action["action_id"]
            if action_id in seen_plan_action_ids:
                raise ValueError(f"ARM execution script repeats action ID: {script_path}:{action_id}")
            seen_plan_action_ids.add(action_id)
            plan_node = prefix + f"plan:{native}:{action_id}"
            audit = action_audits.get(action_id)
            if audit and audit[4] != "source_script":
                graph.unresolved_relation(source_file=source_path(action_path), source_line=audit[1],
                    relation={"action_id": action_id, "script_event_id": native},
                    reason="intervention action ID collides with source script plan")
                audit = None
            if audit:
                if audit[3] != action["type"]:
                    raise ValueError(f"ARM action audit and script type differ: {action_path}:{audit[1]}")
                if audit[5] != action:
                    raise ValueError(f"ARM action audit and execution script action differ: {action_path}:{audit[1]}")
                matched_action_audits.add(action_id)
            graph.add_node(plan_node, "ActionPlan", action["type"], scheduled_tick,
                {"native_event_id": native, "native_action_id": action_id,
                 "source_event_index": event_index, "source_action_index": index,
                 "target_scope": action_target_scope(action),
                 "target_entity_id": action.get("entity_id"),
                 "raw": action, "coordinate_owner": "source_script_plan",
                 "execution_script_role": "saved_branch_execution_script",
                 "parent_script_source_file": parent_script_ref,
                 "audit_status": audit[2] if audit else "not_recorded",
                 "audit_source_file": source_path(action_path) if audit else None,
                 "audit_source_line": audit[1] if audit else None}, source_path(script_path))
            graph.add_edge("script_declares_action", script_node, plan_node,
                           source_file=source_path(script_path))
            if audit:
                graph.add_edge("action_plan_audit", plan_node, audit[0],
                    source_file=source_path(action_path), source_line=audit[1])
            if scheduled_tick is not None:
                at_time(plan_node, scheduled_tick, script_path, role="scheduled_tick")
            actual = script_dispatches.get(action_id)
            if actual:
                graph.add_edge("action_plan_dispatch", plan_node, actual,
                               source_file=source_path(script_path))
    missing_script_plans = {action_id for action_id, audit in action_audits.items()
                            if audit[4] == "source_script" and action_id not in matched_action_audits}
    if missing_script_plans:
        raise ValueError(f"ARM source-script command lacks execution plan: {action_path}:{sorted(missing_script_plans)}")

    if chain_nodes.exists():
        for line, row in source_rows(chain_nodes):
            native = row["node_id"]
            if native in native_index:
                continue
            kind = row.get("kind")
            if kind == "script_emission":
                node = prefix + "script:" + row.get("event_id", native)
                if node not in graph.node_index:
                    intent = row.get("intent")
                    graph.add_node(node, "ScriptEvent", intent, row.get("tick"),
                        {"native_event_id": row.get("event_id"), "native_chain_node_id": native,
                         "intent": intent, "category_status": "unresolved_source_intent" if intent is None else "source_intent",
                         "script_emission_ontology_flag": row.get("ontology_event"),
                         "raw": row}, source_path(chain_nodes), line)
                    if row.get("tick") is not None:
                        at_time(node, row["tick"], chain_nodes, line, "dispatch_tick")
                native_index[native] = node
            elif kind == "observed_action_dispatch":
                action_native = row.get("action_id")
                if action_native in native_index:
                    native_index[native] = native_index[action_native]
                else:
                    graph.unresolved_relation(source_file=source_path(chain_nodes), source_line=line,
                        relation=row, reason="chain action dispatch absent from raw actions")
            elif kind == "measured_business_predicate_onset":
                native_event = row.get("event_id")
                if native_event in event_ref:
                    native_index[native] = event_ref[native_event]
                else:
                    graph.unresolved_relation(source_file=source_path(chain_nodes), source_line=line,
                        relation=row, reason="chain onset absent from objective")
            elif kind == "measured_geometry_support":
                node = prefix + "geometry:" + native
                graph.add_node(node, "ContextAnchor", "measured_geometry_support", row.get("tick"),
                    {"anchor_kind": "geometry_support",
                     "source_family": "measured_geometry_support", "raw": row},
                    source_path(chain_nodes), line)
                at_time(node, row.get("tick"), chain_nodes, line)
                native_index[native] = node
            elif kind == "local_business_observation":
                reference = row["evidence_ref"]
                result_path = branch_root / reference
                if reference != "objective/local_business_result.json":
                    raise ValueError(f"local business result source differs: {chain_nodes}:{line}")
                result = read_json(result_path)
                if result["observation_id"] != native or result["onset_tick"] != row["tick"]:
                    raise ValueError(f"local business result identity differs: {chain_nodes}:{line}")
                node = prefix + "local_result:" + native
                graph.add_node(node, "ContextAnchor", "local_business_measurement", row["tick"],
                    {"anchor_kind": "local_business_measurement", "source_family": "local_business_measurement",
                     "native_observation_id": native, "result": result,
                     "authority_scope": result["authority_scope"]}, source_path(result_path))
                at_time(node, row["tick"], result_path)
                native_index[native] = node
            else:
                graph.unresolved_relation(source_file=source_path(chain_nodes), source_line=line,
                    relation=row, reason="chain node is a projection; endpoint requires objective source")
    relation_path = branch_root / "chain/relations.jsonl"
    if relation_path.exists():
        for line, row in source_rows(relation_path):
            kind = row.get("relation_type", row.get("relation", row.get("type")))
            raw_source = row.get("source", row.get("from"))
            raw_target = row.get("target", row.get("to"))
            source = native_index.get(raw_source)
            target = native_index.get(raw_target)
            if source and target:
                try:
                    graph.add_edge(kind, source, target, source_file=source_path(relation_path),
                        source_line=line, native_relation_id=row.get("relation_id"),
                        attrs={"native_relation_type": kind,
                               "raw": {key: value for key, value in row.items()
                                       if key != "causal_claim"}})
                except ContractError:
                    raise
                except ValueError as exc:
                    graph.unresolved_relation(source_file=source_path(relation_path), source_line=line,
                        relation=row, reason=f"chain relation violates structure: {exc}")
            elif ((source or raw_source in boundary_refs) and (target or raw_target in boundary_refs)):
                node = prefix + f"boundary_relation:{line}"
                graph.add_node(node, "ContextAnchor", "script_chain_context", None,
                    {"anchor_kind": "script_chain_context", "source_family": "script_chain_context",
                     "relation_type": kind, "declared_source": raw_source, "declared_target": raw_target,
                     "boundary_references": [boundary_refs[ref] for ref in (raw_source, raw_target) if ref in boundary_refs],
                     "relation_status": "unobserved_endpoint_outside_saved_window"},
                    source_path(relation_path), line)
            else:
                graph.unresolved_relation(source_file=source_path(relation_path), source_line=line,
                    relation=row, reason="chain endpoint outside available objective/raw sources")
    if profile.startswith("X"):
        from Dataset.world_model.graph.event_derivation import materialize_x_events
        graph.scope["registered_X_events"] = materialize_x_events(graph, prefix)
    return graph.close("complete_with_unresolved" if graph.unresolved else "complete")
