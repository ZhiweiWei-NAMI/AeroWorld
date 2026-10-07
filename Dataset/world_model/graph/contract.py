"""Authoritative structure and catalog compiler for the label temporal graph.

Category indices are append-only within CONTRACT_VERSION. Removing a category or
changing an edge signature requires a version bump and a full graph rebuild; a
partial rebuild would leave partitions and model indices on different contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Dataset.semantic_truth.entity_scope import roster_index
from Dataset.world_model.graph.entity_identity import read_roster

import orjson
import yaml


REPO = Path(__file__).resolve().parents[3]
CONTRACT_VERSION = "label-graph-7"
NODE_TYPES = (
    "TimePoint", "EntityIdentity", "EntityState", "PredicateAssertion",
    "PredicateTransition", "PredicateEpisodeOutcome", "EventOccurrence",
    "EventPhase", "EventOutcome", "ScriptEvent", "ActionPlan",
    "CommandDispatch", "ContextAnchor",
)
AT_TIME_ROLES = ("target_tick", "detection_tick", "phase_tick", "terminal_tick",
                 "trigger_tick", "dispatch_tick", "scheduled_tick")
# A RelativeState is an edge. Distinct source rows remain distinct edge records.
EDGE_ENDPOINTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "identity_state": (("EntityIdentity",), ("EntityState",)),
    "compute_node_host": (("EntityIdentity",), ("EntityIdentity",)),
    "matrix_scope_entity": (("ContextAnchor",), ("EntityIdentity",)),
    "compute_state_node": (("ContextAnchor",), ("EntityIdentity",)),
    "controller_state": (("ContextAnchor",), ("EntityIdentity",)),
    "supplement_subject": (("ContextAnchor",), ("EntityIdentity",)),
    "at_time": (tuple(t for t in NODE_TYPES if t not in {"TimePoint", "EntityIdentity"}), ("TimePoint",)),
    "same_entity_next": (("EntityState",), ("EntityState",)),
    "predicate_argument": (("PredicateAssertion",), ("EntityIdentity", "EntityState", "ContextAnchor")),
    "same_predicate_next": (("PredicateAssertion",), ("PredicateAssertion",)),
    "candidate_scope_change": (("PredicateAssertion", "ContextAnchor"), ("PredicateAssertion", "ContextAnchor")),
    "transition_from": (("PredicateAssertion",), ("PredicateTransition",)),
    "transition_to": (("PredicateTransition",), ("PredicateAssertion",)),
    "predicate_episode_outcome": (("PredicateAssertion", "PredicateTransition", "ContextAnchor"), ("PredicateEpisodeOutcome",)),
    "event_participant": (("EventOccurrence", "ScriptEvent"), ("EntityIdentity", "EntityState", "ContextAnchor")),
    "assertion_supports_event": (("PredicateAssertion",), ("EventOccurrence",)),
    "transition_supports_event": (("PredicateTransition",), ("EventOccurrence",)),
    "assertion_supports_phase": (("PredicateAssertion",), ("EventPhase",)),
    "transition_supports_phase": (("PredicateTransition",), ("EventPhase",)),
    "event_phase": (("EventOccurrence",), ("EventPhase",)),
    "event_outcome": (("EventOccurrence",), ("EventOutcome",)),
    "assertion_supports_outcome": (("PredicateAssertion",), ("EventOutcome", "PredicateEpisodeOutcome")),
    "transition_supports_outcome": (("PredicateTransition",), ("EventOutcome", "PredicateEpisodeOutcome")),
    "projection_equivalent": (("PredicateAssertion",), ("PredicateAssertion",)),
    "source_conflict": (("PredicateAssertion",), ("PredicateAssertion",)),
    "continuity_break_from": (("PredicateAssertion",), ("ContextAnchor",)),
    "continuity_break_to": (("ContextAnchor",), ("PredicateAssertion",)),
    "predicate_onset_evidence": (("PredicateTransition",), ("EventOccurrence",)),
    "predicate_terminal_evidence": (("PredicateTransition",), ("EventOccurrence",)),
    "predicate_escalation_support_evidence": (("PredicateTransition", "PredicateAssertion"), ("EventOccurrence",)),
    "has_observed_outcome": (("EventOccurrence",), ("EventOutcome",)),
    "predicate_onset_assertion": (("ContextAnchor",), ("PredicateAssertion", "PredicateTransition")),
    "script_dependency": (("ScriptEvent", "EventOccurrence", "ActionPlan", "CommandDispatch"), ("ScriptEvent", "EventOccurrence", "ActionPlan", "CommandDispatch")),
    "script_declared_causal": (("ScriptEvent", "EventOccurrence", "ActionPlan", "CommandDispatch", "ContextAnchor"), ("ScriptEvent", "EventOccurrence", "ActionPlan", "CommandDispatch", "ContextAnchor")),
    "script_declared_context": (("ScriptEvent", "EventOccurrence", "ActionPlan", "CommandDispatch", "ContextAnchor"), ("ScriptEvent", "EventOccurrence", "ActionPlan", "CommandDispatch", "ContextAnchor")),
    "geometry_trigger_support": (("PredicateAssertion", "PredicateTransition", "ContextAnchor"), ("ScriptEvent", "EventOccurrence")),
    "predicate_supports_event": (("PredicateAssertion", "PredicateTransition"), ("ScriptEvent", "EventOccurrence")),
    "observed_dispatch": (("ScriptEvent", "ActionPlan"), ("CommandDispatch",)),
    "action_target": (("ActionPlan", "CommandDispatch"), ("EntityIdentity", "EntityState")),
    "action_plan_dispatch": (("ActionPlan",), ("CommandDispatch",)),
    "action_plan_audit": (("ActionPlan",), ("ContextAnchor",)),
    "script_declares_action": (("ScriptEvent",), ("ActionPlan",)),
    "action_recorded_outcome": (("CommandDispatch",), ("EventOutcome", "PredicateEpisodeOutcome", "EntityState")),
    "observed_outcome": (("ScriptEvent", "EventOccurrence", "CommandDispatch"), ("EventOutcome", "PredicateEpisodeOutcome", "PredicateAssertion")),
    "patch_then_measured_onset": (("CommandDispatch",), ("PredicateAssertion",)),
    "RelativeState": (("EntityState",), ("EntityState",)),
}
TRUTH = {"true", "false", "unknown", "out_of_scope"}
ENTITY_CATEGORIES = (
    "uav", "vehicle", "pedestrian", "airspace_corridor", "facility", "prop",
    "crowd_anchor", "facade_anchor", "vehicle_anchor", "traffic_light",
    "compute_node",
)


def declared_formal_samples() -> list[str]:
    """Enumerate episode directories whose manifests declare the same episode ID."""
    root = REPO / "aw_data/render_ready_episodes_capture_filtered"
    episodes: list[str] = []
    for episode in sorted(root.iterdir()):
        if not episode.is_dir():
            continue
        manifest_path = episode / "episode_manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("episode_id") != episode.name:
            raise ValueError(f"formal episode manifest identity differs from directory: {manifest_path}")
        episodes.append(episode.name)
    return episodes


def declared_arm_samples() -> list[Path]:
    """Enumerate four-level ARM branches by their manifest, independent of objective files."""
    root = REPO / "arms"
    if not root.is_dir():
        raise FileNotFoundError(root)
    branches: list[Path] = []
    for branch in sorted(root.glob("*/*/*/*")):
        if not branch.is_dir():
            continue
        manifest_path = branch / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("arm_id") != branch.name:
            raise ValueError(f"ARM manifest identity differs from branch directory: {manifest_path}")
        branches.append(branch)
    return branches


def declared_samples() -> tuple[list[str], list[Path]]:
    """Return formal and ARM sample identities from their own manifests."""
    return declared_formal_samples(), declared_arm_samples()


def _declared_script_paths(branches: list[Path] | None = None) -> list[Path]:
    """Resolve script path declarations without requiring referenced files to exist."""
    sources = set((REPO / "Dataset/scenarios").rglob("event_script.json"))
    for branch in (declared_arm_samples() if branches is None else branches):
        manifest_path = branch / "manifest.json"
        manifest = read_json(manifest_path)
        direct = (manifest.get("sources") or {}).get("script")
        referred = (manifest.get("source_refs") or {}).get("script")
        if direct is not None and referred is not None and direct != referred:
            raise ValueError(f"conflicting script references: {manifest_path}")
        reference = direct if direct is not None else referred
        if isinstance(reference, dict):
            reference = reference.get("path")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"missing script reference: {manifest_path}")
        script_path = REPO / reference
        sources.add(script_path)
    return sorted(sources)


def _script_sources() -> list[Path]:
    """Resolve the current executable script set, including ARM-local revisions."""
    sources = _declared_script_paths()
    for script_path in sources:
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
    return sources


def _script_categories() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    intents: dict[str, set[str]] = {}
    actions: dict[str, set[str]] = {}
    for path in _script_sources():
        source = source_path(path)
        for event in read_json(path).get("events", []):
            intent = event.get("intent")
            if not isinstance(intent, str) or not intent:
                raise ValueError(f"script event has no intent: {source}#{event.get('event_id')}")
            intents.setdefault(intent, set()).add(source)
            for action in event.get("actions", []):
                category = action.get("type")
                if not isinstance(category, str) or not category:
                    raise ValueError(f"script action has no type: {source}#{event.get('event_id')}")
                actions.setdefault(category, set()).add(source)
    return intents, actions


CONTEXT_STRUCTURAL_CATEGORIES = (
    "candidate_added", "candidate_removed", "predicate_continuity_break",
    "compute_comm_predicate_matrix", "domain_state_supplement", "compute_comm_supplement",
    "local_business_measurement", "measured_business_predicate_onset",
    "measured_geometry_support", "script_chain_context", "domain_state_observation",
    "arm_script_plan",
    "arm_action_audit", "formal_weather_meta", "sumo_controller_state",
)
CONTEXT_PREDICATE_STATE_FAMILIES = (
    "predicate_contract_agent_proximity_geometry",
    "predicate_contract_aircraft_corridor_geometry",
    "predicate_contract_aircraft_geometry",
    "predicate_contract_aircraft_pair_geometry",
    "predicate_contract_aircraft_pedestrian_geometry",
    "predicate_contract_aircraft_plan",
    "predicate_contract_aircraft_protected_region_geometry",
    "predicate_contract_aircraft_region_geometry",
    "predicate_contract_aircraft_structure_geometry",
    "predicate_contract_aircraft_vehicle_geometry",
    "predicate_contract_corridor_geometry",
    "predicate_contract_region_runtime_state",
    "predicate_contract_restricted_airspace_state",
    "predicate_contract_vehicle_pedestrian_geometry",
)
CONTEXT_ANCHOR_KINDS = (
    "candidate_scope_change", "predicate_continuity_break", "compute_comm_matrix",
    "supplement_state", "domain_state_observation", "local_business_measurement",
    "predicate_onset", "geometry_support", "script_chain_context",
    "script_parameters", "command_audit",
)
CONTEXT_OBSERVED_DOMAIN_EXTENSIONS = (
    "road_segment_state", "signal_queue_lane_state", "traffic_signal_state",
    "control_response_state",
)
CONTEXT_OBSERVABLE_COMPLETION_FAMILIES = (
    "hazmat_response_state", "lockdown_control", "lockdown_region_state", "observable_communication_state",
    "observable_control_state", "observable_environment", "observable_ground_surface",
    "observable_incident_state", "observable_mission_state", "observable_navigation_state",
    "observable_pedestrian_response", "observable_security_state", "observable_sensor_state",
    "observable_vehicle_response", "observable_pad_facility",
)


def strip_source_digests(value: Any) -> Any:
    """Apply the established integrity exclusions to an original source root."""
    from Dataset.world_model.graph.structure_schema import project_record_for_publication
    if isinstance(value, dict):
        return project_record_for_publication(value)
    if isinstance(value, list):
        return [strip_source_digests(item) for item in value]
    return value


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def iter_jsonl(path: Path):
    with path.open("rb") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                yield line_no, orjson.loads(line)


def source_path(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def canonical_predicate_id(raw_id: str) -> str:
    if raw_id.startswith("x.geometry.trig_"):
        return "x.geometry.entity_proximity"
    if raw_id.startswith("l4_business."):
        return "l4." + raw_id.split(".", 1)[1]
    if raw_id.startswith("l4."):
        return raw_id
    if raw_id in L4_DEFINITIONS:
        return "l4." + raw_id
    return raw_id


L4_DEFINITIONS = read_json(REPO / "Dataset/semantic_rules/profiles/l4_arm_business_semantics.json")["definitions"]


def _l4_roles(name: str) -> list[dict[str, str]]:
    pair_vehicle = {"near_pair", "braking_active", "yielding_active", "braking_stopped", "yield_response"}
    uav_vehicle = {"uav_vehicle_proximity", "uav_vehicle_contact"}
    uav_pedestrian = {"aircraft_near_pedestrian"}
    pedestrian = {"landing_zone_occupied", "pedestrian_clear_of_landing_zone"}
    vehicle = {"vehicle_emergency_stop_active", "vehicle_physically_stopped", "sensor_fault", "minimal_risk_maneuver", "declared_stopped_safe", "physically_stopped_safe", "civilian_yielding"}
    ambulance = {"priority_request_active", "priority_request_acknowledged"}
    ambulance_vehicle = {"ambulance_approaching_vehicle", "yield_clearance_satisfied", "priority_passage_completed", "ambulance_geometry_passed"}
    if name in pair_vehicle:
        return [{"key": "first", "class": "world:GroundVehicle"}, {"key": "second", "class": "world:GroundVehicle"}]
    if name in uav_vehicle:
        return [{"key": "aircraft", "class": "world:UnmannedAircraft"}, {"key": "vehicle", "class": "world:GroundVehicle"}]
    if name in uav_pedestrian:
        return [{"key": "aircraft", "class": "world:UnmannedAircraft"}, {"key": "pedestrian", "class": "world:Pedestrian"}]
    if name in pedestrian:
        return [{"key": "pedestrian", "class": "world:Pedestrian"}]
    if name in vehicle:
        return [{"key": "vehicle", "class": "world:GroundVehicle"}]
    if name in ambulance:
        return [{"key": "ambulance", "class": "world:GroundVehicle"}]
    if name in ambulance_vehicle:
        return [{"key": "ambulance", "class": "world:GroundVehicle"}, {"key": "vehicle", "class": "world:GroundVehicle"}]
    return [{"key": "aircraft", "class": "world:UnmannedAircraft"}]


def _x_subject_classes(name: str) -> list[str]:
    if name == "x.motion.moving":
        return ["world:UnmannedAircraft", "world:Pedestrian", "world:GroundVehicle"]
    if name == "x.pedestrian.fallen" or name.endswith("incident_state.dispatch_active"):
        return ["world:Pedestrian"]
    if name == "x.facility.multiple_requests" or name.startswith("x.runtime.facility_state."):
        return ["world:LandingPad"]
    if name.endswith("communication_state.station_unavailable"):
        return ["world:CommunicationStation", "world:UnmannedAircraft"]
    return ["world:UnmannedAircraft"]


def compile_catalog() -> list[dict[str, Any]]:
    """Compile declared vocabulary from source contracts, never from demo windows."""
    from Dataset.world_model.graph.entity_identity import ENTITY_TARGET_ACTIONS, NON_ENTITY_ACTION_SCOPES
    from Dataset.world_model.graph.field_contract import UNIT_DEFINITIONS

    rows: list[dict[str, Any]] = [{"kind": "schema", "id": "label_temporal_graph", "version": CONTRACT_VERSION,
        "time_unit": "simulation_tick", "source": "Dataset/world_model/graph/contract.py"}]
    rows.extend({"kind": "unit_definition", "id": name, **definition,
                 "source": "Dataset/world_model/graph/field_contract.py"}
                for name, definition in sorted(UNIT_DEFINITIONS.items()))
    rows.extend({"kind": "action_target_scope", "id": name, "target_scope": scope,
                 "source": "Dataset/tools/batch_generate.py:action_realizations"}
                for name, scope in sorted(NON_ENTITY_ACTION_SCOPES.items()))
    rows.extend({"kind": "action_result_target", "id": name,
                 "source": "arms/*/*/*/*/raw/actions.jsonl:paired_action_result"}
                for name in sorted(ENTITY_TARGET_ACTIONS))
    rows.extend({"kind": "graph_edge_role", "id": name,
                 "edge_type": "matrix_scope_entity", "source": "Dataset/semantic_simulation/compute_comm.py"}
                for name in ("compute_node", "uav"))
    rows.extend({"kind": "graph_edge_role", "id": name, "edge_type": "at_time",
                 "source": "Dataset/world_model/graph/formal.py:at_time + Dataset/world_model/graph/arm.py:at_time"}
                for name in AT_TIME_ROLES)
    rows.extend({"kind": "node_type", "id": item, "index": i, "source": "Dataset/world_model/graph/contract.py"}
                for i, item in enumerate(NODE_TYPES))
    rows.extend({"kind": "edge_type", "id": name, "index": i, "source_types": sources,
                 "target_types": targets, "source": "Dataset/world_model/graph/contract.py"}
                for i, (name, (sources, targets)) in enumerate(EDGE_ENDPOINTS.items()))
    rows.extend({"kind": "entity_category", "id": item,
                 "source": ("Dataset/semantic_truth/world_truth.py:compute_node"
                            if item == "compute_node" else
                            "Dataset/semantic_rules/profiles/entity_scope_classes.json")}
                for item in ENTITY_CATEGORIES)
    core_path = REPO / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"
    core = read_json(core_path)
    for p in core["templates"]:
        rows.append({"kind": "predicate", "id": p["id"], "family": p["ontology_module"],
            "source_system": "formal", "raw_ids": [p["id"]], "roles": p["argument_roles"],
            "arity": len(p["argument_roles"]), "distinct_role_sets": p.get("grounding_spec", {}).get("distinct_role_sets", []),
            "unordered_role_sets": p.get("grounding_spec", {}).get("unordered_role_sets", []),
            "execution_status": p["implementation_status"], "time_granularity": "5 ticks",
            "rule": strip_source_digests({"expression": p.get("expression"), "evaluation_spec": p.get("evaluation_spec"),
                                         "state_contract": p.get("state_contract"), "grounding_spec": p.get("grounding_spec")}),
            "source": source_path(core_path), "instance_count": None})
        for field in p.get("state_contract", {}).get("required_fields", []):
            rows.append({"kind": "rule_field", "id": p["id"] + ":" + field["field"],
                "predicate_id": p["id"], "field": field["field"], "unit": field.get("unit"),
                "source": field.get("source"), "execution_status": p["implementation_status"]})
    l4_path = REPO / "Dataset/semantic_rules/profiles/l4_arm_business_semantics.json"
    for name, definition in L4_DEFINITIONS.items():
        roles = _l4_roles(name)
        rows.append({"kind": "predicate", "id": "l4." + name, "family": "l4_business", "source_system": "L4",
            "raw_ids": [name], "roles": roles, "arity": len(roles),
            "distinct_role_sets": [["first", "second"]] if name in {"near_pair", "braking_active", "yielding_active", "braking_stopped", "yield_response"} else [],
            "unordered_role_sets": [["first", "second"]] if name in {"near_pair", "braking_active", "yielding_active", "braking_stopped", "yield_response"} else [],
            "execution_status": "declared_only" if name == "yield_response" else "source_executable",
            "time_granularity": "ARM dense/sampled", "rule": definition,
            "source": source_path(l4_path), "instance_count": None})
    x_raw: dict[str, dict[str, Any]] = {}
    for path in sorted((REPO / "Dataset/scenarios/X_cross_layer").glob("*/event_script.json")):
        script = read_json(path)
        scene_path = path.with_name("scene_setup.json")
        scene_entities = roster_index(read_roster(scene_path))
        for trigger in script.get("triggers", []):
            tid = trigger.get("trigger_id")
            if isinstance(tid, str) and trigger.get("type") == "entity_proximity":
                raw_id = "x.geometry." + tid
                first = scene_entities.get(trigger.get("entity_a"))
                second = scene_entities.get(trigger.get("entity_b"))
                if first is None or second is None:
                    raise ValueError(f"trigger endpoint absent from scene: {path}#{tid}")
                x_raw[raw_id] = {"source": source_path(path), "entity_a": trigger.get("entity_a"),
                    "entity_b": trigger.get("entity_b"), "operator": trigger.get("operator"),
                    "metric": trigger.get("metric"), "distance_m": trigger.get("distance_m"),
                    "horizontal_distance_m": trigger.get("horizontal_distance_m"),
                    "vertical_distance_m": trigger.get("vertical_distance_m"),
                    "min_true_ticks": trigger.get("min_true_ticks"),
                    "entity_a_category": first.get("category"), "entity_b_category": second.get("category"),
                    "entity_a_ontology": first.get("semantic_scope", {}).get("ontology_class_id"),
                    "entity_b_ontology": second.get("semantic_scope", {}).get("ontology_class_id")}
    rows.extend({"kind": "raw_predicate_mapping", "id": raw_id, "canonical_id": "x.geometry.entity_proximity",
                 "rule": rule, "source": rule["source"]} for raw_id, rule in sorted(x_raw.items()))
    # The X business vocabulary is fixed in the current executable rule family.
    x_names = (
        "x.geometry.entity_proximity", "x.motion.moving", "x.motion.descending", "x.pedestrian.fallen",
        "x.facility.multiple_requests",
        "x.runtime.communication_state.communication_unavailable", "x.runtime.communication_state.station_unavailable",
        "x.runtime.control_state.deconfliction_active", "x.runtime.control_state.diversion_active",
        "x.runtime.control_state.evasion_active", "x.runtime.control_state.reroute_active",
        "x.runtime.control_state.safe_hold_active", "x.runtime.facility_state.allocation_failed",
        "x.runtime.facility_state.allocation_stale", "x.runtime.facility_state.arbitration_active",
        "x.runtime.facility_state.priority_granted", "x.runtime.incident_state.dispatch_active",
        "x.runtime.incident_state.requires_reroute", "x.runtime.mission_state.hazard_managed",
        "x.runtime.mission_state.unsafe", "x.runtime.navigation_state.geofence_violation",
        "x.runtime.navigation_state.gnss_spoofed", "x.runtime.navigation_state.mission_recovered",
        "x.runtime.navigation_state.route_recovered", "x.runtime.navigation_state.route_uncertain",
        "x.runtime.navigation_state.visual_navigation_degraded",
        "x.runtime.navigation_state.visual_navigation_recovered",
        "x.runtime.navigation_state.visual_relocalization",
    )
    for name in x_names:
        rows.append({"kind": "predicate", "id": name, "family": name.split(".")[1], "source_system": "X",
            "raw_ids": sorted(x_raw) if name == "x.geometry.entity_proximity" else [name],
            "roles": ([{"key": "first", "class": "world:Entity"}, {"key": "second", "class": "world:Entity"}]
                      if name == "x.geometry.entity_proximity" else
                      [{"key": "entity", "class": "world:Entity", "allowed_classes": _x_subject_classes(name)}]),
            "arity": 2 if name == "x.geometry.entity_proximity" else 1,
            "distinct_role_sets": [["first", "second"]] if name == "x.geometry.entity_proximity" else [],
            "unordered_role_sets": [], "execution_status": "source_executable", "time_granularity": "ARM dense/sampled",
            "rule": "X source script and branch predicate record", "source": "Dataset/scenarios/X_cross_layer/",
            "instance_count": None})
    event_path = REPO / "Dataset/knowledge_graph/events/catalog.yaml"
    event_catalog = yaml.safe_load(event_path.read_text(encoding="utf-8"))
    for family, module in event_catalog["modules"].items():
        for item in module["events"]:
            rows.append({"kind": "event", "id": item["identifier"], "family": family, "roles": item["roles"],
                         "source_system": "ontology", "execution_status": "catalog_declared",
                         "source": source_path(event_path), "instance_count": None})
    for family in ("uav_vehicle_contact_observed", "uav_pedestrian_clearance_observed",
                   "intersection_braking_observed", "ambulance_priority_passage_observed", "av_safe_stop_observed"):
        profiles = read_json(l4_path)["profiles"]
        onset = {p["onset_predicate"] for p in profiles.values() if p["event_family"] == "l4_" + family}
        if len(onset) != 1:
            raise ValueError(f"L4 event family has no unique onset signature: {family}")
        roles = _l4_roles(next(iter(onset)).removeprefix("l4_business."))
        rows.append({"kind": "event", "id": "l4." + family, "family": "l4_business", "roles": roles,
                     "role_policy": {role["key"]: "actor" for role in roles},
                     "source_system": "L4", "execution_status": "source_executable",
                     "source": source_path(l4_path), "instance_count": None})
    script_intents, action_types = _script_categories()
    realized_intents: dict[str, set[str]] = {}
    for path in sorted((REPO / "aw_data/render_ready_episodes_capture_filtered").glob("*/event_realization.jsonl")):
        for line, realization in iter_jsonl(path):
            intent = realization.get("intent")
            if not isinstance(intent, str) or not intent:
                raise ValueError(f"realized script event has no intent: {source_path(path)}:{line}")
            realized_intents.setdefault(intent, set()).add(f"{source_path(path)}:{line}")
    for intent in sorted(set(script_intents) | set(realized_intents)):
        sources = script_intents.get(intent)
        rows.append({"kind": "script_event_category", "id": intent,
                     "execution_status": "source_executable" if sources else "observed_extension",
                     "source": sorted(sources)[0] if sources else sorted(realized_intents[intent])[0],
                     "source_scripts": sorted(sources) if sources else [],
                     "realization_sources": sorted(realized_intents.get(intent, set())),
                     "source_count": len(sources or realized_intents[intent]), "instance_count": None})
    for category, sources in sorted(action_types.items()):
        rows.append({"kind": "command_category", "id": category, "execution_status": "source_executable",
                     "semantic_use": "capture_request" if category == "capture_screenshot" else "simulation_control",
                     "source": sorted(sources)[0], "source_count": len(sources), "instance_count": None})
    domain_profile_path = REPO / "Dataset/semantic_rules/profiles/domain_state_supplement_profile.json"
    domain_families = read_json(domain_profile_path)["observation_families"]
    source_families = sorted(set(CONTEXT_STRUCTURAL_CATEGORIES) |
                             set(CONTEXT_PREDICATE_STATE_FAMILIES) |
                             set(domain_families) | set(CONTEXT_OBSERVED_DOMAIN_EXTENSIONS) |
                             set(CONTEXT_OBSERVABLE_COMPLETION_FAMILIES))
    for index, family in enumerate(source_families):
        if family in CONTEXT_OBSERVABLE_COMPLETION_FAMILIES:
            source = "Dataset/semantic_simulation/observable_state_completion.py"
            status = "source_executable"
        elif family in CONTEXT_OBSERVED_DOMAIN_EXTENSIONS:
            source = ("Dataset/semantic_simulation/control_response_state.py"
                      if family == "control_response_state" else "Dataset/semantic_simulation/domain_state.py")
            status = "source_executable"
        elif family in domain_families:
            source = source_path(domain_profile_path)
            status = "source_executable"
        elif family in CONTEXT_PREDICATE_STATE_FAMILIES:
            source = "Dataset/semantic_simulation/predicate_state_computers.py"
            status = "source_executable"
        else:
            source = "Dataset/world_model/graph/formal.py;Dataset/world_model/graph/arm.py"
            status = "source_executable"
        if family in {"candidate_added", "candidate_removed"}:
            allowed_anchor_kinds = ["candidate_scope_change"]
        elif family == "predicate_continuity_break":
            allowed_anchor_kinds = ["predicate_continuity_break"]
        elif family == "compute_comm_predicate_matrix":
            allowed_anchor_kinds = ["compute_comm_matrix"]
        elif family in {"formal_weather_meta", "sumo_controller_state", "domain_state_supplement", "compute_comm_supplement", *domain_families,
                        *CONTEXT_OBSERVED_DOMAIN_EXTENSIONS, *CONTEXT_OBSERVABLE_COMPLETION_FAMILIES}:
            allowed_anchor_kinds = ["supplement_state"]
        elif family == "arm_action_audit":
            allowed_anchor_kinds = ["command_audit"]
        elif family == "local_business_measurement":
            allowed_anchor_kinds = ["local_business_measurement"]
        elif family == "measured_business_predicate_onset":
            allowed_anchor_kinds = ["predicate_onset"]
        elif family == "measured_geometry_support":
            allowed_anchor_kinds = ["geometry_support"]
        elif family == "script_chain_context":
            allowed_anchor_kinds = ["script_chain_context"]
        elif family == "arm_script_plan":
            allowed_anchor_kinds = ["script_parameters"]
        else:
            allowed_anchor_kinds = ["domain_state_observation"]
        rows.append({"kind": "context_category", "id": family, "index": index,
                     "allowed_anchor_kinds": allowed_anchor_kinds,
                     "source": source, "execution_status": status, "instance_count": None})
    for index, anchor_kind in enumerate(CONTEXT_ANCHOR_KINDS):
        rows.append({"kind": "context_anchor_kind", "id": anchor_kind, "index": index,
                     "source": "Dataset/world_model/graph/formal.py;Dataset/world_model/graph/arm.py"})
    for category in ("onset", "escalation_support", "terminal"):
        rows.append({"kind": "event_phase_category", "id": category,
                     "source": "Dataset/semantic_rules/schema/event_occurrence.schema.json",
                     "execution_status": "source_executable", "instance_count": None})
    rows.append({"kind": "predicate_episode_outcome_category", "id": "x.predicate_episode_outcome",
                 "source": "Dataset/world_model/graph/arm.py", "execution_status": "source_executable",
                 "instance_count": None})
    rows.append({"kind": "time_category", "id": "simulation_tick",
                 "source": "Dataset/world_model/graph/contract.py", "execution_status": "source_executable",
                 "instance_count": None})
    ontology = sorted({role["class"] for row in rows if row["kind"] in {"predicate", "event"}
                       for role in row.get("roles", [])})
    rows.extend({"kind": "ontology_class", "id": item, "source": "predicate/event role contracts"} for item in ontology)
    return rows


def write_catalog(path: Path, rows: list[dict[str, Any]] | None = None) -> None:
    rows = compile_catalog() if rows is None else rows
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    write_catalog(REPO / "design/belief/contract.jsonl")
