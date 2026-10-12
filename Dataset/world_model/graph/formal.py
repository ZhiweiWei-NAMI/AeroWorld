"""Stream one formal episode into the canonical label graph."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from Dataset.world_model.graph.contract import CONTEXT_PREDICATE_STATE_FAMILIES, ENTITY_CATEGORIES, REPO, iter_jsonl, read_json, source_path, strip_source_digests
from Dataset.world_model.graph.entity_identity import (
    BUSINESS_SOURCE_OBJECTS, ENTITY_TARGET_ACTIONS, IdentityResolver,
    action_target_scope, identity_node_id, read_roster, binding_identity_namespace,
    declared_world_bindings,
)
from Dataset.world_model.graph.controller_scope import controller_evidence
from Dataset.world_model.graph.field_paths import iter_field_values
from Dataset.semantic_truth.formal_clock import validate_formal_frame_clock
from Dataset.world_model.graph.records import DEFAULT_ROOT, GraphWriter
from Dataset.world_model.graph.sample_identity import (
    formal_scope, formal_source_family, validate_record_identity,
)
from Dataset.semantic_truth.entity_scope import participant_binding_allowed
from Dataset.semantic_simulation.compute_comm import SCHEMA_VERSION as COMPUTE_RAW_SCHEMA_VERSION


def _capture_request_sources(render_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path = render_root / "episode_manifest.json"
    manifest = read_json(manifest_path)
    script_ref = manifest["source_event_script_path"]
    scene_ref = manifest["source_scene_setup_path"]
    script_path, scene_path = REPO / script_ref, REPO / scene_ref
    script, scene = read_json(script_path), read_json(scene_path)
    cameras: dict[str, int] = {}
    for index, camera in enumerate(scene["cameras"]):
        camera_id = camera["camera_id"]
        if camera_id in cameras:
            raise ValueError(f"source scene repeats camera ID: {scene_path}:{camera_id}")
        cameras[camera_id] = index
    requests: dict[tuple[str, str], dict[str, Any]] = {}
    for event_index, event in enumerate(script["events"]):
        for action_index, action in enumerate(event.get("actions", [])):
            if action["type"] != "capture_screenshot":
                continue
            key = event["event_id"], action["action_id"]
            camera_id = action["camera_id"]
            if key in requests or camera_id not in cameras:
                raise ValueError(f"capture request has repeated action or unknown camera: {script_path}:{key}")
            requests[key] = {"scenario_id": script["scenario_id"],
                "source_script_file": source_path(script_path),
                "source_scene_file": source_path(scene_path),
                "script_event_index": event_index, "script_action_index": action_index,
                "camera_id": camera_id, "camera_index": cameras[camera_id],
                "source_script_request": action}
    return requests


def materialize_formal(episode_id: str, output_root: Path = DEFAULT_ROOT, *,
                       replace_generated: bool = False) -> dict[str, int]:
    truth_root = REPO / "aw_data/objective_semantic_truth" / episode_id
    render_root = REPO / "aw_data/render_ready_episodes_capture_filtered" / episode_id
    if not render_root.is_dir():
        raise FileNotFoundError(f"formal source missing: {episode_id}")
    base_file = truth_root / "world_truth_graph_base.json"
    delta_file = truth_root / "world_truth_graph_deltas.jsonl"
    frame_file = render_root / "truth_frames.jsonl"
    roster_file = render_root / "global_entity_roster.json"
    scope = formal_scope(episode_id, source_root=truth_root,
                         start_tick=0, end_tick=900)
    from Dataset.world_model.graph.source_inventory import CURRENT_INVENTORY
    inventory = CURRENT_INVENTORY
    missing_sources = {REPO / row["source_file"]: row["source_family"]
                       for _, row in iter_jsonl(inventory)
                       if row["kind"] == "source_declaration"
                       and row["identity_status"] == "sample"
                       and row["declared_sample"] == episode_id
                       and row["delivery_status"] == "source_missing"}
    base = read_json(base_file) if base_file.exists() else None
    if base is None:
        if base_file not in missing_sources:
            raise FileNotFoundError(base_file)
    else:
        if base.get("schema_version") != "3.1.0":
            raise ValueError(f"formal world truth version mismatch: {base_file}")
        if base["episode_id"] != episode_id or base["candidate_set_complete"] is not True:
            raise ValueError(f"candidate source mismatch: {episode_id}")
        validate_record_identity(base, scope, source_family="formal_world_truth_graph_base",
                                 source_file=source_path(base_file))

    def source_rows(path: Path):
        if not path.exists() and path in missing_sources:
            return
        family = formal_source_family(path, episode_id)
        for line, row in iter_jsonl(path):
            validate_record_identity(row, scope, source_family=family,
                                     source_file=source_path(path), source_line=line)
            yield line, row

    roster = read_roster(roster_file)
    roster_entities = {item["entity_id"]: item for item in roster["entities"]}
    controllers = controller_evidence(episode_id)
    resolver = IdentityResolver(scope["sample_key"])
    for index, item in enumerate(roster["entities"]):
        resolver.declare(item, source_file=source_path(roster_file),
                         source_path=f"entities[{index}]", authority="formal_roster")
    for controller_id, line in controllers.sumo_first_line.items():
        resolver.declare({"entity_id": controller_id, "category": "sumo_tls_controller",
                          "ontology_class_id": "world:TrafficSignal"},
            source_file=controllers.source_file,
            source_path=f"truth_frames.jsonl:{line}:sumo_traffic_light_states.{controller_id}",
            authority="truth_frame_sumo_tls", namespace="sumo_tls_controller")
    for index, item in enumerate(base["scope_entities"] if base is not None else ()):
        namespace = "compute_node" if item.get("scope_type") == "compute_node" else "sample"
        resolver.declare(item, source_file=source_path(base_file),
                         source_path=f"scope_entities[{index}]", authority="world_scope",
                         namespace=namespace, inherited=True)
    world_binding_claims = declared_world_bindings(resolver, base, source_path(base_file)) if base is not None else {}
    world_binding_class_origins = {entity: (ontology, source_path(base_file), None)
                                   for entity, (ontology, _) in world_binding_claims.items()}
    crowd_ids = {entity for item in base["initial_assertions"] for role, entity in item["bindings"].items()
                 if item["binding_ontology_classes"][role] == "world:Crowd"} if base is not None else set()
    deltas_by_tick: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    previous_delta: str | None = None
    for line, delta in source_rows(delta_file):
        if (delta.get("schema_version") != "3.1.0" or delta["episode_id"] != episode_id
                or delta.get("previous_delta_id") != previous_delta):
            raise ValueError(f"world delta chain diverges at {delta_file}:{line}")
        previous_delta = delta["delta_id"]
        deltas_by_tick[int(delta["tick"])].append((line, delta))
        added = [(i, op["assertion"]) for i, op in enumerate(delta["operations"])
                 if op["operation"] == "add_predicate_assertion"]
        crowd_ids.update(entity for _, item in added for role, entity in item["bindings"].items()
                         if item["binding_ontology_classes"][role] == "world:Crowd")
        claims = declared_world_bindings(resolver, {"initial_assertions": [r for _, r in added]},
            source_path(delta_file), declaration_paths=[f"line[{line}].operations[{i}].assertion" for i, _ in added])
        world_binding_claims.update(claims)
        world_binding_class_origins.update((entity, (ontology, source_path(delta_file), line))
                                          for entity, (ontology, _) in claims.items())
    from Dataset.world_model.graph.entity_identity import crowd_cohort_authorities
    crowd_authorities = crowd_cohort_authorities(episode_id, crowd_ids)
    for entity, declaration in crowd_authorities.items():
        resolver.declare({"entity_id": entity, "ontology_class_id": "world:Crowd"},
            source_file=source_path(declaration["path"]), source_path=f"line[{declaration['line']}].values.crowd_id",
            authority="registered_crowd_cohort")
    compute_file = truth_root / "compute_state.jsonl"
    compute_nodes: dict[str, dict[str, Any]] = {}
    compute_ticks: set[tuple[str, int]] = set()
    compute_rows = list(source_rows(compute_file))
    for line, row in compute_rows:
        host, node_id, kind, tick = (row.get(key) for key in
                                     ("entity_id", "node_id", "node_kind", "tick"))
        host_row = roster_entities.get(host)
        if (row.get("schema_name") != "compute_state" or row.get("schema_version") != COMPUTE_RAW_SCHEMA_VERSION
                or host_row is None or host_row["entity_category"] not in {"uav", "vehicle"}
                or node_id != "compute_node:" + host
                or kind != host_row["entity_category"] + "_edge_node"
                or type(tick) is not int or tick not in range(0, 901, 5)):
            raise ValueError(f"compute identity source differs: {compute_file}:{line}")
        if (node_id, tick) in compute_ticks:
            raise ValueError(f"duplicate compute node/tick: {compute_file}:{line}")
        compute_ticks.add((node_id, tick))
        prior = compute_nodes.get(node_id)
        if prior is not None and (prior["host"], prior["node_kind"]) != (host, kind):
            raise ValueError(f"compute node host or kind changed: {compute_file}:{line}")
        if prior is None:
            compute_nodes[node_id] = {"host": host, "node_kind": kind, "first_line": line}
    compute_class_file = REPO / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"
    compute_class_paths = [f"templates[{index}].argument_roles[{role_index}].class"
                           for index, template in enumerate(read_json(compute_class_file)["templates"])
                           if template["id"].startswith("compute.")
                           for role_index, role in enumerate(template["argument_roles"])
                           if role["class"].removeprefix("world:") == "ComputeNode"]
    if compute_nodes and not compute_class_paths:
        raise ValueError("compute-state producer has no current ComputeNode role declaration")
    for node_id, evidence in compute_nodes.items():
        resolver.declare({"entity_id": node_id, "scope_type": "compute_node",
                          "ontology_class_id": "world:ComputeNode"},
            source_file=source_path(compute_file),
            source_path=f"line[{evidence['first_line']}].node_id",
            authority="compute_state", namespace="compute_node", inherited=base is not None)
    communication_file = compute_file.with_name("communication_state.jsonl")
    communication_rows = list(source_rows(communication_file))
    state_records = {compute_file: compute_rows, communication_file: communication_rows}
    handover_sources = {}
    for state_line, state in communication_rows:
        key = state["tick"], state["handover_id"], state["session_id"]
        if key in handover_sources:
            raise ValueError(f"duplicate same-tick handover/session state: {communication_file}:{state_line}")
        handover_sources[key] = communication_file, state_line, state
    business_sources: dict[tuple[int, str, str], tuple[Path, int, dict[str, Any]]] = {}
    for state_file, rows in state_records.items():
        family = state_file.stem
        for line, row in rows:
            for ontology, (record_family, field, _) in BUSINESS_SOURCE_OBJECTS.items():
                if record_family != family:
                    continue
                value = row.get(field)
                if value is None or type(value) is str and value in {"none", "unknown", "UNKNOWN"}:
                    continue
                if type(value) is not str or not value:
                    raise ValueError(f"business object ID differs: {state_file}:{line}:{field}")
                key = row["tick"], ontology, value
                if key not in business_sources:
                    business_sources[key] = state_file, line, row
    graph = GraphWriter(scope, output_root, replace_generated=replace_generated)
    for path, family in sorted(missing_sources.items()):
        if path.exists():
            raise ValueError(f"source appeared after structural delivery: {path}")
        graph.unresolved_relation(source_file=source_path(path), source_line=None,
            relation={"source_family": family, "source_presence": "source_missing"},
            reason="Declared source file is absent; its records and supervision are not fabricated.")
    for conflict in resolver.conflicts:
        graph.unresolved_relation(source_file=conflict["sources"][-1], source_line=None,
                                  relation=conflict, reason="entity identity declaration conflict")
    manifest_file = truth_root / "manifest.json"
    manifest = read_json(manifest_file) if manifest_file.exists() else None
    if manifest is not None:
        validate_record_identity(manifest, scope, source_family="formal_objective_manifest",
                                 source_file=source_path(manifest_file))
    for key in ("episode_root", "output_dir"):
        declared = manifest.get(key) if manifest is not None else None
        if isinstance(declared, str) and not Path(declared).exists():
            graph.unresolved_relation(source_file=source_path(manifest_file), source_line=None,
                relation={"field": key, "declared_path": declared},
                reason="saved source path no longer exists; current episode is not proven equivalent")
    supplement_root = truth_root
    supplement_predicate_file = supplement_root / "compute_predicate_truth.jsonl"
    supplement_rows = [(line, row) for line, row in source_rows(supplement_predicate_file)] if supplement_predicate_file.exists() else []
    supplement_by_key: dict[tuple[str, str, int], list[tuple[int, dict]]] = defaultdict(list)
    for supplement_line, supplement_row in supplement_rows:
        supplement_by_key[(supplement_row["predicate_id"],
            json.dumps(supplement_row["bindings"], sort_keys=True, separators=(",", ":")),
            supplement_row["tick"])].append((supplement_line, supplement_row))
    truth_file = truth_root / "predicate_truth.jsonl"
    detection_by_tuple_tick: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    evidence_ontology: dict[str, tuple[str, str, int | None]] = {}
    evidence_ontology.update(world_binding_class_origins)
    for item in base["scope_entities"] if base is not None else ():
        if item.get("scope_type") != "compute_node" and item.get("ontology_class_id"):
            evidence_ontology[item["entity_id"]] = (item["ontology_class_id"], source_path(base_file), None)
    for truth_line, truth_row in source_rows(truth_file):
        if truth_row.get("schema_version") != "3.1.0":
            raise ValueError(f"formal predicate truth version mismatch: {truth_file}:{truth_line}")
        detection_by_tuple_tick[(truth_row["tuple_id"], truth_row["tick"])].append((truth_line, truth_row))
        for role, entity in truth_row.get("bindings", {}).items():
            ontology = (truth_row.get("binding_ontology_classes") or {}).get(role)
            if ontology and ontology != "world:ComputeNode":
                previous = evidence_ontology.get(entity)
                if previous and previous[0] != ontology:
                    from Dataset.semantic_truth.facility_scope import ontology_class_is_a
                    if ontology_class_is_a(previous[0], ontology):
                        continue
                    if ontology_class_is_a(ontology, previous[0]):
                        evidence_ontology[entity] = (ontology, source_path(truth_file), truth_line)
                        continue
                    graph.unresolved_relation(source_file=source_path(truth_file), source_line=truth_line,
                        relation={"entity_id": entity, "class_a": previous[0], "class_b": ontology},
                        reason="source ontology class claims differ")
                else:
                    evidence_ontology[entity] = (ontology, source_path(truth_file), truth_line)
    supplement_ref: dict[str, str] = {}
    supplement_matched: set[str] = set()
    prefix = scope["sample_key"] + "|"
    ident = lambda entity, namespace="sample": identity_node_id(scope["sample_key"], namespace, entity)
    state_id = lambda entity, tick: prefix + f"s:{entity}@{tick}"
    pred_id = lambda tuple_id, tick: prefix + f"p:{tuple_id}@{tick}"
    time_id = lambda tick: prefix + f"t:{tick}"
    transition_id = lambda native: prefix + "tr:" + native
    event_id = lambda native: prefix + "ev:" + native
    seen_identity: set[tuple[str, str]] = set()
    state_at_tick: dict[str, str] = {}
    state_lookup: dict[tuple[int, str], str] = {}
    previous_state: dict[str, tuple[int, str]] = {}
    all_assertions: dict[tuple[str, int], str] = {}
    world_by_binding: dict[tuple[str, str, int], str] = {}
    tuple_at_tick: dict[tuple[str, int], str] = {}
    prev_candidate: dict[str, tuple[int, str]] = {}
    evidence_origin: dict[str, tuple[str, int | None]] = {}
    known_ontology: dict[str, str | None] = {}
    identity_event_scope: dict[str, tuple[bool, bool]] = {}
    identity_categories: dict[str, str | None] = {}
    compute_host_by_node: dict[str, str] = {}

    def add_identity(entity: str, category: str | None, ontology: str | None, source_file: Path,
                     source_line: int | None = None, raw: dict | None = None,
                     completeness: str = "declared",
                     source_entity_index: int | None = None,
                     source_scope_entity_index: int | None = None,
                     explicit_namespace: str | None = None,
                     source_scope_type: str | None = None,
                     source_host_entity_id: str | None = None,
                     source_node_kind: str | None = None,
                     ontology_basis_path: str | None = None) -> str:
        namespace = (explicit_namespace or (raw or {}).get("identity_namespace") or
                     ("compute_node" if (raw or {}).get("scope_type") == "compute_node" else "sample"))
        if (namespace, entity) in seen_identity:
            return ident(entity, namespace)
        resolved = resolver.resolve(entity, source_file=source_path(source_file),
                                    source_line=source_line, source_path="entity_id",
                                    namespace=namespace)
        roster_authority = completeness == "declared"
        roster_index = {entity: raw} if roster_authority and raw is not None else {}
        event_actor_allowed = participant_binding_allowed(entity, "actor", roster_index)
        event_non_actor_allowed = participant_binding_allowed(entity, "region", roster_index)
        cat = category if category in ENTITY_CATEGORIES else None
        raw_category = category if category in ENTITY_CATEGORIES else ((raw or {}).get("entity_category") or (raw or {}).get("category"))
        source_declares_ontology = ((raw or {}).get("semantic_scope") or {}).get("ontology_class_id") == ontology or (raw or {}).get("ontology_class_id") == ontology
        ontology_source = ([source_path(source_file), source_line] if source_declares_ontology
                           else list(evidence_ontology[entity][1:]) if namespace == "sample" and entity in evidence_ontology else None)
        if ontology_basis_path is not None:
            ontology_source = [source_path(compute_class_file), None]
        graph.add_node(ident(entity, namespace), "EntityIdentity", category if category in ENTITY_CATEGORIES else None, None,
            {"entity_id": entity, "entity_category": cat, "source_entity_category": raw_category,
             "identity_namespace": namespace,
             "source_host_entity_id": (source_host_entity_id or (raw or {}).get("source_entity_id")) if namespace == "compute_node" else None,
             "source_scope_type": source_scope_type or (raw or {}).get("scope_type"),
             "source_node_kind": source_node_kind,
             "ontology_class": ontology,
             "ontology_source": ontology_source if ontology else None,
             "ontology_basis_path": ontology_basis_path,
             "raw_category_status": "resolved" if cat else "unregistered" if raw_category else "missing",
             "ontology_status": "resolved" if ontology else "missing",
             "classification_status": "resolved" if cat and ontology else "unresolved",
             "identity_status": resolved.status, "identity_basis": resolved.authority,
             "identity_basis_path": resolved.authority_path,
             "identity_inherited_from": list(resolved.inherited_from),
             "identity_reason": resolved.reason,
             "source_completeness": completeness,
             "source_entity_index": source_entity_index,
             "source_scope_entity_index": source_scope_entity_index,
             "event_actor_allowed": event_actor_allowed,
             "event_non_actor_allowed": event_non_actor_allowed, "raw": raw},
            source_path(source_file), source_line)
        seen_identity.add((namespace, entity))
        if namespace == "sample":
            known_ontology[entity] = ontology
            identity_event_scope[entity] = event_actor_allowed, event_non_actor_allowed
            identity_categories[entity] = cat
        return ident(entity, namespace)

    def binding_namespace(kind: str, category: str, role: str) -> str:
        registry = graph.validator.predicates if kind == "predicate" else graph.validator.events
        declaration = registry.get(category)
        roles = [] if declaration is None else [item for item in declaration["roles"] if item["key"] == role]
        if len(roles) != 1:
            raise ValueError(f"binding role absent or duplicated in contract: {kind}:{category}:{role}")
        ontology = roles[0]["class"]
        ontology = ontology if ":" in ontology else "world:" + ontology
        return "compute_node" if ontology == "world:ComputeNode" else binding_identity_namespace(ontology)

    def add_time(tick: int, source_file: Path, source_line: int | None = None,
                 raw: dict | None = None) -> str:
        key = time_id(tick)
        if key not in graph.node_index:
            graph.add_node(key, "TimePoint", "simulation_tick", tick,
                           {"tick": tick, **({"raw": raw} if raw is not None else {})},
                           source_path(source_file), source_line)
        return key

    def at_time(node: str, tick: int, source_file: Path, line: int | None = None,
                role: str = "target_tick") -> None:
        add_time(tick, source_file, line)
        graph.add_edge("at_time", node, time_id(tick), role=role,
                       source_file=source_path(source_file), source_line=line)

    def argument_target(kind: str, category: str, role: str, entity: str, tick: int,
                        source_file: Path, line: int | None, *,
                        claimed_class: str | None = None, prefer_state: bool = False,
                        record_bindings: dict | None = None
                        ) -> tuple[str | None, dict[str, Any]]:
        registry = graph.validator.predicates if kind == "predicate" else graph.validator.events
        declaration = registry[category]
        roles = [item for item in declaration["roles"] if item["key"] == role]
        if len(roles) != 1:
            raise ValueError(f"binding role absent or duplicated: {kind}:{category}:{role}")
        role_class = roles[0]["class"]
        ontology = role_class if ":" in role_class else "world:" + role_class
        object_rule = BUSINESS_SOURCE_OBJECTS.get(ontology)
        if object_rule is not None:
            _, field, per_tick = object_rule
            source_evidence = {}
            if (kind == "event" and category == "communication.communication_handover_event"
                    and role in {"source_station", "target_station"}):
                if record_bindings is None:
                    raise ValueError("handover station binding lacks its event handover/session")
                field = ("handover.previous_station_id" if role == "source_station"
                         else "handover.current_station_id")
                evidence = handover_sources.get((tick, record_bindings["handover"], record_bindings["session"]))
                if evidence is not None and evidence[2]["handover"][field.split(".")[1]] != entity:
                    raise ValueError(f"event station differs from its handover/session state: {source_file}:{line}:{role}")
                source_evidence = {"event_category": category, "event_role": role,
                                   "handover_id": record_bindings["handover"],
                                   "session_id": record_bindings["session"]}
            else:
                evidence = business_sources.get((tick, ontology, entity))
            if evidence is None:
                graph.unresolved_relation(source_file=source_path(source_file), source_line=line,
                    relation={"role": role, "object_id": entity, "ontology_class": ontology, "tick": tick},
                    reason="business binding lacks its exact same-tick state ID")
                return None, {}
            state_file, state_line, state = evidence
            key = prefix + "object:" + ontology + ":" + entity
            if key not in graph.node_index:
                object_tick = tick if per_tick else None
                graph.add_node(key, "ContextAnchor", "compute_comm_supplement", object_tick,
                    {"anchor_kind": "supplement_state", "source_family": "compute_comm_supplement",
                     "entity_id": entity, "ontology_class": ontology,
                     "source_object_field": field, "source_object_record_family": state_file.stem,
                     "source_object_temporality": "one_tick" if per_tick else "stable_id",
                     "source_parent_fields": {name: state[name] for name in
                         ("entity_id", "node_id", "station_id", "task_id", "session_id", "flow_id")
                         if name in state},
                     "ontology_basis": "current predicate/event role contract and exact state producer ID"},
                    source_path(state_file), state_line)
                if object_tick is not None:
                    at_time(key, object_tick, state_file, state_line)
            return key, {"source_object_evidence": {"source_file": source_path(state_file),
                            "source_line": state_line, "source_field": field, **source_evidence}}
        namespace = binding_namespace(kind, category, role)
        if namespace == "sumo_tls_controller" and entity in roster_entities:
            namespace = "sample"
        if (namespace, entity) not in seen_identity:
            add_identity(entity, None, claimed_class, source_file, line,
                raw={"identity_namespace": namespace} if namespace != "sample" else None,
                completeness="binding_only")
        target = (state_lookup.get((tick, entity), ident(entity))
                  if prefer_state and namespace == "sample" else ident(entity, namespace))
        return target, {}

    for roster_entity_index, item in enumerate(roster["entities"]):
        entity = item["entity_id"]
        category = item["entity_category"]
        ontology = (item.get("semantic_scope") or {}).get("ontology_class_id") or evidence_ontology.get(entity, (None,))[0]
        add_identity(entity, category, ontology, roster_file,
                     raw=strip_source_digests(item), source_entity_index=roster_entity_index)
    for scope_entity_index, item in enumerate(base["scope_entities"] if base is not None else ()):
        entity = item["entity_id"]
        namespace = "compute_node" if item.get("scope_type") == "compute_node" else "sample"
        if (namespace, entity) not in seen_identity:
            add_identity(entity, item["scope_type"], item.get("ontology_class_id"), base_file,
                         raw=item, completeness="world_scope",
                         source_scope_entity_index=scope_entity_index)
        elif item.get("ontology_class_id") and resolver.resolve(entity,
                source_file=source_path(base_file), source_line=None,
                source_path="scope_entities",
                namespace="compute_node" if item.get("scope_type") == "compute_node" else "sample").status == "resolved":
            if namespace == "sample":
                known_ontology[entity] = item["ontology_class_id"]
        if item["scope_type"] == "compute_node":
            host = item.get("source_entity_id")
            if (not isinstance(host, str) or not host or entity != "compute_node:" + host
                    or host not in {entry["entity_id"] for entry in roster["entities"]}):
                raise ValueError(f"compute-node host declaration differs: {base_file}:scope_entities[{scope_entity_index}]")
            if entity in compute_host_by_node:
                raise ValueError(f"duplicate compute-node identity: {base_file}:scope_entities[{scope_entity_index}]")
            compute_host_by_node[entity] = host
            graph.add_edge("compute_node_host", ident(entity, "compute_node"), ident(host),
                source_file=source_path(base_file), source_line=None,
                attrs={"source_scope_entity_index": scope_entity_index,
                       "source_path": f"scope_entities[{scope_entity_index}].source_entity_id"})
    for node_id, evidence in compute_nodes.items():
        host, line = evidence["host"], evidence["first_line"]
        if node_id in compute_host_by_node:
            if compute_host_by_node[node_id] != host:
                raise ValueError(f"compute-state host differs from world scope: {compute_file}:{line}")
            continue
        add_identity(node_id, "compute_node", "world:ComputeNode", compute_file, line,
            completeness="compute_state", explicit_namespace="compute_node",
            source_scope_type="compute_node", source_host_entity_id=host,
            source_node_kind=evidence["node_kind"], ontology_basis_path=compute_class_paths[0])
        compute_host_by_node[node_id] = host
        graph.add_edge("compute_node_host", ident(node_id, "compute_node"), ident(host),
            source_file=source_path(compute_file), source_line=line,
            attrs={"node_id_field": "node_id", "host_field": "entity_id",
                   "node_kind_field": "node_kind",
                   "producer_basis": "Dataset/semantic_simulation/compute_comm.py:_build_compute_rows"})
    for controller_id, line in sorted(controllers.sumo_first_line.items()):
        add_identity(controller_id, None, "world:TrafficSignal", frame_file, line,
            raw={"entity_id": controller_id, "category": "sumo_tls_controller",
                 "scope_type": "sumo_tls_controller", "identity_namespace": "sumo_tls_controller"},
            completeness="truth_frame_sumo_tls")

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for item in base["initial_assertions"] if base is not None else ():
        key = (item["predicate_id"], item["tuple_id"])
        if key in candidates:
            raise ValueError(f"duplicate base candidate: {key}")
        candidates[key] = item
        evidence_origin[item["tuple_id"]] = (source_path(base_file), None)

    frame_count = 0
    for line, frame in source_rows(frame_file):
        if frame.get("episode_id") != episode_id:
            raise ValueError(f"formal frame sequence diverges: {frame_file}:{line}")
        validate_formal_frame_clock(frame, expected_tick=frame_count,
                                    source=f"{frame_file}:{line}")
        tick = frame["tick"]
        frame_count += 1
        add_time(tick, frame_file, line,
                 {"tick": tick, **({"entity_motion_state": frame["entity_motion_state"]}
                                  if "entity_motion_state" in frame else {})})
        controller_states = frame.get("sumo_traffic_light_states")
        if controller_states:
            controller_node = prefix + f"sumo_controllers:{tick}"
            graph.add_node(controller_node, "ContextAnchor", "sumo_controller_state", tick,
                {"anchor_kind": "supplement_state", "source_family": "sumo_controller_state",
                 "source_layer": "formal_truth_frame", "controller_ids": sorted(controller_states),
                 "raw": {"tick": frame["tick"],
                         "sumo_traffic_light_states": controller_states}},
                source_path(frame_file), line)
            at_time(controller_node, tick, frame_file, line)
            for controller_id in sorted(controller_states):
                if ("sumo_tls_controller", controller_id) not in seen_identity:
                    raise ValueError(f"SUMO controller lacks declared identity: {frame_file}:{line}:{controller_id}")
                graph.add_edge("controller_state", controller_node, ident(controller_id, "sumo_tls_controller"),
                    role="sumo_tls_controller", source_file=source_path(frame_file), source_line=line,
                    attrs={"source_path": f"sumo_traffic_light_states.{controller_id}"})
        state_at_tick.clear()
        for entity_index, entity_row in enumerate(frame["entities"]):
            entity = entity_row["entity_id"]
            if ("sample", entity) not in seen_identity:
                category = entity_row.get("entity_category") or entity_row.get("label_class")
                ontology = (entity_row.get("semantic_scope") or {}).get("ontology_class_id") or evidence_ontology.get(entity, (None,))[0]
                add_identity(entity, category, ontology, frame_file, line,
                             completeness="frame_only")
            frame_category = entity_row.get("entity_category")
            known_category = identity_categories[entity]
            if frame_category in ENTITY_CATEGORIES and known_category and frame_category != known_category:
                graph.unresolved_relation(source_file=source_path(frame_file), source_line=line,
                    relation={"entity_id": entity, "identity_category": known_category,
                              "frame_category": frame_category},
                    reason="frame and stable roster categories differ")
            state_identity = resolver.resolve(entity, source_file=source_path(frame_file),
                source_line=line, source_path=f"entities[{entity_index}].entity_id")
            state_identity_status = ("unresolved" if frame_category in ENTITY_CATEGORIES
                and known_category and frame_category != known_category else state_identity.status)
            sid = state_id(entity, tick)
            graph.add_node(sid, "EntityState", known_category, tick,
                {"entity_id": entity, "ontology_class": known_ontology.get(entity),
                 "identity_status": state_identity_status,
                 "identity_ref": ident(entity) if state_identity_status == "resolved" else None,
                 "identity_basis": state_identity.authority,
                 "identity_basis_path": state_identity.authority_path,
                 "entity_category": known_category, "source_entity_category": frame_category,
                 "raw": entity_row, "source_entity_index": entity_index,
                 "event_actor_allowed": identity_event_scope[entity][0],
                 "event_non_actor_allowed": identity_event_scope[entity][1],
                 "field_context": {"source_file": source_path(frame_file), "source_line": line,
                                   "value_tick": tick}}, source_path(frame_file), line)
            state_at_tick[entity] = sid
            state_lookup[(tick, entity)] = sid
            if (state_identity_status == "resolved" and
                    graph.node_index[ident(entity)]["attrs"]["identity_status"] == "resolved"):
                graph.add_edge("identity_state", ident(entity), sid, source_file=source_path(frame_file), source_line=line)
            at_time(sid, tick, frame_file, line)
            previous = previous_state.get(entity)
            if (previous and previous[0] == tick - 1
                    and graph.node_index[sid]["attrs"]["identity_status"] == "resolved"
                    and graph.node_index[previous[1]]["attrs"]["identity_status"] == "resolved"):
                graph.add_edge("same_entity_next", previous[1], sid,
                               source_file=source_path(frame_file), source_line=line)
            previous_state[entity] = (tick, sid)
        for entity_row in frame["entities"]:
            vehicle = entity_row.get("sumo_vehicle") or {}
            peer = vehicle.get("leading_vehicle_id")
            value = vehicle.get("following_distance_m")
            subject = entity_row["entity_id"]
            if peer and peer != subject and isinstance(value, (int, float)):
                if peer in state_at_tick:
                    graph.add_edge("RelativeState", state_at_tick[subject], state_at_tick[peer],
                        role="follower_to_leader", source_file=source_path(frame_file), source_line=line,
                        attrs={"field": "sumo_vehicle.following_distance_m", "value": value,
                               "unit": "m", "unit_status": "declared",
                               "unit_basis": "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#regulation.vehicle_violates_following_distance",
                               "metric": "same_lane_longitudinal_center_gap",
                               "value_tick": tick, "peer_field": "sumo_vehicle.leading_vehicle_id"})
                else:
                    graph.unresolved_relation(source_file=source_path(frame_file), source_line=line,
                        relation={"entity_id": subject, "leading_vehicle_id": peer, "tick": tick},
                        reason="explicit following peer lacks same-tick state")
        if tick % 5:
            continue
        scope_changes: list[tuple[str, str, str | None, int]] = []
        for delta_line, delta in deltas_by_tick.get(tick, []):
            for operation in delta["operations"]:
                name = operation["operation"]
                item = operation.get("assertion", operation)
                key = (item["predicate_id"], item["tuple_id"])
                if name == "add_predicate_assertion":
                    if key in candidates:
                        raise ValueError(f"duplicate added candidate at {tick}: {key}")
                    candidates[key] = item
                    scope_changes.append(("candidate_added", item["tuple_id"], None, delta_line))
                elif name == "remove_predicate_assertion":
                    current = candidates.get(key)
                    if current is None or current["value"] != operation["from_value"]:
                        raise ValueError(f"invalid removed candidate at {tick}: {key}")
                    prior = prev_candidate.get(item["tuple_id"])
                    scope_changes.append(("candidate_removed", item["tuple_id"], prior[1] if prior else None, delta_line))
                    del candidates[key]
                elif name in {"set_predicate_value", "refresh_predicate_evidence"}:
                    current = candidates.get(key)
                    expected_value = (
                        operation["from_value"]
                        if name == "set_predicate_value"
                        else operation["value"]
                    )
                    if current is None or current["value"] != expected_value:
                        raise ValueError(f"invalid {name} operation at {tick}: {key}")
                    if name == "set_predicate_value":
                        current.update(value=operation["to_value"],
                                       truth_state_update_tick=tick)
                    current.update(missing_source_record=operation["missing_source_record"],
                                   observations=operation["observations"],
                                   source_refs=operation["source_refs"], evidence_update_tick=tick)
                else:
                    raise ValueError(f"unsupported world operation {name}")
                evidence_origin[item["tuple_id"]] = (source_path(delta_file), delta_line)
        for category, tuple_id, previous, delta_line in scope_changes:
            anchor = prefix + f"scope:{category}:{tuple_id}@{tick}"
            graph.add_node(anchor, "ContextAnchor", category, tick,
                {"anchor_kind": "candidate_scope_change", "source_family": category,
                 "tuple_id": tuple_id,
                 "scope_change": category}, source_path(delta_file), delta_line)
            at_time(anchor, tick, delta_file, delta_line)
            if category == "candidate_removed" and previous:
                graph.add_edge("candidate_scope_change", previous, anchor, role="removed",
                               source_file=source_path(delta_file), source_line=delta_line)
        for key in sorted(candidates):
            item = candidates[key]
            tuple_id = item["tuple_id"]
            node = pred_id(tuple_id, tick)
            origin_file, origin_line = evidence_origin[tuple_id]
            bindings = item["bindings"]
            aliases = []
            for truth_line, truth_row in detection_by_tuple_tick.get((tuple_id, tick), []):
                if truth_row["predicate_id"] == item["predicate_id"] and truth_row["value"] == item["value"]:
                    evidence = truth_row.get("evidence") or {}
                    observations = evidence.get("observations") or []
                    expected_values = {
                        "world_truth_value": item["value"],
                        "world_observations": item["observations"],
                        "truth_state_update_tick": item["truth_state_update_tick"],
                        "evidence_update_tick": item["evidence_update_tick"],
                        "missing_source_record": item["missing_source_record"],
                        "bindings": item["bindings"],
                        "binding_ontology_classes": item["binding_ontology_classes"],
                    }
                    expected_path = (
                        "normalized.grounded_roles__"
                        + "__".join(item["ontology_argument_roles"])
                    )
                    if (
                        truth_row.get("episode_id") != episode_id
                        or truth_row.get("tick") != tick
                        or truth_row.get("tuple_id") != tuple_id
                        or truth_row.get("bindings") != bindings
                        or truth_row.get("source_class") != "deterministic_derived"
                        or observations != [{"path": expected_path, "value": expected_values}]
                        or evidence.get("source_refs") != item["source_refs"]
                        or evidence.get("missing_requirements") != item["missing_source_record"]
                        or truth_row.get("truth_state_update_tick")
                        != item["truth_state_update_tick"]
                        or truth_row.get("evidence_update_tick")
                        != item["evidence_update_tick"]
                    ):
                        raise ValueError(
                            f"formal predicate evidence differs from world truth: "
                            f"{truth_file}:{truth_line}"
                        )
                    aliases.append({"source_file": source_path(truth_file), "source_line": truth_line,
                                    "native_id": truth_row["truth_id"], "source_view": truth_row})
            supplement_key = (item["predicate_id"], json.dumps(bindings, sort_keys=True, separators=(",", ":")), tick)
            for supplement_line, supplement_row in supplement_by_key.get(supplement_key, []):
                if supplement_row["value"] == item["value"]:
                    supplement_matched.add(supplement_row["predicate_truth_id"])
                    supplement_ref[supplement_row["predicate_truth_id"]] = node
                    aliases.append({"source_file": source_path(supplement_predicate_file),
                                    "source_line": supplement_line,
                                    "native_id": supplement_row["predicate_truth_id"],
                                    "source_view": supplement_row})
                else:
                    graph.unresolved_relation(source_file=source_path(supplement_predicate_file),
                        source_line=supplement_line,
                        relation={"predicate_truth_id": supplement_row["predicate_truth_id"],
                                  "saved_value": supplement_row["value"], "world_value": item["value"]},
                        reason="supplement predicate truth conflicts with full-world candidate")
            graph.add_node(node, "PredicateAssertion", item["predicate_id"], tick,
                {"predicate_id": item["predicate_id"], "tuple_id": tuple_id,
                 "native_assertion_id": item["assertion_id"], "bindings": bindings,
                 "binding_ontology_classes": item.get("binding_ontology_classes", {}),
                 "truth_status": item["value"], "candidate_present": True,
                 "truth_state_update_tick": item["truth_state_update_tick"],
                 "evidence_update_tick": item["evidence_update_tick"],
                 "evidence": item.get("observations", []), "source_refs": item.get("source_refs", []),
                 "source_aliases": aliases,
                 "missing_source_record": item.get("missing_source_record", []),
                 "candidate_authority": item.get("candidate_authority")},
                origin_file, origin_line, rule_ref="Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#" + item["predicate_id"])
            all_assertions[(tuple_id, tick)] = node
            world_by_binding[(item["predicate_id"], json.dumps(bindings, sort_keys=True, separators=(",", ":")), tick)] = node
            tuple_at_tick[(tuple_id, tick)] = node
            at_time(node, tick, frame_file, line)
            for role, entity in bindings.items():
                target, evidence = argument_target("predicate", item["predicate_id"], role,
                    entity, tick, REPO / origin_file, origin_line,
                    claimed_class=item.get("binding_ontology_classes", {}).get(role), prefer_state=True)
                if target is not None:
                    graph.add_edge("predicate_argument", node, target, role=role, attrs=evidence,
                                   source_file=origin_file, source_line=origin_line)
            previous = prev_candidate.get(tuple_id)
            if previous and previous[0] == tick - 5:
                graph.add_edge("same_predicate_next", previous[1], node,
                               source_file=origin_file, source_line=origin_line)
            prev_candidate[tuple_id] = (tick, node)
        for category, tuple_id, _, delta_line in scope_changes:
            if category == "candidate_added":
                anchor = prefix + f"scope:{category}:{tuple_id}@{tick}"
                graph.add_edge("candidate_scope_change", anchor, pred_id(tuple_id, tick), role="added",
                               source_file=source_path(delta_file), source_line=delta_line)
    if frame_count != 901:
        raise ValueError(f"formal frame count {frame_count}, expected 901: {frame_file}")
    if set(deltas_by_tick) - set(range(0, 901, 5)):
        raise ValueError("world delta outside formal capture ticks")

    weather_file = render_root / "weather_meta.jsonl"
    weather_count = 0
    for line, weather in source_rows(weather_file):
        tick = weather.get("tick")
        if type(tick) is not int or tick != weather_count:
            raise ValueError(f"weather tick sequence differs: {weather_file}:{line}")
        node = prefix + f"weather:{tick}"
        graph.add_node(node, "ContextAnchor", "formal_weather_meta", tick,
            {"anchor_kind": "supplement_state", "source_family": "formal_weather_meta",
             "source_layer": "formal_weather_meta", "raw": weather},
            source_path(weather_file), line)
        at_time(node, tick, weather_file, line)
        weather_count += 1
    if weather_count != frame_count:
        raise ValueError(f"weather and truth-frame coverage differ: {weather_file}")

    # Supplement labels are an independent saved authority. Equal rows become source
    # aliases; conflicting and extra rows remain visible as separate assertions.
    previous_supplement: dict[tuple[str, str], tuple[int, str]] = {}
    for supplement_line, row in supplement_rows:
        native = row["predicate_truth_id"]
        if native in supplement_matched:
            continue
        tick = int(row["tick"])
        bindings = row["bindings"]
        key = (row["predicate_id"], json.dumps(bindings, sort_keys=True, separators=(",", ":")), tick)
        world_node = world_by_binding.get(key)
        status = "conflict" if world_node else ("saved_unchecked" if base is None else "unresolved")
        node = prefix + "supp_pred:" + native
        graph.add_node(node, "PredicateAssertion", row["predicate_id"], tick,
            {"predicate_id": row["predicate_id"], "native_predicate_truth_id": native,
             "bindings": bindings, "binding_ontology_classes": row.get("binding_ontology_classes", {}),
             "truth_status": row["value"],
             "candidate_present": None if base is None else world_node is not None,
             "world_comparison_status": ("world_source_missing" if base is None else
                                         "conflict" if world_node else "outside_candidate_scope"),
             "assertion_layer": "compute_comm_supplement", "source_refs": row.get("source_refs", []),
             "raw": row, "comparison_world_node": world_node},
            source_path(supplement_predicate_file), supplement_line, rule_ref=row.get("rule_id"),
            source_replay_status=status)
        at_time(node, tick, supplement_predicate_file, supplement_line)
        supplement_ref[native] = node
        supplement_key = (row["predicate_id"], key[1])
        previous = previous_supplement.get(supplement_key)
        if previous is not None and previous[0] == tick - 5:
            graph.add_edge("same_predicate_next", previous[1], node,
                source_file=source_path(supplement_predicate_file), source_line=supplement_line,
                attrs={"assertion_layer": "compute_comm_supplement", "sample_step_ticks": 5},
                source_replay_status=status)
        previous_supplement[supplement_key] = (tick, node)
        for role, entity in bindings.items():
            target, evidence = argument_target("predicate", row["predicate_id"], role,
                entity, tick, supplement_predicate_file, supplement_line,
                claimed_class=row.get("binding_ontology_classes", {}).get(role))
            if target is not None:
                graph.add_edge("predicate_argument", node, target, role=role, attrs=evidence,
                    source_file=source_path(supplement_predicate_file), source_line=supplement_line,
                    source_replay_status=status)
        if world_node:
            graph.add_edge("source_conflict", world_node, node,
                           source_file=source_path(supplement_predicate_file), source_line=supplement_line,
                           attrs={"world_value": graph.node_index[world_node]["attrs"]["truth_status"],
                                  "supplement_value": row["value"]}, source_replay_status="conflict")
        elif base is not None:
            graph.unresolved_relation(source_file=source_path(supplement_predicate_file),
                source_line=supplement_line,
                relation={"predicate_truth_id": native, "predicate_id": row["predicate_id"],
                          "bindings": bindings, "tick": tick},
                reason="supplement label outside full-world candidate scope")

    truth_ref: dict[str, str] = {}
    for line, row in source_rows(truth_file):
        target = all_assertions.get((row["tuple_id"], row["tick"]))
        matching = (target is not None and graph.node_index[target]["category_id"] == row["predicate_id"] and
                    graph.node_index[target]["attrs"]["truth_status"] == row["value"])
        if matching:
            truth_ref[row["truth_id"]] = target
            continue
        node = prefix + "det_pred:" + row["truth_id"]
        graph.add_node(node, "PredicateAssertion", row["predicate_id"], row["tick"],
            {"predicate_id": row["predicate_id"], "tuple_id": row["tuple_id"],
             "native_truth_id": row["truth_id"], "bindings": row["bindings"],
             "binding_ontology_classes": row.get("binding_ontology_classes", {}),
             "truth_status": row["value"], "candidate_present": target is not None,
             "assertion_layer": "detection_projection", "raw": row,
             "comparison_world_node": target}, source_path(truth_file), line,
            source_replay_status="conflict" if target else "unresolved")
        at_time(node, row["tick"], truth_file, line)
        truth_ref[row["truth_id"]] = node
        for role, entity in row["bindings"].items():
            target, evidence = argument_target("predicate", row["predicate_id"], role,
                entity, row["tick"], truth_file, line,
                claimed_class=row.get("binding_ontology_classes", {}).get(role))
            if target is not None:
                graph.add_edge("predicate_argument", node, target, role=role, attrs=evidence,
                    source_file=source_path(truth_file), source_line=line)
        if target is not None:
            graph.add_edge("source_conflict", target, node, source_file=source_path(truth_file),
                           source_line=line, attrs={"world_value": graph.node_index[target]["attrs"]["truth_status"],
                                                     "detection_value": row["value"]},
                           source_replay_status="conflict")
        else:
            graph.unresolved_relation(source_file=source_path(truth_file), source_line=line,
                relation={"truth_id": row["truth_id"], "tuple_id": row["tuple_id"], "tick": row["tick"]},
                reason="detection truth has no full-world candidate")

    continuity_file = truth_root / "predicate_continuity_breaks.jsonl"
    if continuity_file.exists():
        for line, row in source_rows(continuity_file):
            if len(row["evidence_truth_ids"]) != 2:
                raise ValueError(f"continuity break needs two truth IDs: {continuity_file}:{line}")
            node = prefix + "continuity:" + row["break_id"]
            graph.add_node(node, "ContextAnchor", "predicate_continuity_break", row["to_tick"],
                {"anchor_kind": "predicate_continuity_break",
                 "source_family": "predicate_continuity_break",
                 "native_break_id": row["break_id"],
                 "predicate_id": row["predicate_id"],
                 "tuple_id": row["tuple_id"], "bindings": row["bindings"],
                 "from_tick": row["from_tick"], "to_tick": row["to_tick"],
                 "from_value": row["from_value"], "to_value": row["to_value"],
                 "reason": row["reason"], "evidence_truth_ids": row["evidence_truth_ids"],
                 "raw": row}, source_path(continuity_file), line)
            at_time(node, row["to_tick"], continuity_file, line)
            for native, kind, direction in zip(row["evidence_truth_ids"],
                ("continuity_break_from", "continuity_break_to"), ("before", "after")):
                endpoint = truth_ref.get(native)
                if endpoint is None:
                    graph.unresolved_relation(source_file=source_path(continuity_file), source_line=line,
                        relation={"break_id": row["break_id"], "truth_id": native},
                        reason="continuity break evidence truth ID absent")
                elif direction == "before":
                    graph.add_edge(kind, endpoint, node, source_file=source_path(continuity_file), source_line=line)
                else:
                    graph.add_edge(kind, node, endpoint, source_file=source_path(continuity_file), source_line=line)

    transition_ref: dict[str, str] = {}
    transition_file = truth_root / "predicate_transitions.jsonl"
    for line, row in source_rows(transition_file):
        target = transition_id(row["transition_id"])
        graph.add_node(target, "PredicateTransition", row["predicate_id"], row["to_tick"],
            {"native_transition_id": row["transition_id"], "tuple_id": row["tuple_id"],
             "bindings": row["bindings"], "from_tick": row["from_tick"],
             "to_tick": row["to_tick"], "from_value": row["from_value"], "to_value": row["to_value"],
             "direction": row["direction"], "evidence_truth_ids": row.get("evidence_truth_ids", [])},
            source_path(transition_file), line,
            rule_ref="Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json#" + row["predicate_id"])
        at_time(target, row["to_tick"], transition_file, line)
        transition_ref[row["transition_id"]] = target
        left = all_assertions.get((row["tuple_id"], row["from_tick"]))
        right = all_assertions.get((row["tuple_id"], row["to_tick"]))
        for kind, endpoint in (("transition_from", left), ("transition_to", right)):
            if endpoint is None:
                graph.unresolved_relation(source_file=source_path(transition_file), source_line=line,
                    relation={"transition_id": row["transition_id"], "edge_type": kind},
                    reason="formal transition assertion missing")
            elif kind == "transition_from":
                graph.add_edge(kind, endpoint, target, source_file=source_path(transition_file), source_line=line)
            else:
                graph.add_edge(kind, target, endpoint, source_file=source_path(transition_file), source_line=line)

    objective_event_ref: dict[str, str] = {}
    event_file = truth_root / "event_occurrences.jsonl"
    for line, row in source_rows(event_file):
        native = row["event_id"]
        node = event_id(native)
        graph.add_node(node, "EventOccurrence", row["event_type_id"], row["detection_tick"],
            {"native_event_id": native, "bindings": row["bindings"],
             "trigger_tick": row["trigger_tick"], "start_tick": row["start_tick"],
             "end_tick": row["end_tick"], "detection_tick": row["detection_tick"],
             "rule_id": row["rule_id"], "support_combination": row.get("supporting_parameter_refs", []),
             "source_refs": row.get("source_refs", [])},
            source_path(event_file), line, rule_ref=row["rule_id"])
        objective_event_ref[native] = node
        at_time(node, row["detection_tick"], event_file, line, "detection_tick")
        for role, entity in row["bindings"].items():
            target, evidence = argument_target("event", row["event_type_id"], role,
                entity, row["detection_tick"], event_file, line, record_bindings=row["bindings"])
            if target is not None:
                graph.add_edge("event_participant", node, target, role=role, attrs=evidence,
                    source_file=source_path(event_file), source_line=line)
        for tid in row.get("supporting_transition_ids", []):
            endpoint = transition_ref.get(tid)
            if endpoint:
                graph.add_edge("transition_supports_event", endpoint, node,
                    source_file=source_path(event_file), source_line=line, rule_ref=row["rule_id"])
            else:
                graph.unresolved_relation(source_file=source_path(event_file), source_line=line,
                    relation={"event_id": native, "transition_id": tid}, reason="support transition missing")
        for tid in row.get("supporting_truth_ids", []):
            endpoint = truth_ref.get(tid)
            if endpoint:
                graph.add_edge("assertion_supports_event", endpoint, node,
                    source_file=source_path(event_file), source_line=line, rule_ref=row["rule_id"])
            else:
                graph.unresolved_relation(source_file=source_path(event_file), source_line=line,
                    relation={"event_id": native, "truth_id": tid}, reason="support assertion missing")
        for index, phase in enumerate(row.get("event_phases", [])):
            phase_tick = phase["tick"]
            phase_node = prefix + f"phase:{native}:{index}"
            graph.add_node(phase_node, "EventPhase", phase["phase_kind"], phase_tick,
                           {"phase_index": index, "native_event_id": native, "raw": phase},
                           source_path(event_file), line)
            at_time(phase_node, phase_tick, event_file, line, "phase_tick")
            graph.add_edge("event_phase", node, phase_node, role=phase["phase_kind"],
                           source_file=source_path(event_file), source_line=line)
            tid = phase.get("transition_id")
            if tid in transition_ref:
                graph.add_edge("transition_supports_phase", transition_ref[tid], phase_node,
                               source_file=source_path(event_file), source_line=line)
    outcome_file = truth_root / "event_outcomes.jsonl"
    for line, row in source_rows(outcome_file):
        node = prefix + "out:" + row["outcome_id"]
        terminal = row.get("terminal_tick")
        graph.add_node(node, "EventOutcome", row["event_type_id"], terminal,
            {"native_outcome_id": row["outcome_id"], "native_event_id": row["event_id"],
             "status": row["status"], "lifecycle_status": row.get("lifecycle_status"),
             "lifecycle_phase_evidence": row.get("lifecycle_phase_evidence", []),
             "supporting_truth_ids": row.get("supporting_truth_ids", [])},
            source_path(outcome_file), line)
        if terminal is not None:
            at_time(node, terminal, outcome_file, line, "terminal_tick")
        source = objective_event_ref.get(row["event_id"])
        if source:
            graph.add_edge("event_outcome", source, node, source_file=source_path(outcome_file), source_line=line)
        for tid in row.get("supporting_truth_ids", []):
            endpoint = truth_ref.get(tid)
            if endpoint:
                graph.add_edge("assertion_supports_outcome", endpoint, node,
                               source_file=source_path(outcome_file), source_line=line)
            else:
                graph.unresolved_relation(source_file=source_path(outcome_file), source_line=line,
                    relation={"outcome_id": row["outcome_id"], "truth_id": tid},
                    reason="outcome support assertion missing")
        for phase_index, phase_evidence in enumerate(row.get("lifecycle_phase_evidence", [])):
            for tid in phase_evidence.get("supporting_truth_ids", []):
                endpoint = truth_ref.get(tid)
                if endpoint:
                    graph.add_edge("assertion_supports_outcome", endpoint, node,
                        role=f"lifecycle_phase:{phase_index}",
                        source_file=source_path(outcome_file), source_line=line)
                else:
                    graph.unresolved_relation(source_file=source_path(outcome_file), source_line=line,
                        relation={"outcome_id": row["outcome_id"], "truth_id": tid,
                                  "phase_index": phase_index}, reason="lifecycle support assertion missing")
            transition_native = phase_evidence.get("entry_transition_id")
            if transition_native:
                endpoint = transition_ref.get(transition_native)
                if endpoint:
                    graph.add_edge("transition_supports_outcome", endpoint, node,
                        role=f"lifecycle_phase:{phase_index}",
                        source_file=source_path(outcome_file), source_line=line)
                else:
                    graph.unresolved_relation(source_file=source_path(outcome_file), source_line=line,
                        relation={"outcome_id": row["outcome_id"], "entry_transition_id": transition_native},
                        reason="lifecycle support transition missing")

    matrix_file = supplement_root / "compute_predicate_matrix.jsonl"
    if matrix_file.exists():
        for line, row in source_rows(matrix_file):
            tick = row["tick"]
            node = prefix + "supp_matrix:" + row["matrix_row_id"]
            graph.add_node(node, "ContextAnchor", "compute_comm_predicate_matrix", tick,
                {"anchor_kind": "compute_comm_matrix",
                 "source_family": "compute_comm_predicate_matrix",
                 "native_matrix_row_id": row["matrix_row_id"],
                 "raw": row},
                source_path(matrix_file), line, rule_ref=row.get("rule_id"))
            at_time(node, tick, matrix_file, line)
            for scope_index, scope_value in enumerate(row["scope_values"]):
                scope_type = scope_value.get("scope_type")
                scope_id = scope_value.get("scope_entity_id")
                if scope_type == "compute_node":
                    host = compute_host_by_node.get(scope_id)
                    if host is None or scope_id != "compute_node:" + host:
                        graph.unresolved_relation(source_file=source_path(matrix_file), source_line=line,
                            relation={"scope_index": scope_index, "scope_entity_id": scope_id},
                            reason="matrix compute scope lacks typed world-scope identity")
                        continue
                elif scope_type != "uav" or ("sample", scope_id) not in seen_identity:
                    graph.unresolved_relation(source_file=source_path(matrix_file), source_line=line,
                        relation={"scope_index": scope_index, "scope_type": scope_type,
                                  "scope_entity_id": scope_id},
                        reason="matrix scope lacks declared identity mapping")
                    continue
                graph.add_edge("matrix_scope_entity", node,
                    ident(scope_id, "compute_node" if scope_type == "compute_node" else "sample"), role=scope_type,
                    source_file=source_path(matrix_file), source_line=line,
                    attrs={"source_scope_index": scope_index,
                           "source_path": f"scope_values[{scope_index}].scope_entity_id"})
    supplement_event_file = supplement_root / "compute_events.jsonl"
    if supplement_event_file.exists():
        from Dataset.semantic_truth.core_semantic_registry import get_core_event_occurrence_types
        core_event_types = {r["event_type"] for r in get_core_event_occurrence_types()}
        for line, row in source_rows(supplement_event_file):
            native = row["event_id"]
            tick = row["trigger_tick"]
            node = prefix + "supp_ev:" + native
            if row["event_type_id"] in core_event_types:
                # This producer records a single-predicate onset. The core event
                # requires its own complete support/hold rule and is built above.
                graph.add_node(node, "ContextAnchor", "measured_business_predicate_onset", tick,
                    {"anchor_kind": "predicate_onset", "source_family": "measured_business_predicate_onset",
                     "predicate_id": row["source_predicate_id"], "bindings": row["bindings"],
                     "native_event_id": native, "native_declared_event_type": row["event_type_id"],
                     "authority_scope": "business predicate rising onset; core event support is evaluated separately",
                     "raw": row}, source_path(supplement_event_file), line, rule_ref=row["rule_id"])
                at_time(node, tick, supplement_event_file, line)
                for ref in row["source_refs"]:
                    if ref not in supplement_ref:
                        raise ValueError(f"business onset lacks its current predicate: {supplement_event_file}:{line}:{ref}")
                    graph.add_edge("predicate_onset_assertion", node, supplement_ref[ref],
                                   source_file=source_path(supplement_event_file), source_line=line)
                continue
            graph.add_node(node, "EventOccurrence", row["event_type_id"], tick,
                {"native_event_id": native, "bindings": row["bindings"],
                 "terminal_rule_status": "not_declared",
                 "rule_id": row["rule_id"], "derivation_kind": row.get("derivation_kind"),
                 "source_predicate_id": row.get("source_predicate_id"),
                 "source_refs": row.get("source_refs", []), "raw": row},
                source_path(supplement_event_file), line, rule_ref=row["rule_id"])
            at_time(node, tick, supplement_event_file, line, "trigger_tick")
            for role, entity in row["bindings"].items():
                target, evidence = argument_target("event", row["event_type_id"], role,
                    entity, tick, supplement_event_file, line, record_bindings=row["bindings"])
                if target is not None:
                    graph.add_edge("event_participant", node, target, role=role, attrs=evidence,
                        source_file=source_path(supplement_event_file), source_line=line)
            support = []
            for ref in row.get("source_refs", []):
                endpoint = supplement_ref.get(ref)
                if endpoint:
                    support.append(endpoint)
                    graph.add_edge("assertion_supports_event", endpoint, node,
                                   source_file=source_path(supplement_event_file), source_line=line,
                                   rule_ref=row["rule_id"])
                elif ref.startswith("predicate_truth:"):
                    graph.unresolved_relation(source_file=source_path(supplement_event_file), source_line=line,
                        relation={"event_id": native, "source_ref": ref}, reason="supplement event predicate ID absent")
            transition = row.get("transition")
            if transition:
                before = next((item for item in support if graph.node_index[item]["tick"] == transition["from_tick"]), None)
                after = next((item for item in support if graph.node_index[item]["tick"] == transition["to_tick"]), None)
                if before and after:
                    transition_node = prefix + "supp_tr:" + native
                    graph.add_node(transition_node, "PredicateTransition", row["source_predicate_id"],
                        transition["to_tick"],
                        {"native_event_id": native, "source_predicate_id": row["source_predicate_id"],
                         "from_tick": transition["from_tick"], "to_tick": transition["to_tick"],
                         "from_value": transition["from_value"], "to_value": transition["to_value"],
                         "transition_authority": "compute_comm_supplement_event"},
                        source_path(supplement_event_file), line, rule_ref=row["rule_id"])
                    at_time(transition_node, transition["to_tick"], supplement_event_file, line)
                    graph.add_edge("transition_from", before, transition_node,
                                   source_file=source_path(supplement_event_file), source_line=line)
                    graph.add_edge("transition_to", transition_node, after,
                                   source_file=source_path(supplement_event_file), source_line=line)
                    graph.add_edge("transition_supports_event", transition_node, node,
                                   source_file=source_path(supplement_event_file), source_line=line,
                                   rule_ref=row["rule_id"])
                else:
                    graph.unresolved_relation(source_file=source_path(supplement_event_file), source_line=line,
                        relation={"event_id": native, "transition": transition},
                        reason="supplement transition endpoints absent from source predicate IDs")

    render_manifest = read_json(render_root / "episode_manifest.json")
    script_file = REPO / render_manifest["source_event_script_path"]
    script = read_json(script_file)
    triggers = {trigger["trigger_id"]: trigger for trigger in script["triggers"]}
    plan_events: dict[str, tuple[str, dict]] = {}
    plan_actions: dict[tuple[str, str], tuple[str, dict]] = {}
    for event_index, event in enumerate(script["events"]):
        native = event["event_id"]
        trigger = triggers[event["trigger_ref"]]
        scheduled_tick = trigger["tick"] if trigger["type"] == "tick" else None
        plan_node = prefix + "script_plan:" + native
        if native in plan_events:
            raise ValueError(f"source script repeats event ID: {script_file}:{native}")
        graph.add_node(plan_node, "ScriptEvent", event.get("intent"), scheduled_tick,
            {"native_event_id": native, "intent": event.get("intent"),
             "category_status": "source_intent" if event.get("intent") else "unresolved_source_intent",
             "source_event_index": event_index, "raw": event,
             "execution_script_role": "formal_authoring_script", "source": "authoring_plan",
             "scheduled_tick": scheduled_tick, "script_scheduled": trigger["type"] == "tick",
             "trigger_ref": event["trigger_ref"]}, source_path(script_file))
        plan_events[native] = plan_node, event
        if scheduled_tick is not None:
            at_time(plan_node, scheduled_tick, script_file, role="scheduled_tick")
        for action_index, action in enumerate(event["actions"]):
            action_native = action["action_id"]
            key = native, action_native
            if key in plan_actions:
                raise ValueError(f"source event repeats action ID: {script_file}:{key}")
            action_node = prefix + f"action_plan:{native}:{action_native}"
            graph.add_node(action_node, "ActionPlan", action["type"], scheduled_tick,
                {"native_event_id": native, "native_action_id": action_native,
                 "source_event_index": event_index, "source_action_index": action_index,
                 "raw": action, "execution_script_role": "formal_authoring_script",
                 "target_scope": action_target_scope(action),
                 "target_entity_id": action.get("entity_id"), "audit_status": "not_recorded",
                 "audit_source_file": None, "audit_source_line": None}, source_path(script_file))
            plan_actions[key] = action_node, action
            graph.add_edge("script_declares_action", plan_node, action_node,
                           source_file=source_path(script_file))
            if scheduled_tick is not None:
                at_time(action_node, scheduled_tick, script_file, role="scheduled_tick")

    script_ref: dict[str, str] = {}
    realization_file = render_root / "event_realization.jsonl"
    capture_requests: dict[tuple[str, str], dict[str, Any]] | None = None
    for line, row in source_rows(realization_file):
        for leaf in iter_field_values(row, source_family="formal_event_realization"):
            if leaf.entity_id is None:
                continue
            reference = resolver.resolve(leaf.entity_id, source_file=source_path(realization_file),
                source_line=line, source_path=leaf.raw_path,
                source_status=leaf.identity_status, source_reason=leaf.identity_reason)
            if reference.status != "resolved":
                graph.unresolved_relation(source_file=source_path(realization_file), source_line=line,
                    relation={"entity_id": leaf.entity_id, "raw_path": leaf.raw_path,
                              "declared_entity_id": leaf.declared_entity_id},
                    reason=reference.reason or reference.status)
        native = row["event_id"]
        node = prefix + "script:" + native
        dispatch = row.get("dispatch_tick")
        graph.add_node(node, "ScriptEvent", row["intent"], dispatch,
            {"native_event_id": native, "intent": row.get("intent"), "dispatch_tick": dispatch,
             "intent_stage": row.get("intent_stage"),
             "causal_chain_id": row.get("causal_chain_id"),
             "trigger_ref": row.get("trigger_ref"), "priority": row.get("priority"),
             "max_fire_count": row.get("max_fire_count"),
             "target_roles": row.get("target_roles"),
             "action_realizations": row.get("action_realizations", []),
             "log_category_status": "not_recorded_in_event_realization",
             "display_overlay_status": "not_recorded_in_event_realization",
             "result_tick": row.get("result_tick"), "evidence_tick": row.get("evidence_tick"),
             "source": "event_realization", "raw": row}, source_path(realization_file), line)
        script_ref[native] = node
        planned = plan_events.get(native)
        plan_matches = (planned is not None and row["scenario_id"] == script["scenario_id"]
                        and row["intent"] == planned[1].get("intent"))
        if plan_matches:
            graph.add_edge("script_dependency", planned[0], node,
                attrs={"correspondence": "scenario_event_id_intent",
                       "parameter_match_status": "not_proven_by_realization"},
                source_file=source_path(realization_file), source_line=line)
        elif planned is not None:
            graph.unresolved_relation(source_file=source_path(realization_file), source_line=line,
                relation={"event_id": native, "source_plan_file": source_path(script_file),
                          "source_plan_intent": planned[1].get("intent"), "realization_intent": row["intent"]},
                reason="current authoring event and saved realization identity differ")
        else:
            graph.unresolved_relation(source_file=source_path(realization_file), source_line=line,
                relation={"event_id": native, "source_plan_file": source_path(script_file)},
                reason="saved realized event is absent from the current authoring script")
        if dispatch is not None:
            at_time(node, dispatch, realization_file, line, "dispatch_tick")
        for index, action in enumerate(row.get("action_realizations", [])):
            if action.get("status") != "ok":
                raise ValueError(f"formal command has no confirmed dispatch: {realization_file}:{line}:{index}")
            action_native = action["action_id"]
            action_node = prefix + f"action:{native}:{action_native}"
            action_tick = action.get("dispatch_tick")
            issued = plan_actions.get((native, action_native)) if plan_matches else None
            issued_action = None
            if issued is not None:
                realized_target = action.get("entity_id")
                if (realized_target == "" and issued[1].get("entity_id") is None
                        and action_target_scope(issued[1]) is not None):
                    realized_target = None
                if issued[1]["type"] != action["action_type"] or issued[1].get("entity_id") != realized_target:
                    raise ValueError(f"issued command differs from its realized action: {realization_file}:{line}:{index}")
                issued_action = issued[1]
            capture_source = {}
            if action["action_type"] == "capture_screenshot":
                if capture_requests is None:
                    capture_requests = _capture_request_sources(render_root)
                capture_source = capture_requests[(native, action_native)]
                if capture_source["scenario_id"] != row["scenario_id"]:
                    raise ValueError(f"capture request scenario differs from realization: {realization_file}:{line}:{index}")
            graph.add_node(action_node, "CommandDispatch", action["action_type"], action_tick,
                {"native_action_id": action_native, "native_event_id": native,
                 "source_action_index": index, "target_scope": action_target_scope(action),
                 "target_entity_id": action.get("entity_id"),
                 "source_action": action, "current_authored_action": issued_action,
                 "current_authoring_source": {"source_file": source_path(script_file), "event_id": native, "action_id": action_native},
                 "issued_parameter_status": "not_recorded_by_capture",
                 "recorded_dispatch": {"type": action["action_type"], "entity_id": action.get("entity_id")},
                 "dispatch_status": "ok",
                 "semantic_use": "capture_request" if action["action_type"] == "capture_screenshot" else "simulation_control",
                **capture_source},
                source_path(realization_file), line)
            planned_action = plan_actions.get((native, action_native))
            if plan_matches and planned_action is not None and planned_action[1]["type"] == action["action_type"]:
                graph.add_edge("script_dependency", planned_action[0], action_node,
                    attrs={"correspondence": "scenario_event_intent_action_id_type",
                           "parameter_match_status": "not_proven_by_realization"},
                    source_file=source_path(realization_file), source_line=line)
            elif plan_matches:
                graph.unresolved_relation(source_file=source_path(realization_file), source_line=line,
                    relation={"event_id": native, "action_id": action_native,
                              "source_plan_file": source_path(script_file),
                              "action_type": action["action_type"]},
                    reason=("saved realized action is absent from the current authoring event"
                            if planned_action is None else
                            "saved realized action type differs from the current authoring action"))
            graph.add_edge("observed_dispatch", node, action_node,
                           source_file=source_path(realization_file), source_line=line)
            if action_tick is not None:
                at_time(action_node, action_tick, realization_file, line, "dispatch_tick")
            target = action.get("entity_id")
            if action["action_type"] in ENTITY_TARGET_ACTIONS and not target:
                graph.unresolved_relation(source_file=source_path(realization_file), source_line=line,
                    relation={"action_id": action_native, "source_action_index": index},
                    reason="dispatched entity-target command lacks target ID")
            if target:
                if ("sample", target) not in seen_identity:
                    add_identity(target, None, None, realization_file, line, completeness="action_only")
                graph.add_edge("action_target", action_node, ident(target), role="target",
                               source_file=source_path(realization_file), source_line=line)

    chain_file = REPO / "aw_data/event_chains" / episode_id / "event_chain_relations.jsonl"
    if chain_file.exists():
        for line, row in source_rows(chain_file):
            kind = row["relation_type"]
            if kind not in {"script_declared_causal", "script_declared_context"}:
                raise ValueError(f"unrecognized formal chain relation: {chain_file}:{line}:{kind}")
            endpoints = []
            for side in ("source_endpoint", "target_endpoint"):
                endpoint = row[side]
                native = endpoint.get("event_id")
                if endpoint.get("kind") == "script_event" and native in script_ref:
                    endpoints.append(script_ref[native])
                elif endpoint.get("kind") == "context_anchor":
                    anchor = prefix + "context:" + str(row["relation_id"]) + ":" + side
                    if anchor not in graph.node_index:
                        ticks = row.get("runtime_alignment", {}).get("source_dispatch_ticks", [])
                        anchor_tick = ticks[0] if len(ticks) == 1 else None
                        graph.add_node(anchor, "ContextAnchor", "script_chain_context",
                            anchor_tick, {"anchor_kind": "script_chain_context",
                                          "source_family": "script_chain_context",
                                          "native_intent": endpoint.get("intent"),
                                          "source_endpoint": endpoint,
                                          "source_endpoint_side": side},
                            source_path(chain_file), line)
                        if anchor_tick is not None:
                            at_time(anchor, anchor_tick, chain_file, line)
                    endpoints.append(anchor)
                else:
                    endpoints.append(None)
            if all(endpoints):
                graph.add_edge(kind, endpoints[0], endpoints[1], source_file=source_path(chain_file),
                               source_line=line, native_relation_id=row.get("relation_id"),
                               attrs={"runtime_alignment": row.get("runtime_alignment"),
                                      "validation_status": row.get("validation_status"),
                                      "declaration_only": True,
                                      "raw": strip_source_digests({
                                          key: value for key, value in row.items()
                                          if key != "causal_claim"})})
            else:
                graph.unresolved_relation(source_file=source_path(chain_file), source_line=line,
                    relation=row, reason="formal chain endpoint not in current source realization")

    for folder, names in (("domain_state_supplement", ("domain_state_observations.jsonl",)),
                          ("compute_comm_supplement", ("compute_state.jsonl", "communication_state.jsonl"))):
        # These ledgers were produced in the same run as world truth and events.
        supplement_root = truth_root
        for name in names:
            path = supplement_root / name
            if not path.exists():
                continue
            for line, row in state_records[path] if path in state_records else source_rows(path):
                tick = row.get("tick")
                node = prefix + f"supplement:{folder}:{name}:{line}"
                source_family = row.get("observation_family") or folder
                gnss_subject = None
                if folder == "domain_state_supplement" and source_family == "gnss_navigation":
                    gnss_subject = resolver.resolve(
                        row.get("subject_id"), source_file=source_path(path),
                        source_line=line, source_path="subject_id",
                        source_status=("source_reference" if row.get("subject_category") == "uav"
                                       else "unresolved"),
                        source_reason=(None if row.get("subject_category") == "uav"
                                       else "GNSS observation subject is not declared as a UAV"),
                    )
                    if gnss_subject.status == "resolved":
                        identity_node = graph.node_index.get(gnss_subject.identity_node_id)
                        if identity_node is None or identity_node["category_id"] != "uav":
                            gnss_subject = resolver.resolve(
                                row.get("subject_id"), source_file=source_path(path),
                                source_line=line, source_path="subject_id",
                                source_status="unresolved",
                                source_reason=("GNSS subject has no materialized sample identity"
                                               if identity_node is None else
                                               "GNSS subject roster category is not UAV"),
                            )
                graph.add_node(node, "ContextAnchor", source_family, tick,
                    {"anchor_kind": "domain_state_observation" if source_family in CONTEXT_PREDICATE_STATE_FAMILIES else "supplement_state",
                     "source_family": source_family,
                     "source_layer": folder,
                     "row_family": row.get("family"),
                     "observation_family": row.get("observation_family"),
                     "subject_identity_ref": gnss_subject.identity_node_id if gnss_subject else None,
                     "subject_identity_status": gnss_subject.status if gnss_subject else None,
                     "subject_identity_reason": gnss_subject.reason if gnss_subject else None,
                     "raw": row, "field_context": {"source_file": source_path(path),
                     "source_line": line, "value_tick": tick}}, source_path(path), line)
                if isinstance(tick, int):
                    at_time(node, tick, path, line)
                if gnss_subject is not None:
                    if gnss_subject.status == "resolved":
                        graph.add_edge("supplement_subject", node, gnss_subject.identity_node_id,
                                       source_file=source_path(path), source_line=line)
                    else:
                        graph.unresolved_relation(source_file=source_path(path), source_line=line,
                            relation={"subject_id": row.get("subject_id"),
                                      "observation_family": source_family},
                            reason=gnss_subject.reason or "GNSS subject identity is unresolved")
                if name == "compute_state.jsonl":
                    compute_id = row.get("node_id")
                    host_id = row.get("entity_id")
                    if compute_host_by_node.get(compute_id) != host_id:
                        graph.unresolved_relation(source_file=source_path(path), source_line=line,
                            relation={"node_id": compute_id, "entity_id": host_id},
                            reason="compute state lacks matching typed world-scope host")
                    else:
                        graph.add_edge("compute_state_node", node, ident(compute_id, "compute_node"),
                            source_file=source_path(path), source_line=line)
                if folder == "domain_state_supplement" and isinstance(tick, int):
                    values = row.get("values", {})
                    subject = row.get("subject_id")
                    for peer_field, fields, role in (
                        ("nearest_pad_id", (("nearest_pad_distance_m", "euclidean_3d"),
                                            ("nearest_pad_xy_distance_m", "horizontal_euclidean"),
                                            ("nearest_pad_altitude_delta_m", "signed_altitude_delta")), "subject_to_nearest_pad"),
                        ("nearest_uav_id", (("nearest_uav_distance_m", "euclidean_3d"),), "subject_to_nearest_uav"),
                    ):
                        peer = values.get(peer_field)
                        if not isinstance(peer, str) or peer in {"", "UNKNOWN", "unknown"}:
                            continue
                        endpoints = (state_lookup.get((tick, subject)), state_lookup.get((tick, peer)))
                        for field, metric in fields:
                            value = values.get(field)
                            if not isinstance(value, (int, float)):
                                continue
                            if all(endpoints):
                                graph.add_edge("RelativeState", endpoints[0], endpoints[1], role=role,
                                    source_file=source_path(path), source_line=line,
                                    attrs={"field": "values." + field, "value": value,
                                           "unit": "m", "unit_status": "declared",
                                           "unit_basis": "Dataset/semantic_simulation/domain_state.py#meter_coordinate_nearest_entity",
                                           "metric": metric,
                                           "value_tick": tick, "peer_field": "values." + peer_field})
                            else:
                                metric_node = prefix + f"recorded_metric:{name}:{line}:{field}"
                                graph.add_node(metric_node, "ContextAnchor", "measured_geometry_support", tick,
                                    {"anchor_kind": "geometry_support", "source_family": "measured_geometry_support",
                                     "subject_id": subject, "peer_id": peer, "field": "values." + field,
                                     "recorded_value": value, "metric": metric,
                                     "state_availability": [endpoint is not None for endpoint in endpoints],
                                     "geometry_status": "recorded_metric_without_complete_pose_pair",
                                     "recomputed_geometry": False}, source_path(path), line)
                                at_time(metric_node, tick, path, line)
    return graph.close("complete_with_unresolved" if graph.unresolved else "complete")
