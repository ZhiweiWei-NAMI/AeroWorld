"""Resolve a source entity reference inside one graph sample."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from Dataset.semantic_truth.entity_scope import roster_index


# The action producer records these two operations without an entity target.
# Their effects are carried by the action/result pair, not by a roster entity.
NON_ENTITY_ACTION_SCOPES = {
    "capture_screenshot": "capture_request",
    "set_weather": "environment",
}
ENTITY_TARGET_ACTIONS = frozenset({
    "set_visual_state", "move_entity", "set_runtime_state", "spawn_entity", "remove_entity",
    "set_pedestrian_activity",
})

# Exact IDs written by the compute/communication producer.  A stable object
# retains its identity through changing state; tick IDs denote one state-time object.
BUSINESS_SOURCE_OBJECTS = {
    "world:ComputeTask": ("compute_state", "task_id", False),
    "world:ComputeQueue": ("compute_state", "queue_id", False),
    "world:ComputeResource": ("compute_state", "resource_id", False),
    "world:TaskExecution": ("compute_state", "execution_id", True),
    "world:ExecutionFailure": ("compute_state", "failure_state_id", False),
    "world:CommunicationStation": ("communication_state", "station_id", False),
    "world:DataFlow": ("communication_state", "flow_id", False),
    "world:ChannelAllocation": ("communication_state", "channel_allocation_id", False),
    "world:CommunicationSession": ("communication_state", "session_id", False),
    "world:Handover": ("communication_state", "handover_id", False),
    "world:Retransmission": ("communication_state", "retransmission_id", False),
    "world:TransmissionAttempt": ("communication_state", "transmission_attempt_id", True),
    "world:Message": ("communication_state", "message_id", True),
}

# SUMO traffic-light keys are identities of simulator controllers, not scene
# entities with coincident string IDs.  Other source-backed predicate objects
# remain in the sample namespace unless their producer has a narrower mapping.
ONTOLOGY_IDENTITY_NAMESPACES = {
    "world:TrafficSignal": "sumo_tls_controller",
}


def binding_identity_namespace(ontology_class: str | None) -> str:
    """Return the graph namespace fixed by a predicate role's source class."""
    return ONTOLOGY_IDENTITY_NAMESPACES.get(str(ontology_class), "sample")


def crowd_cohort_authorities(episode_id: str, wanted: set[str]) -> dict:
    """Read the exact cohort IDs and members written by the domain producer."""
    from Dataset.world_model.graph.contract import REPO, iter_jsonl
    path = REPO / "aw_data/objective_semantic_truth" / episode_id / "domain_state_observations.jsonl"
    result = {}
    if not wanted:
        return result
    for line, row in iter_jsonl(path):
        if row.get("observation_family") != "crowd_evacuation":
            continue
        values = row["values"]
        entity = values["crowd_id"]
        if entity not in wanted:
            continue
        if row["episode_id"] != episode_id or values["crowd_ontology_class_id"] != "world:Crowd":
            raise ValueError(f"crowd cohort authority differs: {path}:{line}")
        members = values["cohort_member_ids"]
        if not members or len(set(members)) != len(members) or values["cohort_member_count"] != len(members):
            raise ValueError(f"crowd cohort membership is incomplete: {path}:{line}")
        result.setdefault(entity, {"path": path, "line": line, "members": members, "tick": row["tick"]})
        if result.keys() >= wanted:
            break
    return result


def declared_world_bindings(resolver, base: dict, source: str, *,
                            declaration_paths: list[str] | None = None) -> dict[str, tuple[str, str]]:
    """Register logical objects grounded by an executable world template.

    Physical actor existence still requires its roster. This reads actual typed
    declarations for regions, corridors, regulations and UTM objects.
    """
    from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates
    from Dataset.semantic_truth.facility_scope import ontology_class_is_a
    templates = {r["id"]: r for r in get_core_predicate_templates()}
    claims = {}
    physical = ("world:UnmannedAircraft", "world:GroundVehicle", "world:Pedestrian")
    for index, assertion in enumerate(base["initial_assertions"]):
        template = templates[assertion["predicate_id"]]
        grounding = template["grounding_spec"]
        for role, entity in assertion["bindings"].items():
            ontology = assertion["binding_ontology_classes"][role]
            if ontology == "world:ComputeNode" or ontology in BUSINESS_SOURCE_OBJECTS or any(
                    ontology_class_is_a(ontology, actor) for actor in physical):
                continue
            provenance = assertion["binding_provenance"][role]
            rule = grounding["role_sources"][role]
            expected = "scope_entity" if rule["kind"] == "scope_entity" else rule.get("path", "")
            if (provenance["entity_id"] != entity or provenance["ontology_class_id"] != ontology
                    or provenance["binding_source"] != expected):
                raise ValueError(f"world object grounding differs: {source}:initial_assertions[{index}]:{role}")
            pointer = (f"initial_assertions[{index}]" if declaration_paths is None else declaration_paths[index]) + f".binding_provenance.{role}"
            previous = claims.get(entity)
            if previous is not None and previous[0] != ontology:
                if ontology_class_is_a(previous[0], ontology):
                    continue
                if not ontology_class_is_a(ontology, previous[0]):
                    continue  # Conflicting source claims remain graph diagnostics.
            claims[entity] = ontology, pointer
    for entity, (ontology, pointer) in claims.items():
        if ontology == "world:Crowd":
            # A typed predicate binding refers to the producer's cohort; its
            # identity and membership come from the actual domain record.
            continue
        namespace = binding_identity_namespace(ontology)
        if namespace != "sample" and resolver.resolve(entity, source_file=source,
                source_line=None, source_path=pointer, namespace="sample").status == "resolved":
            namespace = "sample"
        if resolver.resolve(entity, source_file=source, source_path=pointer,
                            source_line=None, namespace=namespace).status == "resolved":
            continue
        resolver.declare({"entity_id": entity, "ontology_class_id": ontology},
            source_file=source, source_path=pointer, authority="world_binding_grounding",
            namespace=namespace)
    return claims


def arm_grounded_object_bindings(resolver, rows, source: str) -> None:
    """Declare saved, typed logical regions without inventing physical poses.

    Frozen ARM grounding records are independent authorities. Physical actors
    still require a roster and business objects their execution-state records.
    """
    from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates
    from Dataset.semantic_truth.facility_scope import ontology_class_is_a
    templates = {r["id"]: r for r in get_core_predicate_templates()}
    physical = ("world:UnmannedAircraft", "world:GroundVehicle", "world:Pedestrian")
    for line, row in rows:
        template = templates.get(row["predicate_id"])
        if template is None:
            continue
        roles = {r["key"]: "world:" + r["class"].removeprefix("world:")
                 for r in template["argument_roles"]}
        for index, observation in enumerate(row.get("evidence", {}).get("observations", [])):
            value = observation.get("value")
            if not observation.get("path", "").startswith("normalized.grounded_roles__"):
                continue
            if not isinstance(value, dict) or value.get("bindings") != row["bindings"]:
                raise ValueError(f"ARM grounded bindings differ: {source}:{line}:{index}")
            classes = value["binding_ontology_classes"]
            if classes.keys() != row["bindings"].keys():
                raise ValueError(f"ARM grounded role classes are incomplete: {source}:{line}:{index}")
            for role, entity in row["bindings"].items():
                ontology = classes[role]
                if (ontology in BUSINESS_SOURCE_OBJECTS or ontology == "world:ComputeNode"
                        or any(ontology_class_is_a(ontology, actor) for actor in physical)):
                    continue
                if role not in roles or not ontology_class_is_a(ontology, roles[role]):
                    raise ValueError(f"ARM logical role differs from its executable template: {source}:{line}:{role}")
                namespace = binding_identity_namespace(ontology)
                for ns in ("sample", namespace):
                    if resolver.resolve(entity, source_file=source, source_line=line,
                            source_path="bindings." + role, namespace=ns).status == "resolved":
                        break
                else:
                    if namespace == "sumo_tls_controller":
                        continue  # A controller requires an actual truth-frame key.
                    resolver.declare({"entity_id": entity, "ontology_class_id": ontology},
                        source_file=source,
                        source_path=f"line[{line}].evidence.observations[{index}].value.binding_ontology_classes.{role}",
                        authority="saved_ARM_typed_grounding")


@lru_cache(maxsize=4)
def _business_source_rows(source_file: str) -> dict[int, dict[str, Any]]:
    from Dataset.world_model.graph.contract import REPO, iter_jsonl
    return dict(iter_jsonl(REPO / source_file))


def business_object_binding_allowed(target: Mapping[str, Any], bound: str | None,
                                    role_class: str, evidence: Mapping[str, Any] | None,
                                    tick: int | None) -> bool:
    """Verify one business endpoint against its exact, same-tick state ID."""
    ontology = role_class if ":" in role_class else "world:" + role_class
    declaration = BUSINESS_SOURCE_OBJECTS.get(ontology)
    if declaration is None or evidence is None:
        return False
    attrs = target.get("attrs", {})
    if (target.get("node_type") != "ContextAnchor"
            or target.get("category_id") != "compute_comm_supplement"
            or attrs.get("source_family") != "compute_comm_supplement"
            or attrs.get("anchor_kind") != "supplement_state"
            or attrs.get("entity_id") != bound or attrs.get("ontology_class") != ontology):
        return False
    sample_key = target.get("sample_key", "")
    if not sample_key.startswith("formal:"):
        return False
    family, field, per_tick = declaration
    allowed_fields = {field}
    station_roles = {"handover.previous_station_id": "source_station",
                     "handover.current_station_id": "target_station"}
    if ontology == "world:CommunicationStation":
        allowed_fields.update(station_roles)
    original_field = attrs.get("source_object_field")
    evidence_field = evidence.get("source_field")
    source_file = "aw_data/objective_semantic_truth/" + sample_key.removeprefix("formal:") + "/" + family + ".jsonl"
    line = evidence.get("source_line")
    if (target.get("source_file") != source_file or original_field not in allowed_fields
            or evidence.get("source_file") != source_file or evidence_field not in allowed_fields
            or type(line) is not int or line < 1):
        return False
    rows = _business_source_rows(source_file)
    first = rows.get(target.get("source_line"))
    def value(record, path):
        current = record
        for token in path.split("."):
            if not isinstance(current, Mapping) or token not in current:
                return None
            current = current[token]
        return current
    if first is None or value(first, original_field) != bound:
        return False
    row = rows.get(line)
    if evidence_field in station_roles and (
            evidence.get("event_category") != "communication.communication_handover_event"
            or evidence.get("event_role") != station_roles[evidence_field]
            or row is None or row.get("handover_id") != evidence.get("handover_id")
            or row.get("session_id") != evidence.get("session_id")):
        return False
    return (row is not None and row.get("schema_name") == family
            and row.get("episode_id") == sample_key.removeprefix("formal:")
            and value(row, evidence_field) == bound and type(value(row, evidence_field)) is str
            and row.get("tick") == tick
            and (target.get("tick") == tick if per_tick else target.get("tick") is None))


def action_target_scope(action: Mapping[str, Any]) -> str | None:
    """Return the declared non-entity scope for a target-free action."""
    kind = action.get("action_type") or action.get("type")
    if action.get("entity_id") not in (None, ""):
        return None
    return NON_ENTITY_ACTION_SCOPES.get(kind)


def action_result_target(action: Mapping[str, Any], result: Mapping[str, Any]) -> str | None:
    """Carry the declared command target into its paired ARM result."""
    if action.get("type") not in ENTITY_TARGET_ACTIONS or result.get("entity_id") not in (None, ""):
        return None
    entity_id = action.get("entity_id")
    return entity_id if isinstance(entity_id, str) and entity_id else None


def read_roster(path: Path) -> dict[str, Any]:
    """Read a roster before duplicate JSON object keys can be overwritten."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key in {path}: {key}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError(f"roster is not an object: {path}")
    if not isinstance(payload.get("entities"), list):
        raise ValueError(f"roster lacks an entity list: {path}")
    roster_index(payload)
    return payload


@dataclass(frozen=True)
class EntityReference:
    sample_key: str
    raw_id: str | None
    namespace: str
    lifecycle_id: str | None
    status: str
    identity_node_id: str | None
    source_file: str
    source_line: int | None
    source_path: str
    authority: str | None
    authority_path: str | None
    inherited_from: tuple[dict[str, str], ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ValueIdentity:
    status: str
    node_id: str | None
    local_index: int | None
    basis: str | None
    basis_path: str | None
    reason: str | None


def identity_node_id(sample_key: str, namespace: str, raw_id: str) -> str:
    """Keep equal raw IDs in independent source namespaces separate."""
    if not namespace or not raw_id:
        raise ValueError("identity node requires a namespace and raw ID")
    return f"{sample_key}|i:{namespace}:{raw_id}"


def resolve_value_identity(leaf: Any, owner: Mapping[str, Any],
                           identities: Mapping[tuple[str, str], tuple[int | None, Mapping[str, Any]]]) -> ValueIdentity:
    """Map one typed source leaf to a sample-local identity, if proven."""
    if leaf.identity_status == "unresolved":
        return ValueIdentity("unresolved", None, None, None, None, leaf.identity_reason)
    if leaf.entity_id is None:
        return ValueIdentity("not_applicable", None, None, None, None, None)
    namespace = leaf.entity_namespace or "sample"
    if namespace not in {"sample", "compute_node", "sumo_tls_controller"}:
        return ValueIdentity("unresolved", None, None, None, None,
                             f"entity namespace {namespace} has no graph identity mapping")
    found = identities.get((namespace, leaf.entity_id))
    if found is None:
        return ValueIdentity("unresolved", None, None, None, None,
                             "entity absent from sample identity roster")
    local_index, identity = found
    attrs = identity["attrs"]
    if attrs.get("identity_namespace") != namespace:
        return ValueIdentity("unresolved", None, None, None, None,
                             "source and graph identity namespaces differ")
    if namespace == "compute_node" and (identity.get("category_id") != "compute_node"
                                         or attrs.get("source_scope_type") != "compute_node"
                                         or not attrs.get("source_host_entity_id")):
        return ValueIdentity("unresolved", None, None, None, None,
                             "compute-node identity lacks typed world-scope authority")
    if namespace == "sumo_tls_controller" and attrs.get("source_scope_type") != "sumo_tls_controller":
        return ValueIdentity("unresolved", None, None, None, None,
                             "SUMO controller identity lacks truth-frame authority")
    if attrs.get("identity_status") != "resolved":
        return ValueIdentity("unresolved", None, None, None, None,
                             attrs.get("identity_reason") or "identity declaration unresolved")
    owner_attrs = owner["attrs"]
    if (owner["node_type"] == "ContextAnchor" and
            owner_attrs.get("source_layer") == "domain_state_supplement" and
            owner_attrs.get("observation_family") == "gnss_navigation" and
            leaf.field_path.startswith("values.")):
        raw = owner_attrs.get("raw") or {}
        if raw.get("subject_category") != "uav" or identity.get("category_id") != "uav":
            return ValueIdentity("unresolved", None, None, None, None,
                                 "GNSS subject category differs from UAV roster identity")
        if (owner_attrs.get("subject_identity_status") != "resolved" or
                owner_attrs.get("subject_identity_ref") != identity["node_id"]):
            return ValueIdentity("unresolved", None, None, None, None,
                                 owner_attrs.get("subject_identity_reason") or
                                 "GNSS observation subject identity is unresolved")
    if owner["node_type"] == "EntityState":
        if owner_attrs.get("identity_status") != "resolved":
            return ValueIdentity("unresolved", None, None, None, None,
                                 "state entity identity unresolved")
        if owner_attrs.get("identity_ref") != identity["node_id"]:
            return ValueIdentity("unresolved", None, None, None, None,
                                 "state identity reference differs from source entity")
    return ValueIdentity("resolved", identity["node_id"], local_index,
                         attrs.get("identity_basis"), attrs.get("identity_basis_path"), None)


class IdentityResolver:
    """A roster-backed identity index; branch inheritance is explicit input."""

    def __init__(self, sample_key: str):
        self.sample_key = sample_key
        self._entries: dict[tuple[str, str, str | None], dict[str, Any]] = {}
        self._seen_by_source: dict[str, set[tuple[str, str, str | None]]] = {}
        self.conflicts: list[dict[str, Any]] = []

    def declare(self, item: dict[str, Any], *, source_file: str,
                source_path: str, authority: str, namespace: str = "sample",
                lifecycle_id: str | None = None, inherited: bool = False) -> None:
        entity_id = item.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(f"entity declaration lacks ID: {source_file}:{source_path}")
        key = (namespace, entity_id, lifecycle_id)
        seen = self._seen_by_source.setdefault(source_file, set())
        if key in seen:
            raise ValueError(f"duplicate entity declaration in {source_file}: {entity_id}")
        seen.add(key)
        category = item.get("entity_category") or item.get("category") or item.get("scope_type")
        ontology = (item.get("semantic_scope") or {}).get("ontology_class_id") or item.get("ontology_class_id")
        prior = self._entries.get(key)
        entry = {"category": category, "ontology": ontology,
                 "source_file": source_file, "source_path": source_path,
                 "authority": authority, "inherited": inherited, "status": "resolved"}
        if prior is None:
            self._entries[key] = entry
            return
        differences = {name: (prior[name], entry[name]) for name in ("category", "ontology")
                       if prior[name] is not None and entry[name] is not None and prior[name] != entry[name]}
        if differences:
            prior["status"] = "conflict"
            prior["reason"] = f"declarations differ: {differences}"
            self.conflicts.append({"entity_id": entity_id, "namespace": namespace,
                                   "sources": [prior["source_file"], source_file],
                                   "paths": [prior["source_path"], source_path],
                                   "differences": differences})
        elif inherited:
            prior.setdefault("inherited_from", []).append(
                {"source_file": source_file, "source_path": source_path})
        elif prior["source_file"] != source_file:
            # Multiple source declarations require an explicit shared-roster
            # or inheritance relationship; equal labels alone are insufficient.
            prior["status"] = "unresolved"
            prior["reason"] = "independent declarations have no identity mapping"

    def resolve(self, raw_id: str | None, *, source_file: str,
                source_line: int | None, source_path: str,
                namespace: str = "sample", lifecycle_id: str | None = None,
                source_status: str = "source_reference",
                source_reason: str | None = None) -> EntityReference:
        key = (namespace, raw_id, lifecycle_id) if raw_id is not None else None
        entry = self._entries.get(key) if key is not None else None
        status = ("unresolved" if source_status == "unresolved" or entry is None else entry["status"])
        reason = source_reason or ("entity absent from declared roster" if entry is None else entry.get("reason"))
        if status == "resolved" and (namespace not in {"sample", "compute_node", "sumo_tls_controller"}
                                     or lifecycle_id is not None):
            status = "unresolved"
            reason = "namespace or lifecycle has no declared graph identity node mapping"
        node_id = (identity_node_id(self.sample_key, namespace, raw_id)
                   if status == "resolved" and namespace in {"sample", "compute_node", "sumo_tls_controller"}
                   and lifecycle_id is None else None)
        return EntityReference(self.sample_key, raw_id, namespace, lifecycle_id,
                               status, node_id, source_file, source_line,
                               source_path, entry["source_file"] if entry else None,
                               entry["source_path"] if entry else None,
                               tuple(entry.get("inherited_from", ())) if entry else (), reason)
