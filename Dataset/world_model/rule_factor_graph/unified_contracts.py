"""Executable predicate contracts and explicit shared-state schemas.

The frozen semantic registry remains the authority for predicate meaning.  This
module adds state ownership to its exact source declarations; predicate names
and entity identifiers never select neural parameters.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from Dataset.semantic_truth.core_semantic_registry import get_core_predicate_templates


REGISTRY_SOURCE = "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"
CONTRACT_SOURCE = "Dataset/semantic_rules/predicates/world_truth_predicate_contracts.yaml"
TRUTH_DOMAIN = ("false", "true", "unknown", "out_of_scope")
SUPPORTED_OPERATORS = frozenset({
    "all", "direct_truth", "changed", "present", "truthy", "falsy", "equals",
    "in", "gt", "gte", "lt", "lte", "gt_constant", "gt_fields",
    "absolute_difference_gt", "outside_range",
})

WORLD_NAMESPACE = "https://w3id.org/aeroworld/ontology/world#"
CORE_NAMESPACE = "https://w3id.org/aeroworld/ontology/core#"
DOMAIN_NAMESPACE = "https://w3id.org/aeroworld/ontology/domain/"
# Exact namespace aliases used by the current runtime producer.  These are
# representations of the authored domain IRIs, not newly declared world classes.
# The pairs were checked against all 109 executable argument roles: authored
# argumentClass IRIs + compute_comm.py:47 and world_truth.py:1779 converters.
RUNTIME_CLASS_ALIASES = {
    "world:StopLine": DOMAIN_NAMESPACE + "mobility_regulation#StopLine",
    "world:SpeedLimitRegulation": DOMAIN_NAMESPACE + "mobility_regulation#SpeedLimitRegulation",
    "world:FollowingDistanceRule": DOMAIN_NAMESPACE + "mobility_regulation#FollowingDistanceRule",
    "world:Crowd": DOMAIN_NAMESPACE + "agent_interaction_safety#Crowd",
    "world:TransmissionAttempt": DOMAIN_NAMESPACE + "communication_process#TransmissionAttempt",
    "world:ChannelAllocation": DOMAIN_NAMESPACE + "communication_process#ChannelAllocation",
    "world:CommunicationSession": DOMAIN_NAMESPACE + "communication_process#CommunicationSession",
    "world:Handover": DOMAIN_NAMESPACE + "communication_process#Handover",
    "world:DataFlow": DOMAIN_NAMESPACE + "communication_process#DataFlow",
    "world:Retransmission": DOMAIN_NAMESPACE + "communication_process#Retransmission",
    "world:ComputeNode": DOMAIN_NAMESPACE + "computation_process#ComputeNode",
    "world:ComputeTask": DOMAIN_NAMESPACE + "computation_process#ComputeTask",
    "world:ComputeQueue": DOMAIN_NAMESPACE + "computation_process#ComputeQueue",
    "world:GnssService": DOMAIN_NAMESPACE + "positioning_navigation#GnssService",
    "world:SpoofingSignal": DOMAIN_NAMESPACE + "positioning_navigation#SpoofingSignal",
    "world:EmergencyIsolationZone": DOMAIN_NAMESPACE + "operational_constraint_emergency#EmergencyIsolationZone",
    "world:AirspaceAuthorization": DOMAIN_NAMESPACE + "utm#AirspaceAuthorization",
    "world:FlightPlan": DOMAIN_NAMESPACE + "utm#FlightPlan",
    "world:OperationalIntent": DOMAIN_NAMESPACE + "utm#OperationalIntent",
    "world:UTMSystem": DOMAIN_NAMESPACE + "utm#UTMSystem",
}


@lru_cache(maxsize=1)
def ontology_authority() -> Graph:
    root = Path(__file__).resolve().parents[3] / "Dataset/knowledge_graph"
    # world_core.ttl imports the core ontology; all domain modules import world.
    # Load this local closure explicitly so its disjointness axioms participate.
    graph = Graph().parse(root / "core.ttl")
    graph.parse(root / "world_core.ttl")
    for path in sorted((root / "domain").glob("*.ttl")):
        if not path.name.endswith(".shacl.ttl"):
            graph.parse(path)
    return graph


def canonical_class_uri(value: str) -> str:
    if value in RUNTIME_CLASS_ALIASES:
        uri = RUNTIME_CLASS_ALIASES[value]
    elif value.startswith("world:"):
        uri = WORLD_NAMESPACE + value.split(":", 1)[1]
    elif value.startswith("aw:"):
        uri = CORE_NAMESPACE + value.split(":", 1)[1]
    elif value.startswith("https://w3id.org/aeroworld/ontology/"):
        uri = value
    else:
        raise ContractError(f"class needs an explicit namespace or alias declaration: {value!r}")
    if (URIRef(uri), RDF.type, OWL.Class) not in ontology_authority():
        raise ContractError(f"class is not declared in authoritative ontology: {uri}")
    return uri


def canonical_class_is_a(actual: str, expected: str) -> bool:
    actual_uri, expected_uri = URIRef(canonical_class_uri(actual)), URIRef(canonical_class_uri(expected))
    agenda, visited = [actual_uri], set()
    while agenda:
        current = agenda.pop()
        if current == expected_uri:
            return True
        if current not in visited:
            visited.add(current)
            agenda.extend(parent for parent in ontology_authority().objects(current, RDFS.subClassOf)
                          if isinstance(parent, URIRef))
    return False


def disjoint_class_pairs(classes: tuple[str, ...], graph: Graph | None = None) -> tuple[tuple[str, str], ...]:
    """Return conflicts entailed by explicit OWL disjoint class declarations.

    Multiple incomparable types alone do not contradict an OWL ontology.
    Disjointness applies to their RDFS subclasses as well.
    """
    graph = ontology_authority() if graph is None else graph
    ancestors = {}
    for value in classes:
        uri = URIRef(canonical_class_uri(value))
        reached, agenda = set(), [uri]
        while agenda:
            current = agenda.pop()
            if current not in reached:
                reached.add(current)
                agenda.extend(parent for parent in graph.objects(current, RDFS.subClassOf)
                              if isinstance(parent, URIRef))
        ancestors[value] = reached
    disjoint = {frozenset((left, right)) for left, right in graph.subject_objects(OWL.disjointWith)}
    for declaration in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        members = graph.value(declaration, OWL.members)
        if members is not None:
            values = tuple(graph.items(members))
            disjoint.update(frozenset((left, right)) for i, left in enumerate(values)
                            for right in values[i+1:])
    return tuple((left, right) for i, left in enumerate(classes) for right in classes[i+1:]
        if any(frozenset((a, b)) in disjoint for a in ancestors[left] for b in ancestors[right]))


class ContractError(ValueError):
    """A declared contract or its grounding is malformed."""


class UnsupportedOperatorError(ContractError):
    """The AST contains an operator without executable semantics."""


@dataclass(frozen=True)
class FieldSchema:
    field: str
    data_type: str
    unit: str
    source: str
    categories: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ArgumentRole:
    key: str
    ontology_class: str
    declared_class_uri: str = ""


@dataclass(frozen=True)
class PredicateContract:
    predicate_id: str
    roles: tuple[ArgumentRole, ...]
    required_fields: tuple[FieldSchema, ...]
    threshold_parameters: tuple[str, ...]
    evaluation_spec: Mapping[str, Any]
    grounding_spec: Mapping[str, Any]
    operators: tuple[str, ...]
    executable: bool
    semantic_mode: str
    physical_formula: str
    semantic_ambiguity: str
    source_refs: tuple[str, ...]
    template: Mapping[str, Any]

    @property
    def id(self) -> str:
        return self.predicate_id


def walk_expression(spec: Mapping[str, Any]):
    """Yield every AST node, including nested boolean operators."""
    if not isinstance(spec, Mapping) or not isinstance(spec.get("op"), str):
        raise ContractError("evaluation node must declare an operator")
    yield spec
    if "args" in spec:
        args = spec["args"]
        if not isinstance(args, (list, tuple)):
            raise ContractError("operator args must be an explicit sequence")
        for child in args:
            yield from walk_expression(child)


def expression_fields(spec: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(node[name]) for node in walk_expression(spec)
        for name in ("field", "left", "right") if name in node
    ))


def _field_type(unit: str) -> str:
    if unit == "bool":
        return "boolean"
    if unit == "four-value":
        return "four_value"
    if unit == "identifier":
        return "identifier"
    if unit == "state-code":
        return "categorical"
    if unit in {"m", "m/s", "aircraft", "vehicle", "ratio", "lux", "degC"}:
        return "numeric"
    raise ContractError(f"schema unit requires a declared type adapter: {unit!r}")


def contract_from_template(template: Mapping[str, Any]) -> PredicateContract:
    """Compile schema metadata without changing source semantics."""
    executable = template["implementation_status"] == "executable_l1"
    roles = []
    aw = Namespace("https://w3id.org/aeroworld/ontology/core#")
    world = Namespace(WORLD_NAMESPACE)
    nodes = tuple(ontology_authority().subjects(aw.predicateIdentifier, Literal(template["id"])))
    declared_roles = {}
    for node in nodes:
        for role_node in ontology_authority().objects(node, aw.declaresArgumentRole):
            declared_roles[str(ontology_authority().value(role_node, aw.argumentRoleKey))] = str(
                ontology_authority().value(role_node, world.argumentClass))
    for role in template["argument_roles"]:
        runtime_class = role["class"] if ":" in role["class"] else f"world:{role['class']}"
        # Nonexec vocabulary can contain classes outside the executable alias
        # map; its full authored class is metadata, never executable grounding.
        declared_uri = declared_roles.get(role["key"])
        if executable:
            canonical = canonical_class_uri(runtime_class)
            if declared_uri is not None and canonical != declared_uri:
                raise ContractError(f"role alias disagrees with exact authored class: {role['key']}")
            declared_uri = canonical if declared_uri is None else declared_uri
        roles.append(ArgumentRole(role["key"], runtime_class, declared_uri or ""))
    roles = tuple(roles)
    source_refs = (f"{REGISTRY_SOURCE}#predicate={template['id']}",
                   f"{CONTRACT_SOURCE}#predicate={template['id']}")
    if not executable:
        # Vocabulary-only declarations have no state schema or evaluation AST.
        # Preserve the explicit reason instead of manufacturing executable data.
        return PredicateContract(template["id"], roles, (), (), {}, {}, (), False,
            "non_executable", "not_declared", "none_declared", source_refs, template)
    spec = template["evaluation_spec"]
    nodes = tuple(walk_expression(spec))
    missing = {node["op"] for node in nodes} - SUPPORTED_OPERATORS
    if executable and missing:
        raise UnsupportedOperatorError(f"unsupported AST operators: {sorted(missing)}")
    state = template["state_contract"]
    schemas = []
    for field in state["required_fields"]:
        path = field["field"]
        category_values = []
        for node in nodes:
            if node.get("field") == path:
                if node["op"] == "equals":
                    category_values.append(node["value"])
                elif node["op"] == "in":
                    category_values.extend(node["values"])
        if field["unit"] == "four-value":
            category_values = list(TRUTH_DOMAIN)
        schemas.append(FieldSchema(path, _field_type(field["unit"]), field["unit"],
                                   field["source"], tuple(dict.fromkeys(category_values))))
    direct = any(node["op"] == "direct_truth" for node in nodes)
    return PredicateContract(
        template["id"], roles, tuple(schemas),
        tuple(state["threshold_parameters"]), spec, template["grounding_spec"],
        tuple(sorted({node["op"] for node in nodes})), executable,
        "direct_observation" if direct else "declared_formula",
        "not_declared" if direct else "declared",
        "producer_contract_requires_review" if direct else "none_declared",
        source_refs, template,
    )


def load_unified_contracts(*, executable_only: bool = False) -> dict[str, PredicateContract]:
    contracts = {template["id"]: contract_from_template(template)
                 for template in get_core_predicate_templates()}
    if len(contracts) != 79 or sum(c.executable for c in contracts.values()) != 72:
        raise ContractError("registry drift: expected fixed 79/72/7 partition")
    return {key: value for key, value in contracts.items()
            if value.executable or not executable_only}


def inventory(contracts: Mapping[str, PredicateContract] | None = None) -> dict[str, Any]:
    contracts = load_unified_contracts() if contracts is None else contracts
    selected = tuple(c for c in contracts.values() if c.executable)
    operators = Counter(node["op"] for contract in selected
                        for node in walk_expression(contract.evaluation_spec))
    return {"declared": len(contracts), "executable": len(selected),
            "non_executable": len(contracts) - len(selected),
            "direct_observation": sum(c.semantic_mode == "direct_observation" for c in selected),
            "operator_node_counts": dict(sorted(operators.items())),
            "executable_predicate_ids": [c.predicate_id for c in selected]}


# Supplemental ownership contract.  Each key is an exact producer source plus
# field source, and each value is a declared role dependency.  The aircraft
# geometry producer computes speed from the aircraft alone, even when it emits
# the value into one row per corridor: predicate_state_computers.py:562-618.
# The following-distance source binds both vehicles: world_truth.py:719-741.
FIELD_OWNER_DECLARATIONS = {
    ("predicate_contract_aircraft_region_geometry", "record:values.restricted_region_active"):
        ("restricted_region",),
    ("predicate_contract_aircraft_region_geometry", "record:values.signed_distance_to_boundary_m"):
        ("actor", "restricted_region"),
    ("predicate_contract_aircraft_protected_region_geometry", "record:values.inside_protected_airspace"):
        ("aircraft", "protected_region"),
    ("predicate_contract_aircraft_corridor_geometry", "record:values.speed_mps"): ("actor",),
    ("predicate_contract_aircraft_corridor_geometry", "record:values.aircraft_inside_corridor"):
        ("actor", "corridor"),
    ("predicate_contract_aircraft_geometry", "context:geometry.home_pad_id"): ("aircraft",),
    ("predicate_contract_aircraft_geometry", "context:geometry.xy_distance_to_home_pad_m"):
        ("aircraft", "home_pad"),
    ("predicate_contract_aircraft_geometry", "context:geometry.assigned_landing_zone_id"): ("aircraft",),
    ("predicate_contract_aircraft_geometry", "context:geometry.xy_distance_to_assigned_landing_zone_m"):
        ("aircraft", "landing_zone"),
    ("truth_frames.jsonl#sumo_vehicle", "context:truth.lane_id"): ("vehicle",),
    ("truth_frames.jsonl#sumo_vehicle", "context:truth.speed_mps"): ("vehicle",),
    ("scene_setup.json#crosswalk_anchor", "context:truth.in_crosswalk"): ("pedestrian",),
    ("truth_frames.jsonl#sumo_vehicle.controlling_signal_id", "context:truth.controlling_signal_id"):
        ("vehicle",),
    # These values describe one vehicle's interval crossing, including the
    # previous signal state; a signal-only owner would change their meaning.
    ("truth_frames.jsonl#sumo_vehicle.stop_line_id", "context:truth.crossed_stop_line"):
        ("vehicle", "stop_line", "signal"),
    ("truth_frames.jsonl#sumo_vehicle.stop_line_id", "context:truth.controlling_signal_state"):
        ("vehicle", "stop_line", "signal"),
    ("truth_frames.jsonl#sumo_vehicle.speed_limit_regulation_id", "context:truth.speed_mps"):
        ("vehicle",),
    ("truth_frames.jsonl#sumo_vehicle.speed_limit_regulation_id", "context:truth.posted_speed_limit_mps"):
        ("regulation",),
    ("truth_frames.jsonl#sumo_vehicle.leading_vehicle_id", "context:truth.following_distance_m"):
        ("following_vehicle", "leading_vehicle"),
    ("predicate_contract_aircraft_pair_geometry", "record:values.pair_distance_m"):
        ("first_aircraft", "second_aircraft"),
    ("predicate_contract_aircraft_structure_geometry", "record:values.distance_m"):
        ("aircraft", "structure"),
    ("predicate_contract_aircraft_vehicle_geometry", "record:values.distance_m"):
        ("aircraft", "vehicle"),
    ("predicate_contract_aircraft_pedestrian_geometry", "record:values.distance_m"):
        ("aircraft", "pedestrian"),
    ("predicate_contract_vehicle_pedestrian_geometry", "record:values.distance_m"):
        ("vehicle", "pedestrian"),
    ("pad_facility:values.contention_pairs", "record:pair_contention"):
        ("pad", "first_aircraft", "second_aircraft"),
    ("gnss_navigation", "record:values.quality_level"): ("actor",),
    ("gnss_navigation", "record:values.gnss_spoofed"): ("actor",),
    # Keep the declared link tuple.  The current producer reads an actor flag,
    # which does not establish the ontology's station-specific link impact.
    # This discrepancy stays visible in the semantic audit; ownership must not
    # silently turn the declared link predicate into an actor-only proposition.
    ("security_command", "record:values.jamming_indicator"): ("actor", "station"),
    ("airspace_authorization_log.jsonl", "record:status"): ("authorization",),
    ("flight_plan_log.jsonl", "record:approval_status"): ("flight_plan",),
    ("operational_intent_pair_log.jsonl", "record:conflict_active"):
        ("first_intent", "second_intent"),
    ("deconfliction_log.jsonl", "record:resolution_status"): ("system", "aircraft"),
}

# Source references for the supplemental declarations are producer code, not
# inferred field-name similarity.  The scope_entity and single-role cases below
# get ownership directly from the frozen grounding contract.
OWNERSHIP_SOURCE_REFS = (
    "Dataset/semantic_simulation/predicate_state_computers.py:395",
    "Dataset/semantic_simulation/predicate_state_computers.py:529",
    "Dataset/semantic_simulation/predicate_state_computers.py:562",
    "Dataset/semantic_simulation/predicate_state_computers.py:635",
    "Dataset/semantic_simulation/predicate_state_computers.py:2361",
    "Dataset/semantic_simulation/domain_state.py:1247",
    "Dataset/semantic_simulation/domain_state.py:1885",
    "Dataset/semantic_simulation/domain_state.py:2856",
    "Dataset/tools/sumo_ground_flow/road_signal_context.py:118",
    "Dataset/semantic_simulation/utm_state.py:175",
    "Dataset/semantic_truth/world_truth.py:697",
    "Dataset/semantic_truth/l0_supplement.py:1170",
)

OWNER_SOURCE_EVIDENCE = {
    "predicate_contract_aircraft_region_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:529",
        "Dataset/semantic_simulation/predicate_state_computers.py:536"),
    "predicate_contract_aircraft_protected_region_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:576",),
    "predicate_contract_aircraft_corridor_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:604",
        "Dataset/semantic_simulation/predicate_state_computers.py:611"),
    "predicate_contract_aircraft_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:395",
        "Dataset/semantic_simulation/predicate_state_computers.py:457"),
    "truth_frames.jsonl#sumo_vehicle": ("Dataset/semantic_truth/world_truth.py:697",),
    "scene_setup.json#crosswalk_anchor": ("Dataset/semantic_truth/world_truth.py:756",
        "Dataset/semantic_truth/l0_supplement.py:1195"),
    "truth_frames.jsonl#sumo_vehicle.controlling_signal_id": (
        "Dataset/tools/sumo_ground_flow/road_signal_context.py:155",),
    "truth_frames.jsonl#sumo_vehicle.stop_line_id": (
        "Dataset/tools/sumo_ground_flow/road_signal_context.py:118",
        "Dataset/semantic_truth/l0_supplement.py:287"),
    "truth_frames.jsonl#sumo_vehicle.speed_limit_regulation_id": (
        "Dataset/tools/sumo_ground_flow/road_signal_context.py:129",),
    "truth_frames.jsonl#sumo_vehicle.leading_vehicle_id": (
        "Dataset/semantic_truth/l0_supplement.py:1170",),
    "predicate_contract_aircraft_pair_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:635",),
    "predicate_contract_aircraft_structure_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:2361",),
    "predicate_contract_aircraft_vehicle_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:2361",),
    "predicate_contract_aircraft_pedestrian_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:2361",),
    "predicate_contract_vehicle_pedestrian_geometry": (
        "Dataset/semantic_simulation/predicate_state_computers.py:2361",),
    "pad_facility:values.contention_pairs": ("Dataset/semantic_simulation/domain_state.py:1885",),
    "gnss_navigation": ("Dataset/semantic_simulation/domain_state.py:1247",
                        "Dataset/semantic_simulation/domain_state.py:3749"),
    "security_command": ("Dataset/semantic_simulation/domain_state.py:2856",
                         "Dataset/semantic_simulation/domain_state.py:2882"),
    "airspace_authorization_log.jsonl": ("Dataset/semantic_simulation/utm_state.py:175",),
    "flight_plan_log.jsonl": ("Dataset/semantic_simulation/utm_state.py:211",),
    "operational_intent_pair_log.jsonl": ("Dataset/semantic_simulation/utm_state.py:254",),
    "deconfliction_log.jsonl": ("Dataset/semantic_simulation/utm_state.py:228",),
}


def field_owner_source_refs(contract: PredicateContract, field: str) -> tuple[str, ...]:
    source = contract.grounding_spec["field_sources"][field]
    authority = contract.grounding_spec["candidate_authority"]["source"]
    if (authority, source) in FIELD_OWNER_DECLARATIONS:
        return OWNER_SOURCE_EVIDENCE[authority] + contract.source_refs
    return contract.source_refs


def canonical_state_field(contract: PredicateContract, field: str) -> str:
    """Collapse only source-proven projections of the same physical state."""
    authority = contract.grounding_spec["candidate_authority"]["source"]
    source = contract.grounding_spec["field_sources"][field]
    if (authority, source) in {
        ("predicate_contract_aircraft_region_geometry", "record:values.restricted_region_active"),
        ("predicate_contract_region_runtime_state", "record:values.restricted_region_active"),
    }:
        return "state.restricted_region_active"
    return field


def field_owner_roles(contract: PredicateContract, field: str) -> tuple[str, ...]:
    """Resolve exact ownership from role authorities and producer declarations.

    Unresolved multi-role fields raise an error.  New contracts can provide
    evidence-backed ownership through grounding_spec.field_owner_roles.
    """
    grounding = contract.grounding_spec
    if field not in grounding["field_sources"]:
        raise ContractError(f"field lacks a source declaration: {field}")
    explicit = grounding.get("field_owner_roles", {})
    if field in explicit:
        owners = tuple(explicit[field])
    else:
        source = grounding["field_sources"][field]
        authority = grounding["candidate_authority"]["source"]
        declaration = FIELD_OWNER_DECLARATIONS.get((authority, source))
        if declaration is not None:
            owners = declaration
        elif source.startswith("join:"):
            parts = source.split(":", 3)
            if len(parts) != 4:
                raise ContractError(f"malformed join source: {source}")
            owners = (parts[2],)
        elif source.startswith("context:"):
            scope_roles = tuple(role for role, value in grounding["role_sources"].items()
                                if value["kind"] == "scope_entity")
            if len(scope_roles) == 1:
                owners = scope_roles
            elif len(contract.roles) == 1:
                owners = (contract.roles[0].key,)
            else:
                raise ContractError(f"field needs explicit ownership: {authority}:{source}")
        elif len(contract.roles) == 1:
            owners = (contract.roles[0].key,)
        else:
            raise ContractError(f"field needs explicit ownership: {authority}:{source}")
    declared = {role.key for role in contract.roles}
    if not owners or len(set(owners)) != len(owners) or not set(owners) <= declared:
        raise ContractError(f"invalid explicit owner roles for {field}: {owners}")
    return owners
