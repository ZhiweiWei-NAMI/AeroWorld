"""Bind the explicit corridor rule manifest to exact L0 identities."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdflib import Graph, Literal, Namespace
import yaml

from .contracts import (
    EdgeProvenance, FactorTemplate, GroundFactor, GroundedSlice, GroundingError,
    PredicateEvidence, RoleBinding, StateKey, UnsupportedGroundingError,
    Variable, VariableKind,
)

PREDICATE_ORDER = (
    "airspace.aircraft_occupies_airspace_corridor",
    "airspace.aircraft_moving_in_corridor",
    "airspace.aircraft_stationary_in_corridor",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AW = Namespace("https://w3id.org/aeroworld/ontology/core#")
_WORLD = Namespace("https://w3id.org/aeroworld/ontology/world#")
_DOMAIN = Namespace("https://w3id.org/aeroworld/ontology/domain/airspace_operations#")


def _at_path(record: Mapping[str, Any], path: str):
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise GroundingError(f"missing declared grounding field: {path}")
        value = value[part]
    return value


def _identity(value, name):
    if not isinstance(value, str) or not value or "|" in value:
        raise GroundingError(f"{name} must be an explicit nonempty entity identity")
    return value


def _number(value, name, *, unknown_allowed=False):
    if value is None and unknown_allowed:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise GroundingError(f"{name} must be finite numeric data")
    if value < 0:
        raise GroundingError(f"{name} must be nonnegative")
    return float(value)


def _refs(values):
    if not isinstance(values, (tuple, list)) or any(not isinstance(ref, str) or not ref for ref in values):
        raise GroundingError("source_refs must contain nonempty source references")
    return tuple(dict.fromkeys(values))


def _line_refs(repo_root, relative_path, marker):
    lines = (repo_root / relative_path).read_text().splitlines()
    matches = tuple(f"{relative_path}:{i}" for i, line in enumerate(lines, 1) if marker(line))
    if not matches:
        raise GroundingError(f"authority source marker is absent in {relative_path}")
    return matches


def load_manifest(repo_root: Path | None = None) -> dict:
    """Load explicit mappings and fail if YAML or ontology authority has drifted."""
    repo_root = _REPO_ROOT if repo_root is None else Path(repo_root)
    manifest = json.loads((repo_root / "Dataset/world_model/rule_factor_graph/manifest.json").read_text())
    if manifest["schema"] != "aeroworld.rule-factor-graph.phase1.v1":
        raise GroundingError("unsupported rule graph manifest schema")
    if tuple(manifest["predicate_order"]) != PREDICATE_ORDER:
        raise GroundingError("manifest predicate order differs from the declared O,M,S scope")
    contract_path = manifest["authority"]["predicate_contract"]
    ontology_path = manifest["authority"]["ontology"]
    contracts = yaml.safe_load((repo_root / contract_path).read_text())
    ontology = Graph().parse(repo_root / ontology_path, format="turtle")
    role_classes = tuple((role, spec["ontology_class"]) for role, spec in manifest["roles"].items())
    if role_classes != (("actor", "world:UnmannedAircraft"), ("corridor", "world:AirspaceCorridor")):
        raise GroundingError("manifest changes the declared actor/corridor ontology roles")
    expected_owners = {
        "geometry.aircraft_inside_corridor": ["actor", "corridor"],
        "geometry.speed_mps": ["actor"],
    }
    if {key: value["owner_roles"] for key, value in manifest["state_fields"].items()} != expected_owners:
        raise GroundingError("manifest state field ownership differs from the shared-state contract")
    if manifest["threshold"] != {
        "field": "aircraft_stationary_speed_threshold_mps",
        "summary_path": "governed_parameters.aircraft_stationary_speed_threshold_mps",
    }:
        raise GroundingError("manifest must read the governed stationary threshold from summary")
    templates = manifest["templates"]
    if tuple(spec["predicate_id"] for spec in templates) != PREDICATE_ORDER:
        raise GroundingError("manifest templates do not match predicate order")
    for spec in templates:
        predicate_id = spec["predicate_id"]
        if predicate_id not in contracts["predicate_execution"]["executable_l1_predicate_ids"]:
            raise UnsupportedGroundingError(f"predicate is not executable: {predicate_id}")
        rule = contracts["predicates"][predicate_id]
        grounding = contracts["grounding_specs"][predicate_id]
        if grounding["candidate_authority"] != manifest["authority"]["candidate"]:
            raise GroundingError(f"candidate authority differs for {predicate_id}")
        expected_roles = {
            role: {"kind": "record", "path": value["record_path"]}
            for role, value in manifest["roles"].items()
        }
        if grounding["role_sources"] != expected_roles:
            raise GroundingError(f"role source mapping differs for {predicate_id}")
        field_sources = {
            field: f"record:{manifest['state_fields'][field]['record_path']}"
            for field in spec["input_fields"] if field in manifest["state_fields"]
        }
        if grounding["field_sources"] != field_sources or rule["evaluation"] != spec["evaluation"]:
            raise GroundingError(f"executable field/evaluation contract differs for {predicate_id}")
        fields = tuple(field[0] for field in rule["fields"])
        threshold_fields = tuple(rule.get("thresholds", ()))
        if fields + threshold_fields != tuple(spec["input_fields"]):
            raise GroundingError(f"template dependencies differ for {predicate_id}")
        node = _DOMAIN[spec["ontology_node"]]
        if (node, _AW.predicateIdentifier, Literal(predicate_id)) not in ontology:
            raise GroundingError(f"manifest ontology node does not declare {predicate_id}")
        actual_roles = {}
        for role_node in ontology.objects(node, _AW.declaresArgumentRole):
            key = ontology.value(role_node, _AW.argumentRoleKey)
            klass = ontology.value(role_node, _WORLD.argumentClass)
            if key is None or klass is None:
                raise GroundingError(f"ontology role lacks exact class provenance for {predicate_id}")
            actual_roles[str(key)] = str(klass)
        expected_classes = {role: str(_WORLD[klass.removeprefix("world:")]) for role, klass in role_classes}
        if actual_roles != expected_classes:
            raise GroundingError(f"ontology argument classes differ for {predicate_id}")
        spec["source_refs"] = list(_line_refs(
            repo_root, contract_path, lambda line, p=predicate_id: line.startswith(f"  {p}:")))
        spec["source_refs"] += list(_line_refs(
            repo_root, ontology_path, lambda line, n=spec["ontology_node"]: line == f"dom:{n}"))
    return manifest


def _build_graph(grounded: GroundedSlice, manifest: dict):
    bindings = (RoleBinding("actor", grounded.actor_id, grounded.actor_class),
                RoleBinding("corridor", grounded.corridor_id, grounded.corridor_class))
    sources = dict(grounded.variable_source_refs)
    fields = tuple(manifest["state_fields"]) + (manifest["threshold"]["field"],) + PREDICATE_ORDER
    if len(sources) != len(grounded.variable_source_refs) or set(sources) != set(fields):
        raise GroundingError("slice must carry source references for every exact variable field")
    keys = {}
    variables = []
    values = {"geometry.aircraft_inside_corridor": grounded.inside,
              "geometry.speed_mps": grounded.speed_mps}
    for field, spec in manifest["state_fields"].items():
        owner_bindings = tuple(binding for binding in bindings if binding.role in spec["owner_roles"])
        owner = "|".join(binding.entity_id for binding in owner_bindings)
        key = StateKey(owner, field, grounded.episode_id, grounded.tick, grounded.branch_id)
        keys[field] = key
        variables.append(Variable(key, VariableKind.STATE, values[field], values[field] is not None,
                                  owner_bindings, sources[field]))
    threshold = manifest["threshold"]["field"]
    keys[threshold] = StateKey(grounded.episode_id, threshold, grounded.episode_id,
                               grounded.tick, grounded.branch_id)
    variables.append(Variable(keys[threshold], VariableKind.EXOGENOUS, grounded.tau_mps, True, (), sources[threshold]))
    factors, edges = [], []
    role_classes = tuple((binding.role, binding.ontology_class) for binding in bindings)
    for i, spec in enumerate(manifest["templates"]):
        predicate = spec["predicate_id"]
        key = StateKey(grounded.owner_id, predicate, grounded.episode_id, grounded.tick, grounded.branch_id)
        variables.append(Variable(key, VariableKind.PREDICATE, grounded.predicate_values[i],
                                  grounded.predicate_observed[i], bindings, sources[predicate]))
        template = FactorTemplate(spec["template_id"], spec["factor_type"], predicate,
                                  tuple(spec["input_fields"]), role_classes, tuple(spec["source_refs"]),
                                  tuple(tuple(manifest["state_fields"][field]["owner_roles"])
                                        if field in manifest["state_fields"] else ()
                                        for field in spec["input_fields"]))
        inputs = tuple(keys[field] for field in spec["input_fields"])
        factor_id = f"{template.template_id}|{grounded.episode_id}|{grounded.tick}|{grounded.branch_id}|{grounded.owner_id}"
        factor = GroundFactor(factor_id, template, inputs, key, bindings, template.source_refs)
        factors.append(factor)
        edges.extend(EdgeProvenance(source, key, factor_id,
                                   tuple(dict.fromkeys(sources[source.field] + template.source_refs)))
                     for source in inputs)
    return tuple(variables), tuple(factors), tuple(edges)


def validate_grounded_slice(grounded: GroundedSlice) -> None:
    """Check complete graph dependencies, exact owners, classes, times and values."""
    if not isinstance(grounded, GroundedSlice):
        raise GroundingError("inference requires a typed GroundedSlice")
    if not grounded.applicable:
        raise UnsupportedGroundingError("out_of_scope is separate from Boolean predicate truth")
    manifest = load_manifest()
    if grounded.actor_class != manifest["roles"]["actor"]["ontology_class"] or grounded.corridor_class != manifest["roles"]["corridor"]["ontology_class"]:
        raise GroundingError("slice ontology classes differ from declared role classes")
    if grounded.inside is not None and type(grounded.inside) is not bool:
        raise GroundingError("inside corridor must be Boolean or explicit unknown")
    _number(grounded.speed_mps, "speed_mps", unknown_allowed=True)
    _number(grounded.tau_mps, "tau_mps")
    if not _refs(grounded.source_refs):
        raise GroundingError("grounded slice requires source provenance")
    expected_authority = tuple(dict.fromkeys(ref for spec in manifest["templates"] for ref in spec["source_refs"]))
    if grounded.authority_source_refs != expected_authority:
        raise GroundingError("slice authority references differ from exact executable rule sources")
    for _field, refs in grounded.variable_source_refs:
        _refs(refs)
    all_variable_refs = {ref for _field, refs in grounded.variable_source_refs for ref in refs}
    if set(grounded.source_refs) != all_variable_refs | set(grounded.authority_source_refs):
        raise GroundingError("slice source references differ from declared variable/authority sources")
    expected = _build_graph(grounded, manifest)
    if (grounded.variables, grounded.factors, grounded.edges) != expected:
        raise GroundingError("typed graph differs from exact rule/state owner dependencies")
    grounded._validate_graph()


def ground_corridor_stop_slice(
    record: Mapping[str, Any], summary: Mapping[str, Any], *,
    expected_episode_id: str | None = None, expected_actor_id: str | None = None,
    expected_corridor_id: str | None = None, expected_tick: int | None = None,
    branch_id: str = "factual", record_source: str | None = None,
    summary_source: str | None = None,
    predicate_evidence: Sequence[PredicateEvidence] | None = None,
) -> GroundedSlice:
    """Ground one exact L0 actor/corridor record; absent predicates stay unknown.

    Source strings may include ``path:line``. A measured NaN is invalid; only
    explicit null state fields represent unknown samples. Boolean predicate
    observations, when supplied, carry their own exact typed role bindings.
    """
    manifest = load_manifest()
    if record.get("schema_name") != "domain_state_observation" or record.get("annotation_layer") != "L0":
        raise GroundingError("grounding requires the declared L0 domain observation schema")
    if record.get("observation_family") != manifest["authority"]["candidate"]["source"]:
        raise UnsupportedGroundingError("record is outside the declared corridor candidate authority")
    episode = _identity(_at_path(record, "episode_id"), "episode_id")
    if summary.get("episode_id") != episode:
        raise GroundingError("summary and record episode identities differ")
    tick = _at_path(record, "tick")
    if type(tick) is not int or tick < 0:
        raise GroundingError("record tick must be a nonnegative integer")
    _identity(branch_id, "branch_id")
    if "branch_id" in record:
        if record["branch_id"] != branch_id:
            raise GroundingError("record branch differs from the requested branch")
    elif branch_id != "factual":
        raise UnsupportedGroundingError("nonfactual grounding requires explicit matching record branch provenance")
    bindings = []
    for role, spec in manifest["roles"].items():
        entity = _identity(_at_path(record, spec["record_path"]), role)
        klass = _at_path(record, spec["class_path"])
        if klass != spec["ontology_class"]:
            raise GroundingError(f"{role} class must be exactly {spec['ontology_class']}, got {klass!r}")
        bindings.append(RoleBinding(role, entity, klass))
    actor, corridor = bindings
    owner = f"{actor.entity_id}|{corridor.entity_id}"
    if _at_path(record, "subject_id") != owner:
        raise GroundingError("record subject owner differs from exact actor/corridor bindings")
    if "owner_id" in record and record["owner_id"] != owner:
        raise GroundingError("record owner_id differs from exact actor/corridor bindings")
    for name, expected, actual in (
        ("episode", expected_episode_id, episode), ("actor", expected_actor_id, actor.entity_id),
        ("corridor", expected_corridor_id, corridor.entity_id), ("tick", expected_tick, tick),
    ):
        if expected is not None and expected != actual:
            raise GroundingError(f"requested {name} identity differs from the source record")
    inside = _at_path(record, manifest["state_fields"]["geometry.aircraft_inside_corridor"]["record_path"])
    if inside is not None and type(inside) is not bool:
        raise GroundingError("inside corridor must be Boolean or explicit unknown")
    speed = _number(_at_path(record, manifest["state_fields"]["geometry.speed_mps"]["record_path"]),
                    "speed_mps", unknown_allowed=True)
    tau = _number(_at_path(summary, manifest["threshold"]["summary_path"]), "tau_mps")
    record_refs = _refs(_at_path(record, "source_refs"))
    for source in (record_source, summary_source):
        if source is not None:
            if not isinstance(source, str) or not source:
                raise GroundingError("explicit source path references must be nonempty strings")
    variable_sources = {}
    for field, spec in manifest["state_fields"].items():
        field_ref = () if record_source is None else (f"{record_source}#field={spec['record_path']}",)
        variable_sources[field] = tuple(dict.fromkeys(record_refs + field_ref))
    threshold = manifest["threshold"]
    variable_sources[threshold["field"]] = (() if summary_source is None else
        (f"{summary_source}#{threshold['summary_path']}",))
    for spec in manifest["templates"]:
        variable_sources[spec["predicate_id"]] = tuple(spec["source_refs"])
    authority_refs = tuple(dict.fromkeys(ref for spec in manifest["templates"] for ref in spec["source_refs"]))
    predicate_values = [None, None, None]
    seen = set()
    for evidence in (() if predicate_evidence is None else predicate_evidence):
        if not isinstance(evidence, PredicateEvidence):
            raise GroundingError("predicate evidence must carry typed identities and role bindings")
        if evidence.predicate_id not in PREDICATE_ORDER:
            raise UnsupportedGroundingError(f"predicate outside declared scope: {evidence.predicate_id}")
        if evidence.predicate_id in seen:
            raise GroundingError("duplicate predicate evidence")
        seen.add(evidence.predicate_id)
        if (evidence.episode_id, evidence.tick, evidence.branch_id) != (episode, tick, branch_id):
            raise GroundingError("predicate evidence crosses episode, tick, or branch identities")
        if evidence.bindings != tuple(bindings):
            raise GroundingError("predicate evidence role identities/classes differ from the L0 record")
        if evidence.value == "out_of_scope":
            raise UnsupportedGroundingError("out_of_scope cannot be encoded as false or unknown")
        if evidence.value not in {"true", "false", "unknown"}:
            raise GroundingError("predicate evidence requires explicit true/false/unknown status")
        value = None if evidence.value == "unknown" else evidence.value == "true"
        predicate_values[PREDICATE_ORDER.index(evidence.predicate_id)] = value
        variable_sources[evidence.predicate_id] = tuple(dict.fromkeys(
            variable_sources[evidence.predicate_id] + _refs(evidence.source_refs)))
    values = tuple(predicate_values)
    refs = tuple(dict.fromkeys(ref for variable_refs in variable_sources.values() for ref in variable_refs))
    grounded = GroundedSlice(
        episode, branch_id, tick, actor.entity_id, corridor.entity_id, owner,
        actor.ontology_class, corridor.ontology_class, values,
        tuple(value is not None for value in values), inside, speed, tau, True,
        refs, variable_source_refs=tuple(variable_sources.items()), authority_source_refs=authority_refs,
    )
    variables, factors, edges = _build_graph(grounded, manifest)
    complete = GroundedSlice(
        episode, branch_id, tick, actor.entity_id, corridor.entity_id, owner,
        actor.ontology_class, corridor.ontology_class, values,
        tuple(value is not None for value in values), inside, speed, tau, True,
        grounded.source_refs, variables, factors, edges,
        grounded.variable_source_refs, grounded.authority_source_refs,
    )
    validate_grounded_slice(complete)
    return complete
