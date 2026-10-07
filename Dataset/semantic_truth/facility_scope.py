"""Generation-time authority and ontology applicability for facility scopes."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from rdflib import Graph, URIRef
from rdflib.namespace import RDFS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "facility_scope_contract.json"
)
DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "schema"
    / "facility_scope_contract.schema.json"
)
WORLD_NAMESPACE = "https://w3id.org/aeroworld/ontology/world#"
FACILITY_SOURCE_CATEGORIES = frozenset(
    {"facility", "ground_station", "airspace_constraint", "hazard_zone"}
)


class FacilityScopeError(ValueError):
    """Raised when a facility lacks one unambiguous semantic subtype."""


@lru_cache(maxsize=1)
def load_facility_scope_contract() -> dict[str, Any]:
    contract = json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8-sig"))
    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8-sig"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(contract),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(item) for item in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise FacilityScopeError(f"facility scope contract is invalid: {details}")
    subtypes = contract["subtypes"]
    if set(subtypes) != {
        "charging_station",
        "landing_pad",
        "communication_base_station",
        "ground_control_station",
        "radio_tower",
        "no_fly_zone",
        "barrier",
    }:
        raise FacilityScopeError(
            "facility scope contract must declare exactly seven subtypes"
        )
    asset_owners: dict[str, str] = {}
    kind_owners: dict[str, str] = {}
    for subtype, spec in subtypes.items():
        service_kind = str(spec["service_kind"])
        capacity = spec.get("service_capacity")
        if service_kind in {"charging", "landing"} and not isinstance(capacity, int):
            raise FacilityScopeError(
                f"{subtype} must declare an integer service_capacity"
            )
        if service_kind == "none" and capacity is not None:
            raise FacilityScopeError(f"{subtype} cannot declare service_capacity")
        for asset_id in spec["source_asset_ids"]:
            previous = asset_owners.setdefault(str(asset_id), subtype)
            if previous != subtype:
                raise FacilityScopeError(
                    f"asset {asset_id} maps to multiple facility subtypes"
                )
        for entity_kind in spec["source_entity_kinds"]:
            previous = kind_owners.setdefault(str(entity_kind), subtype)
            if previous != subtype:
                raise FacilityScopeError(
                    f"entity kind {entity_kind} maps to multiple facility subtypes"
                )
    contract["asset_subtype_index"] = asset_owners
    contract["entity_kind_subtype_index"] = kind_owners
    return contract


def semantic_scope_for_source(
    *,
    category: str,
    asset_id: str,
    declared_subtype: str | None = None,
) -> dict[str, Any] | None:
    """Create the explicit roster identity at the authoritative generation step."""

    normalized_category = str(category).strip().lower()
    if normalized_category not in FACILITY_SOURCE_CATEGORIES:
        return None
    contract = load_facility_scope_contract()
    subtype = str(declared_subtype or "").strip()
    if not subtype:
        subtype = str(contract["asset_subtype_index"].get(str(asset_id), ""))
    if not subtype:
        raise FacilityScopeError(
            f"facility source category={category!r} asset_id={asset_id!r} lacks an explicit subtype"
        )
    spec = contract["subtypes"].get(subtype)
    if not isinstance(spec, Mapping):
        raise FacilityScopeError(f"unknown facility scope subtype: {subtype!r}")
    allowed_assets = {str(value) for value in spec["source_asset_ids"]}
    if allowed_assets and str(asset_id) not in allowed_assets:
        raise FacilityScopeError(
            f"asset {asset_id!r} is not declared for facility subtype {subtype!r}"
        )
    if spec.get("requires_explicit_declaration") is True and not declared_subtype:
        raise FacilityScopeError(
            f"facility subtype {subtype!r} requires an explicit source declaration"
        )
    result: dict[str, Any] = {
        "scope_type": "facility",
        "scope_subtype": subtype,
        "ontology_class_id": str(spec["ontology_class_id"]),
        "authority_id": str(contract["contract_id"]),
    }
    if "service_capacity" in spec:
        result["service_capacity"] = int(spec["service_capacity"])
    return result


def canonical_entity_kind_for_subtype(subtype: str) -> str:
    contract = load_facility_scope_contract()
    spec = contract["subtypes"].get(str(subtype))
    if not isinstance(spec, Mapping):
        raise FacilityScopeError(f"unknown facility scope subtype: {subtype!r}")
    kinds = [str(value) for value in spec["source_entity_kinds"]]
    if len(kinds) != 1:
        raise FacilityScopeError(
            f"facility subtype {subtype!r} must have exactly one canonical entity kind"
        )
    return kinds[0]


def validate_roster_facility_scope(entity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an already materialized roster identity without inferring it."""

    category = str(
        entity.get("entity_category") or entity.get("category") or ""
    ).lower()
    if category not in {"facility", "ground_station"}:
        raise FacilityScopeError(
            f"entity {entity.get('entity_id')!r} is not a facility source category"
        )
    raw_scope = entity.get("semantic_scope")
    if not isinstance(raw_scope, Mapping):
        raise FacilityScopeError(
            f"facility entity {entity.get('entity_id')!r} lacks semantic_scope"
        )
    contract = load_facility_scope_contract()
    subtype = raw_scope.get("scope_subtype")
    spec = contract["subtypes"].get(subtype)
    if not isinstance(spec, Mapping):
        raise FacilityScopeError(
            f"facility entity {entity.get('entity_id')!r} has invalid scope_subtype {subtype!r}"
        )
    expected = {
        "scope_type": "facility",
        "scope_subtype": str(subtype),
        "ontology_class_id": str(spec["ontology_class_id"]),
        "authority_id": str(contract["contract_id"]),
    }
    if "service_capacity" in spec:
        expected["service_capacity"] = int(spec["service_capacity"])
    if dict(raw_scope) != expected:
        raise FacilityScopeError(
            f"facility entity {entity.get('entity_id')!r} semantic_scope differs from authority: "
            f"expected={expected}, actual={dict(raw_scope)}"
        )
    entity_kind = str(entity.get("entity_kind") or entity.get("entity_type") or "")
    allowed_kinds = {str(value) for value in spec["source_entity_kinds"]}
    if entity_kind not in allowed_kinds:
        raise FacilityScopeError(
            f"facility entity {entity.get('entity_id')!r} kind {entity_kind!r} is not valid "
            f"for subtype {subtype!r}"
        )
    return expected


@lru_cache(maxsize=1)
def load_world_ontology() -> Graph:
    graph = Graph()
    graph.parse(PROJECT_ROOT / "Dataset" / "knowledge_graph" / "world_core.ttl")
    return graph


def compact_world_class_uri(class_id: str) -> URIRef:
    value = str(class_id)
    if value.startswith("world:"):
        return URIRef(WORLD_NAMESPACE + value.split(":", 1)[1])
    return URIRef(value)


def ontology_class_is_a(actual_class_id: str, expected_class_id: str) -> bool:
    """Return exact/subclass compatibility using the authoritative world ontology."""

    actual = compact_world_class_uri(actual_class_id)
    expected = compact_world_class_uri(expected_class_id)
    agenda = [actual]
    visited: set[URIRef] = set()
    graph = load_world_ontology()
    while agenda:
        current = agenda.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == expected:
            return True
        agenda.extend(
            parent
            for parent in graph.objects(current, RDFS.subClassOf)
            if isinstance(parent, URIRef)
        )
    return False
