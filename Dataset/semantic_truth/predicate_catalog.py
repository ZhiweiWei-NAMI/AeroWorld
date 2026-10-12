"""Dashboard projection of the ontology-authoritative V3 predicate registry."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .core_semantic_registry import load_core_semantic_registry
from .provenance import stable_identifier


CATALOG_SCHEMA_NAME = "aeroworld_predicate_catalog"
CATALOG_SCHEMA_VERSION = "3.0.0"


def get_predicate_catalog() -> list[dict[str, Any]]:
    """Return one dashboard entry for each of the 79 authoritative predicates."""

    registry = load_core_semantic_registry()
    catalog_id = str(registry["registry_id"])
    catalog_version = str(registry["schema_version"])
    truth_values = list(registry["truth_values"])
    rows: list[dict[str, Any]] = []
    for template in sorted(registry["templates"], key=lambda item: str(item["id"])):
        predicate_id = str(template["id"])
        status = str(template["implementation_status"])
        if status == "executable_l1":
            state_contract = template["state_contract"]
            sources = deepcopy(state_contract["required_fields"])
            rule: dict[str, Any] | None = deepcopy(template["evaluation_spec"])
            unknown_policy: dict[str, Any] | None = {
                "unknown_if_missing": deepcopy(state_contract["unknown_if_missing"]),
                "missing_record_field": state_contract["missing_record_field"],
            }
            threshold_parameters = deepcopy(state_contract["threshold_parameters"])
            non_executable_contract = None
        elif status == "non_executable":
            sources = []
            rule = None
            unknown_policy = None
            threshold_parameters = []
            non_executable_contract = deepcopy(template["non_executable_contract"])
        else:
            raise ValueError(
                f"unsupported predicate implementation status: {predicate_id}:{status}"
            )
        rows.append(
            {
                "schema_name": "aeroworld_predicate_catalog_entry",
                "schema_version": CATALOG_SCHEMA_VERSION,
                "catalog_id": catalog_id,
                "catalog_version": catalog_version,
                "catalog_entry_id": stable_identifier(
                    "predicate_catalog",
                    catalog_id,
                    catalog_version,
                    predicate_id,
                ),
                "api_id": predicate_id,
                "kind": "boolean_predicate",
                "binding_kind": template["tuple_type"],
                "unit": "truth_value",
                "value_range": truth_values,
                "ontology_module": template["ontology_module"],
                "scope_types": deepcopy(template["scope_types"]),
                "argument_roles": deepcopy(template["argument_roles"]),
                "implementation_status": status,
                "sources": sources,
                "rule": rule,
                "expression": template["expression"],
                "unknown_policy": unknown_policy,
                "threshold_parameters": threshold_parameters,
                "non_executable_contract": non_executable_contract,
            }
        )
    return rows


def audit_contract_predicate_references(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare event-family references with the sole V3 predicate vocabulary."""

    families = contract.get("event_families")
    if not isinstance(families, Mapping):
        raise ValueError("contract.event_families must be an object")
    referenced = {
        str(predicate_id)
        for family in families.values()
        if isinstance(family, Mapping)
        for predicate_id in family.get("required_predicates", ())
        if isinstance(predicate_id, str)
    }
    authoritative = {entry["api_id"] for entry in get_predicate_catalog()}
    non_authoritative = sorted(referenced - authoritative)
    return {
        "passed": not non_authoritative,
        "authoritative_predicate_count": len(authoritative),
        "referenced_predicate_count": len(referenced),
        "non_authoritative_predicate_ids": non_authoritative,
    }


__all__ = [
    "CATALOG_SCHEMA_NAME",
    "CATALOG_SCHEMA_VERSION",
    "audit_contract_predicate_references",
    "get_predicate_catalog",
]
