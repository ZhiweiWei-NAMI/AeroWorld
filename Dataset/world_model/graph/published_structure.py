"""Compose closed schemas with the semantic producer's publication projection."""

from copy import deepcopy
from typing import Any

from Dataset.semantic_truth.provenance import INTEGRITY_MEMBERS


def published_record_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove exactly the fields omitted by without_integrity_metadata.

    This composes an existing producer schema; it does not infer types from
    records. Raw ARM grammars keep their complete frozen source structure.
    """
    result = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                for member in INTEGRITY_MEMBERS:
                    properties.pop(member, None)
            if isinstance(value.get("required"), list):
                value["required"] = [member for member in value["required"]
                                     if member not in INTEGRITY_MEMBERS]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result)
    return result
