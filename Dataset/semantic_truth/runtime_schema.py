"""Exact runtime-record JSON Schema validation shared by builders and gates."""

from __future__ import annotations

from functools import cache
import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "semantic_rules" / "schema"


@cache
def _validator(schema_file: str) -> Draft202012Validator:
    path = SCHEMA_ROOT / schema_file
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def runtime_schema_errors(
    record: Mapping[str, Any], schema_file: str
) -> tuple[str, ...]:
    """Return stable, path-qualified violations for one current runtime row."""

    errors = sorted(
        _validator(schema_file).iter_errors(record),
        key=lambda item: list(item.absolute_path),
    )
    return tuple(
        f"{''.join(f'[{part!r}]' for part in error.absolute_path) or '$'}: "
        f"{error.message}"
        for error in errors
    )


__all__ = ["runtime_schema_errors"]
