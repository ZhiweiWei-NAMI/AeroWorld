"""Pure recursive source guard for validated runtime-state subtrees.

The episode/render producers use ``engine_whole``. The scenario regenerator
uses ``regenerate_split``, whose ASCII tokenization and split-string check are
intentionally different. Callers choose the policy for the subtree they
actually consume; this guard does not validate runtime-state field values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal


RuntimeStateSourceGuardPolicy = Literal["engine_whole", "regenerate_split"]
RUNTIME_STATE_SOURCE_GUARD_POLICIES = frozenset({"engine_whole", "regenerate_split"})

_ENGINE_FORBIDDEN_FIELDS = frozenset({
    "event_id", "source_event_id", "event_realization", "semantic_role",
    "task_id", "event_type_id", "predicate_id", "predicate",
    "expected_event", "dynamic_label", "dynamic_labels", "event_trace",
    "active_event_ids",
})
_REGENERATE_FORBIDDEN_FIELDS = _ENGINE_FORBIDDEN_FIELDS | {"scenario_plan"}


def _normalized_token(value: Any, policy: RuntimeStateSourceGuardPolicy) -> str:
    text = str(value or "").strip().casefold()
    if policy == "engine_whole":
        return "".join(character for character in text if character.isalnum())
    return re.sub(r"[^a-z0-9]+", "", text)


_FORBIDDEN_TOKENS = {
    "engine_whole": frozenset(
        _normalized_token(field, "engine_whole") for field in _ENGINE_FORBIDDEN_FIELDS
    ),
    "regenerate_split": frozenset(
        _normalized_token(field, "regenerate_split")
        for field in _REGENERATE_FORBIDDEN_FIELDS
    ),
}


def _string_is_path(value: str) -> bool:
    text = str(value or "").strip()
    if ".json" in text.casefold():
        return True
    if not text:
        return False
    if "\\" in text or "/" in text or "://" in text:
        return True
    return len(text) >= 3 and text[0].isalpha() and text[1] == ":"


def _string_has_forbidden_token(
    value: str, policy: RuntimeStateSourceGuardPolicy, forbidden: frozenset[str]
) -> bool:
    if _normalized_token(value, policy) in forbidden:
        return True
    if policy == "regenerate_split":
        return any(
            _normalized_token(token, policy) in forbidden
            for token in re.split(r"[\s,;:=]+", str(value or ""))
        )
    return False


def forbidden_runtime_state_paths(
    value: Any, *, policy: RuntimeStateSourceGuardPolicy, path: str = ""
) -> list[str]:
    """Return forbidden key/value paths in traversal order for one subtree."""

    if policy not in RUNTIME_STATE_SOURCE_GUARD_POLICIES:
        raise ValueError(f"unsupported runtime-state source guard policy: {policy!r}")
    forbidden = _FORBIDDEN_TOKENS[policy]
    paths: list[str] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                key_text = str(key)
                child_path = f"{item_path}.{key_text}" if item_path else key_text
                if _normalized_token(key_text, policy) in forbidden:
                    paths.append(child_path)
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")
        elif isinstance(item, str):
            if _string_has_forbidden_token(item, policy, forbidden) or _string_is_path(item):
                paths.append(item_path or "<string_value>")

    walk(value, path)
    return paths
