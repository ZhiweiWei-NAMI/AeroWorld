"""Pure schema fragments shared by record grammars with the same producer rule.

The runtime-state producer validates enum and command strings with
``value.strip().lower()`` and preserves the original string.  The patterns
below describe that raw-string domain for the current project Python runtime
(Unicode 13.0.0) and the contract's lowercase ASCII member names.  They use
only escapes supported by both ECMA-262 and Python ``re``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection

from Dataset.tools.runtime_state_source_guard import (
    RUNTIME_STATE_SOURCE_GUARD_POLICIES,
    RuntimeStateSourceGuardPolicy,
)

__all__ = [
    "runtime_enum_string_rule", "nonblank_python_string_rule",
    "stripped_lower_phrase_string_rule",
    "runtime_state_source_guard_rule",
]


_UNICODE_VERSION = "13.0.0"
# Exactly the code points removed by Python 3.10 str.strip() / str.isspace().
# Explicit escapes avoid the different ECMA-262 and Python meanings of \s.
_PYTHON_STRIP_SPACE = (
    r"\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680"
    r"\u2000-\u200a\u2028-\u2029\u202f\u205f\u3000"
)
_SPACE = f"[{_PYTHON_STRIP_SPACE}]"
_ASCII_MEMBER = re.compile(r"[a-z0-9_]+\Z")
_ASCII_PHRASE = re.compile(r"[a-z0-9_]+(?: [a-z0-9_]+)*\Z")


def _require_unicode_contract() -> None:
    if unicodedata.unidata_version != _UNICODE_VERSION:
        raise RuntimeError(
            "runtime enum patterns require a review of Python's Unicode "
            f"mapping: expected {_UNICODE_VERSION}, got "
            f"{unicodedata.unidata_version}"
        )


def _raw_member_pattern(member: str) -> str:
    parts = []
    for character in member:
        if character == "k":
            # Kelvin sign is the sole non-ASCII code point whose lower()
            # equals one ASCII letter under this runtime's Unicode tables.
            parts.append(r"[kK\u212a]")
        elif "a" <= character <= "z":
            parts.append(f"[{character}{character.upper()}]")
        else:
            parts.append(character)
    return "".join(parts)


def runtime_enum_string_rule(allowed: Collection[str]) -> dict[str, str]:
    """Match raw strings whose ``strip().lower()`` is an allowed member.

    ``allowed`` must contain only the current runtime contract's canonical
    lowercase ASCII identifiers.  Command arrays can reuse this string rule
    as their ``items`` schema; the caller owns the scalar/array branch.
    """

    _require_unicode_contract()
    if isinstance(allowed, str):
        raise ValueError("allowed must be a collection of member strings")
    members = tuple(allowed)
    if not members or any(
        type(member) is not str or _ASCII_MEMBER.fullmatch(member) is None
        for member in members
    ):
        raise ValueError("allowed must contain lowercase ASCII identifiers")
    alternatives = "|".join(_raw_member_pattern(member) for member in sorted(set(members)))
    return {
        "type": "string",
        "pattern": f"^{_SPACE}*(?:{alternatives}){_SPACE}*$",
    }


def stripped_lower_phrase_string_rule(allowed: Collection[str]) -> dict[str, str]:
    """Match raw strings whose ``strip().lower()`` is an ASCII phrase.

    Canonical members may contain single internal ASCII spaces. The runtime
    enum helper retains its stricter identifier-only input contract.
    """

    _require_unicode_contract()
    if isinstance(allowed, str):
        raise ValueError("allowed must be a collection of member strings")
    members = tuple(allowed)
    if not members or any(
        type(member) is not str or _ASCII_PHRASE.fullmatch(member) is None
        for member in members
    ):
        raise ValueError("allowed must contain lowercase ASCII words or phrases")
    alternatives = "|".join(_raw_member_pattern(member) for member in sorted(set(members)))
    return {
        "type": "string",
        "pattern": f"^{_SPACE}*(?:{alternatives}){_SPACE}*$",
    }


def nonblank_python_string_rule() -> dict[str, str]:
    """Match strings accepted by the producer's ``bool(value.strip())``."""

    _require_unicode_contract()
    return {"type": "string", "pattern": f"[^{_PYTHON_STRIP_SPACE}]"}


def runtime_state_source_guard_rule(
    policy: RuntimeStateSourceGuardPolicy,
) -> dict[str, dict[str, str]]:
    """Declare the project validator's guard on one consumed object subtree."""

    if policy not in RUNTIME_STATE_SOURCE_GUARD_POLICIES:
        raise ValueError(f"unsupported runtime-state source guard policy: {policy!r}")
    return {"x-runtime-state-source-guard": {"policy": policy}}
