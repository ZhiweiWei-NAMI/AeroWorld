"""Shared semantic truth model types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "1.0.0"
TRUTH_TRUE = "true"
TRUTH_FALSE = "false"
TRUTH_UNKNOWN = "unknown"
TRUTH_OUT_OF_SCOPE = "out_of_scope"
BOOLEAN_TRUTHS = {TRUTH_TRUE, TRUTH_FALSE}
BREAK_TRUTHS = {TRUTH_UNKNOWN, TRUTH_OUT_OF_SCOPE}
MISSING = object()


class SemanticCompileError(ValueError):
    """Raised when rules or authoritative inputs cannot be compiled safely."""


@dataclass(frozen=True)
class EvalResult:
    value: Any
    source_refs: tuple[str, ...] = ()
    observations: tuple[tuple[str, Any], ...] = ()
    missing: tuple[str, ...] = ()

    def with_value(self, value: Any) -> "EvalResult":
        return EvalResult(
            value=value,
            source_refs=self.source_refs,
            observations=self.observations,
            missing=self.missing,
        )


@dataclass
class TickContext:
    tick: int
    tick_present: bool
    episode_id: str
    entities: dict[str, dict[str, Any]]
    frame_entities: dict[str, dict[str, Any]]
    roster_entities: dict[str, dict[str, Any]]
    static_entities: dict[str, dict[str, Any]]
    truth_frames_name: str
    roster_name: str
    static_name: str
