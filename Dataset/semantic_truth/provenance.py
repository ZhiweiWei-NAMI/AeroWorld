"""Canonical serialization, deterministic identifiers, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import EvalResult, SemanticCompileError


# Integrity metadata emitted by the semantic producers. These exact members
# are omitted at publication; source paths, rule identifiers and values remain.
INTEGRITY_MEMBERS = frozenset({
    "sha256", "input_digest", "parameter_digest", "seed_digest", "rule_digest",
    "base_graph_digest", "delta_digest", "graph_digest", "state_digest",
    "record_digest", "replayed_state_digest", "source_state_digest",
    "source_digest", "rule_digests", "parameter_digests", "input_digests",
    "stage_contract_digest", "candidate_match_digest", "closure_digest",
    "manifest_digest", "domain_input_digest", "context_input_digest",
    "global_detection_record_digests", "resolved_source_sha256",
    "sumo_roster_never_active_digest", "render_vehicle_authority_conflict_digest",
    "l0_state_profile_sha256", "sumo_authority_digest", "runtime_schedule_sha256",
    "event_fire_tick_digest", "multiset_digest", "trajectory_state_digest",
    "restricted_region_activity_digest", "geometry_input_digest",
    "predicate_state_input_digest", "profile_digest", "raw_input_digest",
    "source_scene_setup_digest", "entity_roster_digest", "charging_service_plan_digest",
    "source_digest_multiset_by_log",
})


def without_integrity_metadata(value: Any) -> Any:
    """Project producer records for publication without changing source values."""
    if isinstance(value, Mapping):
        return {
            key: without_integrity_metadata(item)
            for key, item in value.items() if key not in INTEGRITY_MEMBERS
        }
    if isinstance(value, (list, tuple)):
        return [without_integrity_metadata(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def digest_object(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def stable_identifier(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SemanticCompileError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SemanticCompileError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise SemanticCompileError(
                    f"{path}:{line_number}: JSONL record must be an object"
                )
            yield value


def unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def merge_results(value: Any, results: Iterable[EvalResult]) -> EvalResult:
    refs: list[str] = []
    observations: list[tuple[str, Any]] = []
    missing: list[str] = []
    for result in results:
        refs.extend(result.source_refs)
        observations.extend(result.observations)
        missing.extend(result.missing)
    observation_map = {
        (path, canonical_json(observation_value)): (path, observation_value)
        for path, observation_value in observations
    }
    return EvalResult(
        value=value,
        source_refs=unique_sorted(refs),
        observations=tuple(
            observation_map[key] for key in sorted(observation_map)
        ),
        missing=unique_sorted(missing),
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(without_integrity_metadata(row)))
            handle.write("\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            without_integrity_metadata(value),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
