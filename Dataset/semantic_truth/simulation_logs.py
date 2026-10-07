"""Build the eight first-class V7 deterministic producer logs.

The V7 logs are a source-ledger layer. They prove that every governed producer
row was materialized with stable provenance, but they do not project predicate
truth. Grounded L1 predicate truth is owned by ``world_truth.py`` and validated
by the V3 world-truth validator.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence

from Dataset.semantic_truth.provenance import canonical_json, digest_object, stable_identifier


FORMAL_TICKS = tuple(range(0, 901, 5))
TRUTH_VALUES = {"true", "false", "unknown", "out_of_scope"}
LOG_FILE_NAMES = (
    "compute_log.jsonl",
    "communication_log.jsonl",
    "charging_log.jsonl",
    "battery_log.jsonl",
    "utm_log.jsonl",
    "gnss_log.jsonl",
    "weather_log.jsonl",
    "facility_log.jsonl",
)


class SimulationLogError(ValueError):
    """Raised when a simulation log cannot be reconciled to its state source."""


def build_simulation_logs(
    *,
    episode_id: str,
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    compute_predicate_rows: Sequence[Mapping[str, Any]],
    domain_rows: Sequence[Mapping[str, Any]],
    utm_records: Mapping[str, Sequence[Mapping[str, Any]]],
    weather_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Project state producers into stable, line-addressable log records."""

    logs: dict[str, list[dict[str, Any]]] = {name: [] for name in LOG_FILE_NAMES}

    for state in compute_rows:
        node_id = str(state["node_id"])
        tick = int(state["tick"])
        logs["compute_log.jsonl"].append(
            _state_log_row(
                episode_id=episode_id,
                log_kind="compute",
                tick=tick,
                scope_type="compute_node",
                scope_entity_id=node_id,
                state=state,
            )
        )

    for state in communication_rows:
        entity_id = str(state["entity_id"])
        tick = int(state["tick"])
        logs["communication_log.jsonl"].append(
            _state_log_row(
                episode_id=episode_id,
                log_kind="communication",
                tick=tick,
                scope_type="uav",
                scope_entity_id=entity_id,
                state=state,
            )
        )

    for state in domain_rows:
        if state.get("episode_id") != episode_id:
            continue
        family = str(state.get("observation_family") or "")
        tick = int(state["tick"])
        subject_id = str(state["subject_id"])
        if family == "pad_facility":
            is_charging_station = (
                _state_values(state).get("facility_subtype") == "charging_station"
            )
            facility_row = _state_log_row(
                episode_id=episode_id,
                log_kind="facility",
                tick=tick,
                scope_type="facility",
                scope_entity_id=subject_id,
                state=state,
            )
            logs["facility_log.jsonl"].append(facility_row)
            if is_charging_station:
                logs["charging_log.jsonl"].append(
                    _state_log_row(
                        episode_id=episode_id,
                        log_kind="charging",
                        tick=tick,
                        scope_type="facility",
                        scope_entity_id=subject_id,
                        state=state,
                    )
                )
        elif family == "payload_energy":
            logs["battery_log.jsonl"].append(
                _state_log_row(
                    episode_id=episode_id,
                    log_kind="battery",
                    tick=tick,
                    scope_type="uav",
                    scope_entity_id=subject_id,
                    state=state,
                )
            )
        elif family == "gnss_navigation":
            logs["gnss_log.jsonl"].append(
                _state_log_row(
                    episode_id=episode_id,
                    log_kind="gnss",
                    tick=tick,
                    scope_type="uav",
                    scope_entity_id=subject_id,
                    state=state,
                )
            )

    for source_name, states in sorted(utm_records.items()):
        for state in states:
            tick = int(state["tick"])
            scope_type, scope_entity_id = _utm_source_scope(
                episode_id=episode_id,
                source_name=source_name,
                state=state,
            )
            logs["utm_log.jsonl"].append(
                _state_log_row(
                    episode_id=episode_id,
                    log_kind="utm",
                    tick=tick,
                    scope_type=scope_type,
                    scope_entity_id=scope_entity_id,
                    state=state,
                    source_record_kind=source_name.removesuffix(".jsonl"),
                )
            )

    weather_by_tick: dict[int, Mapping[str, Any]] = {}
    for state in weather_rows:
        tick = state.get("tick")
        if tick not in FORMAL_TICKS:
            continue
        if int(tick) in weather_by_tick:
            raise SimulationLogError(f"duplicate weather row at formal tick {tick}")
        weather_by_tick[int(tick)] = state
    if set(weather_by_tick) != set(FORMAL_TICKS):
        raise SimulationLogError("weather log lacks the exact formal tick set")
    weather_input_digest = digest_object(weather_rows)
    for tick, state in sorted(weather_by_tick.items()):
        weather_state = {
            "episode_id": episode_id,
            "tick": tick,
            "schema_name": "weather_state",
            "schema_version": "1.0.0",
            "model_id": "aeroworld_weather_state_materialization",
            "model_version": "1.0.0",
            "input_digest": weather_input_digest,
            "parameter_digest": weather_input_digest,
            "seed_digest": weather_input_digest,
            "source_class": "simulated_derived",
            "source_refs": [f"weather_meta.jsonl#tick={tick}"],
            "values": copy.deepcopy(dict(state)),
        }
        logs["weather_log.jsonl"].append(
            _state_log_row(
                episode_id=episode_id,
                log_kind="weather",
                tick=tick,
                scope_type="scene",
                scope_entity_id=f"scene:{episode_id}",
                state=weather_state,
                source_state_digest=digest_object(state),
            )
        )

    for file_name, rows in logs.items():
        rows.sort(key=_log_sort_key)
        for line_number, row in enumerate(rows, start=1):
            row["source_refs"] = [f"simulation_logs/{file_name}#line={line_number}"]
            row["log_id"] = stable_identifier(
                "simulation_log",
                episode_id,
                file_name,
                line_number,
                row["source_state_digest"],
            )
    validate_simulation_log_source_coverage(
        logs=logs,
        compute_rows=compute_rows,
        communication_rows=communication_rows,
        domain_rows=domain_rows,
        utm_records=utm_records,
        weather_rows=tuple(weather_by_tick.values()),
    )
    return logs


def validate_simulation_log_source_coverage(
    *,
    logs: Mapping[str, Sequence[Mapping[str, Any]]],
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    domain_rows: Sequence[Mapping[str, Any]],
    utm_records: Mapping[str, Sequence[Mapping[str, Any]]],
    weather_rows: Sequence[Mapping[str, Any]],
) -> None:
    if set(logs) != set(LOG_FILE_NAMES):
        raise SimulationLogError("simulation log set is incomplete")
    expected = {
        "compute_log.jsonl": [dict(row) for row in compute_rows],
        "communication_log.jsonl": [dict(row) for row in communication_rows],
        "facility_log.jsonl": [
            dict(row)
            for row in domain_rows
            if row.get("observation_family") == "pad_facility"
        ],
        "charging_log.jsonl": [
            dict(row)
            for row in domain_rows
            if row.get("observation_family") == "pad_facility"
            and _state_values(row).get("facility_subtype") == "charging_station"
        ],
        "battery_log.jsonl": [
            dict(row)
            for row in domain_rows
            if row.get("observation_family") == "payload_energy"
        ],
        "gnss_log.jsonl": [
            dict(row)
            for row in domain_rows
            if row.get("observation_family") == "gnss_navigation"
        ],
        "utm_log.jsonl": [
            dict(row)
            for name in sorted(utm_records)
            for row in utm_records[name]
        ],
        "weather_log.jsonl": [dict(row) for row in weather_rows],
    }
    for file_name in LOG_FILE_NAMES:
        actual: list[Mapping[str, Any]] = []
        for log_row in logs[file_name]:
            state = log_row.get("state")
            if not isinstance(state, Mapping):
                raise SimulationLogError(
                    f"{file_name} contains a log row without a source state"
                )
            # Weather logs wrap the source row in a materialized state object;
            # all other logs retain the producer row directly.
            if file_name == "weather_log.jsonl" and isinstance(
                state.get("values"), Mapping
            ):
                state = state["values"]
            actual.append(state)
        if sorted(canonical_json(row) for row in actual) != sorted(
            canonical_json(row) for row in expected[file_name]
        ):
            raise SimulationLogError(
                f"{file_name} does not cover its producer state exactly"
            )


def summarize_simulation_log_source_coverage(
    *,
    logs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return the exact source-ledger coverage summary stored beside V7 logs."""

    if set(logs) != set(LOG_FILE_NAMES):
        raise SimulationLogError("simulation log set is incomplete")
    record_count_by_log = {
        file_name: len(tuple(rows)) for file_name, rows in sorted(logs.items())
    }
    record_count = sum(record_count_by_log.values())
    if record_count <= 0:
        raise SimulationLogError("simulation logs contain no producer records")
    source_digest_multiset_by_log: dict[str, dict[str, Any]] = {}
    for file_name, rows in sorted(logs.items()):
        digests = sorted(str(row.get("source_state_digest")) for row in rows)
        source_digest_multiset_by_log[file_name] = {
            "record_count": len(digests),
            "multiset_digest": digest_object(digests),
        }
    return {
        "schema_name": "simulation_log_source_coverage_reconciliation",
        "schema_version": "2.0.0",
        "status": "PASS",
        "record_count": record_count,
        "record_count_by_log": record_count_by_log,
        "source_digest_multiset_by_log": source_digest_multiset_by_log,
        "log_count": len(logs),
    }


def _state_log_row(
    *,
    episode_id: str,
    log_kind: str,
    tick: int,
    scope_type: str,
    scope_entity_id: str,
    state: Mapping[str, Any],
    source_record_kind: str | None = None,
    source_state_digest: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema_name": "aeroworld_simulation_log",
        "schema_version": "1.0.0",
        "episode_id": episode_id,
        "log_kind": log_kind,
        "tick": tick,
        "scope_type": scope_type,
        "scope_entity_id": scope_entity_id,
        "source_record_kind": source_record_kind
        or str(state.get("schema_name") or "state"),
        "source_class": str(state.get("source_class") or "simulated_derived"),
        "parameter_source_class": (
            "standard_referenced_parameter"
            if log_kind in {"compute", "communication"}
            else "deterministic_simulation"
        ),
        "model_id": str(
            state.get("model_id") or "aeroworld_deterministic_domain_state"
        ),
        "model_version": str(state.get("model_version") or "1.0.0"),
        "input_digest": str(state.get("input_digest") or digest_object(state)),
        "parameter_digest": str(state.get("parameter_digest") or digest_object(state)),
        "seed_digest": str(
            state.get("seed_digest")
            or state.get("input_digest")
            or digest_object(state)
        ),
        "source_state_digest": source_state_digest or digest_object(state),
        "state": copy.deepcopy(dict(state)),
        "source_refs": [],
    }
    return row


def _utm_source_scope(
    *,
    episode_id: str,
    source_name: str,
    state: Mapping[str, Any],
) -> tuple[str, str]:
    uav_id = state.get("uav_id")
    if isinstance(uav_id, str) and uav_id:
        return "uav", uav_id
    if source_name == "operational_intent_pair_log.jsonl":
        first = state.get("first_uav_id")
        second = state.get("second_uav_id")
        if isinstance(first, str) and first and isinstance(second, str) and second:
            return "scene", f"scene:{episode_id}"
    raise SimulationLogError(f"{source_name}: UTM row lacks explicit source scope")


def _state_values(state: Mapping[str, Any]) -> Mapping[str, Any]:
    values = state.get("values")
    if not isinstance(values, Mapping):
        raise SimulationLogError("domain state row lacks values")
    return values


def _log_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["tick"]),
        str(row["scope_type"]),
        str(row["scope_entity_id"]),
        str(row["source_record_kind"]),
        str(row["source_state_digest"]),
    )


def serialize_simulation_log(rows: Iterable[Mapping[str, Any]]) -> str:
    from Dataset.semantic_truth.provenance import canonical_json, without_integrity_metadata

    return "".join(f"{canonical_json(without_integrity_metadata(row))}\n" for row in rows)


__all__ = [
    "LOG_FILE_NAMES",
    "SimulationLogError",
    "build_simulation_logs",
    "serialize_simulation_log",
    "summarize_simulation_log_source_coverage",
    "validate_simulation_log_source_coverage",
]
