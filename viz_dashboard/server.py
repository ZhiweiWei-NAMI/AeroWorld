#!/usr/bin/env python3
"""Read-only data service for the semantic truth dashboard."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
from threading import Lock
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from jsonschema import Draft202012Validator


DASHBOARD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.semantic_truth.provenance import open_artifact_text

from Dataset.semantic_truth.event_chain import (  # noqa: E402
    RUNTIME_PASS_VALIDATION_STATUSES,
    validate_event_chain_artifacts,
)
from Dataset.semantic_truth.predicate_catalog import get_predicate_catalog  # noqa: E402
from viz_dashboard.arm_review import ArmReviewError, catalog as arm_catalog, group_detail as arm_group_detail  # noqa: E402
from viz_dashboard.arm_profile import (  # noqa: E402
    ArmProfileError,
    profile_overview as arm_profile_overview,
)
from viz_dashboard import arm_viz  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "aw_data" / "objective_semantic_truth"
EVENT_CHAIN_ROOT = PROJECT_ROOT / "aw_data" / "event_chains"
EVENT_CHAIN_CAPTURE_ROOT = (
    PROJECT_ROOT / "aw_data" / "render_ready_episodes_capture_filtered"
)
REGISTRY_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "predicates"
    / "core_semantic_predicate_templates.json"
)
OBJECTIVE_CONTRACT_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "epi_objective_semantic_contract.json"
)
EPISODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*__seed\d{2}$")
FORMAL_TICKS = tuple(range(0, 901, 5))
SMALL_JSONL_FILES = (
    "event_occurrences.jsonl",
    "predicate_transitions.jsonl",
    "event_outcomes.jsonl",
    "predicate_continuity_breaks.jsonl",
    "semantic_graph_deltas.jsonl",
)
BUNDLE_FILES = (
    "manifest.json",
    "world_truth_summary.json",
    "epi_closure.json",
    "semantic_graph_base.json",
    *SMALL_JSONL_FILES,
)
EVENT_CHAIN_FILES = ("event_chain_manifest.json", "event_chain_relations.jsonl")
PHYSICAL_LOG_FILES = ("event_trace.jsonl", "event_realization.jsonl")
EVENT_CHAIN_SCHEMA_VERSION = "2.0.0"
EVENT_CHAIN_RELATION_TYPES = {"script_declared_causal", "script_declared_context"}
EVENT_CHAIN_SCHEMA_ROOT = PROJECT_ROOT / "Dataset" / "semantic_rules" / "schema"
EVENT_CHAIN_MANIFEST_SCHEMA_PATH = (
    EVENT_CHAIN_SCHEMA_ROOT / "event_chain_manifest.schema.json"
)
EVENT_CHAIN_RELATION_SCHEMA_PATH = (
    EVENT_CHAIN_SCHEMA_ROOT / "event_chain_relation.schema.json"
)
WEB_MERCATOR_ORIGIN_SHIFT = 20037508.342789244


class DashboardDataError(RuntimeError):
    """Raised when a declared truth artifact is malformed or unavailable."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DashboardDataError(f"JSON artifact root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open_artifact_text(path, encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DashboardDataError(
                        f"JSONL record must be an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"cannot read JSONL artifact {path}: {exc}") from exc
    return rows


def _load_event_chain_schema(path: Path) -> Draft202012Validator:
    schema = read_json(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


_EVENT_CHAIN_MANIFEST_VALIDATOR = _load_event_chain_schema(
    EVENT_CHAIN_MANIFEST_SCHEMA_PATH
)
_EVENT_CHAIN_RELATION_VALIDATOR = _load_event_chain_schema(
    EVENT_CHAIN_RELATION_SCHEMA_PATH
)


def _event_chain_schema_errors(
    validator: Draft202012Validator,
    payload: dict[str, Any],
    *,
    location: str,
) -> list[str]:
    errors: list[str] = []
    for error in sorted(
        validator.iter_errors(payload), key=lambda item: list(item.absolute_path)
    ):
        suffix = "".join(f"[{part!r}]" for part in error.absolute_path)
        errors.append(f"{location}{suffix}: {error.message}")
    return errors


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


@lru_cache(maxsize=1024)
def unit_for_parameter(name: str) -> str:
    suffixes = (
        ("_mps", "m/s"),
        ("_ms", "ms"),
        ("_hz", "Hz"),
        ("_wh", "Wh"),
        ("_w", "W"),
        ("_bytes", "byte"),
        ("_deg", "deg"),
        ("_m", "m"),
        ("_c", "°C"),
        ("_ratio", "ratio"),
        ("_ticks", "tick"),
        ("_vehicles", "vehicle"),
    )
    for suffix, unit in suffixes:
        if name.endswith(suffix):
            return unit
    return "scalar"


def unique(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def keyed_unique(
    values: Iterable[dict[str, Any]], key: str = "id"
) -> list[dict[str, Any]]:
    result: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for value in values:
        result[str(value[key])] = value
    return list(result.values())


def threshold_operator(rule: Any, threshold_name: str) -> str | None:
    if isinstance(rule, dict):
        if rule.get("threshold") == threshold_name:
            return str(rule.get("op")) if rule.get("op") else None
        for value in rule.values():
            operator = threshold_operator(value, threshold_name)
            if operator:
                return operator
    if isinstance(rule, list):
        for value in rule:
            operator = threshold_operator(value, threshold_name)
            if operator:
                return operator
    return None


def _build_registry_payload() -> dict[str, Any]:
    registry = read_json(REGISTRY_PATH)
    contract = read_json(OBJECTIVE_CONTRACT_PATH)
    truth_values = list(registry["truth_values"])
    predicates = get_predicate_catalog()

    mode_rows = [
        row
        for row in predicates
        if re.match(r"^operation\.mode\.[^.]+$", row["api_id"])
    ]
    if mode_rows:
        predicates = [row for row in predicates if row not in mode_rows]
        parameterized = dict(mode_rows[0])
        parameterized["api_id"] = "operation.mode.{mode}"
        parameterized["mode_values"] = [
            row["api_id"].removeprefix("operation.mode.") for row in mode_rows
        ]
        predicates.append(parameterized)
        predicates.sort(key=lambda row: row["api_id"])

    family_contracts = contract.get("event_families", {})
    event_rules: list[dict[str, Any]] = []
    for source_rule in registry["event_occurrence_types"]:
        rule = dict(source_rule)
        family = family_contracts.get(rule["event_family_id"], {})
        rule["required_response"] = family.get("required_response")
        rule["start_condition"] = family.get("start_condition")
        rule["terminal_condition"] = family.get("terminal_condition")
        rule["failure_condition"] = family.get("failure_condition")
        rule["unknown_policy"] = family.get("unknown_policy")
        rule["required_state_fields"] = family.get("required_state_fields", [])
        rule["semantic_rationale"] = family.get("semantic_rationale")
        event_rules.append(rule)

    event_types = sorted({row["event_type"] for row in event_rules})
    event_families = sorted({row["event_family_id"] for row in event_rules})
    governance = registry["parameter_governance"]
    return {
        "schema_name": registry["schema_name"],
        "schema_version": registry["schema_version"],
        "registry_id": registry["registry_id"],
        "predicates": predicates,
        "event_rules": event_rules,
        "truth_values": truth_values,
        "governance": governance,
        "transition_rule": registry["transition_rule"],
        "counts": {
            "predicates": len(predicates),
            "registry_predicates": registry["template_count"],
            "event_rules": len(event_rules),
            "event_types": len(event_types),
            "event_families": len(event_families),
            **registry["event_counts"],
        },
    }


_REGISTRY_LOCK = Lock()
_REGISTRY_CACHE: dict[str, Any] | None = None
_REGISTRY_FINGERPRINT: tuple[tuple[int, int], tuple[int, int]] | None = None


def registry_payload() -> dict[str, Any]:
    global _REGISTRY_CACHE, _REGISTRY_FINGERPRINT
    try:
        registry_stat = REGISTRY_PATH.stat()
        contract_stat = OBJECTIVE_CONTRACT_PATH.stat()
    except OSError as exc:
        raise DashboardDataError(
            f"cannot stat semantic registry artifacts: {exc}"
        ) from exc
    fingerprint = (
        (registry_stat.st_size, registry_stat.st_mtime_ns),
        (contract_stat.st_size, contract_stat.st_mtime_ns),
    )
    with _REGISTRY_LOCK:
        if _REGISTRY_CACHE is not None and _REGISTRY_FINGERPRINT == fingerprint:
            return _REGISTRY_CACHE
    payload = _build_registry_payload()
    with _REGISTRY_LOCK:
        _REGISTRY_CACHE = payload
        _REGISTRY_FINGERPRINT = fingerprint
        return payload


def predicate_by_id(predicate_id: str | None) -> dict[str, Any] | None:
    if not predicate_id:
        return None
    for row in registry_payload()["predicates"]:
        if row["api_id"] == predicate_id:
            return row
    return None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    epi_id: str
    seed: str
    closure_status: str
    event_count: int
    event_families: tuple[str, ...]
    event_types: tuple[str, ...]
    truth_distribution: dict[str, int]
    predicate_count: int
    formal_tick_count: int
    anomaly_count: int
    missing_artifacts: tuple[str, ...]
    record_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "epi_id": self.epi_id,
            "seed": self.seed,
            "closure_status": self.closure_status,
            "event_count": self.event_count,
            "event_families": list(self.event_families),
            "event_types": list(self.event_types),
            "truth_distribution": self.truth_distribution,
            "predicate_count": self.predicate_count,
            "formal_tick_count": self.formal_tick_count,
            "anomaly_count": self.anomaly_count,
            "missing_artifacts": list(self.missing_artifacts),
            "record_counts": self.record_counts,
        }


class EpisodeIndex:
    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[EpisodeRecord] | None = None
        self._fingerprint: tuple[tuple[str, str, int, int], ...] | None = None

    @staticmethod
    def _artifact_fingerprint() -> tuple[tuple[str, str, int, int], ...]:
        rows: list[tuple[str, str, int, int]] = []
        filenames = (
            "manifest.json",
            "world_truth_summary.json",
            "epi_closure.json",
            "event_occurrences.jsonl",
        )
        try:
            for episode_dir in sorted(
                DATA_ROOT.iterdir(), key=lambda path: natural_key(path.name)
            ):
                if not episode_dir.is_dir() or not EPISODE_ID_PATTERN.fullmatch(
                    episode_dir.name
                ):
                    continue
                for filename in filenames:
                    path = episode_dir / filename
                    if path.is_file():
                        stat = path.stat()
                        rows.append(
                            (episode_dir.name, filename, stat.st_size, stat.st_mtime_ns)
                        )
                    else:
                        rows.append((episode_dir.name, filename, -1, -1))
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat corpus index artifacts: {exc}"
            ) from exc
        return tuple(rows)

    def records(self) -> list[EpisodeRecord]:
        fingerprint = self._artifact_fingerprint()
        with self._lock:
            if self._records is None or self._fingerprint != fingerprint:
                self._records = self._build()
                self._fingerprint = fingerprint
            return self._records

    def _build(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        required = ("manifest.json", "world_truth_summary.json", "epi_closure.json")
        for episode_dir in sorted(
            DATA_ROOT.iterdir(), key=lambda path: natural_key(path.name)
        ):
            if not episode_dir.is_dir() or not EPISODE_ID_PATTERN.fullmatch(
                episode_dir.name
            ):
                continue
            missing = tuple(
                name for name in required if not (episode_dir / name).is_file()
            )
            if missing:
                epi_id, seed = episode_dir.name.rsplit("__", 1)
                records.append(
                    EpisodeRecord(
                        episode_id=episode_dir.name,
                        epi_id=epi_id,
                        seed=seed,
                        closure_status="INCOMPLETE",
                        event_count=0,
                        event_families=(),
                        event_types=(),
                        truth_distribution={
                            value: 0 for value in registry_payload()["truth_values"]
                        },
                        predicate_count=0,
                        formal_tick_count=0,
                        anomaly_count=len(missing),
                        missing_artifacts=missing,
                        record_counts={},
                    )
                )
                continue
            manifest = read_json(episode_dir / "manifest.json")
            summary = read_json(episode_dir / "world_truth_summary.json")
            closure = read_json(episode_dir / "epi_closure.json")
            events = read_jsonl(episode_dir / "event_occurrences.jsonl")
            families = tuple(sorted({str(row["event_family_id"]) for row in events}))
            types = tuple(sorted({str(row["event_type_id"]) for row in events}))
            truth_distribution = {
                value: int(summary.get("value_counts", {}).get(value, 0))
                for value in registry_payload()["truth_values"]
            }
            counts = {
                key: int(value)
                for key, value in manifest.get("record_counts", {}).items()
                if isinstance(value, int)
            }
            rejected = int(manifest.get("compact_graph_rejected_record_count", 0))
            continuity_breaks = counts.get("predicate_continuity_breaks", 0)
            closure_failure = 0 if closure.get("status") == "PASS" else 1
            records.append(
                EpisodeRecord(
                    episode_id=episode_dir.name,
                    epi_id=str(manifest["epi_id"]),
                    seed=str(manifest["seed_label"]),
                    closure_status=str(closure["status"]),
                    event_count=len(events),
                    event_families=families,
                    event_types=types,
                    truth_distribution=truth_distribution,
                    predicate_count=int(summary["predicate_count"]),
                    formal_tick_count=int(summary["formal_tick_count"]),
                    anomaly_count=rejected + continuity_breaks + closure_failure,
                    missing_artifacts=(),
                    record_counts=counts,
                )
            )
        return records

    def summary(self) -> dict[str, Any]:
        records = self.records()
        registry = registry_payload()
        truth = Counter()
        closure = Counter()
        event_types: set[str] = set()
        event_families: set[str] = set()
        family_episode_counts: Counter[str] = Counter()
        anomaly_count = 0
        for record in records:
            truth.update(record.truth_distribution)
            closure[record.closure_status] += 1
            event_types.update(record.event_types)
            event_families.update(record.event_families)
            family_episode_counts.update(record.event_families)
            anomaly_count += record.anomaly_count
        epis = {record.epi_id for record in records}
        complete = sum(not record.missing_artifacts for record in records)
        corpus_complete = len(records) == 210 and complete == len(records)
        closure_complete = corpus_complete and closure.get("PASS", 0) == len(records)
        return {
            "corpus": {
                "epi_count": len(epis),
                "episode_count": len(records),
                "expected_episode_count": 210,
                "seed_count": len({record.seed for record in records}),
                "predicate_template_count": registry["counts"]["registry_predicates"],
                "event_rule_count": registry["counts"]["event_rules"],
                "event_type_count": len(event_types),
                "event_family_count": len(event_families),
                "tick_start": FORMAL_TICKS[0],
                "tick_end": FORMAL_TICKS[-1],
                "tick_step": 5,
                "closure_status_counts": dict(closure),
            },
            "truth_distribution": {
                value: truth.get(value, 0) for value in registry["truth_values"]
            },
            "generation": {
                "status": "complete" if corpus_complete else "incomplete",
                "completed_episodes": complete,
                "total_episodes": len(records),
                "validation_status": "PASS" if closure_complete else "ATTENTION",
                "updated_at": None,
                "source": str(DATA_ROOT.relative_to(PROJECT_ROOT)),
            },
            "anomaly_count": anomaly_count,
            "event_families": [
                {"id": family, "count": family_episode_counts[family]}
                for family in sorted(event_families)
            ],
            "event_types": sorted(event_types),
            "generated_from_real_artifacts": bool(records),
        }

    def query(self, filters: dict[str, str]) -> dict[str, Any]:
        records = self.records()
        epi_filter = filters.get("epi", "").strip().lower()
        seed_filter = filters.get("seed", "").strip()
        family_filter = filters.get("event_family", "").strip()
        closure_filter = filters.get("closure", "").strip()
        result = [
            record
            for record in records
            if (not epi_filter or epi_filter in record.epi_id.lower())
            and (not seed_filter or seed_filter == record.seed)
            and (not family_filter or family_filter in record.event_families)
            and (not closure_filter or closure_filter == record.closure_status)
        ]

        def item(record: EpisodeRecord) -> dict[str, Any]:
            value = record.as_dict()
            value["transition_count"] = record.record_counts.get(
                "predicate_transitions", 0
            )
            return value

        def facet_rows(values: Iterable[str]) -> list[dict[str, Any]]:
            counts = Counter(values)
            return [
                {"value": value, "label": value, "count": counts[value]}
                for value in sorted(counts, key=natural_key)
            ]

        return {
            "items": [item(record) for record in result],
            "total": len(result),
            "facets": {
                "epis": facet_rows(record.epi_id for record in records),
                "seeds": facet_rows(record.seed for record in records),
                "event_families": facet_rows(
                    family for record in records for family in record.event_families
                ),
                "closure_statuses": facet_rows(
                    record.closure_status for record in records
                ),
            },
        }


EPISODE_INDEX = EpisodeIndex()


def episode_path(episode_id: str) -> Path:
    if not EPISODE_ID_PATTERN.fullmatch(episode_id):
        raise DashboardDataError(f"invalid episode id: {episode_id}")
    path = DATA_ROOT / episode_id
    if not path.is_dir():
        raise DashboardDataError(f"unknown episode id: {episode_id}")
    return path


def _phase_records_for_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for phase in event.get("event_phases", ()):
        if not isinstance(phase, dict):
            raise DashboardDataError("event phase entry must be an object")
        transition_id = phase.get("transition_id")
        tick = phase.get("tick")
        if not isinstance(transition_id, str) or not isinstance(tick, int):
            raise DashboardDataError("event phase lacks transition_id or tick")
        records.append(
            {
                "phase_id": (
                    f"{event['event_id']}:{phase['phase_kind']}:{transition_id}"
                ),
                "episode_id": event["episode_id"],
                "event_id": event["event_id"],
                "event_family_id": event["event_family_id"],
                "event_type_id": event["event_type_id"],
                "phase_kind": phase["phase_kind"],
                "predicate_id": phase["predicate_id"],
                "direction": phase["direction"],
                "tick": tick,
                "bindings": dict(phase.get("predicate_bindings", {})),
                "supporting_transition_ids": [transition_id],
                "supporting_truth_ids": [],
                "source_refs": list(event.get("source_refs", [])),
            }
        )
    return records


def build_bundle_index(bundle: dict[str, Any]) -> dict[str, Any]:
    """Build compact lookup tables for one already-loaded episode bundle."""

    transitions_by_id = {
        row["transition_id"]: row for row in bundle["predicate_transitions"]
    }
    transitions_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["predicate_transitions"]:
        transitions_by_tick[int(row["to_tick"])].append(row)

    def group(
        rows: Iterable[dict[str, Any]], field: str
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if value is not None:
                grouped[str(value)].append(row)
        return dict(grouped)

    events_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    phases_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    phases_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["event_occurrences"]:
        events_by_tick[int(row["trigger_tick"])].append(row)
        for phase in _phase_records_for_event(row):
            phases_by_tick[int(phase["tick"])].append(phase)
            phases_by_event[str(row["event_id"])].append(phase)
    outcomes_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["event_outcomes"]:
        if row.get("terminal_tick") is not None:
            outcomes_by_tick[int(row["terminal_tick"])].append(row)
    semantic_deltas_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["semantic_graph_deltas"]:
        semantic_deltas_by_tick[int(row["tick"])].append(row)

    current_predicates: dict[tuple[str, str], dict[str, Any]] = {}
    for assertion in bundle["semantic_graph_base"]["initial_assertions"]:
        key = (assertion["predicate_id"], assertion["tuple_id"])
        current_predicates[key] = {
            "predicate_id": assertion["predicate_id"],
            "tuple_id": assertion["tuple_id"],
            "bindings": assertion["bindings"],
            "value": assertion["truth_value"],
            "last_update_tick": 0,
            "supporting_transition_id": None,
            "truth_id": assertion["assertion_id"],
            "source_refs": assertion["source_refs"],
        }
    predicate_snapshots: dict[int, tuple[dict[str, Any], ...]] = {}
    for formal_tick in FORMAL_TICKS:
        for delta in semantic_deltas_by_tick.get(formal_tick, []):
            for operation in delta["operations"]:
                key = (operation["predicate_id"], operation["tuple_id"])
                if operation["operation"] == "add_predicate_truth":
                    current_predicates[key] = {
                        "predicate_id": operation["predicate_id"],
                        "tuple_id": operation["tuple_id"],
                        "bindings": operation["bindings"],
                        "value": operation["truth_value"],
                        "last_update_tick": formal_tick,
                        "supporting_transition_id": None,
                        "truth_id": operation["source_truth_id"],
                        "source_refs": [],
                    }
                elif operation["operation"] == "set_predicate_truth":
                    current_predicates[key] = {
                        "predicate_id": operation["predicate_id"],
                        "tuple_id": operation["tuple_id"],
                        "bindings": operation["bindings"],
                        "value": operation["to_value"],
                        "last_update_tick": formal_tick,
                        "supporting_transition_id": operation["source_transition_id"],
                        "truth_id": next(reversed(operation["evidence_truth_ids"])),
                        "source_refs": [],
                    }
                elif operation["operation"] == "invalidate_predicate_truth":
                    if key not in current_predicates:
                        raise DashboardDataError(
                            "semantic graph invalidates an absent predicate tuple"
                        )
                    current_predicates[key] = {
                        **current_predicates[key],
                        "value": "unknown",
                        "last_update_tick": formal_tick,
                        "supporting_transition_id": None,
                        "truth_id": None,
                        "source_refs": [],
                    }
                elif operation["operation"] == "restore_predicate_truth":
                    if (
                        key not in current_predicates
                        or current_predicates[key]["value"] != "unknown"
                    ):
                        raise DashboardDataError(
                            "semantic graph restores a predicate without an unknown interval"
                        )
                    current_predicates[key] = {
                        "predicate_id": operation["predicate_id"],
                        "tuple_id": operation["tuple_id"],
                        "bindings": operation["bindings"],
                        "value": operation["truth_value"],
                        "last_update_tick": formal_tick,
                        "supporting_transition_id": None,
                        "truth_id": operation["source_truth_id"],
                        "source_refs": [],
                    }
                else:
                    raise DashboardDataError(
                        f"unsupported semantic graph operation: {operation['operation']}"
                    )
        predicate_snapshots[formal_tick] = tuple(current_predicates.values())

    return {
        "transitions_by_id": transitions_by_id,
        "transitions_by_tick": dict(transitions_by_tick),
        "events_by_id": {row["event_id"]: row for row in bundle["event_occurrences"]},
        "events_by_tick": dict(events_by_tick),
        "phases_by_event": dict(phases_by_event),
        "phases_by_tick": dict(phases_by_tick),
        "outcomes_by_event": group(bundle["event_outcomes"], "event_id"),
        "outcomes_by_tick": dict(outcomes_by_tick),
        "semantic_deltas_by_tick": dict(semantic_deltas_by_tick),
        "predicate_snapshots": predicate_snapshots,
    }


class EpisodeBundleCache:
    def __init__(self, max_entries: int = 4) -> None:
        self.max_entries = max_entries
        self._lock = Lock()
        self._bundles: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._bundle_fingerprints: dict[str, tuple[tuple[str, int, int], ...]] = {}
        self._evidence: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
        self._evidence_offsets: OrderedDict[str, dict[str, int]] = OrderedDict()
        self._evidence_fingerprints: dict[str, tuple[int, int]] = {}
        self._state_offsets: OrderedDict[str, dict[int, list[int]]] = OrderedDict()
        self._state_fingerprints: dict[str, tuple[int, int]] = {}
        self._state_counts: OrderedDict[str, dict[int, int]] = OrderedDict()
        self._state_records: OrderedDict[tuple[str, int], list[dict[str, Any]]] = (
            OrderedDict()
        )
        self._stats: Counter[str] = Counter()

    @staticmethod
    def _bundle_fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
        rows: list[tuple[str, int, int]] = []
        try:
            for filename in BUNDLE_FILES:
                stat = (root / filename).stat()
                rows.append((filename, stat.st_size, stat.st_mtime_ns))
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat episode artifacts in {root}: {exc}"
            ) from exc
        return tuple(rows)

    def _drop_episode_locked(self, episode_id: str) -> None:
        self._bundles.pop(episode_id, None)
        self._bundle_fingerprints.pop(episode_id, None)
        self._evidence.pop(episode_id, None)
        self._evidence_offsets.pop(episode_id, None)
        self._evidence_fingerprints.pop(episode_id, None)
        self._state_offsets.pop(episode_id, None)
        self._state_fingerprints.pop(episode_id, None)
        self._state_counts.pop(episode_id, None)
        for key in [key for key in self._state_records if key[0] == episode_id]:
            self._state_records.pop(key, None)

    def _trim_locked(self) -> None:
        while len(self._bundles) > self.max_entries:
            evicted, _ = self._bundles.popitem(last=False)
            self._bundle_fingerprints.pop(evicted, None)
            self._evidence.pop(evicted, None)
            self._evidence_offsets.pop(evicted, None)
            self._evidence_fingerprints.pop(evicted, None)
            self._state_offsets.pop(evicted, None)
            self._state_fingerprints.pop(evicted, None)
            self._state_counts.pop(evicted, None)
            for key in [key for key in self._state_records if key[0] == evicted]:
                self._state_records.pop(key, None)
        while len(self._evidence) > self.max_entries:
            evicted, _ = self._evidence.popitem(last=False)
            self._evidence_offsets.pop(evicted, None)
            self._evidence_fingerprints.pop(evicted, None)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                **self._stats,
                "bundle_entries": len(self._bundles),
                "evidence_record_entries": sum(
                    len(rows) for rows in self._evidence.values()
                ),
                "evidence_offset_indexes": len(self._evidence_offsets),
                "state_offset_indexes": len(self._state_offsets),
                "state_record_frames": len(self._state_records),
            }

    def bundle(self, episode_id: str) -> dict[str, Any]:
        root = episode_path(episode_id)
        fingerprint = self._bundle_fingerprint(root)
        with self._lock:
            cached = self._bundles.get(episode_id)
            if (
                cached is not None
                and self._bundle_fingerprints.get(episode_id) == fingerprint
            ):
                self._stats["bundle_hits"] += 1
                self._bundles.move_to_end(episode_id)
                return cached
            if cached is not None:
                self._drop_episode_locked(episode_id)
                self._stats["bundle_invalidations"] += 1
            self._stats["bundle_misses"] += 1
        bundle: dict[str, Any] = {
            "manifest": read_json(root / "manifest.json"),
            "summary": read_json(root / "world_truth_summary.json"),
            "closure": read_json(root / "epi_closure.json"),
            "semantic_graph_base": read_json(root / "semantic_graph_base.json"),
        }
        for filename in SMALL_JSONL_FILES:
            bundle[filename.removesuffix(".jsonl")] = read_jsonl(root / filename)
        bundle["_index"] = build_bundle_index(bundle)
        with self._lock:
            self._bundles[episode_id] = bundle
            self._bundle_fingerprints[episode_id] = fingerprint
            self._bundles.move_to_end(episode_id)
            self._trim_locked()
        return bundle

    def _offset_index(self, episode_id: str, source: Path) -> dict[str, int]:
        try:
            stat = source.stat()
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat evidence artifact {source}: {exc}"
            ) from exc
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._evidence_offsets.get(episode_id)
            if (
                cached is not None
                and self._evidence_fingerprints.get(episode_id) == fingerprint
            ):
                self._stats["evidence_index_hits"] += 1
                self._evidence_offsets.move_to_end(episode_id)
                return cached
            if episode_id in self._evidence_fingerprints:
                self._evidence.pop(episode_id, None)
            self._stats["evidence_index_builds"] += 1

        offsets: dict[str, int] = {}
        line_number = 0
        try:
            with source.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    value = json.loads(line.decode("utf-8-sig"))
                    if not isinstance(value, dict):
                        raise DashboardDataError(
                            f"evidence record must be an object: {source}:{line_number}"
                        )
                    truth_id = value.get("observation_id")
                    if truth_id:
                        offsets[str(truth_id)] = offset
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DashboardDataError(
                f"cannot index evidence artifact {source}:{line_number}: {exc}"
            ) from exc
        with self._lock:
            self._evidence_offsets[episode_id] = offsets
            self._evidence_fingerprints[episode_id] = fingerprint
            self._evidence_offsets.move_to_end(episode_id)
            return offsets

    def evidence(
        self, episode_id: str, truth_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        targets = {truth_id for truth_id in truth_ids if truth_id}
        if not targets:
            return {}
        source = episode_path(episode_id) / "evidence_observations.jsonl"
        offsets = self._offset_index(episode_id, source)
        with self._lock:
            episode_cache = self._evidence.setdefault(episode_id, {})
            self._evidence.move_to_end(episode_id)
            missing = targets - episode_cache.keys()
        if missing:
            found: dict[str, dict[str, Any]] = {}
            line_number = 0
            try:
                with source.open("rb") as stream:
                    for truth_id in missing:
                        offset = offsets.get(truth_id)
                        if offset is None:
                            continue
                        stream.seek(offset)
                        line_number += 1
                        value = json.loads(stream.readline().decode("utf-8-sig"))
                        if (
                            not isinstance(value, dict)
                            or value.get("observation_id") != truth_id
                        ):
                            raise DashboardDataError(
                                f"stale evidence offset for {truth_id} in {source}"
                            )
                        value["record_ref"] = {
                            "file": source.relative_to(PROJECT_ROOT).as_posix(),
                            "byte_offset": offset,
                        }
                        found[truth_id] = value
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DashboardDataError(
                    f"cannot seek evidence artifact {source}:{line_number}: {exc}"
                ) from exc
            with self._lock:
                episode_cache.update(found)
                self._stats["evidence_record_reads"] += len(found)
                self._trim_locked()
        with self._lock:
            return {
                truth_id: self._evidence.get(episode_id, {}).get(truth_id)
                for truth_id in targets
                if self._evidence.get(episode_id, {}).get(truth_id) is not None
            }

    def _state_offset_index(
        self, episode_id: str, source: Path
    ) -> dict[int, list[int]]:
        try:
            stat = source.stat()
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat L0 state artifact {source}: {exc}"
            ) from exc
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._state_offsets.get(episode_id)
            if (
                cached is not None
                and self._state_fingerprints.get(episode_id) == fingerprint
            ):
                self._stats["state_index_hits"] += 1
                self._state_offsets.move_to_end(episode_id)
                return cached
            if episode_id in self._state_fingerprints:
                for key in [key for key in self._state_records if key[0] == episode_id]:
                    self._state_records.pop(key, None)
                self._state_counts.pop(episode_id, None)
            self._stats["state_index_builds"] += 1

        offsets: dict[int, list[int]] = defaultdict(list)
        tick_pattern = re.compile(rb'"tick"\s*:\s*(\d+)')
        try:
            with source.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    match = tick_pattern.search(line)
                    if match is None:
                        raise DashboardDataError(
                            f"L0 state record has no tick: {source}@{offset}"
                        )
                    offsets[int(match.group(1))].append(offset)
        except OSError as exc:
            raise DashboardDataError(
                f"cannot index L0 state artifact {source}: {exc}"
            ) from exc
        indexed = dict(offsets)
        with self._lock:
            self._state_offsets[episode_id] = indexed
            self._state_fingerprints[episode_id] = fingerprint
            self._state_offsets.move_to_end(episode_id)
            return indexed

    def state_observations(self, episode_id: str, tick: int) -> list[dict[str, Any]]:
        source = episode_path(episode_id) / "l0_predicate_state.jsonl"
        offsets = self._state_offset_index(episode_id, source).get(tick, [])
        source_ref = source.relative_to(PROJECT_ROOT).as_posix()
        cache_key = (episode_id, tick)
        with self._lock:
            cached = self._state_records.get(cache_key)
            if cached is not None:
                self._stats["state_record_hits"] += 1
                self._state_records.move_to_end(cache_key)
                return cached
        rows: list[dict[str, Any]] = []
        try:
            with source.open("rb") as stream:
                for offset in offsets:
                    if stream.tell() != offset:
                        stream.seek(offset)
                    value = json.loads(stream.readline().decode("utf-8-sig"))
                    if not isinstance(value, dict) or int(value["tick"]) != tick:
                        raise DashboardDataError(
                            f"stale L0 state offset in {source}@{offset}"
                        )
                    value["record_ref"] = {
                        "file": source_ref,
                        "byte_offset": offset,
                    }
                    rows.append(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DashboardDataError(
                f"cannot seek L0 state artifact {source}: {exc}"
            ) from exc
        with self._lock:
            self._stats["state_record_reads"] += len(rows)
            self._state_records[cache_key] = rows
            self._state_records.move_to_end(cache_key)
            while len(self._state_records) > self.max_entries * 8:
                self._state_records.popitem(last=False)
        return rows

    def state_tick_counts(self, episode_id: str) -> dict[int, int]:
        source = episode_path(episode_id) / "l0_predicate_state.jsonl"
        offsets = self._state_offset_index(episode_id, source)
        fingerprint = self._state_fingerprints.get(episode_id)
        with self._lock:
            cached = self._state_counts.get(episode_id)
            if cached is not None:
                self._state_counts.move_to_end(episode_id)
                self._stats["state_count_hits"] += 1
                return cached
        counts = {
            tick: len(offsets)
            for tick, offsets in sorted(offsets.items())
        }
        with self._lock:
            # The offset index is fingerprinted; if it changed while the
            # counts were being built, discard the derived cache entry and
            # let the next request rebuild it from the fresh offsets.
            if fingerprint == self._state_fingerprints.get(episode_id):
                self._state_counts[episode_id] = counts
                self._state_counts.move_to_end(episode_id)
        return counts


def capture_episode_path(episode_id: str) -> Path:
    if not EPISODE_ID_PATTERN.fullmatch(episode_id):
        raise DashboardDataError(f"invalid capture episode id: {episode_id}")
    path = EVENT_CHAIN_CAPTURE_ROOT / episode_id
    if not path.is_dir():
        raise DashboardDataError(f"capture episode is unavailable: {episode_id}")
    return path


class TruthFrameCache:
    """Seek exact-tick render truth without loading an episode into memory."""

    def __init__(self, max_entries: int = 4, max_frames: int = 32) -> None:
        self.max_entries = max_entries
        self.max_frames = max_frames
        self._lock = Lock()
        self._offsets: OrderedDict[str, dict[int, int]] = OrderedDict()
        self._fingerprints: dict[str, tuple[int, int]] = {}
        self._frames: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()

    def _offset_index(self, episode_id: str, source: Path) -> dict[int, int]:
        try:
            stat = source.stat()
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat render truth artifact {source}: {exc}"
            ) from exc
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._offsets.get(episode_id)
            if cached is not None and self._fingerprints.get(episode_id) == fingerprint:
                self._offsets.move_to_end(episode_id)
                return cached
            if episode_id in self._fingerprints:
                for key in [key for key in self._frames if key[0] == episode_id]:
                    self._frames.pop(key, None)

        offsets: dict[int, int] = {}
        tick_pattern = re.compile(rb'"tick"\s*:\s*(\d+)')
        try:
            with source.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    match = tick_pattern.search(line)
                    if match is None:
                        raise DashboardDataError(
                            f"render truth frame has no tick: {source}@{offset}"
                        )
                    frame_tick = int(match.group(1))
                    if frame_tick in offsets:
                        raise DashboardDataError(
                            f"duplicate render truth tick {frame_tick}: {source}"
                        )
                    offsets[frame_tick] = offset
        except OSError as exc:
            raise DashboardDataError(
                f"cannot index render truth artifact {source}: {exc}"
            ) from exc

        with self._lock:
            self._offsets[episode_id] = offsets
            self._fingerprints[episode_id] = fingerprint
            self._offsets.move_to_end(episode_id)
            while len(self._offsets) > self.max_entries:
                evicted, _ = self._offsets.popitem(last=False)
                self._fingerprints.pop(evicted, None)
                for key in [key for key in self._frames if key[0] == evicted]:
                    self._frames.pop(key, None)
        return offsets

    def frame(self, episode_id: str, tick: int) -> dict[str, Any]:
        source = capture_episode_path(episode_id) / "truth_frames.jsonl"
        offsets = self._offset_index(episode_id, source)
        if tick not in offsets:
            raise DashboardDataError(
                f"render truth has no exact frame for {episode_id} at tick {tick}"
            )
        cache_key = (episode_id, tick)
        with self._lock:
            cached = self._frames.get(cache_key)
            if cached is not None:
                self._frames.move_to_end(cache_key)
                return cached

        offset = offsets[tick]
        try:
            with source.open("rb") as stream:
                stream.seek(offset)
                value = json.loads(stream.readline().decode("utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DashboardDataError(
                f"cannot seek render truth artifact {source}: {exc}"
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("episode_id") != episode_id
            or value.get("tick") != tick
            or not isinstance(value.get("entities"), list)
        ):
            raise DashboardDataError(
                f"stale or malformed render truth frame in {source}@{offset}"
            )
        frame = {
            **value,
            "record_ref": {
                "file": source.relative_to(PROJECT_ROOT).as_posix(),
                "byte_offset": offset,
            },
        }
        with self._lock:
            self._frames[cache_key] = frame
            self._frames.move_to_end(cache_key)
            while len(self._frames) > self.max_frames:
                self._frames.popitem(last=False)
        return frame


class PhysicalLogCache:
    """Cache the two small physical event logs without touching frame data."""

    def __init__(self, max_entries: int = 4) -> None:
        self.max_entries = max_entries
        self._lock = Lock()
        self._bundles: OrderedDict[str, dict[str, list[dict[str, Any]]]] = OrderedDict()
        self._fingerprints: dict[str, tuple[tuple[str, int, int], ...]] = {}

    @staticmethod
    def _fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
        rows: list[tuple[str, int, int]] = []
        try:
            for filename in PHYSICAL_LOG_FILES:
                stat = (root / filename).stat()
                rows.append((filename, stat.st_size, stat.st_mtime_ns))
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat physical event logs in {root}: {exc}"
            ) from exc
        return tuple(rows)

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        records = read_jsonl(path)
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        return [
            {**record, "record_ref": {"file": relative, "line": index}}
            for index, record in enumerate(records, 1)
        ]

    def bundle(self, episode_id: str) -> dict[str, list[dict[str, Any]]]:
        root = capture_episode_path(episode_id)
        fingerprint = self._fingerprint(root)
        with self._lock:
            cached = self._bundles.get(episode_id)
            if cached is not None and self._fingerprints.get(episode_id) == fingerprint:
                self._bundles.move_to_end(episode_id)
                return cached
        bundle = {
            filename.removesuffix(".jsonl"): self._read_records(root / filename)
            for filename in PHYSICAL_LOG_FILES
        }
        with self._lock:
            self._bundles[episode_id] = bundle
            self._fingerprints[episode_id] = fingerprint
            self._bundles.move_to_end(episode_id)
            while len(self._bundles) > self.max_entries:
                evicted, _ = self._bundles.popitem(last=False)
                self._fingerprints.pop(evicted, None)
        return bundle


def event_chain_episode_path(episode_id: str, data_root: Path) -> Path:
    if not EPISODE_ID_PATTERN.fullmatch(episode_id):
        raise DashboardDataError(f"invalid event-chain episode id: {episode_id}")
    if not data_root.is_dir():
        raise DashboardDataError(
            f"event-chain artifact root does not exist: {data_root}"
        )
    path = data_root / episode_id
    if not path.is_dir():
        raise DashboardDataError(f"event-chain artifact is unavailable: {episode_id}")
    return path


def _event_chain_record_digest(record: dict[str, Any], field: str) -> str:
    payload = dict(record)
    payload.pop(field, None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _event_chain_histogram(
    relations: list[dict[str, Any]], first: str, second: str | None = None
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for relation in relations:
        value: Any = relation.get(first)
        if second is not None:
            value = value.get(second) if isinstance(value, dict) else None
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def validate_event_chain_bundle(
    episode_id: str, manifest: dict[str, Any], relations: list[dict[str, Any]]
) -> None:
    """Reject incomplete or cross-contaminated Stage-A authored-chain artifacts."""

    errors = _event_chain_schema_errors(
        _EVENT_CHAIN_MANIFEST_VALIDATOR,
        manifest,
        location="event_chain_manifest",
    )
    for index, relation in enumerate(relations):
        errors.extend(
            _event_chain_schema_errors(
                _EVENT_CHAIN_RELATION_VALIDATOR,
                relation,
                location=f"event_chain_relations[{index}]",
            )
        )
    if errors:
        raise DashboardDataError(
            "invalid Stage-A event-chain artifact schema: " + "; ".join(errors)
        )
    errors.extend(validate_event_chain_artifacts(manifest, relations))
    identity = (
        manifest.get("episode_id"),
        manifest.get("epi_id"),
        manifest.get("causal_chain_id"),
    )
    if manifest.get("schema_name") != "event_chain_manifest":
        errors.append("manifest schema_name is not event_chain_manifest")
    if manifest.get("schema_version") != EVENT_CHAIN_SCHEMA_VERSION:
        errors.append("manifest schema version is unsupported")
    if identity[0] != episode_id or not all(
        isinstance(value, str) and value for value in identity
    ):
        errors.append("manifest identity is incomplete or differs from request")
    if manifest.get("runtime_validation_status") not in {"PASS", "FAIL"}:
        errors.append("runtime validation status is invalid")

    roots = manifest.get("roots")
    if (
        not isinstance(roots, list)
        or not roots
        or any(not isinstance(root, dict) for root in roots)
    ):
        errors.append("manifest must declare one or more well-formed roots")
        roots = []
    root_ids: set[str] = set()
    for root in roots:
        root_id = root.get("event_id")
        if (
            root.get("kind") != "script_event"
            or not isinstance(root_id, str)
            or not root_id
            or root.get("candidate_event_ids") != [root_id]
            or not isinstance(root.get("intent"), str)
            or not re.fullmatch(r"\d+\..+", str(root.get("intent_stage", "")))
        ):
            errors.append("root endpoint is malformed")
        else:
            root_ids.add(root_id)

    relation_ids: set[str] = set()
    target_ids: set[str] = set()
    source_ids: set[str] = set()
    for index, relation in enumerate(relations):
        location = f"relations[{index}]"
        relation_identity = (
            relation.get("episode_id"),
            relation.get("epi_id"),
            relation.get("causal_chain_id"),
        )
        if relation.get("schema_name") != "event_chain_relation":
            errors.append(f"{location}: invalid schema_name")
        if relation.get("schema_version") != EVENT_CHAIN_SCHEMA_VERSION:
            errors.append(f"{location}: unsupported schema version")
        if relation_identity != identity:
            errors.append(f"{location}: manifest identity mismatch")
        relation_id = relation.get("relation_id")
        if not isinstance(relation_id, str) or not relation_id:
            errors.append(f"{location}: relation_id is absent")
        elif relation_id in relation_ids:
            errors.append(f"{location}: duplicate relation_id")
        else:
            relation_ids.add(relation_id)
        if relation.get("relation_type") not in EVENT_CHAIN_RELATION_TYPES:
            errors.append(f"{location}: physical or unknown relation type is forbidden")

        target = relation.get("target_endpoint")
        if not isinstance(target, dict):
            errors.append(f"{location}: target endpoint is absent")
            continue
        target_id = target.get("event_id")
        if (
            target.get("kind") != "script_event"
            or not isinstance(target_id, str)
            or not target_id
            or target.get("candidate_event_ids") != [target_id]
            or not isinstance(target.get("intent"), str)
            or not re.fullmatch(r"\d+\..+", str(target.get("intent_stage", "")))
        ):
            errors.append(f"{location}: target endpoint is malformed")
        elif target_id in target_ids:
            errors.append(f"{location}: duplicate target event")
        else:
            target_ids.add(target_id)

        source = relation.get("source_endpoint")
        resolution = relation.get("source_resolution")
        alignment = relation.get("runtime_alignment")
        if (
            not isinstance(source, dict)
            or not isinstance(resolution, dict)
            or not isinstance(alignment, dict)
        ):
            errors.append(
                f"{location}: source resolution or runtime alignment is absent"
            )
            continue
        candidates = source.get("candidate_event_ids", [])
        if not isinstance(candidates, list) or any(
            not isinstance(value, str) or not value for value in candidates
        ):
            errors.append(f"{location}: source candidates are malformed")
        else:
            source_ids.update(candidates)
        if len(candidates) != len(set(candidates)):
            errors.append(f"{location}: duplicate source candidate")
        if not isinstance(alignment.get("source_dispatch_ticks"), list):
            errors.append(f"{location}: source dispatch ticks are absent")
        target_tick = alignment.get("target_dispatch_tick")
        if target_tick is not None and (
            not isinstance(target_tick, int) or target_tick not in FORMAL_TICKS
        ):
            errors.append(f"{location}: target dispatch tick violates the formal grid")
        if relation.get("runtime_strict_edge_eligible") is True and not (
            resolution.get("status") == "resolved_unique"
            and alignment.get("status") == "aligned_strict"
            and alignment.get("declaration_consistent") is True
        ):
            errors.append(f"{location}: runtime strict eligibility is invalid")
        if relation.get("relation_type") == "script_declared_context" and not (
            alignment.get("policy") == "any_anchor_at_or_before_target"
            and (
                (
                    source.get("kind") == "context_anchor"
                    and resolution.get("status") == "resolved_context"
                )
                or (
                    not source
                    and resolution.get("status") == "source_declaration_missing"
                )
            )
        ):
            errors.append(
                f"{location}: context relation has a non-context source or policy"
            )

    node_ids = root_ids | target_ids
    if source_ids - node_ids:
        errors.append("source candidates include undeclared event nodes")
    if manifest.get("root_count") != len(roots):
        errors.append("root count mismatch")
    if manifest.get("relation_count") != len(relations):
        errors.append("relation count mismatch")
    if manifest.get("node_count") != len(node_ids):
        errors.append("node count mismatch")

    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        errors.append("manifest summary is absent")
    else:
        expected_summaries = {
            "source_resolution_counts": _event_chain_histogram(
                relations, "source_resolution", "method"
            ),
            "runtime_alignment_counts": _event_chain_histogram(
                relations, "runtime_alignment", "status"
            ),
            "validation_status_counts": _event_chain_histogram(
                relations, "validation_status"
            ),
            "runtime_strict_edge_eligible_count": sum(
                relation.get("runtime_strict_edge_eligible") is True
                for relation in relations
            ),
        }
        for field, expected in expected_summaries.items():
            if summary.get(field) != expected:
                errors.append(f"manifest summary mismatch: {field}")

    provenance = manifest.get("input_provenance")
    if not isinstance(provenance, dict):
        errors.append("input provenance is absent")

    runtime_audit = manifest.get("runtime_audit")
    expected_runtime_status = (
        "PASS"
        if isinstance(runtime_audit, dict)
        and runtime_audit.get("status") == "PASS"
        and all(
            relation.get("validation_status") in RUNTIME_PASS_VALIDATION_STATUSES
            for relation in relations
        )
        else "FAIL"
    )
    if manifest.get("runtime_validation_status") != expected_runtime_status:
        errors.append("runtime validation status disagrees with relation audit")
    if errors:
        raise DashboardDataError(
            "invalid Stage-A event-chain artifact: " + "; ".join(errors)
        )


class EventChainCache:
    """Small cache isolated from the physical semantic-truth episode cache."""

    def __init__(
        self,
        data_root: Path = EVENT_CHAIN_ROOT,
        capture_root: Path = EVENT_CHAIN_CAPTURE_ROOT,
        project_root: Path = PROJECT_ROOT,
        max_entries: int = 4,
    ) -> None:
        self.data_root = data_root
        self.capture_root = capture_root
        self.project_root = project_root
        self.max_entries = max_entries
        self._lock = Lock()
        self._bundles: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._fingerprints: dict[str, tuple[tuple[str, int, int], ...]] = {}

    @staticmethod
    def _fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
        rows: list[tuple[str, int, int]] = []
        try:
            for filename in EVENT_CHAIN_FILES:
                stat = (root / filename).stat()
                rows.append((filename, stat.st_size, stat.st_mtime_ns))
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat event-chain artifacts in {root}: {exc}"
            ) from exc
        return tuple(rows)

    def bundle(self, episode_id: str) -> dict[str, Any]:
        root = event_chain_episode_path(episode_id, self.data_root)
        try:
            actual_entries = {path.name for path in root.iterdir()}
        except OSError as exc:
            raise DashboardDataError(
                f"cannot enumerate event-chain artifacts in {root}: {exc}"
            ) from exc
        expected_entries = set(EVENT_CHAIN_FILES)
        if actual_entries != expected_entries or any(
            not (root / filename).is_file() for filename in EVENT_CHAIN_FILES
        ):
            raise DashboardDataError(
                "event-chain artifact entries must be exactly "
                f"{sorted(expected_entries)}, found {sorted(actual_entries)}"
            )
        fingerprint = self._fingerprint(root)
        with self._lock:
            cached = self._bundles.get(episode_id)
            if cached is not None and self._fingerprints.get(episode_id) == fingerprint:
                self._bundles.move_to_end(episode_id)
                return cached
        manifest = read_json(root / EVENT_CHAIN_FILES[0])
        relations = read_jsonl(root / EVENT_CHAIN_FILES[1])
        validate_event_chain_bundle(episode_id, manifest, relations)
        bundle = {"manifest": manifest, "relations": relations}
        with self._lock:
            self._bundles[episode_id] = bundle
            self._fingerprints[episode_id] = fingerprint
            self._bundles.move_to_end(episode_id)
            while len(self._bundles) > self.max_entries:
                evicted, _ = self._bundles.popitem(last=False)
                self._fingerprints.pop(evicted, None)
        return bundle


def _declared_project_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DashboardDataError(f"{label} must be a project-relative path")
    declared = Path(value)
    if declared.is_absolute():
        raise DashboardDataError(f"{label} must be project-relative: {value}")
    resolved = (PROJECT_ROOT / declared).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise DashboardDataError(f"{label} escapes the project root: {value}") from exc
    if not resolved.is_file():
        raise DashboardDataError(f"{label} is unavailable: {value}")
    return resolved


def _finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DashboardDataError(f"{label} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise DashboardDataError(f"{label} must be finite: {value!r}")
    return number


def _position_enu(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 2:
        raise DashboardDataError(f"{label} must contain at least x and y")
    position = [
        _finite_float(value[0], label=f"{label}[0]"),
        _finite_float(value[1], label=f"{label}[1]"),
    ]
    position.append(
        _finite_float(value[2], label=f"{label}[2]") if len(value) > 2 else 0.0
    )
    return position


def _bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (
        left[2] < right[0]
        or left[0] > right[2]
        or left[3] < right[1]
        or left[1] > right[3]
    )


class SceneMapCache:
    """Project authoritative map geometry into one fixed episode ROI."""

    def __init__(self, max_entries: int = 16) -> None:
        self.max_entries = max_entries
        self._lock = Lock()
        self._episode_payloads: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._episode_fingerprints: dict[str, tuple[tuple[str, int, int], ...]] = {}
        self._map_geometry: dict[str, dict[str, Any]] = {}
        self._map_fingerprints: dict[str, tuple[tuple[str, int, int], ...]] = {}

    @staticmethod
    def _fingerprint(paths: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
        rows: list[tuple[str, int, int]] = []
        try:
            for path in paths:
                stat = path.stat()
                rows.append(
                    (
                        path.relative_to(PROJECT_ROOT).as_posix(),
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
        except OSError as exc:
            raise DashboardDataError(f"cannot stat scene-map artifact: {exc}") from exc
        return tuple(rows)

    @staticmethod
    def _map_paths(map_id: str) -> dict[str, Path]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", map_id):
            raise DashboardDataError(f"invalid map id: {map_id}")
        package_path = (
            PROJECT_ROOT
            / "Config"
            / "LowAltitude"
            / "Maps"
            / map_id
            / "map_package.json"
        )
        package = read_json(package_path)
        if package.get("map_id") != map_id:
            raise DashboardDataError(f"map package id mismatch: {package_path}")
        traffic_bundle_value = package.get("traffic_bundle_dir")
        if not isinstance(traffic_bundle_value, str) or not traffic_bundle_value:
            raise DashboardDataError(
                f"map package lacks traffic_bundle_dir: {package_path}"
            )
        traffic_bundle = (PROJECT_ROOT / traffic_bundle_value).resolve()
        try:
            traffic_bundle.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise DashboardDataError(
                f"traffic bundle escapes project root: {traffic_bundle_value}"
            ) from exc
        source_geojson = package.get("source_geojson")
        if not isinstance(source_geojson, dict):
            raise DashboardDataError(
                f"map package lacks source_geojson: {package_path}"
            )
        return {
            "package": package_path,
            "roads": traffic_bundle / "road_center_samples.csv",
            "transform": traffic_bundle / "road_geojson_transform.json",
            "buildings": _declared_project_path(
                source_geojson.get("building"), label="building GeoJSON"
            ),
            "bounds": _declared_project_path(
                source_geojson.get("bounds"), label="bounds GeoJSON"
            ),
        }

    @staticmethod
    def _load_roads(path: Path) -> list[dict[str, Any]]:
        grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        widths: dict[str, float] = {}
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                required = {"edge_id", "s_m", "x_m", "y_m", "width_m"}
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise DashboardDataError(
                        f"road center samples lack fields {sorted(missing)}: {path}"
                    )
                for line_number, row in enumerate(reader, 2):
                    edge_id = str(row.get("edge_id") or "")
                    if not edge_id:
                        raise DashboardDataError(
                            f"road sample has no edge_id: {path}:{line_number}"
                        )
                    grouped[edge_id].append(
                        (
                            _finite_float(
                                row.get("s_m"), label=f"{path}:{line_number}:s_m"
                            ),
                            _finite_float(
                                row.get("x_m"), label=f"{path}:{line_number}:x_m"
                            ),
                            _finite_float(
                                row.get("y_m"), label=f"{path}:{line_number}:y_m"
                            ),
                        )
                    )
                    width = _finite_float(
                        row.get("width_m"), label=f"{path}:{line_number}:width_m"
                    )
                    widths[edge_id] = max(widths.get(edge_id, 0.0), width)
        except OSError as exc:
            raise DashboardDataError(
                f"cannot read road center samples {path}: {exc}"
            ) from exc
        if not grouped:
            raise DashboardDataError(f"road center samples are empty: {path}")
        return [
            {
                "edge_id": edge_id,
                "width_m": widths[edge_id],
                "points_enu_m": [
                    [x_m, y_m]
                    for _, x_m, y_m in sorted(samples, key=lambda item: item[0])
                ],
            }
            for edge_id, samples in sorted(
                grouped.items(), key=lambda item: natural_key(item[0])
            )
        ]

    @staticmethod
    def _mercator_xy(lon: Any, lat: Any) -> tuple[float, float]:
        longitude = _finite_float(lon, label="GeoJSON longitude")
        latitude = max(
            -85.05112878,
            min(85.05112878, _finite_float(lat, label="GeoJSON latitude")),
        )
        x_m = longitude * WEB_MERCATOR_ORIGIN_SHIFT / 180.0
        radians = math.radians(latitude)
        y_m = (
            math.log(math.tan(math.pi * 0.25 + 0.5 * radians))
            * WEB_MERCATOR_ORIGIN_SHIFT
            / math.pi
        )
        return x_m, y_m

    @classmethod
    def _load_buildings(
        cls, building_path: Path, bounds_path: Path, transform_path: Path
    ) -> list[dict[str, Any]]:
        bounds = read_json(bounds_path)
        features = bounds.get("features")
        if not isinstance(features, list) or not features:
            raise DashboardDataError(f"bounds GeoJSON has no features: {bounds_path}")
        properties = features[0].get("properties")
        bbox = properties.get("bbox") if isinstance(properties, dict) else None
        if not isinstance(bbox, list) or len(bbox) < 4:
            raise DashboardDataError(
                f"bounds GeoJSON lacks WebMercator bbox: {bounds_path}"
            )
        center_x = 0.5 * (
            _finite_float(bbox[0], label="bounds bbox min x")
            + _finite_float(bbox[2], label="bounds bbox max x")
        )
        center_y = 0.5 * (
            _finite_float(bbox[1], label="bounds bbox min y")
            + _finite_float(bbox[3], label="bounds bbox max y")
        )

        transform = read_json(transform_path)
        matrix = transform.get("matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        ):
            raise DashboardDataError(
                f"road GeoJSON transform matrix is malformed: {transform_path}"
            )
        coefficients = [
            [
                _finite_float(value, label="GeoJSON transform coefficient")
                for value in row
            ]
            for row in matrix
        ]

        def transform_coordinate(raw: Any) -> list[float]:
            if not isinstance(raw, list) or len(raw) < 2:
                raise DashboardDataError(
                    f"building GeoJSON coordinate is malformed: {raw!r}"
                )
            mercator_x, mercator_y = cls._mercator_xy(raw[0], raw[1])
            local_x = mercator_x - center_x
            local_y = mercator_y - center_y
            return [
                local_x * coefficients[0][0]
                + local_y * coefficients[1][0]
                + coefficients[2][0],
                local_x * coefficients[0][1]
                + local_y * coefficients[1][1]
                + coefficients[2][1],
            ]

        geojson = read_json(building_path)
        building_features = geojson.get("features")
        if not isinstance(building_features, list):
            raise DashboardDataError(
                f"building GeoJSON has no feature list: {building_path}"
            )
        buildings: list[dict[str, Any]] = []
        for feature_index, feature in enumerate(building_features):
            if not isinstance(feature, dict):
                raise DashboardDataError(
                    f"building feature is malformed: {building_path}:{feature_index}"
                )
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                raise DashboardDataError(
                    f"building geometry is malformed: {building_path}:{feature_index}"
                )
            geometry_type = geometry.get("type")
            coordinates = geometry.get("coordinates")
            if geometry_type == "Polygon":
                polygon_groups = [coordinates]
            elif geometry_type == "MultiPolygon":
                polygon_groups = coordinates
            else:
                raise DashboardDataError(
                    f"unsupported building geometry {geometry_type!r}: "
                    f"{building_path}:{feature_index}"
                )
            if not isinstance(polygon_groups, list):
                raise DashboardDataError(
                    f"building coordinates are malformed: {building_path}:{feature_index}"
                )
            feature_properties = feature.get("properties")
            metadata = (
                feature_properties if isinstance(feature_properties, dict) else {}
            )
            base_id = str(metadata.get("id") or feature_index)
            for polygon_index, rings in enumerate(polygon_groups):
                if not isinstance(rings, list) or not rings:
                    raise DashboardDataError(
                        f"building polygon has no rings: {building_path}:{feature_index}"
                    )
                transformed_rings: list[list[list[float]]] = []
                for ring in rings:
                    if not isinstance(ring, list) or len(ring) < 4:
                        raise DashboardDataError(
                            f"building ring is malformed: {building_path}:{feature_index}"
                        )
                    transformed_rings.append(
                        [transform_coordinate(coordinate) for coordinate in ring]
                    )
                all_points = [point for ring in transformed_rings for point in ring]
                building_bbox = [
                    min(point[0] for point in all_points),
                    min(point[1] for point in all_points),
                    max(point[0] for point in all_points),
                    max(point[1] for point in all_points),
                ]
                buildings.append(
                    {
                        "building_id": (
                            base_id
                            if len(polygon_groups) == 1
                            else f"{base_id}:{polygon_index}"
                        ),
                        "floor": metadata.get("floor"),
                        "height_m": metadata.get("height"),
                        "name": metadata.get("name_zh") or metadata.get("name_en"),
                        "rings_enu_m": transformed_rings,
                        "_bbox": building_bbox,
                    }
                )
        if not buildings:
            raise DashboardDataError(f"building GeoJSON is empty: {building_path}")
        return buildings

    def _load_map_geometry(self, map_id: str) -> tuple[dict[str, Any], dict[str, Path]]:
        paths = self._map_paths(map_id)
        fingerprint = self._fingerprint(paths.values())
        with self._lock:
            cached = self._map_geometry.get(map_id)
            if cached is not None and self._map_fingerprints.get(map_id) == fingerprint:
                return cached, paths
        geometry = {
            "roads": self._load_roads(paths["roads"]),
            "buildings": self._load_buildings(
                paths["buildings"], paths["bounds"], paths["transform"]
            ),
        }
        with self._lock:
            self._map_geometry[map_id] = geometry
            self._map_fingerprints[map_id] = fingerprint
        return geometry, paths

    @staticmethod
    def _validated_roi(plan: dict[str, Any], source: Path) -> list[float]:
        crop = plan.get("runtime_spatial_crop")
        if not isinstance(crop, dict) or crop.get("enabled") is not True:
            raise DashboardDataError(
                f"scenario plan lacks enabled runtime_spatial_crop: {source}"
            )
        raw_bbox = crop.get("bbox_enu_m")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise DashboardDataError(
                f"runtime_spatial_crop bbox is malformed: {source}"
            )
        bbox = [
            _finite_float(value, label=f"runtime ROI bbox[{index}]")
            for index, value in enumerate(raw_bbox)
        ]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise DashboardDataError(f"runtime ROI bbox is inverted: {source}")
        return bbox

    @staticmethod
    def _capture_boundary(
        setup: dict[str, Any], boundary_id: str, source: Path
    ) -> dict[str, Any]:
        matches: list[dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            if value.get("boundary_id") == boundary_id and isinstance(
                value.get("polygon_enu_m"), list
            ):
                matches.append(value)
            for nested in value.values():
                visit(nested)

        visit(setup)
        if len(matches) != 1:
            raise DashboardDataError(
                f"expected one authoritative boundary {boundary_id!r}, found "
                f"{len(matches)} in {source}"
            )
        boundary = matches[0]
        polygon = [
            _position_enu(point, label=f"capture boundary {boundary_id}")[:2]
            for point in boundary["polygon_enu_m"]
        ]
        if len(polygon) < 3:
            raise DashboardDataError(
                f"capture boundary has fewer than three points: {source}"
            )
        boundary_role = boundary.get("boundary_role")
        if not isinstance(boundary_role, str) or not boundary_role:
            raise DashboardDataError(f"capture boundary lacks its role: {source}")
        return {
            "boundary_id": boundary_id,
            "boundary_role": boundary_role,
            "polygon_enu_m": polygon,
            "center_enu_m": boundary.get("center_enu_m"),
            "source_entity_id": boundary.get("source_entity_id"),
        }

    @staticmethod
    def _select_roads(
        roads: list[dict[str, Any]], bbox: list[float]
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for road in roads:
            margin = max(4.0, float(road["width_m"]) * 0.5 + 2.0)
            expanded = [
                bbox[0] - margin,
                bbox[1] - margin,
                bbox[2] + margin,
                bbox[3] + margin,
            ]
            points = road["points_enu_m"]
            runs: list[list[list[float]]] = []
            run: list[list[float]] = []
            for index, point in enumerate(points):
                inside = (
                    expanded[0] <= point[0] <= expanded[2]
                    and expanded[1] <= point[1] <= expanded[3]
                )
                if inside:
                    if not run and index:
                        run.append(points[index - 1])
                    run.append(point)
                    continue
                if run:
                    run.append(point)
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)
            for run_index, dense_points in enumerate(runs):
                if len(dense_points) < 2:
                    continue
                simplified = [dense_points[0]]
                for point in dense_points[1:-1]:
                    if (
                        math.hypot(
                            point[0] - simplified[-1][0],
                            point[1] - simplified[-1][1],
                        )
                        >= 2.5
                    ):
                        simplified.append(point)
                simplified.append(dense_points[-1])
                selected.append(
                    {
                        "id": f"{road['edge_id']}:{run_index}",
                        "edge_id": road["edge_id"],
                        "width_m": road["width_m"],
                        "points_enu_m": simplified,
                    }
                )
        return selected

    @staticmethod
    def _corridors(
        roster: list[dict[str, Any]], bbox: list[float]
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for entity in roster:
            if entity.get("entity_category") != "airspace_corridor":
                continue
            entity_id = str(entity.get("entity_id") or "")
            if not entity_id:
                raise DashboardDataError("airspace corridor entity has no entity_id")
            match = re.fullmatch(r"(.+)_([0-9]+)", entity_id)
            group_id = match.group(1) if match else entity_id
            order = int(match.group(2)) if match else 0
            groups[group_id].append((order, entity))
        corridors: list[dict[str, Any]] = []
        for group_id, members in sorted(
            groups.items(), key=lambda item: natural_key(item[0])
        ):
            ordered = sorted(members, key=lambda item: (item[0], item[1]["entity_id"]))
            points = [
                _position_enu(
                    entity.get("initial_position_enu_m"),
                    label=f"corridor {entity['entity_id']} position",
                )
                for _, entity in ordered
            ]
            corridor_bbox = [
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
            ]
            if not _bbox_intersects(corridor_bbox, bbox):
                continue
            corridors.append(
                {
                    "id": group_id,
                    "entity_ids": [entity["entity_id"] for _, entity in ordered],
                    "points_enu_m": points,
                    "altitude_range_m": [
                        min(point[2] for point in points),
                        max(point[2] for point in points),
                    ],
                }
            )
        return corridors

    @staticmethod
    def _constraint_records(
        roster: list[dict[str, Any]], boundary: dict[str, Any]
    ) -> list[dict[str, Any]]:
        constraints: list[dict[str, Any]] = []
        for entity in roster:
            kind = str(entity.get("entity_kind") or "")
            if not kind.endswith(".no_fly_zone"):
                continue
            entity_id = str(entity.get("entity_id") or "")
            record = {
                "entity_id": entity_id,
                "entity_category": entity.get("entity_category"),
                "entity_kind": kind,
                "position_enu_m": _position_enu(
                    entity.get("initial_position_enu_m"),
                    label=f"no-fly entity {entity_id} position",
                ),
            }
            if entity_id == boundary["boundary_id"]:
                record["polygon_enu_m"] = boundary["polygon_enu_m"]
            constraints.append(record)
        return constraints

    def payload(self, episode_id: str) -> dict[str, Any]:
        root = capture_episode_path(episode_id)
        manifest_path = root / "episode_manifest.json"
        plan_path = root / "scenario_plan.json"
        roster_path = root / "global_entity_roster.json"
        manifest = read_json(manifest_path)
        plan = read_json(plan_path)
        if (
            manifest.get("episode_id") != episode_id
            or plan.get("episode_id") != episode_id
        ):
            raise DashboardDataError(f"scene-map episode id mismatch: {episode_id}")
        map_id = str(plan.get("map_id") or "")
        if manifest.get("map_id") != map_id:
            raise DashboardDataError(f"scene-map map id mismatch: {episode_id}")
        map_paths = self._map_paths(map_id)
        source_setup_path = _declared_project_path(
            manifest.get("source_scene_setup_path"), label="source scene setup"
        )
        episode_fingerprint = self._fingerprint(
            (
                manifest_path,
                plan_path,
                roster_path,
                source_setup_path,
                *map_paths.values(),
            )
        )
        with self._lock:
            cached = self._episode_payloads.get(episode_id)
            if (
                cached is not None
                and self._episode_fingerprints.get(episode_id) == episode_fingerprint
            ):
                self._episode_payloads.move_to_end(episode_id)
                return cached

        roster_payload = read_json(roster_path)
        setup = read_json(source_setup_path)
        boundary_id = str(plan.get("capture_boundary_id") or "")
        if manifest.get("capture_boundary_id") != boundary_id:
            raise DashboardDataError(f"scene-map boundary id mismatch: {episode_id}")
        roster = roster_payload.get("entities")
        if not isinstance(roster, list):
            raise DashboardDataError(
                f"global entity roster is malformed: {roster_path}"
            )
        bbox = self._validated_roi(plan, plan_path)
        boundary = self._capture_boundary(setup, boundary_id, source_setup_path)
        map_geometry, map_paths = self._load_map_geometry(map_id)
        buildings = [
            {key: value for key, value in building.items() if key != "_bbox"}
            for building in map_geometry["buildings"]
            if _bbox_intersects(building["_bbox"], bbox)
        ]
        roads = self._select_roads(map_geometry["roads"], bbox)
        corridors = self._corridors(roster, bbox)
        constraints = self._constraint_records(roster, boundary)
        no_fly_ids = {record["entity_id"] for record in constraints}
        boundary["classification"] = (
            "no_fly_zone" if boundary_id in no_fly_ids else "capture_boundary"
        )
        roster_counts = Counter(
            str(entity.get("entity_category") or "unknown") for entity in roster
        )
        payload = {
            "schema_name": "scene_map_dashboard_view",
            "schema_version": "1.0.0",
            "episode_id": episode_id,
            "map_id": map_id,
            "roi": {
                "bbox_enu_m": bbox,
                "width_m": bbox[2] - bbox[0],
                "height_m": bbox[3] - bbox[1],
            },
            "capture_boundary": boundary,
            "constraints": constraints,
            "roads": roads,
            "buildings": buildings,
            "corridors": corridors,
            "roster_counts": dict(sorted(roster_counts.items())),
            "counts": {
                "road_segments": len(roads),
                "buildings": len(buildings),
                "corridors": len(corridors),
                "no_fly_zones": len(constraints),
            },
            "sources": [
                path.relative_to(PROJECT_ROOT).as_posix()
                for path in (
                    plan_path,
                    roster_path,
                    source_setup_path,
                    map_paths["roads"],
                    map_paths["buildings"],
                    map_paths["bounds"],
                    map_paths["transform"],
                )
            ],
        }
        with self._lock:
            self._episode_payloads[episode_id] = payload
            self._episode_fingerprints[episode_id] = episode_fingerprint
            self._episode_payloads.move_to_end(episode_id)
            while len(self._episode_payloads) > self.max_entries:
                evicted, _ = self._episode_payloads.popitem(last=False)
                self._episode_fingerprints.pop(evicted, None)
        return payload


EPISODE_CACHE = EpisodeBundleCache()
EVENT_CHAIN_CACHE = EventChainCache()
PHYSICAL_LOG_CACHE = PhysicalLogCache()
TRUTH_FRAME_CACHE = TruthFrameCache()
SCENE_MAP_CACHE = SceneMapCache()


def normalize_tick(raw_tick: str | int | None) -> int:
    try:
        tick = int(raw_tick if raw_tick is not None else 0)
    except (TypeError, ValueError) as exc:
        raise DashboardDataError(f"tick must be an integer: {raw_tick!r}") from exc
    if tick not in FORMAL_TICKS:
        raise DashboardDataError("tick must be in 0..900 and divisible by 5")
    return tick


def transition_tick(transition: dict[str, Any]) -> int:
    return int(transition["to_tick"])


_ARRAY_INDEX_SUFFIX_RE = re.compile(r"\[\d+\]$")


def flatten_state_values(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [
            row
            for key, nested in value.items()
            for row in flatten_state_values(
                nested, f"{prefix}.{key}" if prefix else str(key)
            )
        ]
    if isinstance(value, list):
        return [
            row
            for index, nested in enumerate(value)
            for row in flatten_state_values(nested, f"{prefix}[{index}]")
        ]
    leaf = _ARRAY_INDEX_SUFFIX_RE.sub("", prefix.rsplit(".", 1)[-1])
    return [{"name": prefix, "value": value, "unit": unit_for_parameter(leaf)}]


def timeline_payload(episode_id: str, tick: int) -> dict[str, Any]:
    bundle = EPISODE_CACHE.bundle(episode_id)
    events = bundle["event_occurrences"]
    index = bundle["_index"]

    current = index["predicate_snapshots"][tick]

    exact_transitions = index["transitions_by_tick"].get(tick, [])
    exact_events = index["events_by_tick"].get(tick, [])
    exact_phases = index["phases_by_tick"].get(tick, [])
    exact_outcomes = index["outcomes_by_tick"].get(tick, [])

    states: list[dict[str, Any]] = []
    for record in EPISODE_CACHE.state_observations(episode_id, tick):
        measurements = flatten_state_values(record["values"])
        primary = next(
            (
                row
                for row in measurements
                if isinstance(row["value"], (int, float))
                and not isinstance(row["value"], bool)
            ),
            measurements[0]
            if measurements
            else {"name": "empty", "value": None, "unit": "scalar"},
        )
        states.append(
            {
                "id": record["observation_id"],
                "api_id": record["observation_family"],
                "name": record["observation_family"],
                "value": primary["value"],
                "unit": primary["unit"],
                "primary_path": primary["name"],
                "measurements": measurements,
                "subject_id": record["subject_id"],
                "subject_kind": record.get("subject_kind"),
                "source_refs": record["source_refs"],
                "record_ref": record["record_ref"],
                "tick": record["tick"],
                "quality": record["quality"],
            }
        )

    truth_frame = TRUTH_FRAME_CACHE.frame(episode_id, tick)
    truth_record_ref = truth_frame["record_ref"]
    entity_states: list[dict[str, Any]] = []
    for entity_index, entity in enumerate(truth_frame["entities"]):
        if not isinstance(entity, dict) or not isinstance(entity.get("entity_id"), str):
            raise DashboardDataError(
                f"render truth entity is malformed for {episode_id} at tick {tick}"
            )
        entity_id = entity["entity_id"]
        entity_states.append(
            {
                "id": f"render_truth:{episode_id}:{tick}:{entity_id}",
                "api_id": "render.truth_entity",
                "name": entity.get("entity_kind") or entity.get("entity_type"),
                "value": entity.get("state"),
                "unit": "scalar",
                "primary_path": "state",
                "values": entity,
                "subject_id": entity_id,
                "subject_kind": (
                    entity.get("entity_category") or entity.get("label_class")
                ),
                "source_refs": [
                    f'{truth_record_ref["file"]}#tick={tick}#entity={entity_id}'
                ],
                "record_ref": {
                    **truth_record_ref,
                    "entity_index": entity_index,
                },
                "tick": tick,
                "quality": {
                    "status": "complete",
                    "authority": "render_truth",
                },
                "state_scope": "exact_tick",
                "state_authority": "truth_frame",
            }
        )

    all_event_markers = [
        {
            "event_id": row["event_id"],
            "tick": row["trigger_tick"],
            "event_family_id": row["event_family_id"],
            "event_type_id": row["event_type_id"],
            "continuity_status": row["continuity_status"],
        }
        for row in events
    ]
    semantic_milestones = [
        {
            "event_id": event["event_id"],
            "event_family_id": event["event_family_id"],
            "event_trigger_tick": event["trigger_tick"],
            "transition_id": phase["transition_id"],
            "predicate_id": phase["predicate_id"],
            "tuple_id": phase["predicate_tuple_id"],
            "bindings": phase["predicate_bindings"],
            "tick": phase["tick"],
            "direction": phase["direction"],
            "phase_kind": phase["phase_kind"],
        }
        for event in events
        for phase in event.get("event_phases", [])
    ]
    event_markers = [row for row in all_event_markers if int(row["tick"]) <= tick]
    transition_markers = [
        {
            "transition_id": row["transition_id"],
            "tick": row["to_tick"],
            "predicate_id": row["predicate_id"],
            "tuple_id": row["tuple_id"],
            "direction": row["direction"],
        }
        for row in bundle["predicate_transitions"]
    ]
    state_track = [
        {"tick": state_tick, "record_count": record_count}
        for state_tick, record_count in EPISODE_CACHE.state_tick_counts(
            episode_id
        ).items()
    ]
    return {
        "episode_id": episode_id,
        "tick": tick,
        "formal_ticks": FORMAL_TICKS,
        "window": {"start": max(0, tick - 25), "end": min(900, tick + 25)},
        "states": states,
        "entity_states": entity_states,
        "state_track": state_track,
        "predicates": sorted(
            ({**row, "tick": tick} for row in current),
            key=lambda row: (row["predicate_id"], row["tuple_id"]),
        ),
        "transitions": exact_transitions,
        "events": exact_events,
        "phases": exact_phases,
        "outcomes": exact_outcomes,
        "event_markers": event_markers,
        "all_event_markers": all_event_markers,
        "semantic_milestones": semantic_milestones,
        "transition_markers": transition_markers,
        "counts": {
            "states": len(states),
            "entity_states": len(entity_states),
            "predicates": len(current),
            "transitions": len(exact_transitions),
            "events": len(exact_events),
            "phases": len(exact_phases),
            "outcomes": len(exact_outcomes),
        },
    }


def evidence_unit(predicate_id: str | None, observation_path: str) -> str:
    predicate = predicate_by_id(predicate_id)
    if predicate is None:
        leaf = re.sub(r"\[\d+\]", "", observation_path.rsplit(".", 1)[-1])
        return unit_for_parameter(leaf)
    leaf = re.sub(r"\[\d+\]", "", observation_path.rsplit(".", 1)[-1])
    source_units = {str(source["unit"]) for source in predicate["sources"]}
    if leaf == "value" and len(source_units) == 1:
        return source_units.pop()
    for source in predicate["sources"]:
        field = str(source["field"])
        if field.endswith(leaf) or leaf.endswith(field.rsplit(".", 1)[-1]):
            return str(source["unit"])
    return unit_for_parameter(leaf)


def event_chain_payload(episode_id: str) -> dict[str, Any]:
    """Build a display projection for authored Stage-A relations only."""

    bundle = EVENT_CHAIN_CACHE.bundle(episode_id)
    manifest = bundle["manifest"]
    relations = bundle["relations"]
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for root in manifest["roots"]:
        root_id = root["event_id"]
        nodes_by_id[root_id] = {
            "id": root_id,
            "intent": root["intent"],
            "intent_stage": root["intent_stage"],
            "dispatch_tick": root["dispatch_tick"],
            "runtime_status": root["runtime_status"],
            "is_root": True,
            "scheduling_provenance": root["scheduling_provenance"],
            "realization_ref": root["realization_ref"],
        }

    edges: list[dict[str, Any]] = []
    for relation in relations:
        target = relation["target_endpoint"]
        alignment = relation["runtime_alignment"]
        resolution = relation["source_resolution"]
        source = relation["source_endpoint"]
        target_id = target["event_id"]
        nodes_by_id[target_id] = {
            "id": target_id,
            "intent": target["intent"],
            "intent_stage": target["intent_stage"],
            "dispatch_tick": alignment["target_dispatch_tick"],
            "runtime_status": (
                "fired"
                if alignment["target_dispatch_tick"] is not None
                else "not_fired"
            ),
            "is_root": False,
            "scheduling_provenance": None,
            "realization_ref": alignment["target_realization_ref"],
            "relation_validation_status": relation["validation_status"],
        }
        classifications = [
            "context"
            if relation["relation_type"] == "script_declared_context"
            else "authored_causal"
        ]
        if source.get("kind") == "intent_stage_set":
            classifications.append("stage_set")
        if alignment["status"] == "co_triggered":
            classifications.append("co_triggered")
        if alignment["status"] == "inverted":
            classifications.append("inverted")
        if relation["runtime_strict_edge_eligible"]:
            classifications.append("strict")
        primary_classification = next(
            (
                value
                for value in (
                    "context",
                    "inverted",
                    "co_triggered",
                    "stage_set",
                    "strict",
                )
                if value in classifications
            ),
            "authored_causal",
        )
        edges.append(
            {
                "id": relation["relation_id"],
                "source_ids": list(source.get("candidate_event_ids", [])),
                "target_id": target_id,
                "relation_type": relation["relation_type"],
                "declared_predecessor_intent": relation["declared_predecessor_intent"],
                "source_kind": (
                    source["kind"] if source else "source_declaration_missing"
                ),
                "resolution_method": resolution["method"],
                "resolution_status": resolution["status"],
                "cardinality": resolution["cardinality"],
                "scheduling_dependency_event_id": resolution[
                    "scheduling_dependency_event_id"
                ],
                "runtime_status": alignment["status"],
                "runtime_policy": alignment["policy"],
                "source_dispatch_ticks": alignment["source_dispatch_ticks"],
                "target_dispatch_tick": alignment["target_dispatch_tick"],
                "delta_ticks_min": alignment["delta_ticks_min"],
                "delta_ticks_max": alignment["delta_ticks_max"],
                "declaration_consistent": alignment["declaration_consistent"],
                "validation_status": relation["validation_status"],
                "runtime_strict_edge_eligible": relation[
                    "runtime_strict_edge_eligible"
                ],
                "classifications": classifications,
                "primary_classification": primary_classification,
            }
        )

    if set(nodes_by_id) != {
        source_id for edge in edges for source_id in edge["source_ids"]
    } | {edge["target_id"] for edge in edges} | {
        root["event_id"] for root in manifest["roots"]
    }:
        raise DashboardDataError(
            "event-chain display projection lost one or more event nodes"
        )
    nodes = sorted(
        nodes_by_id.values(),
        key=lambda node: (
            int(str(node["intent_stage"]).split(".", 1)[0]),
            str(node["intent_stage"]),
            str(node["id"]),
        ),
    )
    classification_counts = Counter(
        classification for edge in edges for classification in edge["classifications"]
    )
    return {
        "schema_name": "authored_event_chain_dashboard_view",
        "schema_version": EVENT_CHAIN_SCHEMA_VERSION,
        "stage": "A",
        "episode_id": episode_id,
        "epi_id": manifest["epi_id"],
        "causal_chain_id": manifest["causal_chain_id"],
        "runtime_validation_status": manifest["runtime_validation_status"],
        "event_script_digest_authority": manifest["input_provenance"][
            "event_script_digest_authority"
        ],
        "nodes": nodes,
        "edges": edges,
        "runtime_audit": manifest["runtime_audit"],
        "counts": {
            "nodes": len(nodes),
            "relations": len(edges),
            "roots": len(manifest["roots"]),
            "runtime_strict": sum(
                edge["runtime_strict_edge_eligible"] for edge in edges
            ),
            "classifications": dict(sorted(classification_counts.items())),
            "runtime_alignment": manifest["summary"]["runtime_alignment_counts"],
            "source_resolution": manifest["summary"]["source_resolution_counts"],
            "validation": manifest["summary"]["validation_status_counts"],
        },
    }


def _bounded_integer(
    value: str | int | None, *, default: int, minimum: int, maximum: int, name: str
) -> int:
    if value in {None, ""}:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DashboardDataError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise DashboardDataError(f"{name} must be in {minimum}..{maximum}")
    return parsed


def _search_tokens(query: str | None) -> tuple[str, ...]:
    text = str(query or "").strip().casefold()
    if len(text) > 120:
        raise DashboardDataError(
            "semantic context query must not exceed 120 characters"
        )
    return tuple(token for token in re.split(r"\s+", text) if token)


def _record_search_text(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True).casefold()


def _matches_search(record: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    if not tokens:
        return True
    text = _record_search_text(record)
    return all(token in text for token in tokens)


def _bound_entities(*records: dict[str, Any] | None) -> set[str]:
    entities: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        bindings = record.get("bindings")
        if isinstance(bindings, dict):
            entities.update(
                value for value in bindings.values() if isinstance(value, str) and value
            )
        targets = record.get("target_ids")
        if isinstance(targets, list):
            entities.update(
                value for value in targets if isinstance(value, str) and value
            )
    return entities


def _physical_log_ticks(record: dict[str, Any], kind: str) -> list[int]:
    fields = (
        ("tick", "activated_tick", "recovered_tick", "source_tick")
        if kind == "event_trace"
        else (
            "dispatch_tick",
            "result_tick",
            "evidence_tick",
            "before_tick",
            "after_tick",
        )
    )
    ticks = [record.get(field) for field in fields]
    if kind == "event_realization":
        ticks.extend(
            action.get(field)
            for action in record.get("action_realizations", [])
            if isinstance(action, dict)
            for field in (
                "dispatch_tick",
                "scheduled_tick",
                "result_tick",
                "evidence_tick",
                "terminal_tick",
            )
        )
    return sorted({int(value) for value in ticks if isinstance(value, int)})


def _physical_log_entities(record: dict[str, Any]) -> set[str]:
    entities = {
        value
        for value in record.get("target_ids", [])
        if isinstance(value, str) and value
    }
    scope = record.get("scope")
    if isinstance(scope, dict):
        entities.update(
            value
            for value in scope.get("entities", [])
            if isinstance(value, str) and value
        )
    for action in record.get("action_realizations", []):
        if (
            isinstance(action, dict)
            and isinstance(action.get("entity_id"), str)
            and action["entity_id"]
        ):
            entities.add(action["entity_id"])
    return entities


def _physical_log_projection(
    record: dict[str, Any], kind: str, selected_tick: int
) -> dict[str, Any]:
    ticks = _physical_log_ticks(record, kind)
    raw = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "source_truth_snapshots_by_tick",
            "render_truth_snapshots_by_tick",
        }
    }
    title = record.get("title")
    if not title and isinstance(record.get("payload"), dict):
        title = record["payload"].get("title")
    return {
        "kind": kind,
        "event_id": record.get("event_id") or record.get("source_event_id"),
        "title": title,
        "intent": record.get("intent"),
        "status": record.get("status") or "materialized",
        "ticks": ticks,
        "nearest_tick": min(ticks, key=lambda value: abs(value - selected_tick))
        if ticks
        else None,
        "target_ids": sorted(_physical_log_entities(record)),
        "action_count": len(record.get("action_realizations", [])),
        "record_ref": record["record_ref"],
        "raw": raw,
    }


def _artifact_file_rows(
    episode_id: str, bundle: dict[str, Any]
) -> list[dict[str, Any]]:
    objective_root = episode_path(episode_id)
    capture_root = capture_episode_path(episode_id)
    declarations = (
        (objective_root / "l0_predicate_state.jsonl", "L0 state", "tick offset"),
        (objective_root / "evidence_observations.jsonl", "evidence", "truth-id offset"),
        (
            objective_root / "predicate_transitions.jsonl",
            "predicate transitions",
            "cached",
        ),
        (objective_root / "semantic_graph_deltas.jsonl", "graph delta", "cached"),
        (objective_root / "event_occurrences.jsonl", "semantic events", "cached"),
        (capture_root / "event_trace.jsonl", "physical trace", "small-log cache"),
        (
            capture_root / "event_realization.jsonl",
            "event realization",
            "small-log cache",
        ),
    )
    count_by_name = {
        "predicate_transitions.jsonl": len(bundle["predicate_transitions"]),
        "semantic_graph_deltas.jsonl": len(bundle["semantic_graph_deltas"]),
        "event_occurrences.jsonl": len(bundle["event_occurrences"]),
    }
    rows: list[dict[str, Any]] = []
    for path, label, access in declarations:
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise DashboardDataError(
                f"cannot stat semantic context source {path}: {exc}"
            )
        rows.append(
            {
                "file": path.relative_to(PROJECT_ROOT).as_posix(),
                "label": label,
                "access": access,
                "size_bytes": size,
                "record_count": count_by_name.get(path.name),
            }
        )
    return rows


def semantic_context_payload(
    episode_id: str, tick: int, query: dict[str, str]
) -> dict[str, Any]:
    """Join one predicate state or transition to its exact source-side context."""

    bundle = EPISODE_CACHE.bundle(episode_id)
    transitions = bundle["predicate_transitions"]
    index = bundle["_index"]
    predicate_id = str(query.get("predicateId") or "").strip() or None
    tuple_id = str(query.get("tupleId") or "").strip() or None
    transition_id = str(query.get("transitionId") or "").strip() or None
    event_id = str(query.get("eventId") or "").strip() or None
    window = _bounded_integer(
        query.get("window"), default=25, minimum=0, maximum=100, name="window"
    )
    limit = _bounded_integer(
        query.get("limit"), default=40, minimum=1, maximum=100, name="limit"
    )
    tokens = _search_tokens(query.get("q"))

    selected_transition: dict[str, Any] | None = None
    if transition_id:
        selected_transition = index["transitions_by_id"].get(transition_id)
        if selected_transition is None:
            raise DashboardDataError(f"unknown transition id: {transition_id}")
        predicate_id = selected_transition["predicate_id"]
        tuple_id = selected_transition["tuple_id"]
        tick = int(selected_transition["to_tick"])

    snapshot = index["predicate_snapshots"][tick]
    selected_assertion = (
        next(
            (
                row
                for row in snapshot
                if (not predicate_id or row["predicate_id"] == predicate_id)
                and (not tuple_id or row["tuple_id"] == tuple_id)
            ),
            None,
        )
        if predicate_id or tuple_id
        else None
    )
    if selected_assertion is not None:
        predicate_id = selected_assertion["predicate_id"]
        tuple_id = selected_assertion["tuple_id"]

    if predicate_id and tuple_id:
        predicate_changes = [
            row
            for row in transitions
            if row["predicate_id"] == predicate_id and row["tuple_id"] == tuple_id
        ]
    else:
        predicate_changes = [
            row for row in transitions if abs(int(row["to_tick"]) - tick) <= window
        ]
    predicate_changes = sorted(
        predicate_changes, key=lambda row: (int(row["to_tick"]), row["predicate_id"])
    )[:100]

    linked_events: list[dict[str, Any]] = []
    selected_event = None
    if event_id:
        selected_event = next(
            (
                row
                for row in bundle["event_occurrences"]
                if row.get("event_id") == event_id
            ),
            None,
        )
        if selected_event is not None:
            linked_events.append(selected_event)
    for event in bundle["event_occurrences"]:
        if selected_event is not None and event is selected_event:
            continue
        transition_match = bool(
            selected_transition
            and selected_transition["transition_id"]
            in event.get("supporting_transition_ids", [])
        )
        phase_match = any(
            phase.get("predicate_id") == predicate_id
            and (not tuple_id or phase.get("predicate_tuple_id") == tuple_id)
            for phase in event.get("event_phases", [])
        )
        tick_match = int(event["trigger_tick"]) == tick
        if transition_match or phase_match or (not predicate_id and not event_id and tick_match):
            linked_events.append(event)

    evidence_ids: list[str] = []
    if selected_transition:
        evidence_ids.extend(selected_transition.get("evidence_truth_ids", []))
    if selected_assertion and selected_assertion.get("truth_id"):
        evidence_ids.append(selected_assertion["truth_id"])
    for event in linked_events:
        evidence_ids.extend(event.get("supporting_truth_ids", []))
    evidence_ids = unique(evidence_ids)[:40]
    evidence = list(EPISODE_CACHE.evidence(episode_id, evidence_ids).values())
    if not evidence:
        bound = _bound_entities(selected_transition, selected_assertion, *linked_events)
        if bound:
            evidence = [
                record
                for record in EPISODE_CACHE.state_observations(episode_id, tick)
                if record.get("subject_id") in bound
            ]
    evidence.sort(key=lambda row: (int(row.get("tick", tick)), row["observation_id"]))
    evidence = [
        {
            **row,
            "measurements": [
                {
                    "path": path,
                    "value": value,
                    "unit": evidence_unit(row.get("metric_id") or predicate_id, path),
                }
                for path, value in row.get("numeric_values", {}).items()
            ],
        }
        for row in evidence
        if _matches_search(row, tokens)
    ][:limit]

    entities = _bound_entities(selected_transition, selected_assertion, *linked_events)
    state_ticks = {tick}
    if selected_transition:
        state_ticks.update(
            (int(selected_transition["from_tick"]), int(selected_transition["to_tick"]))
        )
    state_records: list[tuple[int, dict[str, Any]]] = []
    for state_tick in sorted(state_ticks):
        for record in EPISODE_CACHE.state_observations(episode_id, state_tick):
            if not _matches_search(record, tokens):
                continue
            text = _record_search_text(record)
            score = 0
            if record.get("subject_id") in entities:
                score += 8
            score += 3 * sum(entity.casefold() in text for entity in entities)
            if predicate_id:
                fragments = [
                    fragment
                    for fragment in re.split(r"[._]", predicate_id.casefold())
                    if len(fragment) > 3
                ]
                score += sum(fragment in text for fragment in fragments)
            if tokens:
                score += 10
            if not entities and not predicate_id:
                score += 1
            if score:
                state_records.append((score, record))
    state_records.sort(
        key=lambda item: (-item[0], int(item[1]["tick"]), item[1]["subject_id"])
    )
    state_rows = [
        {
            "observation_id": record["observation_id"],
            "observation_family": record["observation_family"],
            "subject_id": record["subject_id"],
            "subject_kind": record.get("subject_kind"),
            "tick": record["tick"],
            "quality": record["quality"],
            "values": record["values"],
            "source_refs": record["source_refs"],
            "record_ref": record["record_ref"],
            "relevance_score": score,
        }
        for score, record in state_records[:limit]
    ]

    log_rows: list[tuple[int, dict[str, Any]]] = []
    physical_logs = PHYSICAL_LOG_CACHE.bundle(episode_id)
    for kind in ("event_trace", "event_realization"):
        for record in physical_logs[kind]:
            if not _matches_search(record, tokens):
                continue
            ticks = _physical_log_ticks(record, kind)
            distance = min((abs(value - tick) for value in ticks), default=901)
            if not tokens and distance > window:
                continue
            shared_entities = entities & _physical_log_entities(record)
            score = (12 if distance == 0 else 6 if distance <= window else 0) + (
                8 * len(shared_entities)
            )
            if tokens:
                score += 10
            if not entities and not predicate_id and distance <= window:
                score += 1
            if score:
                log_rows.append((score, _physical_log_projection(record, kind, tick)))
    log_rows.sort(
        key=lambda item: (
            -item[0],
            abs((item[1]["nearest_tick"] or tick) - tick),
            item[1]["kind"],
        )
    )
    logs = [{**row, "relevance_score": score} for score, row in log_rows[:limit]]

    semantic_operations: list[dict[str, Any]] = []
    for delta in index["semantic_deltas_by_tick"].get(tick, []):
        for operation in delta["operations"]:
            if predicate_id and operation.get("predicate_id") != predicate_id:
                continue
            if tuple_id and operation.get("tuple_id") != tuple_id:
                continue
            semantic_operations.append(
                {
                    **operation,
                    "record_ref": {
                        "file": (
                            f"aw_data/objective_semantic_truth/{episode_id}/"
                            "semantic_graph_deltas.jsonl"
                        ),
                        "tick": tick,
                        "delta_id": delta["delta_id"],
                    },
                }
            )

    return {
        "schema_name": "semantic_context_dashboard_view",
        "schema_version": "1.0.0",
        "episode_id": episode_id,
        "tick": tick,
        "selection": {
            "predicate_id": predicate_id,
            "tuple_id": tuple_id,
            "transition_id": selected_transition.get("transition_id")
            if selected_transition
            else None,
            "entities": sorted(entities),
        },
        "assertion": {**selected_assertion, "tick": tick}
        if selected_assertion
        else None,
        "transition": selected_transition,
        "predicate_changes": predicate_changes,
        "linked_events": linked_events,
        "semantic_operations": semantic_operations,
        "evidence_records": evidence,
        "state_records": state_rows,
        "physical_logs": logs,
        "files": _artifact_file_rows(episode_id, bundle),
        "query": {
            "text": " ".join(tokens),
            "window": window,
            "limit": limit,
            "state_ticks": sorted(state_ticks),
        },
        "counts": {
            "predicate_changes": len(predicate_changes),
            "linked_events": len(linked_events),
            "semantic_operations": len(semantic_operations),
            "evidence_records": len(evidence),
            "state_records": len(state_rows),
            "physical_logs": len(logs),
        },
    }


def closure_payload(episode_id: str) -> dict[str, Any]:
    bundle = EPISODE_CACHE.bundle(episode_id)
    closure = bundle["closure"]
    contract = read_json(OBJECTIVE_CONTRACT_PATH)
    epi_contract = contract.get("epi_contracts", {}).get(closure["epi_id"], {})
    family_contracts = contract.get("event_families", {})
    missing_fields: list[dict[str, Any]] = []
    for stage in closure.get("stages", []):
        if stage.get("stage_goal_status") not in {"FAIL", "UNKNOWN"}:
            continue
        declared = stage.get("missing_fields")
        fields = (
            declared
            if isinstance(declared, list)
            else family_contracts.get(stage.get("event_family_id"), {}).get(
                "required_state_fields", []
            )
        )
        source = "closure_stage" if isinstance(declared, list) else "contract_candidate"
        for field in fields:
            missing_fields.append(
                {
                    "stage_key": stage.get("stage_key"),
                    "field": field,
                    "reason": stage.get(
                        "status_reason", "required state field is not evidenced"
                    ),
                    "source": source,
                }
            )
    return {
        **closure,
        "motivation": epi_contract.get("motivation"),
        "contract": epi_contract,
        "missing_fields": missing_fields,
    }


def event_lookup(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return bundle["_index"]["events_by_id"]


def transition_lookup(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return bundle["_index"]["transitions_by_id"]


def compact_numeric_label(record: dict[str, Any] | None) -> str:
    if not record:
        return "数值证据未物化"
    values = record.get("numeric_values", {})
    if not values:
        return f"{record.get('subject_id', '对象')} · 无数值字段"
    path, value = next(iter(values.items()))
    name = path.rsplit(".", 1)[-1]
    return f"{name} = {value}"


def semantic_chain_payload(episode_id: str, tick: int) -> dict[str, Any]:
    bundle = EPISODE_CACHE.bundle(episode_id)
    index = bundle["_index"]
    transitions_by_id = transition_lookup(bundle)
    outcomes_by_event = index["outcomes_by_event"]
    phases_by_event = index["phases_by_event"]
    visible_events = [
        row for row in bundle["event_occurrences"] if int(row["trigger_tick"]) <= tick
    ]
    lifecycle_by_event: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for event in visible_events:
        event_id = event["event_id"]
        lifecycle_by_event[event_id] = {
            "phases": [
                row
                for row in phases_by_event.get(event_id, [])
                if int(row["tick"]) <= tick
            ],
            "outcomes": [
                row
                for row in outcomes_by_event.get(event_id, [])
                if row.get("terminal_tick") is None or int(row["terminal_tick"]) <= tick
            ],
        }
    truth_ids = {
        truth_id
        for event in visible_events
        for record in [
            event,
            *lifecycle_by_event[event["event_id"]]["phases"],
            *lifecycle_by_event[event["event_id"]]["outcomes"],
        ]
        for transition_id in record.get("supporting_transition_ids", [])
        for truth_id in transitions_by_id.get(transition_id, {}).get(
            "evidence_truth_ids", []
        )
    }
    truth_ids.update(
        truth_id
        for event in visible_events
        for record in [
            event,
            *lifecycle_by_event[event["event_id"]]["phases"],
            *lifecycle_by_event[event["event_id"]]["outcomes"],
        ]
        for truth_id in record.get("supporting_truth_ids", [])
    )
    evidence = EPISODE_CACHE.evidence(episode_id, truth_ids)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    added_node_ids: list[str] = []
    added_edge_ids: list[str] = []

    def add_node(node: dict[str, Any]) -> None:
        nodes.append(node)
        if int(node.get("tick", -1)) == tick:
            added_node_ids.append(node["id"])

    def add_edge(source: str, target: str, label: str, status: str = "proven") -> None:
        edge_id = f"edge:{source}>{target}"
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "label": label,
                "status": status,
            }
        )
        if target in added_node_ids:
            added_edge_ids.append(edge_id)

    def support_status(
        record: dict[str, Any],
        record_tick: int,
        expected_bindings: dict[str, Any],
        *,
        allow_transition_after_record: bool = False,
    ) -> str:
        transition_ids = record.get("supporting_transition_ids", [])
        truth_ids = record.get("supporting_truth_ids", [])
        if not transition_ids and not truth_ids:
            return "unknown"
        if any(truth_id not in evidence for truth_id in truth_ids):
            return "unknown"
        expected_episode = record.get("episode_id")
        for transition_id in transition_ids:
            transition = transitions_by_id.get(transition_id)
            if transition is None:
                return "unknown"
            transition_bindings = dict(transition.get("bindings", {}))
            if (
                not transition_bindings
                and expected_bindings
            ) or any(
                expected_bindings.get(role) != value
                for role, value in transition_bindings.items()
            ):
                return "unknown"
            if (
                expected_episode is not None
                and transition.get("episode_id") is not None
                and transition["episode_id"] != expected_episode
            ):
                return "unknown"
            if (
                not allow_transition_after_record
                and int(transition["to_tick"]) > record_tick
            ):
                return "unknown"
            transition_truth_ids = transition.get("evidence_truth_ids", [])
            if len(transition_truth_ids) != 2 or any(
                truth_id not in evidence for truth_id in transition_truth_ids
            ):
                return "unknown"
            evidence_ticks = {
                int(evidence[truth_id]["tick"]) for truth_id in transition_truth_ids
            }
            if evidence_ticks != {
                int(transition["from_tick"]),
                int(transition["to_tick"]),
            }:
                return "unknown"
        for truth_id in truth_ids:
            observation = evidence[truth_id]
            # An occurrence may carry truth IDs for a later lifecycle phase
            # (for example the recovery transition).  Those observations are
            # not evidence against the occurrence at its own tick; they are
            # consumed when the later phase/outcome becomes visible.
            if int(observation["tick"]) > record_tick:
                continue
            subject_id = observation.get("subject_id")
            if subject_id is not None:
                subject_parts = set(str(subject_id).split("|"))
                expected_subjects = {str(value) for value in expected_bindings.values()}
                if not subject_parts <= expected_subjects:
                    return "unknown"
        return "proven"

    def add_derivation_path(
        transition_id: str,
        target_node: str,
        edge_label: str,
        path_status: str,
    ) -> None:
        transition = transitions_by_id.get(transition_id)
        if transition is None:
            return
        predicate_node = f"predicate:{transition_id}"
        transition_node = f"transition:{transition_id}"
        evidence_ids = transition.get("evidence_truth_ids", [])
        for truth_id in evidence_ids:
            observation = evidence.get(truth_id)
            evidence_tick = (
                int(observation["tick"]) if observation else int(transition["to_tick"])
            )
            state_node = f"state:{truth_id}"
            add_node(
                {
                    "id": state_node,
                    "stage": "state",
                    "label": compact_numeric_label(observation),
                    "detail": observation.get("subject_id")
                    if observation
                    else truth_id,
                    "tick": evidence_tick,
                    "truth": None,
                    "status": path_status if observation else "unknown",
                    "selection": {"kind": "truth", "id": truth_id, "truthId": truth_id},
                }
            )
            add_edge(state_node, predicate_node, "数值输入", path_status)
        add_node(
            {
                "id": predicate_node,
                "stage": "predicate",
                "label": transition["predicate_id"],
                "detail": f"{transition['from_value']} → {transition['to_value']}",
                "tick": int(transition["to_tick"]),
                "truth": transition["to_value"],
                "status": path_status,
                "selection": {
                    "kind": "predicate",
                    "id": evidence_ids[-1] if evidence_ids else transition_id,
                    "predicateId": transition["predicate_id"],
                    "truthId": evidence_ids[-1] if evidence_ids else None,
                    "transitionId": transition_id,
                },
            }
        )
        add_node(
            {
                "id": transition_node,
                "stage": "transition",
                "label": f"{transition['direction']} @ {transition['from_tick']}→{transition['to_tick']}",
                "detail": transition_id,
                "tick": int(transition["to_tick"]),
                "truth": transition["to_value"],
                "status": path_status,
                "selection": {
                    "kind": "transition",
                    "id": transition_id,
                    "transitionId": transition_id,
                },
            }
        )
        add_edge(predicate_node, transition_node, "相邻 tick 变化", path_status)
        add_edge(transition_node, target_node, edge_label, path_status)

    def add_direct_truth_support(
        record: dict[str, Any],
        target_node: str,
        target_tick: int,
        edge_label: str,
        record_status: str,
    ) -> None:
        transition_truth_ids = {
            truth_id
            for transition_id in record.get("supporting_transition_ids", [])
            for truth_id in transitions_by_id.get(transition_id, {}).get(
                "evidence_truth_ids", []
            )
        }
        for truth_id in record.get("supporting_truth_ids", []):
            if truth_id in transition_truth_ids:
                continue
            observation = evidence.get(truth_id)
            state_node = f"state:{truth_id}"
            add_node(
                {
                    "id": state_node,
                    "stage": "state",
                    "label": compact_numeric_label(observation),
                    "detail": observation.get("subject_id")
                    if observation
                    else truth_id,
                    "tick": int(observation["tick"]) if observation else target_tick,
                    "truth": None,
                    "status": "proven" if observation else "unknown",
                    "selection": {"kind": "truth", "id": truth_id, "truthId": truth_id},
                }
            )
            add_edge(
                state_node,
                target_node,
                edge_label,
                record_status if observation else "unknown",
            )

    for event in visible_events:
        event_id = event["event_id"]
        event_node = f"event:{event_id}"
        event_bindings = dict(event.get("bindings", {}))
        event_status = support_status(
            event,
            int(event["trigger_tick"]),
            event_bindings,
            allow_transition_after_record=True,
        )
        add_node(
            {
                "id": event_node,
                "stage": "event",
                "label": event["event_family_id"],
                "detail": event["event_type_id"],
                "tick": int(event["trigger_tick"]),
                "truth": "occurred",
                "status": event_status,
                "selection": {"kind": "event", "id": event_id, "eventId": event_id},
            }
        )
        for transition_id in event.get("supporting_transition_ids", []):
            add_derivation_path(transition_id, event_node, "触发", event_status)
        add_direct_truth_support(
            event,
            event_node,
            int(event["trigger_tick"]),
            "事件 truth 支持",
            event_status,
        )

        terminal_sources = [event_node]
        phase_node_status: dict[str, str] = {}
        for phase in lifecycle_by_event[event_id]["phases"]:
            phase_node = f"phase:{phase['phase_id']}"
            phase_tick = int(phase["tick"])
            phase_status = support_status(phase, phase_tick, dict(phase["bindings"]))
            phase_node_status[phase_node] = phase_status
            add_node(
                {
                    "id": phase_node,
                    "stage": "phase",
                    "label": phase["phase_kind"],
                    "detail": phase["predicate_id"],
                    "tick": phase_tick,
                    "truth": phase["direction"],
                    "status": phase_status,
                    "selection": {
                        "kind": "phase",
                        "id": phase["phase_id"],
                        "eventId": event_id,
                    },
                }
            )
            add_edge(
                event_node,
                phase_node,
                "阶段",
                "proven" if event_status == phase_status == "proven" else "unknown",
            )
            for transition_id in phase.get("supporting_transition_ids", []):
                add_derivation_path(transition_id, phase_node, "阶段证据", phase_status)
            if phase["phase_kind"] == "terminal":
                terminal_sources = [phase_node]
        for outcome in lifecycle_by_event[event_id]["outcomes"]:
            terminal_tick = outcome.get("terminal_tick")
            outcome_node = f"outcome:{outcome['outcome_id']}"
            is_pending = terminal_tick is None
            outcome_tick = int(event["trigger_tick"] if is_pending else terminal_tick)
            outcome_status = (
                "unknown"
                if is_pending or outcome_tick < int(event["trigger_tick"])
                else support_status(outcome, outcome_tick, event_bindings)
            )
            add_node(
                {
                    "id": outcome_node,
                    "stage": "outcome",
                    "label": outcome["status"],
                    "detail": (
                        f"{outcome['lifecycle_status']} · terminal_tick 未物化"
                        if is_pending
                        else outcome["lifecycle_status"]
                    ),
                    "tick": int(event["trigger_tick"] if is_pending else terminal_tick),
                    "truth": outcome["status"],
                    "status": outcome_status,
                    "selection": {
                        "kind": "outcome",
                        "id": outcome["outcome_id"],
                        "eventId": event_id,
                    },
                }
            )
            for transition_id in outcome.get("supporting_transition_ids", []):
                add_derivation_path(
                    transition_id, outcome_node, "结果证据", outcome_status
                )
            add_direct_truth_support(
                outcome,
                outcome_node,
                int(event["trigger_tick"] if is_pending else terminal_tick),
                "结果 truth 支持",
                outcome_status,
            )
            for source_node in terminal_sources:
                source_status = (
                    event_status
                    if source_node == event_node
                    else phase_node_status.get(source_node)
                )
                lifecycle_edge_proven = (
                    outcome_status == "proven" and source_status == "proven"
                )
                add_edge(
                    source_node,
                    outcome_node,
                    "终止待观测"
                    if is_pending
                    else "终止/结果"
                    if lifecycle_edge_proven
                    else "结果独立物化",
                    "unknown" if is_pending or not lifecycle_edge_proven else "proven",
                )

    delta_rows = index["semantic_deltas_by_tick"].get(tick, [])
    return {
        "episode_id": episode_id,
        "tick": tick,
        "nodes": keyed_unique(nodes),
        "edges": keyed_unique(edges),
        "delta": {
            "added_nodes": unique(added_node_ids),
            "updated_nodes": unique(
                [
                    node["id"]
                    for node in nodes
                    if node["stage"] == "predicate" and node["tick"] == tick
                ]
            ),
            "added_edges": unique(added_edge_ids),
            "source_delta_ids": [row["delta_id"] for row in delta_rows],
            "operations": [
                operation for row in delta_rows for operation in row["operations"]
            ],
        },
    }


def evidence_values(
    predicate_id: str | None, records: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for path, value in record.get("numeric_values", {}).items():
            metric_id = record.get("metric_id") or predicate_id
            rows.append(
                {
                    "name": path,
                    "path": path,
                    "value": value,
                    "unit": evidence_unit(metric_id, path),
                    "tick": record.get("tick"),
                    "subject_id": record.get("subject_id"),
                    "truth_id": record.get("observation_id"),
                    "source_class": record.get("source_class"),
                    "source_ref": next(iter(record.get("source_refs", [])), None),
                }
            )
    return rows


_TRAJECTORY_SOURCE_REF_PATTERN = re.compile(
    r"^trajectories\.jsonl#tick=(\d+)&entity=([^&]+)$"
)


def evidence_provenance_warnings(
    records: Iterable[dict[str, Any]], *, limit: int = 20
) -> list[dict[str, Any]]:
    """Expose trajectory-reference/tick disagreement without rewriting truth.

    A trajectory reference is expected to point at the same tick as its
    persisted observation.  Static references (event scripts, rosters, and
    parameter records) are intentionally ignored.  The warning is advisory
    for the read-only dashboard; release validators remain the authority for
    accepting a corpus.
    """

    warnings: list[dict[str, Any]] = []
    for record in records:
        try:
            observation_tick = int(record.get("tick"))
        except (TypeError, ValueError):
            continue
        mismatches: list[dict[str, Any]] = []
        for ref in record.get("source_refs", []) or []:
            if not isinstance(ref, str):
                continue
            match = _TRAJECTORY_SOURCE_REF_PATTERN.fullmatch(ref)
            if match is None:
                continue
            reference_tick = int(match.group(1))
            if reference_tick != observation_tick:
                mismatches.append(
                    {
                        "reference_tick": reference_tick,
                        "entity_id": match.group(2),
                        "source_ref": ref,
                    }
                )
        if mismatches:
            warnings.append(
                {
                    "code": "trajectory_reference_tick_mismatch",
                    "observation_id": record.get("observation_id"),
                    "observation_tick": observation_tick,
                    "subject_id": record.get("subject_id"),
                    "mismatches": mismatches,
                }
            )
            if len(warnings) >= limit:
                break
    return warnings


def evidence_payload(episode_id: str, query: dict[str, str]) -> dict[str, Any]:
    bundle = EPISODE_CACHE.bundle(episode_id)
    events = event_lookup(bundle)
    transitions = transition_lookup(bundle)
    index = bundle["_index"]
    event = events.get(query.get("eventId", ""))
    selected_transition = transitions.get(
        query.get("transitionId") or query.get("contextTransitionId", "")
    )
    truth_id = query.get("truthId")
    predicate_id = query.get("predicateId")
    tuple_id = query.get("tupleId")
    tick = normalize_tick(query.get("tick", 0))
    requested_kind = (
        "event"
        if query.get("eventId")
        else "transition"
        if query.get("transitionId")
        else "predicate"
        if query.get("predicateId")
        else "truth"
    )
    if query.get("eventId") and event is None:
        raise DashboardDataError(f"unknown event id: {query['eventId']}")

    truth_transition_context = None
    if truth_id:
        truth_transition_context = next(
            (
                candidate
                for candidate in transitions.values()
                if truth_id in candidate.get("evidence_truth_ids", [])
            ),
            None,
        )
        if selected_transition is None and truth_transition_context is not None:
            predicate_id = truth_transition_context["predicate_id"]
            tuple_id = truth_transition_context["tuple_id"]
            if int(truth_transition_context["to_tick"]) <= tick:
                selected_transition = truth_transition_context

    if selected_transition is not None:
        predicate_id = selected_transition["predicate_id"]
        if event is None:
            event = next(
                (
                    candidate
                    for candidate in events.values()
                    if selected_transition["transition_id"]
                    in (
                        set(candidate.get("supporting_transition_ids", []))
                        | {
                            str(phase.get("transition_id"))
                            for phase in candidate.get("event_phases", ())
                            if isinstance(phase, dict)
                        }
                    )
                ),
                None,
            )
    if (
        selected_transition is not None
        and tuple_id
        and selected_transition["tuple_id"] != tuple_id
    ):
        raise DashboardDataError("transitionId and tupleId identify different tuples")
    if predicate_id and selected_transition is None and not truth_id and not tuple_id:
        raise DashboardDataError("predicate evidence selection requires tupleId")
    if selected_transition is not None:
        predicate_id = selected_transition["predicate_id"]
        tuple_id = selected_transition["tuple_id"]

    snapshot = index["predicate_snapshots"][tick]
    selected_assertion = next(
        (row for row in snapshot if truth_id and row["truth_id"] == truth_id),
        None,
    )
    if selected_assertion is None and predicate_id and tuple_id and not truth_id:
        selected_assertion = next(
            (
                row
                for row in snapshot
                if row["predicate_id"] == predicate_id and row["tuple_id"] == tuple_id
            ),
            None,
        )
    if selected_assertion is None and truth_transition_context is not None:
        truth_tick = tick
        truth_value = None
        if tick == int(truth_transition_context["from_tick"]):
            truth_value = truth_transition_context["from_value"]
        elif tick == int(truth_transition_context["to_tick"]):
            truth_value = truth_transition_context["to_value"]
        selected_assertion = {
            "predicate_id": truth_transition_context["predicate_id"],
            "tuple_id": truth_transition_context["tuple_id"],
            "bindings": truth_transition_context["bindings"],
            "value": truth_value,
            "last_update_tick": truth_tick,
            "supporting_transition_id": None,
            "truth_id": truth_id,
            "source_refs": truth_transition_context.get("source_refs", []),
        }
    if selected_assertion is not None:
        predicate_id = selected_assertion["predicate_id"]
        tuple_id = selected_assertion["tuple_id"]
        exact_transition_id = selected_assertion.get("supporting_transition_id")
        if selected_transition is None and exact_transition_id is not None:
            selected_transition = transitions.get(str(exact_transition_id))
            if selected_transition is None:
                raise DashboardDataError(
                    f"snapshot references missing transition: {exact_transition_id}"
                )
            if (
                selected_transition["predicate_id"] != predicate_id
                or selected_transition["tuple_id"] != tuple_id
                or selected_transition.get("bindings")
                != selected_assertion.get("bindings")
                or int(selected_transition["to_tick"])
                != int(selected_assertion["last_update_tick"])
            ):
                raise DashboardDataError(
                    "snapshot supporting transition does not match its exact assertion"
                )

    event_id = event.get("event_id") if event else None
    lifecycle = {
        "occurrence": event,
        "phases": list(index["phases_by_event"].get(event_id, [])),
        "outcomes": list(index["outcomes_by_event"].get(event_id, [])),
    }
    lifecycle_rows = [row for kind in ("phases", "outcomes") for row in lifecycle[kind]]
    if query.get("eventId"):
        supporting_transition_ids = unique(
            event.get("supporting_transition_ids", [])
            + [
                transition_id
                for row in lifecycle_rows
                for transition_id in row.get("supporting_transition_ids", [])
            ]
        )
    else:
        supporting_transition_ids = (
            [selected_transition["transition_id"]] if selected_transition else []
        )
    selected_transitions = [
        transitions[transition_id]
        for transition_id in supporting_transition_ids
        if transition_id in transitions
    ]
    if not selected_transitions and selected_transition is not None:
        selected_transitions = [selected_transition]
        supporting_transition_ids = [selected_transition["transition_id"]]

    trace_truth_ids = unique(
        ([truth_id] if truth_id else [])
        + (event.get("supporting_truth_ids", []) if event else [])
        + [
            evidence_truth_id
            for transition in selected_transitions
            for evidence_truth_id in transition.get("evidence_truth_ids", [])
        ]
    )
    supporting_truth_ids = unique(
        trace_truth_ids
        + [
            evidence_truth_id
            for row in lifecycle_rows
            for evidence_truth_id in row.get("supporting_truth_ids", [])
        ]
    )
    observations = EPISODE_CACHE.evidence(episode_id, supporting_truth_ids)
    ordered_observations = sorted(
        observations.values(), key=lambda row: int(row["tick"])
    )
    trace_observations = sorted(
        (
            observations[evidence_truth_id]
            for evidence_truth_id in trace_truth_ids
            if evidence_truth_id in observations
        ),
        key=lambda row: int(row["tick"]),
    )
    governed = bundle["summary"].get("governed_parameters", {})
    provenance_warnings = evidence_provenance_warnings(ordered_observations)

    def rule_thresholds(definition: dict[str, Any] | None) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "value": governed.get(name),
                "unit": unit_for_parameter(name),
                "operator": threshold_operator(definition["rule"], name),
                "source_ref": f"world_truth_summary.json#governed_parameters.{name}",
            }
            for name in (
                definition.get("threshold_parameters", []) if definition else []
            )
        ]

    derivations: list[dict[str, Any]] = []
    for transition in selected_transitions:
        definition = predicate_by_id(transition["predicate_id"])
        transition_observations = [
            observations[evidence_truth_id]
            for evidence_truth_id in transition.get("evidence_truth_ids", [])
            if evidence_truth_id in observations
        ]
        before_records = [
            row
            for row in transition_observations
            if int(row["tick"]) == int(transition["from_tick"])
        ]
        after_records = [
            row
            for row in transition_observations
            if int(row["tick"]) == int(transition["to_tick"])
        ]
        assertion_truth_id = (
            truth_id
            if truth_id in transition.get("evidence_truth_ids", [])
            else next(reversed(transition.get("evidence_truth_ids", [])), None)
        )
        assertion_observation = observations.get(assertion_truth_id, {})
        assertion_tick = assertion_observation.get("tick", transition["to_tick"])
        assertion_value = (
            transition["from_value"]
            if int(assertion_tick) == int(transition["from_tick"])
            else transition["to_value"]
            if int(assertion_tick) == int(transition["to_tick"])
            else None
        )
        thresholds = rule_thresholds(definition)
        derivations.append(
            {
                "predicate": {
                    "predicate_id": transition["predicate_id"],
                    "value": assertion_value,
                    "truth_id": assertion_truth_id,
                    "tick": assertion_tick,
                    "tuple_id": transition["tuple_id"],
                },
                "predicate_definition": definition,
                "transition": transition,
                "values": evidence_values(
                    transition["predicate_id"], transition_observations
                ),
                "thresholds": thresholds,
                "expression": definition.get("expression") if definition else None,
                "calculation_rule": definition.get("rule") if definition else None,
                "adjacent": {
                    "before": {
                        "tick": transition["from_tick"],
                        "truth": transition["from_value"],
                        "truth_id": before_records[0].get("observation_id")
                        if before_records
                        else None,
                        "values": evidence_values(
                            transition["predicate_id"], before_records
                        ),
                    },
                    "after": {
                        "tick": transition["to_tick"],
                        "truth": transition["to_value"],
                        "truth_id": after_records[0].get("observation_id")
                        if after_records
                        else None,
                        "values": evidence_values(
                            transition["predicate_id"], after_records
                        ),
                    },
                },
                "truth_change": {
                    "from_truth": transition["from_value"],
                    "to_truth": transition["to_value"],
                    "from_tick": transition["from_tick"],
                    "to_tick": transition["to_tick"],
                    "direction": transition["direction"],
                    "strict_adjacent": int(transition["to_tick"])
                    - int(transition["from_tick"])
                    == 5,
                },
            }
        )

    assertion = None
    if selected_assertion is not None and requested_kind != "event":
        assertion_definition = predicate_by_id(selected_assertion["predicate_id"])
        assertion_observation = observations.get(selected_assertion["truth_id"])
        assertion = {
            "predicate": {
                "predicate_id": selected_assertion["predicate_id"],
                "tuple_id": selected_assertion["tuple_id"],
                "bindings": selected_assertion["bindings"],
                "value": selected_assertion["value"],
                "truth_id": selected_assertion["truth_id"],
                "tick": selected_assertion["last_update_tick"],
            },
            "predicate_definition": assertion_definition,
            "values": evidence_values(
                selected_assertion["predicate_id"],
                [assertion_observation] if assertion_observation else [],
            ),
            "thresholds": rule_thresholds(assertion_definition),
            "expression": assertion_definition.get("expression")
            if assertion_definition
            else None,
            "calculation_rule": assertion_definition.get("rule")
            if assertion_definition
            else None,
        }

    source_refs = unique(
        (event.get("source_refs", []) if event else [])
        + (selected_assertion.get("source_refs", []) if selected_assertion else [])
        + [ref for row in selected_transitions for ref in row.get("source_refs", [])]
        + [ref for row in lifecycle_rows for ref in row.get("source_refs", [])]
        + [ref for row in ordered_observations for ref in row.get("source_refs", [])]
    )
    selection_id = (
        query.get("eventId") or query.get("transitionId") or truth_id or predicate_id
    )
    return {
        "episode_id": episode_id,
        "kind": requested_kind,
        "id": selection_id,
        "event": lifecycle,
        "assertion": assertion,
        "derivations": derivations,
        "subject": {
            "ids": unique(
                row.get("subject_id")
                for row in trace_observations
                if row.get("subject_id")
            ),
        },
        "bindings": event.get("bindings", {})
        if event
        else selected_assertion.get("bindings", {})
        if selected_assertion
        else selected_transition.get("bindings", {})
        if selected_transition
        else {},
        "values": evidence_values(predicate_id, trace_observations),
        "source_refs": source_refs,
        "provenance_warnings": provenance_warnings,
        "supporting_truth_ids": supporting_truth_ids,
        "supporting_transition_ids": supporting_transition_ids,
        "provenance": {
            "event_rule_digest": event.get("rule_digest") if event else None,
            "predicate_rule_digests": unique(
                row.get("rule_digest")
                for row in selected_transitions
                if row.get("rule_digest")
            ),
            "continuity_status": event.get("continuity_status") if event else None,
            "evidence_records": ordered_observations,
        },
    }


def episode_detail(episode_id: str) -> dict[str, Any]:
    bundle = EPISODE_CACHE.bundle(episode_id)
    events = bundle["event_occurrences"]
    index_record = next(
        record for record in EPISODE_INDEX.records() if record.episode_id == episode_id
    )
    return {
        **index_record.as_dict(),
        "transition_count": index_record.record_counts.get("predicate_transitions", 0),
        "summary": bundle["summary"],
        "event_types": sorted({row["event_type_id"] for row in events}),
        "events": events,
        "formal_ticks": FORMAL_TICKS,
    }


def operation_status(kind: str) -> dict[str, Any]:
    status = {
        "available": False,
        "status": "interface_ready",
        "kind": kind,
        "message": "UI 与数据接口已就绪；本交付不启动转换、打包、上传或常驻任务。",
        "credential_policy": "SSH credentials are never accepted or stored by this frontend.",
    }
    if kind == "generation":
        status.update(
            {
                "available_inputs": [
                    {
                        "id": "render_ready_capture_filtered",
                        "label": "UE 交接后的 capture-filtered truth",
                        "path": "aw_data/render_ready_episodes_capture_filtered",
                    }
                ],
                "job": None,
                "package": None,
            }
        )
    if kind == "upload":
        status.update(
            {
                "targets": [
                    {
                        "id": "server_managed_target",
                        "label": "服务端托管目标",
                        "directory": "未配置（由部署端注入，不进入前端文件）",
                    }
                ],
                "packages": [],
                "transfer": None,
            }
        )
    return status


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "AeroWorldSemanticDashboard/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DASHBOARD_ROOT), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def end_headers(self) -> None:
        # The dashboard is served directly from the working tree while the
        # semantic contracts and view modules evolve together.  A browser must
        # never retain an older module graph (for example one that still
        # imports a removed view) and render a shell with no data.  API and PNG
        # responses set their own no-store header in send_json/send_png_bytes;
        # apply the same policy to the static HTML/CSS/ES-module graph.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_png_bytes(
        self,
        body: bytes,
        status: HTTPStatus = HTTPStatus.OK,
        cache_control: str = "public, max-age=120",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            if parsed.path == "/":
                self.path = "/index.html"
            return super().do_GET()
        try:
            plot_match = re.fullmatch(
                r"/api/episodes/([^/]+)/trajectory-plot", parsed.path
            )
            if plot_match is not None:
                tick = normalize_tick(parse_qs(parsed.query).get("tick", [None])[-1])
                png = TRAJECTORY_PLOT_CACHE.png_bytes(unquote(plot_match.group(1)), tick)
                self.send_png_bytes(png)
                return
            self.send_json(self.route_get(parsed.path, parse_qs(parsed.query)))
        except DashboardDataError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.log_error(
                "internal data service error: %s: %s", type(exc).__name__, exc
            )
            self.send_json(
                {"error": "internal data service error"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        request_contracts = {
            "/api/operations/generation/start": {"input_id", "mode"},
            "/api/operations/package/build": {"job_id"},
            "/api/operations/upload/start": {"package_id", "target_id"},
        }
        if parsed.path not in request_contracts:
            self.send_json({"error": "unknown API endpoint"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return
        if not 0 <= length <= 65_536:
            self.send_json(
                {"error": "request body exceeds 64 KiB limit"},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self.send_json(
                {"error": f"invalid JSON body: {exc}"}, HTTPStatus.BAD_REQUEST
            )
            return
        if not isinstance(payload, dict):
            self.send_json(
                {"error": "JSON body must be an object"}, HTTPStatus.BAD_REQUEST
            )
            return
        expected_fields = request_contracts[parsed.path]
        if set(payload) != expected_fields:
            self.send_json(
                {
                    "error": "request fields must exactly match the operation contract",
                    "expected_fields": sorted(expected_fields),
                },
                HTTPStatus.BAD_REQUEST,
            )
            return
        if (
            parsed.path == "/api/operations/generation/start"
            and payload["mode"] != "objective_semantic_only"
        ):
            self.send_json(
                {"error": "generation mode must be objective_semantic_only"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        kind = parsed.path.split("/")[-2]
        response = operation_status(kind)
        response["request_received"] = payload
        self.send_json(response, HTTPStatus.NOT_IMPLEMENTED)

    def route_get(self, path: str, query: dict[str, list[str]]) -> Any:
        single = {key: values[-1] for key, values in query.items() if values}
        if path == "/api/health":
            summary = EPISODE_INDEX.summary()
            return {
                "status": "ok",
                "data_root": str(DATA_ROOT),
                "capture_truth_root": str(
                    EVENT_CHAIN_CAPTURE_ROOT.relative_to(PROJECT_ROOT)
                ),
                "corpus_episode_count": summary["corpus"]["episode_count"],
                "expected_episode_count": summary["corpus"]["expected_episode_count"],
                "read_only": True,
                "formal_tick_contract": "0..900 step 5",
                "cache": EPISODE_CACHE.stats(),
            }
        if path == "/api/summary":
            return EPISODE_INDEX.summary()
        if path == "/api/episodes":
            return EPISODE_INDEX.query(single)
        if path == "/api/episodes/compare":
            episode_ids = [value for value in single.get("ids", "").split(",") if value]
            if not 1 <= len(episode_ids) <= 3:
                raise DashboardDataError("compare requires one to three episode ids")
            return {
                "episodes": [episode_detail(episode_id) for episode_id in episode_ids]
            }
        if path == "/api/catalogs":
            return registry_payload()
        if path == "/api/operations/generation":
            return operation_status("generation")
        if path == "/api/operations/upload":
            return operation_status("upload")

        if path == "/api/arms":
            try:
                return {
                    **arm_catalog(PROJECT_ROOT, allow_partial=True),
                    "project_root": str(PROJECT_ROOT),
                }
            except ArmReviewError as exc:
                raise DashboardDataError(str(exc)) from exc
        if path == "/api/arms/profile":
            profile = single.get("profile")
            seed = single.get("seed")
            if not profile or not seed:
                raise DashboardDataError(
                    "arm profile overview requires profile and seed parameters"
                )
            try:
                return arm_profile_overview(PROJECT_ROOT, profile, seed, catalog_data=arm_catalog(PROJECT_ROOT, allow_partial=True))
            except (ArmProfileError, ArmReviewError) as exc:
                raise DashboardDataError(str(exc)) from exc
        if path == "/api/arms/group":
            profile = single.get("profile")
            seed = single.get("seed")
            group = single.get("group")
            if not profile or not seed or not group:
                raise DashboardDataError(
                    "arm group detail requires profile, seed and group parameters"
                )
            try:
                return arm_group_detail(PROJECT_ROOT, profile, seed, group, catalog_data=arm_catalog(PROJECT_ROOT, allow_partial=True))
            except ArmReviewError as exc:
                raise DashboardDataError(str(exc)) from exc
        if path == "/api/arms/viz":
            profile = single.get("profile")
            seed = single.get("seed")
            group = single.get("group")
            tick = normalize_tick(single.get("tick"))
            if not profile or not seed or not group:
                raise DashboardDataError(
                    "arm viz requires profile, seed, group and tick parameters"
                )
            try:
                catalog_data = arm_catalog(PROJECT_ROOT, allow_partial=True)
                detail = arm_group_detail(
                    PROJECT_ROOT, profile, seed, group, catalog_data=catalog_data)
                payload = arm_viz.arm_viz_payload(detail, PROJECT_ROOT)
                payload["tick"] = tick
                payload["graphs"] = [
                    {
                        **arm_viz.arm_predicate_graph(
                            PROJECT_ROOT, arm_detail, tick, arm_detail["path"]),
                        "arm_key": arm_detail["arm_key"],
                        "arm_id": arm_detail["arm_id"],
                        "mode": arm_detail.get("mode"),
                    }
                    for arm_detail in detail["arms"]
                ]
                return payload
            except ArmReviewError as exc:
                raise DashboardDataError(str(exc)) from exc

        match = re.fullmatch(r"/api/episodes/([^/]+)(?:/(.+))?", path)
        if match is None:
            raise DashboardDataError(f"unknown API endpoint: {path}")
        episode_id = unquote(match.group(1))
        action = match.group(2)
        if action is None:
            return episode_detail(episode_id)
        if action == "timeline":
            return timeline_payload(episode_id, normalize_tick(single.get("tick")))
        if action == "scene-map":
            return SCENE_MAP_CACHE.payload(episode_id)
        if action == "semantic-chain":
            return semantic_chain_payload(
                episode_id, normalize_tick(single.get("tick"))
            )
        if action == "event-chain":
            return event_chain_payload(episode_id)
        if action == "semantic-context":
            return semantic_context_payload(
                episode_id, normalize_tick(single.get("tick")), single
            )
        if action == "evidence":
            return evidence_payload(episode_id, single)
        if action == "closure":
            return closure_payload(episode_id)
        if action == "scene-changes":
            return scene_changes_payload(episode_id, normalize_tick(single.get("tick")))
        raise DashboardDataError(f"unknown episode API action: {action}")


TRAJECTORY_PLOT_DPI = 90
TRAJECTORY_PLOT_SIZE_IN = (11.0, 6.4)
TRAJECTORY_PLOT_CATEGORIES = ("vehicle", "pedestrian", "uav")
TRAJECTORY_CATEGORY_COLORS = {
    "vehicle": "#f2b84b",
    "pedestrian": "#62e6a7",
    "uav": "#35e4f2",
}
TRAJECTORY_LEGEND_LABELS = {
    "vehicle": "vehicle (SUMO)",
    "pedestrian": "pedestrian",
    "uav": "UAV",
}


class TrajectorySeriesCache:
    """Parse capture-filtered trajectories.jsonl into per-entity series (LRU).

    The formal corpus stores one record per entity per simulation tick
    (0..900 step 1).  Parsing is lazy per episode and bounded by an LRU so the
    service never holds the full 307 GB corpus in memory.
    """

    def __init__(self, max_entries: int = 4) -> None:
        self.max_entries = max_entries
        self._lock = Lock()
        self._series: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._stats: Counter[str] = Counter()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {**self._stats, "series_entries": len(self._series)}

    def series(self, episode_id: str) -> dict[str, Any]:
        with self._lock:
            cached = self._series.get(episode_id)
            if cached is not None:
                self._series.move_to_end(episode_id)
                self._stats["series_hits"] += 1
                return cached
        parsed = self._parse(episode_id)
        with self._lock:
            self._series[episode_id] = parsed
            self._series.move_to_end(episode_id)
            while len(self._series) > self.max_entries:
                self._series.popitem(last=False)
                self._stats["series_evictions"] += 1
        self._stats["series_parses"] += 1
        return parsed

    @staticmethod
    def _parse(episode_id: str) -> dict[str, Any]:
        source = capture_episode_path(episode_id) / "trajectories.jsonl"
        if not source.is_file():
            raise DashboardDataError(
                f"capture trajectory artifact is unavailable: {source}"
            )
        entities: dict[str, dict[str, Any]] = {}
        tick_pattern = re.compile(rb'"tick"\s*:\s*(\d+)')
        try:
            handle = source.open("rb")
        except OSError as exc:
            raise DashboardDataError(
                f"cannot read trajectory artifact {source}: {exc}"
            ) from exc
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if not (
                    b'"vehicle"' in line
                    or b'"pedestrian"' in line
                    or b'"uav"' in line
                ):
                    continue
                tick_match = tick_pattern.search(line)
                if tick_match is None or int(tick_match.group(1)) % 5 != 0:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DashboardDataError(
                        f"malformed trajectory record {source}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    continue
                tick = record.get("tick")
                category = str(
                    record.get("entity_category") or record.get("label_class") or ""
                )
                if category not in TRAJECTORY_PLOT_CATEGORIES:
                    continue
                entity_id = record.get("entity_id")
                position = record.get("pos_enu")
                if not isinstance(entity_id, str) or not entity_id:
                    continue
                if not isinstance(position, list) or len(position) < 2:
                    continue
                entry = entities.setdefault(
                    entity_id,
                    {"category": category, "points": []},
                )
                entry["points"].append((tick, float(position[0]), float(position[1])))
        for entry in entities.values():
            entry["points"].sort(key=lambda point: point[0])
        return {"episode_id": episode_id, "source": source, "entities": entities}


class TrajectoryPlotCache:
    """Render matplotlib trajectory PNGs from real capture truth, on demand.

    Every pixel is derived from
    aw_data/render_ready_episodes_capture_filtered/<episode>/trajectories.jsonl
    and, when available, the authoritative scene-map geometry.  Nothing is
    interpolated, synthesised or cached across episodes.  Future positions
    (tick > selected) are never drawn.
    """

    def __init__(self, max_entries: int = 96) -> None:
        self.max_entries = max_entries
        self._lock = Lock()
        self._plots: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._stats: Counter[str] = Counter()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {**self._stats, "plot_entries": len(self._plots)}

    def png_bytes(self, episode_id: str, tick: int) -> bytes:
        tick = normalize_tick(tick)
        key = (episode_id, tick)
        with self._lock:
            cached = self._plots.get(key)
            if cached is not None:
                self._plots.move_to_end(key)
                self._stats["plot_hits"] += 1
                return cached
        png = self._render(episode_id, tick)
        with self._lock:
            self._plots[key] = png
            self._plots.move_to_end(key)
            while len(self._plots) > self.max_entries:
                self._plots.popitem(last=False)
            self._stats["plot_renders"] += 1
        return png

    def _render(self, episode_id: str, tick: int) -> bytes:
        episode_path(episode_id)
        series = TRAJECTORY_SERIES_CACHE.series(episode_id)
        figure = Figure(
            figsize=TRAJECTORY_PLOT_SIZE_IN,
            dpi=TRAJECTORY_PLOT_DPI,
            facecolor="#07131a",
        )
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(1, 1, 1)
        axes.set_facecolor("#050b11")
        try:
            basemap_available = self._basemap(axes, episode_id)
            plotted = self._trajectories(axes, series, tick)
            self._finish_axes(axes, series, tick, plotted, basemap_available)
            buffer = BytesIO()
            figure.savefig(
                buffer,
                format="png",
                dpi=TRAJECTORY_PLOT_DPI,
                facecolor=figure.get_facecolor(),
            )
            return buffer.getvalue()
        finally:
            figure.clf()

    @staticmethod
    def _basemap(axes: Any, episode_id: str) -> bool:
        try:
            payload = SCENE_MAP_CACHE.payload(episode_id)
        except DashboardDataError:
            return False
        for road in payload.get("roads", ()):
            points = road.get("points_enu_m") or []
            if len(points) >= 2:
                axes.plot(
                    [float(point[0]) for point in points],
                    [float(point[1]) for point in points],
                    color="#1a3542",
                    linewidth=0.9,
                    alpha=0.9,
                    zorder=1,
                )
        for corridor in payload.get("corridors", ()):
            points = corridor.get("points_enu_m") or []
            if len(points) >= 2:
                axes.plot(
                    [float(point[0]) for point in points],
                    [float(point[1]) for point in points],
                    color="#3a5f72",
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.55,
                    zorder=1,
                )
        boundary = payload.get("capture_boundary") or {}
        polygon = boundary.get("polygon_enu_m")
        if isinstance(polygon, list) and len(polygon) >= 3:
            axes.add_patch(
                MplPolygon(
                    [(float(point[0]), float(point[1])) for point in polygon],
                    closed=True,
                    fill=False,
                    edgecolor="#f2b84b",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.7,
                    zorder=2,
                )
            )
        bbox = (payload.get("roi") or {}).get("bbox_enu_m")
        if isinstance(bbox, list) and len(bbox) == 4:
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            axes.set_xlim(bbox[0] - width * 0.04, bbox[2] + width * 0.04)
            axes.set_ylim(bbox[1] - height * 0.04, bbox[3] + height * 0.04)
            return True
        return False

    @staticmethod
    def _trajectories(axes: Any, series: dict[str, Any], tick: int) -> int:
        plotted = 0
        polylines: dict[str, list[list[tuple[float, float]]]] = {
            category: [] for category in TRAJECTORY_PLOT_CATEGORIES
        }
        current_by_category: dict[str, list[tuple[float, float]]] = {
            category: [] for category in TRAJECTORY_PLOT_CATEGORIES
        }
        for entry in series["entities"].values():
            category = entry["category"]
            if category not in TRAJECTORY_CATEGORY_COLORS:
                continue
            history = [point for point in entry["points"] if point[0] <= tick]
            if not history:
                continue
            plotted += 1
            if len(history) >= 2:
                polylines[category].append([(point[1], point[2]) for point in history])
            if history[-1][0] == tick:
                current_by_category[category].append((history[-1][1], history[-1][2]))
        for category, lines in polylines.items():
            if not lines:
                continue
            axes.add_collection(
                LineCollection(
                    lines,
                    colors=TRAJECTORY_CATEGORY_COLORS[category],
                    linewidths=1.0,
                    alpha=0.8,
                    zorder=3,
                )
            )
        for category, points in current_by_category.items():
            if not points:
                continue
            axes.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=22,
                color=TRAJECTORY_CATEGORY_COLORS[category],
                edgecolors="#04080c",
                linewidths=0.5,
                zorder=5,
            )
        return plotted

    @staticmethod
    def _finish_axes(
        axes: Any,
        series: dict[str, Any],
        tick: int,
        plotted: int,
        basemap_available: bool,
    ) -> None:
        if not basemap_available:
            axes.autoscale_view()
        handles = [
            Line2D([0], [0], color=color, linewidth=2.0, label=label)
            for category, color in TRAJECTORY_CATEGORY_COLORS.items()
            for label in [TRAJECTORY_LEGEND_LABELS[category]]
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                color="#3a5f72",
                linewidth=1.0,
                linestyle="--",
                label="airspace corridor",
            )
        )
        legend = axes.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            facecolor="#0b1720",
            edgecolor="#1e3440",
            labelcolor="#dce8ec",
            framealpha=0.9,
        )
        legend.set_zorder(6)
        axes.set_title(
            f"{series['episode_id']} · capture-filtered trajectory truth · tick {tick}",
            color="#dce8ec",
            fontsize=13,
            pad=12,
        )
        axes.set_xlabel("ENU East (m)", color="#8ba2ad", fontsize=9)
        axes.set_ylabel("ENU North (m)", color="#8ba2ad", fontsize=9)
        axes.tick_params(colors="#8ba2ad", labelsize=8)
        axes.grid(True, color="#12242f", linewidth=0.5, alpha=0.6)
        axes.set_aspect("equal", adjustable="box")
        source = series["source"].relative_to(PROJECT_ROOT).as_posix()
        axes.figure.text(
            0.012,
            0.012,
            f"source: {source} · entities plotted: {plotted} · "
            "corridor geometry: scene map · no future ticks drawn",
            color="#8ba2ad",
            fontsize=8,
            ha="left",
        )


TRAJECTORY_SERIES_CACHE = TrajectorySeriesCache()
TRAJECTORY_PLOT_CACHE = TrajectoryPlotCache()


def scene_changes_payload(episode_id: str, tick: int) -> dict[str, Any]:
    """Discrete semantic-graph changes materialised exactly at ``tick``."""

    tick = normalize_tick(tick)
    bundle = EPISODE_CACHE.bundle(episode_id)
    deltas = bundle["_index"]["semantic_deltas_by_tick"].get(tick, [])
    operations: list[dict[str, Any]] = []
    for delta in deltas:
        for operation in delta["operations"]:
            operations.append(
                {
                    **operation,
                    "record_ref": {
                        "file": "semantic_graph_deltas.jsonl",
                        "delta_id": delta["delta_id"],
                        "tick": tick,
                    },
                }
            )
    counts = Counter(str(operation.get("operation")) for operation in operations)
    return {
        "schema_name": "scene_changes_dashboard_view",
        "schema_version": "1.0.0",
        "episode_id": episode_id,
        "tick": tick,
        "deltas": [
            {
                "delta_id": delta["delta_id"],
                "tick": delta["tick"],
                "operation_count": delta["operation_count"],
                "previous_delta_id": delta.get("previous_delta_id"),
            }
            for delta in deltas
        ],
        "operations": operations,
        "counts": {
            "deltas": len(deltas),
            "operations": len(operations),
            **dict(sorted(counts.items())),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA_ROOT.is_dir():
        raise SystemExit(f"semantic truth data root does not exist: {DATA_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"AERO WORLD semantic dashboard: http://{args.host}:{args.port}")
    print(f"Read-only data root: {DATA_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
