"""Deterministic L0 UTM state derived from the scenario service plan.

The UTM supplement materializes operational declarations only.  It does not
emit predicate truth or semantic events.  Every record is evaluated at the
formal 0..900, five-tick cadence and is reproducible from the episode roster
and authored scenario contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from Dataset.semantic_truth.provenance import (
    canonical_json,
    digest_file,
    digest_object,
    stable_identifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_TICKS = tuple(range(0, 901, 5))
SCHEMA_VERSION = "1.0.0"


class UtmStateError(ValueError):
    """Raised when a UTM L0 input or generated record is incomplete."""


@dataclass(frozen=True)
class UtmEpisodeArtifacts:
    episode_id: str
    output_dir: Path
    files: Mapping[str, str]
    summary: Mapping[str, Any]


def build_utm_episode_artifacts(
    episode_root: Path,
    output_dir: Path,
) -> UtmEpisodeArtifacts:
    """Build the four governed UTM logs for one formal episode."""

    episode_root = episode_root.resolve()
    manifest_path = episode_root / "episode_manifest.json"
    roster_path = episode_root / "global_entity_roster.json"
    if not manifest_path.is_file() or not roster_path.is_file():
        raise UtmStateError(
            f"UTM inputs are missing under {episode_root}: "
            "episode_manifest.json and global_entity_roster.json are required"
        )
    manifest = _load_object(manifest_path)
    episode_id = str(manifest.get("episode_id") or episode_root.name)
    if episode_id != episode_root.name:
        raise UtmStateError(
            f"manifest episode_id {episode_id!r} differs from directory {episode_root.name!r}"
        )
    script_path = _resolve_scenario_event_script(manifest)
    script = _load_object(script_path)
    roster = _load_object(roster_path)
    entities = roster.get("entities")
    if not isinstance(entities, list):
        raise UtmStateError(f"{roster_path}: entities must be an array")
    uavs = sorted(
        (entity for entity in entities if _entity_category(entity) == "uav"),
        key=lambda entity: str(entity.get("entity_id", "")),
    )
    if not uavs:
        raise UtmStateError(f"{roster_path}: no UAV scope entities")

    scenario_id = str(script.get("scenario_id") or manifest.get("scenario_id") or "")
    duration = manifest.get("duration_ticks")
    if not isinstance(duration, int) or duration != FORMAL_TICKS[-1]:
        raise UtmStateError(
            f"{manifest_path}: duration_ticks must be {FORMAL_TICKS[-1]}"
        )
    service_plan = _load_utm_service_plan(
        script,
        roster_uavs={str(entity["entity_id"]): entity for entity in uavs},
    )
    capture_boundary = _capture_boundary_id(script, manifest)
    if capture_boundary is None:
        raise UtmStateError(
            "UTM OperationalRegion authority requires an explicit capture boundary id"
        )
    input_digest = digest_object(
        {
            "episode_manifest": digest_file(manifest_path),
            "entity_roster": digest_file(roster_path),
            "event_script": digest_file(script_path),
            "formal_ticks": FORMAL_TICKS,
        }
    )
    source_refs = [
        str(script_path.relative_to(PROJECT_ROOT)),
        "global_entity_roster.json",
    ]

    declarations: list[dict[str, Any]] = []
    utm_system_id = stable_identifier(
        "utm_system",
        service_plan["model_id"],
        service_plan["model_version"],
    )
    for plan in service_plan["uav_plans"]:
        uav_id = str(plan["uav_id"])
        corridor_id = str(plan["corridor_id"])
        airspace_id = capture_boundary
        declarations.append(
            {
                "uav_id": uav_id,
                "corridor_id": corridor_id,
                "airspace_id": airspace_id,
                "intent_id": stable_identifier(
                    "utm_operational_intent", episode_id, uav_id, corridor_id
                ),
                "plan_id": stable_identifier(
                    "utm_flight_plan", episode_id, uav_id, corridor_id
                ),
                "authorization_id": stable_identifier(
                    "utm_airspace_authorization", episode_id, uav_id, airspace_id
                ),
                "intent_start_tick": int(plan["intent_start_tick"]),
                "intent_end_tick": int(plan["intent_end_tick"]),
                "authorization_schedule": list(plan["authorization_schedule"]),
                "flight_plan_schedule": list(plan["flight_plan_schedule"]),
            }
        )

    authorization_rows: list[dict[str, Any]] = []
    intent_rows: list[dict[str, Any]] = []
    flight_plan_rows: list[dict[str, Any]] = []
    deconfliction_rows: list[dict[str, Any]] = []
    intent_pair_rows: list[dict[str, Any]] = []
    for tick in FORMAL_TICKS:
        active_intents = {
            declaration["uav_id"]: declaration
            for declaration in declarations
            if declaration["intent_start_tick"] <= tick <= declaration["intent_end_tick"]
        }
        conflicts_by_uav: dict[str, list[str]] = {
            declaration["uav_id"]: [] for declaration in declarations
        }
        active_declarations = sorted(
            active_intents.values(), key=lambda row: str(row["uav_id"])
        )
        for index, first in enumerate(active_declarations):
            for second in active_declarations[index + 1 :]:
                if first["corridor_id"] != second["corridor_id"]:
                    continue
                conflicts_by_uav[first["uav_id"]].append(second["uav_id"])
                conflicts_by_uav[second["uav_id"]].append(first["uav_id"])
        for declaration in declarations:
            common = _common(
                episode_id,
                tick,
                input_digest,
                source_refs,
            )
            common.update(
                {
                    "parameter_digest": digest_object(service_plan),
                    "model_id": str(service_plan["model_id"]),
                    "model_version": str(service_plan["model_version"]),
                }
            )
            authorization_rows.append(
                {
                    **common,
                    "schema_name": "utm_airspace_authorization_state",
                    "authorization_id": declaration["authorization_id"],
                    "authorization_ontology_class_id": "world:AirspaceAuthorization",
                    "uav_id": declaration["uav_id"],
                    "uav_ontology_class_id": "world:UnmannedAircraft",
                    "airspace_id": declaration["airspace_id"],
                    "airspace_ontology_class_id": "world:OperationalRegion",
                    "status": _status_at_tick(
                        declaration["authorization_schedule"],
                        field="status",
                        tick=tick,
                    ),
                    "valid_from_tick": FORMAL_TICKS[0],
                    "valid_to_tick": FORMAL_TICKS[-1],
                }
            )
            intent_rows.append(
                {
                    **common,
                    "schema_name": "utm_operational_intent_state",
                    "intent_id": declaration["intent_id"],
                    "intent_ontology_class_id": "world:OperationalIntent",
                    "uav_id": declaration["uav_id"],
                    "corridor_id": declaration["corridor_id"],
                    "intent_start_tick": declaration["intent_start_tick"],
                    "intent_end_tick": declaration["intent_end_tick"],
                    "intent_active": declaration["uav_id"] in active_intents,
                    "conflict_active": bool(conflicts_by_uav[declaration["uav_id"]]),
                    "conflicting_intent_ids": [
                        active_intents[other_id]["intent_id"]
                        for other_id in conflicts_by_uav[declaration["uav_id"]]
                    ],
                }
            )
            flight_plan_rows.append(
                {
                    **common,
                    "schema_name": "utm_flight_plan_state",
                    "plan_id": declaration["plan_id"],
                    "plan_ontology_class_id": "world:FlightPlan",
                    "uav_id": declaration["uav_id"],
                    "uav_ontology_class_id": "world:UnmannedAircraft",
                    "corridor_id": declaration["corridor_id"],
                    "approval_status": _status_at_tick(
                        declaration["flight_plan_schedule"],
                        field="approval_status",
                        tick=tick,
                    ),
                }
            )
            conflicting_uav_ids = sorted(
                conflicts_by_uav[declaration["uav_id"]]
            )
            deconfliction_rows.append(
                {
                    **common,
                    "schema_name": "utm_deconfliction_state",
                    "deconfliction_id": stable_identifier(
                        "utm_deconfliction",
                        episode_id,
                        declaration["uav_id"],
                    ),
                    "system_id": utm_system_id,
                    "system_ontology_class_id": "world:UTMSystem",
                    "system_source_ref": (
                        f"{script_path.relative_to(PROJECT_ROOT)}"
                        "#parameters.utm_service_plan"
                    ),
                    "uav_id": declaration["uav_id"],
                    "uav_ontology_class_id": "world:UnmannedAircraft",
                    "conflicting_uav_ids": conflicting_uav_ids,
                    "conflict_active": bool(conflicting_uav_ids),
                    "resolution_status": "active"
                    if conflicting_uav_ids
                    else "not_required",
                }
            )
        for first, second in combinations(
            sorted(declarations, key=lambda row: str(row["intent_id"])), 2
        ):
            first_active = first["uav_id"] in active_intents
            second_active = second["uav_id"] in active_intents
            conflict_active = bool(
                first_active
                and second_active
                and first["corridor_id"] == second["corridor_id"]
            )
            intent_pair_rows.append(
                {
                    **_common(episode_id, tick, input_digest, source_refs),
                    "schema_name": "utm_operational_intent_pair_state",
                    "parameter_digest": digest_object(service_plan),
                    "model_id": str(service_plan["model_id"]),
                    "model_version": str(service_plan["model_version"]),
                    "first_intent_id": first["intent_id"],
                    "first_intent_ontology_class_id": "world:OperationalIntent",
                    "second_intent_id": second["intent_id"],
                    "second_intent_ontology_class_id": "world:OperationalIntent",
                    "first_uav_id": first["uav_id"],
                    "second_uav_id": second["uav_id"],
                    "first_intent_active": first_active,
                    "second_intent_active": second_active,
                    "same_corridor": first["corridor_id"] == second["corridor_id"],
                    "conflict_active": conflict_active,
                }
            )

    records = {
        "airspace_authorization_log.jsonl": authorization_rows,
        "operational_intent_log.jsonl": intent_rows,
        "operational_intent_pair_log.jsonl": intent_pair_rows,
        "flight_plan_log.jsonl": flight_plan_rows,
        "deconfliction_log.jsonl": deconfliction_rows,
    }
    files = {
        name: "".join(f"{canonical_json(row)}\n" for row in rows)
        for name, rows in records.items()
    }
    summary: dict[str, Any] = {
        "schema_name": "utm_supplement_summary",
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "input_digest": input_digest,
        "parameter_digest": digest_object(service_plan),
        "model_id": str(service_plan["model_id"]),
        "model_version": str(service_plan["model_version"]),
        "formal_tick_count": len(FORMAL_TICKS),
        "uav_count": len(declarations),
        "record_counts": {name: len(rows) for name, rows in records.items()},
        "source_policy": "scenario_contract_plus_entity_roster",
    }
    files["summary.json"] = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return UtmEpisodeArtifacts(
        episode_id=episode_id,
        output_dir=output_dir.resolve(),
        files=files,
        summary=summary,
    )


def write_utm_episode_artifacts(artifacts: UtmEpisodeArtifacts) -> None:
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.files.items():
        (artifacts.output_dir / name).write_text(
            content, encoding="utf-8", newline="\n"
        )


def check_utm_episode_artifacts(artifacts: UtmEpisodeArtifacts) -> list[str]:
    mismatches: list[str] = []
    for name, expected in artifacts.files.items():
        path = artifacts.output_dir / name
        if not path.is_file():
            mismatches.append(f"missing:{name}")
        elif path.read_text(encoding="utf-8") != expected:
            mismatches.append(f"stale:{name}")
    return mismatches


def _common(
    episode_id: str,
    tick: int,
    input_digest: str,
    source_refs: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "tick": tick,
        "source_class": "simulated_derived",
        "source_refs": sorted(set(source_refs)),
        "input_digest": input_digest,
        "rule_id": "utm_supplement.scenario_contract_projection",
        "rule_version": SCHEMA_VERSION,
    }


def _load_utm_service_plan(
    script: Mapping[str, Any],
    *,
    roster_uavs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parameters = script.get("parameters")
    if not isinstance(parameters, Mapping):
        raise UtmStateError("scenario parameters must be an object")
    plan = parameters.get("utm_service_plan")
    if not isinstance(plan, Mapping):
        raise UtmStateError("scenario parameters lack utm_service_plan")
    required_root = {
        "schema_name",
        "schema_version",
        "model_id",
        "model_version",
        "global_uav_flow_policy",
        "uav_plans",
    }
    if set(plan) != required_root:
        raise UtmStateError(
            f"utm_service_plan must have exactly {sorted(required_root)}"
        )
    if (
        plan["schema_name"] != "utm_service_plan"
        or plan["schema_version"] != "1.0.0"
    ):
        raise UtmStateError("utm_service_plan schema identity is invalid")
    if not all(
        isinstance(plan.get(key), str) and plan[key]
        for key in ("model_id", "model_version")
    ):
        raise UtmStateError("utm_service_plan lacks model identity")
    rows = plan["uav_plans"]
    if not isinstance(rows, list) or not rows:
        raise UtmStateError("utm_service_plan.uav_plans must be a non-empty array")
    required_plan = {
        "uav_id",
        "corridor_id",
        "intent_start_tick",
        "intent_end_tick",
        "authorization_schedule",
        "flight_plan_schedule",
    }
    actual_uav_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != required_plan:
            raise UtmStateError(
                f"utm_service_plan.uav_plans[{index}] must have exactly {sorted(required_plan)}"
            )
        uav_id = row["uav_id"]
        corridor_id = row["corridor_id"]
        if not isinstance(uav_id, str) or not uav_id or uav_id in actual_uav_ids:
            raise UtmStateError(f"invalid or duplicate UTM uav_id at plan row {index}")
        if not isinstance(corridor_id, str) or not corridor_id:
            raise UtmStateError(f"UTM plan row {index} lacks corridor_id")
        actual_uav_ids.add(uav_id)
        intent_start_tick = int(row["intent_start_tick"])
        intent_end_tick = int(row["intent_end_tick"])
        if (
            intent_start_tick not in FORMAL_TICKS
            or intent_end_tick not in FORMAL_TICKS
            or intent_start_tick > intent_end_tick
        ):
            raise UtmStateError(f"UTM intent window is invalid for {uav_id}")
        authorization_schedule = _validate_complete_schedule(
            row["authorization_schedule"],
            field="status",
            allowed_values={"active", "revoked", "denied"},
            context=f"authorization:{uav_id}",
        )
        flight_plan_schedule = _validate_complete_schedule(
            row["flight_plan_schedule"],
            field="approval_status",
            allowed_values={"approved", "rejected", "pending"},
            context=f"flight_plan:{uav_id}",
        )
        normalized.append(
            {
                **dict(row),
                "authorization_schedule": authorization_schedule,
                "flight_plan_schedule": flight_plan_schedule,
            }
        )
    unexpected_plan_ids = actual_uav_ids - set(roster_uavs)
    if unexpected_plan_ids:
        raise UtmStateError(
            f"UTM plan references UAVs absent from the roster: {sorted(unexpected_plan_ids)}"
        )
    global_policy = plan["global_uav_flow_policy"]
    required_global_policy = {
        "source_contract",
        "intent_start_tick",
        "intent_end_tick",
        "authorization_status",
        "flight_plan_approval_status",
    }
    if not isinstance(global_policy, Mapping) or set(global_policy) != required_global_policy:
        raise UtmStateError(
            "utm_service_plan.global_uav_flow_policy must have exactly "
            f"{sorted(required_global_policy)}"
        )
    if global_policy["source_contract"] != "uav_global_flow":
        raise UtmStateError("global UTM policy source_contract must be uav_global_flow")
    extra_roster_ids = sorted(set(roster_uavs) - actual_uav_ids)
    for uav_id in extra_roster_ids:
        source = roster_uavs[uav_id].get("uav_global_flow")
        if not isinstance(source, Mapping):
            raise UtmStateError(
                f"roster UAV {uav_id} has no explicit UTM plan or uav_global_flow contract"
            )
        corridor_id = source.get("corridor_id")
        if not isinstance(corridor_id, str) or not corridor_id:
            raise UtmStateError(
                f"roster UAV {uav_id} uav_global_flow lacks corridor_id"
            )
        start_tick = int(global_policy["intent_start_tick"])
        end_tick = int(global_policy["intent_end_tick"])
        if start_tick not in FORMAL_TICKS or end_tick not in FORMAL_TICKS:
            raise UtmStateError("global UTM intent window must use formal ticks")
        normalized.append(
            {
                "uav_id": uav_id,
                "corridor_id": corridor_id,
                "intent_start_tick": start_tick,
                "intent_end_tick": end_tick,
                "authorization_schedule": _validate_complete_schedule(
                    [
                        {
                            "start_tick": FORMAL_TICKS[0],
                            "end_tick": FORMAL_TICKS[-1],
                            "status": global_policy["authorization_status"],
                        }
                    ],
                    field="status",
                    allowed_values={"active", "revoked", "denied"},
                    context=f"global_authorization:{uav_id}",
                ),
                "flight_plan_schedule": _validate_complete_schedule(
                    [
                        {
                            "start_tick": FORMAL_TICKS[0],
                            "end_tick": FORMAL_TICKS[-1],
                            "approval_status": global_policy[
                                "flight_plan_approval_status"
                            ],
                        }
                    ],
                    field="approval_status",
                    allowed_values={"approved", "rejected", "pending"},
                    context=f"global_flight_plan:{uav_id}",
                ),
            }
        )
    return {**dict(plan), "uav_plans": sorted(normalized, key=lambda row: row["uav_id"])}


def _validate_complete_schedule(
    value: Any,
    *,
    field: str,
    allowed_values: set[str],
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise UtmStateError(f"{context} schedule must be a non-empty array")
    required = {"start_tick", "end_tick", field}
    rows: list[dict[str, Any]] = []
    coverage: dict[int, int] = {tick: 0 for tick in FORMAL_TICKS}
    for index, interval in enumerate(value):
        if not isinstance(interval, Mapping) or set(interval) != required:
            raise UtmStateError(
                f"{context}[{index}] must have exactly {sorted(required)}"
            )
        start_tick = int(interval["start_tick"])
        end_tick = int(interval["end_tick"])
        state = str(interval[field])
        if (
            start_tick not in FORMAL_TICKS
            or end_tick not in FORMAL_TICKS
            or start_tick > end_tick
            or state not in allowed_values
        ):
            raise UtmStateError(f"{context}[{index}] has an invalid interval or state")
        for tick in FORMAL_TICKS:
            if start_tick <= tick <= end_tick:
                coverage[tick] += 1
        rows.append(dict(interval))
    invalid_ticks = [tick for tick, count in coverage.items() if count != 1]
    if invalid_ticks:
        raise UtmStateError(
            f"{context} must cover every formal tick exactly once: {invalid_ticks[:8]}"
        )
    return sorted(rows, key=lambda row: int(row["start_tick"]))


def _status_at_tick(
    schedule: Sequence[Mapping[str, Any]],
    *,
    field: str,
    tick: int,
) -> str:
    matches = [
        str(row[field])
        for row in schedule
        if int(row["start_tick"]) <= tick <= int(row["end_tick"])
    ]
    if len(matches) != 1:
        raise UtmStateError(f"schedule has {len(matches)} states at tick {tick}")
    return matches[0]


def _capture_boundary_id(
    script: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str | None:
    parameters = script.get("parameters")
    contract = parameters.get("semantic_event_contract") if isinstance(parameters, Mapping) else None
    boundary = contract.get("capture_boundary") if isinstance(contract, Mapping) else None
    value = boundary.get("boundary_id") if isinstance(boundary, Mapping) else None
    if not isinstance(value, str) or not value:
        value = manifest.get("capture_boundary_id")
    return value if isinstance(value, str) and value else None


def _entity_category(entity: Any) -> str:
    if not isinstance(entity, Mapping):
        return ""
    value = entity.get("entity_category") or entity.get("category")
    return str(value).lower() if isinstance(value, str) else ""


def _resolve_scenario_event_script(manifest: Mapping[str, Any]) -> Path:
    episode_id = manifest.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise UtmStateError("episode manifest lacks episode_id")
    epi_id = episode_id.split("__seed", 1)[0]
    matches = sorted(
        path.resolve()
        for path in (PROJECT_ROOT / "Dataset" / "scenarios").glob(
            f"**/{epi_id}/event_script.json"
        )
        if path.is_file()
    )
    if len(matches) != 1:
        raise UtmStateError(
            "scenario authority must contain exactly one event_script.json "
            f"for {epi_id}: {matches}"
        )
    return matches[0]


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise UtmStateError(f"{path}: root must be an object")
    return value


__all__ = [
    "FORMAL_TICKS",
    "UtmEpisodeArtifacts",
    "UtmStateError",
    "build_utm_episode_artifacts",
    "check_utm_episode_artifacts",
    "write_utm_episode_artifacts",
]
