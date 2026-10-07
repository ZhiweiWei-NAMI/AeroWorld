"""Materialize and validate episode-local deterministic charging service plans."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from Dataset.semantic_truth.facility_scope import validate_roster_facility_scope
from Dataset.semantic_truth.provenance import canonical_json, digest_file, digest_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "aw_data" / "charging_supplement"
PLAN_SCHEMA_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "semantic_rules"
    / "schema"
    / "charging_service_plan.schema.json"
)


class ChargingSupplementError(ValueError):
    """Raised when an authored charging plan cannot become an episode source."""


def materialize_charging_supplement(
    episode_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    plan, summary = build_charging_plan(episode_root)
    output_dir = output_root.resolve() / summary["episode_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output_dir / "charging_service_plan.json", json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    _atomic_write_text(output_dir / "manifest.json", json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return summary


def build_charging_plan(episode_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build from the authored plan and verify its actual episode participants."""
    episode_root = episode_root.resolve()
    manifest_path = episode_root / "episode_manifest.json"
    roster_path = episode_root / "global_entity_roster.json"
    manifest = _load_object(manifest_path)
    roster = _load_object(roster_path)
    episode_id = str(manifest.get("episode_id") or episode_root.name)
    if episode_id != episode_root.name:
        raise ChargingSupplementError(
            f"episode identity mismatch: {episode_id!r} != {episode_root.name!r}"
        )
    source_path = _resolve_scene_setup(manifest, episode_root)
    scene_setup = _load_object(source_path)
    simulation_plans = scene_setup.get("simulation_plans")
    plan = (
        simulation_plans.get("charging_service_plan")
        if isinstance(simulation_plans, Mapping)
        else None
    )
    if not isinstance(plan, Mapping):
        raise ChargingSupplementError(
            f"{source_path}: simulation_plans.charging_service_plan is required"
        )
    plan = json.loads(canonical_json(plan))
    _validate_plan_schema(plan)
    roster_entities = _roster_entities(roster, roster_path)
    charger_ids = {
        entity_id
        for entity_id, entity in roster_entities.items()
        if _facility_subtype(entity) == "charging_station"
    }
    planned_ids = {
        str(row["facility_id"])
        for row in plan["facilities"]
        if isinstance(row, Mapping)
    }
    if planned_ids != charger_ids:
        raise ChargingSupplementError(
            f"{episode_id}: charging plan/roster charger mismatch: "
            f"plan_only={sorted(planned_ids - charger_ids)}, "
            f"roster_only={sorted(charger_ids - planned_ids)}"
        )
    uav_ids = {
        entity_id
        for entity_id, entity in roster_entities.items()
        if _entity_category(entity) == "uav"
    }
    for facility in plan["facilities"]:
        for request in facility["requests"]:
            if request["uav_id"] not in uav_ids:
                raise ChargingSupplementError(
                    f"{episode_id}: charging request references absent UAV "
                    f"{request['uav_id']!r}"
                )
    summary = {
        "schema_name": "charging_supplement_manifest",
        "schema_version": "1.0.0",
        "episode_id": episode_id,
        "source_scene_setup": str(source_path.relative_to(PROJECT_ROOT)),
        "entity_roster": str(roster_path.relative_to(PROJECT_ROOT)),
        "facility_count": len(plan["facilities"]),
        "request_count": sum(len(row["requests"]) for row in plan["facilities"]),
        "model_id": plan["model_id"],
        "model_version": plan["model_version"],
    }
    return plan, summary


def check_charging_supplement(
    episode_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[str]:
    episode_id = episode_root.resolve().name
    output_dir = output_root.resolve() / episode_id
    names = ("charging_service_plan.json", "manifest.json")
    if any(not (output_dir / name).is_file() for name in names):
        return [f"{episode_id}: charging supplement is incomplete"]
    with tempfile.TemporaryDirectory(prefix="aeroworld_charging_check_") as temp:
        expected_root = Path(temp)
        materialize_charging_supplement(episode_root, expected_root)
        return [
            f"{episode_id}: stale {name}"
            for name in names
            if (output_dir / name).read_bytes()
            != (expected_root / episode_id / name).read_bytes()
        ]


def _resolve_scene_setup(manifest: Mapping[str, Any], episode_root: Path) -> Path:
    local = episode_root / "scene_setup.json"
    if local.is_file():
        return local
    value = manifest.get("source_scene_setup_path")
    if not isinstance(value, str) or not value:
        raise ChargingSupplementError("episode manifest lacks source_scene_setup_path")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ChargingSupplementError(f"source scene setup is missing: {path}")
    return path


def _validate_plan_schema(plan: Mapping[str, Any]) -> None:
    schema = _load_object(PLAN_SCHEMA_PATH)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(plan),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        detail = "; ".join(
            f"{'.'.join(str(item) for item in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ChargingSupplementError(f"charging service plan schema violation: {detail}")


def _roster_entities(
    roster: Mapping[str, Any],
    path: Path,
) -> dict[str, dict[str, Any]]:
    raw: Any = roster.get("entities", roster)
    rows: Sequence[Any]
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, Mapping):
        rows = list(raw.values())
    else:
        raise ChargingSupplementError(f"{path}: roster must be an object or array")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ChargingSupplementError(f"{path}: roster entry must be an object")
        entity_id = row.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id or entity_id in result:
            raise ChargingSupplementError(f"{path}: invalid or duplicate entity_id")
        result[entity_id] = row
    return result


def _facility_subtype(entity: Mapping[str, Any]) -> str | None:
    if _entity_category(entity) not in {"facility", "ground_station"}:
        return None
    return str(validate_roster_facility_scope(entity)["scope_subtype"])


def _entity_category(entity: Mapping[str, Any]) -> str:
    category = str(
        entity.get("entity_category")
        or entity.get("category")
        or entity.get("label_class")
        or ""
    ).lower()
    if category in {"aircraft", "drone"}:
        return "uav"
    return category


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ChargingSupplementError(f"required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ChargingSupplementError(f"{path}: root must be an object")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


__all__ = [
    "ChargingSupplementError",
    "DEFAULT_OUTPUT_ROOT",
    "check_charging_supplement",
    "materialize_charging_supplement",
]
