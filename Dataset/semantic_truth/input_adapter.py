"""Authoritative episode input adapter and candidate tuple construction."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import EvalResult, MISSING, SemanticCompileError, TickContext
from .provenance import digest_file, digest_object, load_json, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def deep_get(value: Any, path_parts: Sequence[str]) -> Any:
    current = value
    for part in path_parts:
        if isinstance(current, Mapping):
            if part not in current:
                return MISSING
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            try:
                index = int(part)
            except ValueError:
                return MISSING
            if index < 0 or index >= len(current):
                return MISSING
            current = current[index]
        else:
            return MISSING
    return current


def index_entities(values: Any, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise SemanticCompileError(f"{source}: entities must be an array")
    result: dict[str, dict[str, Any]] = {}
    for index, entity in enumerate(values):
        if not isinstance(entity, Mapping):
            raise SemanticCompileError(f"{source}: entity {index} must be an object")
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise SemanticCompileError(f"{source}: entity {index} lacks entity_id")
        if entity_id in result:
            raise SemanticCompileError(f"{source}: duplicate entity_id {entity_id}")
        result[entity_id] = copy.deepcopy(dict(entity))
    return result


def resolve_scenario_scene_setup(manifest: Mapping[str, Any]) -> Path:
    episode_id = manifest.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise SemanticCompileError("episode manifest lacks episode_id")
    epi_id = episode_id.split("__seed", 1)[0]
    matches = sorted(
        path.resolve()
        for path in (PROJECT_ROOT / "Dataset" / "scenarios").glob(
            f"**/{epi_id}/scene_setup.json"
        )
        if path.is_file()
    )
    if len(matches) != 1:
        raise SemanticCompileError(
            f"scenario authority must contain exactly one scene_setup.json for {epi_id}: {matches}"
        )
    return matches[0]


def scene_setup_geometry(
    scene_setup: Mapping[str, Any],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    """Extract objective geometry from a scene setup.

    No event script, event trace, scheduled label, or semantic intent is read.
    ``polygon_prism`` and axis-aligned ``box_volume`` placements are
    authoritative closed volumes.  ``facade_anchor`` is retained as an
    authoritative point/normal surface anchor for structure-clearance rules.
    A no-fly asset is mapped to the stable restricted-region kind from its
    asset identity, never from its authored role, title, or event label.
    """
    entities = scene_setup.get("entities", [])
    if not isinstance(entities, list):
        raise SemanticCompileError(f"{source_name}: entities must be an array")
    result: dict[str, dict[str, Any]] = {}
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise SemanticCompileError(
                f"{source_name}: entity {index} must be an object"
            )
        placement_mode = entity.get("placement_mode")
        if placement_mode not in {"polygon_prism", "box_volume", "facade_anchor"}:
            continue
        entity_id = entity.get("entity_id")
        placement = entity.get("placement")
        if not isinstance(entity_id, str) or not isinstance(placement, Mapping):
            raise SemanticCompileError(
                f"{source_name}: closed-volume entity {index} is incomplete"
            )
        if placement_mode == "facade_anchor":
            point = placement.get("resolved_position_enu_m") or placement.get(
                "position_enu_m"
            )
            normal = placement.get("outward_normal_enu")
            if (
                not isinstance(point, list)
                or len(point) < 3
                or not all(isinstance(value, (int, float)) for value in point[:3])
                or not isinstance(normal, list)
                or len(normal) < 3
                or not all(isinstance(value, (int, float)) for value in normal[:3])
            ):
                raise SemanticCompileError(
                    f"{source_name}: facade-anchor {entity_id} requires numeric point and normal vectors"
                )
            normal_norm = sum(float(value) ** 2 for value in normal[:3]) ** 0.5
            if normal_norm <= 0.0:
                raise SemanticCompileError(
                    f"{source_name}: facade-anchor {entity_id} has a zero outward normal"
                )
            normalized = [float(value) / normal_norm for value in normal[:3]]
            result[entity_id] = {
                "entity_id": entity_id,
                "category": entity.get("category"),
                "entity_category": entity.get("category"),
                "entity_kind": "structure.facade_anchor",
                "logical_asset_id": entity.get("logical_asset_id"),
                "placement_mode": placement_mode,
                "activation_tick": entity.get("activation_tick", 0),
                "deactivation_tick": entity.get("deactivation_tick"),
                "geometry": {
                    "geometry_kind": "facade_anchor",
                    "point_enu_m": [float(value) for value in point[:3]],
                    "outward_normal_enu": normalized,
                    "stand_off_m": placement.get("stand_off_m"),
                    "building_id": placement.get("building_id"),
                    "building_source": placement.get("building_source"),
                    "active": True,
                },
            }
            continue
        if placement_mode == "polygon_prism":
            required = ("polygon_enu_m", "base_z_m", "height_m")
            missing = [field for field in required if field not in placement]
            if missing:
                raise SemanticCompileError(
                    f"{source_name}: polygon-prism {entity_id} lacks {missing}"
                )
            polygon = placement["polygon_enu_m"]
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise SemanticCompileError(
                    f"{source_name}: polygon-prism {entity_id} has invalid polygon"
                )
            base_z_m = placement["base_z_m"]
            height_m = placement["height_m"]
        else:
            center = placement.get("center_enu_m")
            extent = placement.get("extent_m")
            if (
                not isinstance(center, list)
                or len(center) < 3
                or not isinstance(extent, list)
                or len(extent) < 3
                or not all(isinstance(value, (int, float)) for value in center[:3])
                or not all(isinstance(value, (int, float)) for value in extent[:3])
                or any(float(value) <= 0.0 for value in extent[:3])
            ):
                raise SemanticCompileError(
                    f"{source_name}: box-volume {entity_id} requires positive center/extent vectors"
                )
            cx, cy, cz = (float(value) for value in center[:3])
            ex, ey, ez = (float(value) for value in extent[:3])
            polygon = [
                [cx - ex, cy - ey],
                [cx + ex, cy - ey],
                [cx + ex, cy + ey],
                [cx - ex, cy + ey],
            ]
            base_z_m = cz - ez
            height_m = 2.0 * ez
        logical_asset_id = entity.get("logical_asset_id")
        entity_kind = (
            "airspace.restricted_region"
            if logical_asset_id == "trigger.no_fly.box.v1"
            else "airspace.hazard_region"
            if entity.get("category") == "airspace_constraint"
            else None
        )
        result[entity_id] = {
            "entity_id": entity_id,
            "category": entity.get("category"),
            "entity_category": entity.get("category"),
            "entity_kind": entity_kind,
            "logical_asset_id": logical_asset_id,
            "spawn_policy": entity.get("spawn_policy"),
            "placement_mode": placement_mode,
            "activation_tick": entity.get("activation_tick", 0),
            "deactivation_tick": entity.get("deactivation_tick"),
            "geometry": {
                "polygon_enu_m": copy.deepcopy(polygon),
                "base_z_m": base_z_m,
                "height_m": height_m,
                "active": True,
                "source_shape": placement_mode,
                "constraint_kind": entity_kind,
            },
        }
    return result


def static_entity_for_tick(entity: Mapping[str, Any], tick: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(entity))
    if isinstance(result.get("geometry"), Mapping):
        activation_tick = result.get("activation_tick", 0)
        deactivation_tick = result.get("deactivation_tick")
        active = result["geometry"].get("active", True)
        if isinstance(activation_tick, int):
            active = active and tick >= activation_tick
        if isinstance(deactivation_tick, int):
            active = active and tick < deactivation_tick
        result["geometry"]["active"] = bool(active)
    return result


def build_tick_contexts(
    episode_root: Path,
    profile: Mapping[str, Any],
) -> tuple[
    str,
    list[TickContext],
    list[dict[str, Any]],
    str,
    dict[str, Any],
]:
    inputs = profile["inputs"]
    truth_path = episode_root / inputs["truth_frames"]
    roster_path = episode_root / inputs["entity_roster"]
    explicit_static_path = episode_root / inputs["static_geometry"]
    manifest_path = episode_root / inputs["episode_manifest"]
    if not truth_path.is_file():
        raise SemanticCompileError(
            f"required truth frame file is missing: {truth_path}"
        )
    if not roster_path.is_file():
        raise SemanticCompileError(f"required entity roster is missing: {roster_path}")
    if not manifest_path.is_file():
        raise SemanticCompileError(
            f"required episode manifest is missing: {manifest_path}"
        )

    manifest = load_json(manifest_path)
    roster = load_json(roster_path)
    roster_entities = index_entities(roster.get("entities", []), str(roster_path))

    static_source_path: Path | None = None
    static_source_name = inputs["static_geometry"]
    if explicit_static_path.is_file():
        static_source_path = explicit_static_path
        static = load_json(explicit_static_path)
        static_entities = index_entities(
            static.get("entities", []), str(explicit_static_path)
        )
        static_geometry_authority = {
            "used": False,
            "source_type": "explicit_static_geometry",
            "source_scene_setup_path": None,
            "resolved_source_sha256": None,
        }
    else:
        candidate_path = resolve_scenario_scene_setup(manifest)
        source_value = str(candidate_path.relative_to(PROJECT_ROOT))
        scene_setup = load_json(candidate_path)
        static_entities = scene_setup_geometry(scene_setup, source_value)
        static_source_path = candidate_path
        static_source_name = source_value
        static_geometry_authority = {
            "used": True,
            "source_type": "scenario_authority_scene_setup_polygon_prism",
            "source_scene_setup_path": static_source_name,
            "resolved_source_sha256": digest_file(candidate_path),
        }

    tick_policy = profile["authoritative_tick_policy"]
    expected_ticks = list(
        range(tick_policy["start"], tick_policy["end"] + 1, tick_policy["step"])
    )
    expected_set = set(expected_ticks)
    frames: dict[int, dict[str, Any]] = {}
    observed_episode_ids: set[str] = set()
    for frame in read_jsonl(truth_path):
        tick = frame.get("tick")
        if not isinstance(tick, int) or tick not in expected_set:
            continue
        if tick in frames:
            raise SemanticCompileError(
                f"duplicate authoritative truth frame tick {tick}"
            )
        frames[tick] = frame
        if isinstance(frame.get("episode_id"), str):
            observed_episode_ids.add(frame["episode_id"])

    episode_id = manifest.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        if len(observed_episode_ids) == 1:
            episode_id = next(iter(observed_episode_ids))
        else:
            episode_id = episode_root.name
    if observed_episode_ids and observed_episode_ids != {episode_id}:
        raise SemanticCompileError(
            f"truth frame episode ids do not match episode id {episode_id}"
        )

    contexts: list[TickContext] = []
    for tick in expected_ticks:
        frame = frames.get(tick)
        frame_entities = index_entities(
            frame.get("entities", []) if frame else [],
            f"{truth_path}#tick={tick}",
        )
        tick_static_entities = {
            entity_id: static_entity_for_tick(entity, tick)
            for entity_id, entity in static_entities.items()
        }
        entity_ids = sorted(
            set(roster_entities) | set(tick_static_entities) | set(frame_entities)
        )
        merged_entities: dict[str, dict[str, Any]] = {}
        for entity_id in entity_ids:
            merged: dict[str, Any] = {}
            if entity_id in roster_entities:
                merged = deep_merge(merged, roster_entities[entity_id])
            if entity_id in tick_static_entities:
                merged = deep_merge(merged, tick_static_entities[entity_id])
            if entity_id in frame_entities:
                merged = deep_merge(merged, frame_entities[entity_id])
            merged["entity_id"] = entity_id
            merged_entities[entity_id] = merged
        contexts.append(
            TickContext(
                tick=tick,
                tick_present=frame is not None,
                episode_id=episode_id,
                entities=merged_entities,
                frame_entities=frame_entities,
                roster_entities=roster_entities,
                static_entities=tick_static_entities,
                truth_frames_name=inputs["truth_frames"],
                roster_name=inputs["entity_roster"],
                static_name=static_source_name,
            )
        )

    input_paths: list[Path] = [
        truth_path,
        roster_path,
        explicit_static_path,
        manifest_path,
    ]
    if static_source_path and static_source_path != explicit_static_path:
        input_paths.append(static_source_path)
    input_files: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for path in input_paths:
        normalized = path.resolve()
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        exists = path.is_file()
        if normalized == (static_source_path.resolve() if static_source_path else None):
            display_path = static_source_name
        else:
            display_path = path.name
        input_files.append(
            {
                "path": display_path,
                "exists": exists,
                "sha256": digest_file(path) if exists else None,
            }
        )
    input_digest = digest_object(input_files)
    return (
        episode_id,
        contexts,
        input_files,
        input_digest,
        static_geometry_authority,
    )


def selector_matches(entity: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    for path, test in selector.items():
        if not isinstance(test, Mapping):
            return False
        actual = deep_get(entity, path.split("."))
        if actual is MISSING:
            return False
        if "equals" in test and actual != test["equals"]:
            return False
        if "one_of" in test and actual not in test["one_of"]:
            return False
    return True


def candidate_bindings(
    role_specs: Mapping[str, Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, Any]],
    distinct_pairs: Sequence[Sequence[str]] = (),
    unordered_pairs: Sequence[Sequence[str]] = (),
) -> list[dict[str, str]]:
    role_names = list(role_specs)
    domains: list[list[str]] = []
    for role_name in role_names:
        selector = role_specs[role_name]["selector"]
        domains.append(
            sorted(
                entity_id
                for entity_id, entity in entities.items()
                if selector_matches(entity, selector)
            )
        )
    results: list[dict[str, str]] = []
    for values in itertools.product(*domains):
        binding = dict(zip(role_names, values))
        if any(binding[first] == binding[second] for first, second in distinct_pairs):
            continue
        if any(binding[first] > binding[second] for first, second in unordered_pairs):
            continue
        results.append(binding)
    return results


def resolve_reference(
    reference: str,
    bindings: Mapping[str, str],
    context: TickContext,
) -> EvalResult:
    parts = reference.split(".")
    if len(parts) < 2:
        return EvalResult(value=MISSING, missing=(f"invalid reference {reference}",))
    role_name = parts[0]
    path_parts = parts[1:]
    if role_name not in bindings:
        return EvalResult(value=MISSING, missing=(f"unbound role {role_name}",))
    entity_id = bindings[role_name]
    if entity_id not in context.entities:
        return EvalResult(value=MISSING, missing=(f"missing entity {entity_id}",))

    dynamic_root = path_parts[0] in {
        "truth_pose",
        "render_presence",
        "annotations",
        "state",
        "runtime_visibility",
    }
    sources: list[tuple[Mapping[str, Any], str]] = []
    if entity_id in context.frame_entities:
        sources.append(
            (
                context.frame_entities[entity_id],
                f"{context.truth_frames_name}#tick={context.tick}&entity={entity_id}",
            )
        )
    if not dynamic_root:
        if entity_id in context.static_entities:
            sources.append(
                (
                    context.static_entities[entity_id],
                    f"{context.static_name}#entity={entity_id}",
                )
            )
        if entity_id in context.roster_entities:
            sources.append(
                (
                    context.roster_entities[entity_id],
                    f"{context.roster_name}#entity={entity_id}",
                )
            )
    for source_entity, source_ref in sources:
        value = deep_get(source_entity, path_parts)
        if value is not MISSING:
            return EvalResult(
                value=value,
                source_refs=(source_ref,),
                observations=((reference, value),),
            )
    return EvalResult(value=MISSING, missing=(reference,))
