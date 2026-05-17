#!/usr/bin/env python3
"""Validate the formal UE replay truth synchronization package.

``Dataset/render_ready_episodes_capture_filtered`` is kept as the formal capture
input path, but it must not delete or visibility-filter dynamic P/V/U truth.
This validator compares each formal package against the source render-ready
episode and fails if truth frames, trajectories, or roster entities were removed.
Generator-side mistakes should surface here and be fixed upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
DEFAULT_SOURCE_ROOT = ROOT / "Dataset" / "render_ready_episodes"
PVU_CATEGORIES = {"pedestrian", "vehicle", "uav"}
DISABLED_RENDER_FLAGS = (
    ("vehicle_lane_offsets", "enabled"),
    ("entity_overlap_filter", "enabled"),
    ("pedestrian_roadside_projection", "enabled"),
)

if str(ROOT / "Dataset" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "Dataset" / "tools"))

from filter_render_ready_truth_for_capture import entity_category, loads_jsonl_bytes, read_json  # noqa: E402


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            name = item.strip()
            if name:
                names.append(name)
    return names


def selected_episode_dirs(input_root: Path, names: list[str], limit: int) -> list[Path]:
    all_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "truth_frames.jsonl").exists())
    if names:
        wanted = set(names)
        by_name = {path.name: path for path in all_dirs}
        missing = sorted(name for name in wanted if name not in by_name)
        if missing:
            raise SystemExit(f"Missing formal capture episodes: {missing}")
        all_dirs = [by_name[name] for name in names]
    if limit > 0:
        all_dirs = all_dirs[:limit]
    return all_dirs


def source_episode_dir(formal_dir: Path, source_root: Path, manifest: dict[str, Any]) -> Path:
    generation = dict(manifest.get("generation") or {})
    raw = str(generation.get("truth_filter_source_root") or "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.exists():
            return candidate.resolve()
    return (source_root / formal_dir.name).resolve()


def check_render_config(formal_dir: Path, errors: list[str]) -> None:
    config_path = formal_dir / "render_host_config.json"
    if not config_path.exists():
        errors.append(f"{formal_dir.name}: missing render_host_config.json")
        return
    config = read_json(config_path)
    for section, key in DISABLED_RENDER_FLAGS:
        payload = dict(config.get(section) or {})
        if payload.get(key) is not False:
            errors.append(f"{formal_dir.name}: render_host_config must set {section}.{key}=false")
    road_topology = dict(config.get("road_topology_snap") or {})
    if road_topology.get("enabled") is True and road_topology.get("preserve_truth_xy") is not True:
        errors.append(f"{formal_dir.name}: road_topology_snap must preserve truth XY when enabled")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth_stats(path: Path) -> dict[str, Any]:
    frame_count = 0
    entity_records = 0
    dynamic_pvu_records = 0
    dynamic_pvu_ids: set[str] = set()
    dynamic_id_sets: list[set[str]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frame = loads_jsonl_bytes(stripped)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
            frame_count += 1
            current_dynamic_ids: set[str] = set()
            entities = frame.get("entities") or []
            entity_records += len(entities) if isinstance(entities, list) else 0
            for entity in entities if isinstance(entities, list) else []:
                if not isinstance(entity, dict):
                    continue
                if entity_category(entity) not in PVU_CATEGORIES:
                    continue
                entity_id = str(entity.get("entity_id") or "")
                dynamic_pvu_records += 1
                if entity_id:
                    dynamic_pvu_ids.add(entity_id)
                    current_dynamic_ids.add(entity_id)
            dynamic_id_sets.append(current_dynamic_ids)
    return {
        "frame_count": frame_count,
        "entity_records": entity_records,
        "dynamic_pvu_records": dynamic_pvu_records,
        "dynamic_pvu_ids": dynamic_pvu_ids,
        "dynamic_id_sets": dynamic_id_sets,
    }


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def check_no_filter_policy(formal_dir: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    payload = dict(manifest.get("capture_truth_sync") or manifest.get("capture_visible_truth_filter") or {})
    if not payload:
        errors.append(f"{formal_dir.name}: missing capture_truth_sync/capture_visible_truth_filter metadata")
        return
    if payload.get("filtering_enabled") is not False:
        errors.append(f"{formal_dir.name}: formal capture sync must set filtering_enabled=false")
    if payload.get("dynamic_pvu_filtering_enabled") is not False:
        errors.append(f"{formal_dir.name}: formal capture sync must set dynamic_pvu_filtering_enabled=false")
    stats = dict(payload.get("stats") or {})
    if int(stats.get("removed_truth_entities") or 0) != 0:
        errors.append(f"{formal_dir.name}: formal capture sync removed_truth_entities={stats.get('removed_truth_entities')}")


def validate_episode(formal_dir: Path, source_root: Path, max_errors: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "episode_manifest.json",
        "global_entity_roster.json",
        "truth_frames.jsonl",
        "trajectories.jsonl",
        "scenario_plan.json",
    )
    for name in required:
        if not (formal_dir / name).exists():
            errors.append(f"{formal_dir.name}: missing {name}")
    if errors:
        return {"episode": formal_dir.name, "ok": False, "errors": errors, "warnings": warnings}

    manifest = read_json(formal_dir / "episode_manifest.json")
    source_dir = source_episode_dir(formal_dir, source_root, manifest)
    if not source_dir.exists():
        errors.append(f"{formal_dir.name}: source render-ready episode not found: {source_dir}")
        return {"episode": formal_dir.name, "ok": False, "errors": errors, "warnings": warnings}
    for name in required:
        if not (source_dir / name).exists():
            errors.append(f"{formal_dir.name}: source missing {name}: {source_dir}")
    if errors:
        return {"episode": formal_dir.name, "ok": False, "errors": errors[:max_errors], "warnings": warnings}

    check_no_filter_policy(formal_dir, manifest, errors)
    check_render_config(formal_dir, errors)

    source_roster = read_json(source_dir / "global_entity_roster.json")
    formal_roster = read_json(formal_dir / "global_entity_roster.json")
    source_entities = list(source_roster.get("entities") or [])
    formal_entities = list(formal_roster.get("entities") or [])
    if formal_entities != source_entities:
        source_ids = [str(entity.get("entity_id") or "") for entity in source_entities if isinstance(entity, dict)]
        formal_ids = [str(entity.get("entity_id") or "") for entity in formal_entities if isinstance(entity, dict)]
        missing = sorted(set(source_ids) - set(formal_ids))
        extra = sorted(set(formal_ids) - set(source_ids))
        errors.append(
            f"{formal_dir.name}: global_entity_roster entities differ from source "
            f"source={len(source_entities)} formal={len(formal_entities)} missing={missing[:10]} extra={extra[:10]}"
        )

    source_truth_path = source_dir / "truth_frames.jsonl"
    formal_truth_path = formal_dir / "truth_frames.jsonl"
    source_traj_path = source_dir / "trajectories.jsonl"
    formal_traj_path = formal_dir / "trajectories.jsonl"
    source_truth_sha = sha256_file(source_truth_path)
    formal_truth_sha = sha256_file(formal_truth_path)
    source_traj_sha = sha256_file(source_traj_path)
    formal_traj_sha = sha256_file(formal_traj_path)

    source_stats = truth_stats(source_truth_path)
    formal_stats = truth_stats(formal_truth_path)
    if source_truth_sha != formal_truth_sha:
        errors.append(f"{formal_dir.name}: truth_frames.jsonl differs from source; formal sync must not delete/rewrite truth frames")
        if source_stats["frame_count"] != formal_stats["frame_count"]:
            errors.append(
                f"{formal_dir.name}: frame count changed source={source_stats['frame_count']} formal={formal_stats['frame_count']}"
            )
        if source_stats["entity_records"] != formal_stats["entity_records"]:
            errors.append(
                f"{formal_dir.name}: entity records changed source={source_stats['entity_records']} formal={formal_stats['entity_records']}"
            )
        if source_stats["dynamic_pvu_records"] != formal_stats["dynamic_pvu_records"]:
            errors.append(
                f"{formal_dir.name}: dynamic P/V/U records changed source={source_stats['dynamic_pvu_records']} "
                f"formal={formal_stats['dynamic_pvu_records']}"
            )
        max_frames = min(len(source_stats["dynamic_id_sets"]), len(formal_stats["dynamic_id_sets"]))
        for frame_index in range(max_frames):
            missing = sorted(source_stats["dynamic_id_sets"][frame_index] - formal_stats["dynamic_id_sets"][frame_index])
            if missing:
                errors.append(f"{formal_dir.name}: frame {frame_index} missing dynamic ids after sync: {missing[:10]}")
                break
    if source_traj_sha != formal_traj_sha:
        errors.append(
            f"{formal_dir.name}: trajectories.jsonl differs from source "
            f"source_rows={count_jsonl_rows(source_traj_path)} formal_rows={count_jsonl_rows(formal_traj_path)}"
        )

    pvu_missing = sorted(source_stats["dynamic_pvu_ids"] - formal_stats["dynamic_pvu_ids"])
    if pvu_missing:
        errors.append(f"{formal_dir.name}: dynamic P/V/U ids missing after formal sync: {pvu_missing[:20]}")
    if int(formal_stats["dynamic_pvu_records"]) <= 0:
        warnings.append(f"{formal_dir.name}: no dynamic P/V/U records in formal truth")

    return {
        "episode": formal_dir.name,
        "ok": not errors,
        "errors": errors[:max_errors],
        "warnings": warnings,
        "frame_count": int(formal_stats["frame_count"]),
        "entity_records": int(formal_stats["entity_records"]),
        "dynamic_pvu_records": int(formal_stats["dynamic_pvu_records"]),
        "dynamic_pvu_ids": len(formal_stats["dynamic_pvu_ids"]),
        "source_episode_dir": str(source_dir),
        "truth_sha256": formal_truth_sha,
        "trajectories_sha256": formal_traj_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate formal capture truth sync packages.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    source_root = args.source_root.resolve()
    episodes = selected_episode_dirs(input_root, parse_episode_names(args.episode), int(args.limit or 0))
    if not episodes:
        raise SystemExit("No formal capture episodes selected")

    results = [validate_episode(path, source_root, max(1, int(args.max_errors or 50))) for path in episodes]
    failed = [result for result in results if not result["ok"]]
    summary = {
        "ok": not failed,
        "input_root": str(input_root),
        "source_root": str(source_root),
        "episode_count": len(results),
        "failed_episode_count": len(failed),
        "dynamic_pvu_records": sum(int(result.get("dynamic_pvu_records") or 0) for result in results),
        "dynamic_pvu_ids": sum(int(result.get("dynamic_pvu_ids") or 0) for result in results),
        "entity_records": sum(int(result.get("entity_records") or 0) for result in results),
        "failed_episodes": failed[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
