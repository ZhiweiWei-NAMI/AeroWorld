"""Synchronize render-ready trajectories.jsonl from truth_frames.jsonl.

The render host uses truth_frames.jsonl as the authoritative replay source, but
the package still exposes trajectories.jsonl for inspection and downstream
tools.  This utility keeps that artifact coherent by deriving trajectory rows
directly from the current truth frames instead of copying source episode rows.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover - optional speedup.
    orjson = None

try:
    from convert_to_render_ready import truth_entity_to_trajectory_row
except ModuleNotFoundError:  # pragma: no cover - supports package imports from repo root.
    from Dataset.tools.convert_to_render_ready import truth_entity_to_trajectory_row


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dumps_jsonl_bytes(payload: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def loads_jsonl_bytes(payload: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload.decode("utf-8-sig"))


def iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, loads_jsonl_bytes(stripped)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc


def sync_episode(episode_dir: Path) -> dict[str, Any]:
    truth_path = episode_dir / "truth_frames.jsonl"
    trajectory_path = episode_dir / "trajectories.jsonl"
    manifest_path = episode_dir / "episode_manifest.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"{episode_dir}: missing truth_frames.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError(f"{episode_dir}: missing episode_manifest.json")

    count = 0
    temp_path = trajectory_path.with_suffix(".jsonl.tmp")
    with temp_path.open("wb") as handle:
        for _line_number, frame in iter_jsonl(truth_path):
            for entity in frame.get("entities") or []:
                if not isinstance(entity, dict):
                    continue
                handle.write(dumps_jsonl_bytes(truth_entity_to_trajectory_row(frame, entity)))
                count += 1
    temp_path.replace(trajectory_path)

    manifest = read_json(manifest_path)
    for key in ("record_counts", "canonical_record_counts"):
        record_counts = dict(manifest.get(key) or {})
        record_counts["trajectories"] = count
        manifest[key] = record_counts
    generation = dict(manifest.get("generation") or {})
    generation["trajectory_source"] = "truth_frames.jsonl"
    generation["trajectory_contract"] = "truth_frame_derived_v1"
    manifest["generation"] = generation
    write_json(manifest_path, manifest)
    return {
        "episode": episode_dir.name,
        "trajectory_rows": count,
        "trajectory_path": str(trajectory_path),
    }


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            name = item.strip()
            if name:
                names.append(name)
    return names


def select_episode_dirs(input_root: Path, names: list[str]) -> list[Path]:
    all_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "truth_frames.jsonl").exists())
    if not names:
        return all_dirs
    by_name = {path.name: path for path in all_dirs}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown episode(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive render-ready trajectories.jsonl from truth_frames.jsonl.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    episode_dirs = select_episode_dirs(args.input_root, parse_episode_names(args.episode))
    workers = max(1, int(args.workers or 1))
    if workers == 1 or len(episode_dirs) <= 1:
        results = [sync_episode(path) for path in episode_dirs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_episode = {executor.submit(sync_episode, path): path for path in episode_dirs}
            for future in as_completed(future_to_episode):
                results.append(future.result())
        results.sort(key=lambda item: str(item["episode"]))
    summary = {
        "ok": True,
        "episode_count": len(results),
        "total_trajectory_rows": sum(int(item["trajectory_rows"]) for item in results),
        "results": results,
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
