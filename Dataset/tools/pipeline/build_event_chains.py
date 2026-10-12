#!/usr/bin/env python3
"""Build authored Stage-A event-chain audit artifacts from existing runtime logs."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.semantic_truth.event_chain import (  # noqa: E402
    EventChainBuildResult,
    EventChainError,
    build_event_chain,
    canonical_json,
    sha256_bytes,
)


DEFAULT_CAPTURE_ROOT = (
    PROJECT_ROOT / "aw_data" / "render_ready_episodes_capture_filtered"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "aw_data" / "event_chains"
EPISODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*__seed\d{2}$")
OUTPUT_FILENAMES = {"event_chain_manifest.json", "event_chain_relations.jsonl"}
FORMAL_EPISODE_COUNT = 210
FORMAL_EPI_COUNT = 70
FORMAL_SEEDS = frozenset({"seed00", "seed01", "seed02"})


class EventChainInputError(EventChainError):
    """Raised when a handoff episode does not expose the exact Stage-A inputs."""


@dataclass(frozen=True)
class LoadedEventChainInput:
    episode_id: str
    epi_id: str
    script: Mapping[str, Any]
    trace_rows: tuple[Mapping[str, Any], ...]
    realization_rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]


def _read_object_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventChainInputError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EventChainInputError(f"JSON root must be an object: {path}")
    return value, payload


def _read_jsonl_bytes(path: Path) -> tuple[tuple[dict[str, Any], ...], bytes]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise EventChainInputError(f"cannot read JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EventChainInputError(
                f"invalid JSONL record {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EventChainInputError(
                f"JSONL record must be an object: {path}:{line_number}"
            )
        rows.append(value)
    return tuple(rows), payload


def _project_ref(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace(
            "\\", "/"
        )
    except ValueError as exc:
        raise EventChainInputError(
            f"event-chain input escapes the project root: {path}"
        ) from exc


def _resolve_project_relative(raw: Any, project_root: Path) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise EventChainInputError("episode manifest lacks source_event_script_path")
    declared = Path(value)
    if declared.is_absolute() or ".." in declared.parts:
        raise EventChainInputError(
            "source_event_script_path must be a project-relative path without '..'"
        )
    path = (project_root / declared).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise EventChainInputError(
            f"source_event_script_path escapes project root: {value}"
        ) from exc
    if not path.is_file():
        raise EventChainInputError(f"source event script does not exist: {path}")
    return path


def load_event_chain_input(
    capture_episode_dir: Path, *, project_root: Path = PROJECT_ROOT
) -> LoadedEventChainInput:
    """Load only the manifest, script, trace, and realization Stage-A inputs."""

    episode_id = capture_episode_dir.name
    if EPISODE_ID_PATTERN.fullmatch(episode_id) is None:
        raise EventChainInputError(f"invalid episode directory name: {episode_id}")
    manifest, _manifest_bytes = _read_object_bytes(
        capture_episode_dir / "episode_manifest.json"
    )
    if manifest.get("episode_id") != episode_id:
        raise EventChainInputError(
            f"{episode_id}: episode manifest identity mismatch: {manifest.get('episode_id')!r}"
        )
    epi_id = episode_id.rsplit("__seed", 1)[0]
    if manifest.get("scenario_id") != epi_id:
        raise EventChainInputError(
            f"{episode_id}: manifest scenario_id differs from EPI {epi_id!r}"
        )
    script_path = _resolve_project_relative(
        manifest.get("source_event_script_path"), project_root
    )
    script, script_bytes = _read_object_bytes(script_path)
    trace_path = capture_episode_dir / "event_trace.jsonl"
    realization_path = capture_episode_dir / "event_realization.jsonl"
    trace_rows, trace_bytes = _read_jsonl_bytes(trace_path)
    realization_rows, realization_bytes = _read_jsonl_bytes(realization_path)
    source_seal = manifest.get("source_event_script_digest")
    provenance = {
        "event_script_ref": _project_ref(script_path, project_root),
        "event_script_digest": sha256_bytes(script_bytes),
        "source_manifest_event_script_digest": source_seal,
        "event_trace_ref": _project_ref(trace_path, project_root),
        "event_trace_digest": sha256_bytes(trace_bytes),
        "event_realization_ref": _project_ref(realization_path, project_root),
        "event_realization_digest": sha256_bytes(realization_bytes),
    }
    return LoadedEventChainInput(
        episode_id=episode_id,
        epi_id=epi_id,
        script=script,
        trace_rows=trace_rows,
        realization_rows=realization_rows,
        provenance=provenance,
    )


def build_episode_result(
    capture_episode_dir: Path, *, project_root: Path = PROJECT_ROOT
) -> EventChainBuildResult:
    loaded = load_event_chain_input(capture_episode_dir, project_root=project_root)
    return build_event_chain(
        episode_id=loaded.episode_id,
        epi_id=loaded.epi_id,
        script=loaded.script,
        trace_rows=loaded.trace_rows,
        realization_rows=loaded.realization_rows,
        provenance=loaded.provenance,
    )


def write_event_chain_episode(
    output_episode_dir: Path, result: EventChainBuildResult
) -> None:
    if output_episode_dir.exists():
        raise EventChainInputError(
            f"output episode already exists: {output_episode_dir}"
        )
    output_episode_dir.mkdir(parents=False)
    manifest_path = output_episode_dir / "event_chain_manifest.json"
    relation_path = output_episode_dir / "event_chain_relations.jsonl"
    manifest_path.write_text(
        json.dumps(
            dict(result.manifest),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    relation_path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in result.relations),
        encoding="utf-8",
        newline="\n",
    )


def _require_exact_output_entries(output_episode_dir: Path) -> None:
    entries = {path.name for path in output_episode_dir.iterdir()}
    if entries != OUTPUT_FILENAMES or any(
        not (output_episode_dir / filename).is_file() for filename in OUTPUT_FILENAMES
    ):
        raise EventChainInputError(
            f"staged output entries must be exactly {sorted(OUTPUT_FILENAMES)}, "
            f"found {sorted(entries)}"
        )


def _rename_directory(source: Path, target: Path) -> None:
    source.replace(target)


def _exchange_directories(first: Path, second: Path) -> None:
    """Atomically exchange two existing directories on the Linux server."""

    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(first),
        -100,
        os.fsencode(second),
        2,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{first} <-> {second}",
        )


def _replace_episode_directory(
    staged: Path, destination: Path, *, output_root: Path
) -> None:
    """Atomically install one complete episode without a missing-output window."""

    if destination.parent.resolve() != output_root.resolve():
        raise EventChainInputError(
            f"refusing to replace output outside selected root: {destination}"
        )
    _require_exact_output_entries(staged)
    if not destination.exists():
        _rename_directory(staged, destination)
        return
    if not destination.is_dir():
        raise EventChainInputError(f"output episode is not a directory: {destination}")
    try:
        _exchange_directories(staged, destination)
    except OSError as exc:
        raise EventChainInputError(
            f"atomic replacement failed; prior output is unchanged: {destination}: {exc}"
        ) from exc


def _install_formal_root(
    staged_root: Path, output_root: Path, *, replace_existing: bool
) -> None:
    if output_root.is_symlink():
        raise EventChainInputError(
            f"formal event-chain output root must not be a symlink: {output_root}"
        )
    if output_root.exists():
        if not output_root.is_dir():
            raise EventChainInputError(
                f"formal event-chain output root is not a directory: {output_root}"
            )
        if not replace_existing:
            raise EventChainInputError(
                f"output root exists; pass --replace-existing: {output_root}"
            )
        try:
            _exchange_directories(staged_root, output_root)
        except OSError as exc:
            raise EventChainInputError(
                "atomic formal-corpus replacement failed; prior output is unchanged: "
                f"{output_root}: {exc}"
            ) from exc
        return
    _rename_directory(staged_root, output_root)


def _selected_episode_dirs(capture_root: Path, names: Sequence[str]) -> list[Path]:
    if not capture_root.is_dir():
        raise EventChainInputError(f"capture root does not exist: {capture_root}")
    available = {
        path.name: path
        for path in capture_root.iterdir()
        if path.is_dir() and EPISODE_ID_PATTERN.fullmatch(path.name)
    }
    if names:
        if len(set(names)) != len(names):
            raise EventChainInputError("duplicate --episode selection")
        missing = sorted(set(names) - set(available))
        if missing:
            raise EventChainInputError(f"unknown episodes: {missing}")
        return [available[name] for name in sorted(names)]
    epi_seeds: dict[str, set[str]] = {}
    for episode_id in available:
        epi_id, seed = episode_id.rsplit("__", 1)
        epi_seeds.setdefault(epi_id, set()).add(seed)
    bad_seed_sets = {
        epi_id: sorted(seeds)
        for epi_id, seeds in epi_seeds.items()
        if seeds != FORMAL_SEEDS
    }
    if (
        len(available) != FORMAL_EPISODE_COUNT
        or len(epi_seeds) != FORMAL_EPI_COUNT
        or bad_seed_sets
    ):
        raise EventChainInputError(
            "production build requires the exact formal 210-episode/70-EPI/3-seed "
            "capture set; use explicit --episode selections for diagnostics: "
            f"episodes={len(available)}, epis={len(epi_seeds)}, "
            f"bad_seed_sets={bad_seed_sets}"
        )
    return [available[name] for name in sorted(available)]


def build_selected_episodes(
    *,
    capture_root: Path,
    output_root: Path,
    episode_names: Sequence[str],
    replace_existing: bool,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    selected = _selected_episode_dirs(capture_root, episode_names)
    formal_build = not episode_names
    built = [
        (
            capture_episode_dir,
            build_episode_result(capture_episode_dir, project_root=project_root),
        )
        for capture_episode_dir in selected
    ]
    if formal_build:
        invalid_results = [
            result.manifest["episode_id"]
            for _capture_episode_dir, result in built
            if result.manifest["runtime_validation_status"] != "PASS"
        ]
        if invalid_results:
            raise EventChainInputError(
                "formal build requires runtime PASS for every episode: "
                f"{invalid_results}"
            )
    selected_names = {path.name for path in selected}
    if formal_build:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output_root.parent,
            prefix=f".{output_root.name}.formal-stage.",
        ) as temporary_parent:
            staged_root = Path(temporary_parent) / output_root.name
            staged_root.mkdir()
            for capture_episode_dir, result in built:
                write_event_chain_episode(
                    staged_root / capture_episode_dir.name,
                    result,
                )
            actual_staged_names = {
                path.name for path in staged_root.iterdir() if path.is_dir()
            }
            if actual_staged_names != selected_names or any(
                not path.is_dir() for path in staged_root.iterdir()
            ):
                raise EventChainInputError(
                    "staged formal event-chain episode set differs from the capture set"
                )
            _install_formal_root(
                staged_root,
                output_root,
                replace_existing=replace_existing,
            )
        return [
            {
                "episode_id": capture_episode_dir.name,
                "node_count": result.manifest["node_count"],
                "relation_count": result.manifest["relation_count"],
                "runtime_validation_status": result.manifest[
                    "runtime_validation_status"
                ],
                "event_script_digest_authority": result.manifest["input_provenance"][
                    "event_script_digest_authority"
                ],
            }
            for capture_episode_dir, result in built
        ]

    output_root.mkdir(parents=True, exist_ok=True)
    existing_destinations = sorted(
        capture_episode_dir.name
        for capture_episode_dir, _result in built
        if (output_root / capture_episode_dir.name).exists()
    )
    if existing_destinations and not replace_existing:
        raise EventChainInputError(
            "output episodes exist; pass --replace-existing: "
            f"{existing_destinations}"
        )
    summaries: list[dict[str, Any]] = []
    for capture_episode_dir, result in built:
        episode_id = capture_episode_dir.name
        destination = output_root / episode_id
        if destination.exists() and not replace_existing:
            raise EventChainInputError(
                f"output episode exists; pass --replace-existing: {destination}"
            )
        with tempfile.TemporaryDirectory(
            dir=output_root, prefix=f".{episode_id}.event-chain."
        ) as temporary_root:
            staged = Path(temporary_root) / episode_id
            write_event_chain_episode(staged, result)
            _replace_episode_directory(
                staged,
                destination,
                output_root=output_root,
            )
        summaries.append(
            {
                "episode_id": episode_id,
                "node_count": result.manifest["node_count"],
                "relation_count": result.manifest["relation_count"],
                "runtime_validation_status": result.manifest[
                    "runtime_validation_status"
                ],
                "event_script_digest_authority": result.manifest["input_provenance"][
                    "event_script_digest_authority"
                ],
            }
        )
    return summaries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summaries = build_selected_episodes(
            capture_root=args.capture_root.resolve(),
            output_root=args.output_root.resolve(),
            episode_names=args.episode,
            replace_existing=bool(args.replace_existing),
        )
    except EventChainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    totals = {
        "episodes": len(summaries),
        "nodes": sum(int(row["node_count"]) for row in summaries),
        "relations": sum(int(row["relation_count"]) for row in summaries),
        "runtime_failures": sum(
            row["runtime_validation_status"] != "PASS" for row in summaries
        ),
    }
    print(json.dumps(totals, ensure_ascii=False, sort_keys=True))
    return 0 if totals["runtime_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EventChainInputError",
    "LoadedEventChainInput",
    "build_episode_result",
    "build_selected_episodes",
    "load_event_chain_input",
    "write_event_chain_episode",
]
