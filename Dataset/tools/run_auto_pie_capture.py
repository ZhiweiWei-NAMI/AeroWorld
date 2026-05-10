#!/usr/bin/env python3
"""Reuse UE PIE and run deterministic episode captures.

This wrapper intentionally avoids UI automation for the normal dataset path.
It reuses the existing PIE session on the AirSim RPC port and keeps UE open on
success and failure. Starting or closing UE belongs to explicit operator actions
or rebuild workflows, not ordinary capture chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from batch_render_dataset import discover_source_episodes
from batch_render_dataset import validate_single_capture_selection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMO_SCRIPTS_DIR = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scripts"
DEFAULT_BATCH_SCRIPT = Path(__file__).resolve().parent / "batch_render_dataset.py"

if str(SUMO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUMO_SCRIPTS_DIR))

from aero_sim_client import AeroSimClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse an existing UE PIE session, then render selected episodes in restartable chunks."
    )
    parser.add_argument("--ue-root", type=Path, default=Path(""), help=argparse.SUPPRESS)
    parser.add_argument("--editor-exe", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project", type=Path, default=Path(""), help=argparse.SUPPRESS)
    parser.add_argument("--map", default="", help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", type=Path, default=Path(""), help=argparse.SUPPRESS)
    parser.add_argument("--episodes-root", type=Path, default=Path("Dataset/episodes"))
    parser.add_argument("--render-ready-root", type=Path, default=Path("Dataset/render_ready_episodes"))
    parser.add_argument("--output-root", type=Path, default=Path("Saved/AirSim/episode_render_host"))
    parser.add_argument("--episode", action="append", default=[], help="Episode directory name. Repeatable.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--tick-stride", type=int, default=5)
    parser.add_argument("--camera-role", action="append", default=[], choices=["all", "ground", "uav"])
    parser.add_argument("--camera-id", action="append", default=[])
    parser.add_argument("--modality", action="append", default=[])
    parser.add_argument("--segmentation-backend", choices=["ue_custom_stencil"], default="ue_custom_stencil")
    parser.add_argument("--runtime-uav-control-backend", choices=["airsim_move", "pose_sync"], default="airsim_move")
    parser.add_argument("--semantic-rules-path", type=Path, default=Path("Config/LowAltitude/semantic_stencil_rules.json"))
    parser.add_argument("--semantic-stencil-audit-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--rpc-timeout-s", type=float, default=180.0)
    parser.add_argument("--batch-timeout-s", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--reuse-existing-ue", action="store_true", default=True)
    parser.add_argument("--keep-ue-open", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def chunked(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    size = max(1, int(chunk_size))
    for index in range(0, len(items), size):
        yield items[index : index + size]


def select_episodes(args: argparse.Namespace) -> list[str]:
    episode_dirs = discover_source_episodes(
        args.episodes_root,
        list(args.episode or []),
        include_private=bool(args.include_private),
    )
    episode_dirs = episode_dirs[int(args.start) : args.end]
    if args.max_episodes > 0:
        episode_dirs = episode_dirs[: int(args.max_episodes)]
    return [p.name for p in episode_dirs]


def is_port_open(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def rpc_preflight(args: argparse.Namespace, *, phase: str) -> dict:
    host = str(args.host)
    port = int(args.port)
    if not is_port_open(host, port, timeout_s=2.0):
        raise RuntimeError(f"RPC preflight failed: {host}:{port} is not listening.")
    timeout_s = min(30.0, max(5.0, float(args.rpc_timeout_s)))
    try:
        client = AeroSimClient(host=host, port=port, timeout_value=timeout_s, auto_connect=True)
        capabilities = client.describe_capabilities()
    except Exception as exc:
        raise RuntimeError(
            f"RPC preflight failed for {host}:{port}: AirSim ping/capability call did not complete: {exc}"
        ) from exc
    payload = dict(capabilities.get("payload") or {})
    operations = set(payload.get("operations") or [])
    required_ops = {
        "simAeroDescribeCapabilities",
        "simAeroLoadContext",
        "simAeroApplyFrame",
        "simAeroCaptureWorldCamera",
    }
    missing = sorted(required_ops - operations)
    if missing:
        raise RuntimeError(f"RPC preflight failed: bridge missing required operations: {', '.join(missing)}")
    result = {
        "phase": phase,
        "host": host,
        "port": port,
        "status": "ok",
        "operation_count": len(operations),
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def can_reuse_existing_unreal(args: argparse.Namespace) -> bool:
    return bool(args.reuse_existing_ue) and is_port_open(str(args.host), int(args.port), timeout_s=2.0)


def run_batch(args: argparse.Namespace, episodes: list[str]) -> None:
    command = [
        sys.executable,
        str(DEFAULT_BATCH_SCRIPT),
        "--episodes-root",
        str(args.episodes_root),
        "--render-ready-root",
        str(args.render_ready_root),
        "--output-root",
        str(args.output_root),
        "--tick-stride",
        str(int(args.tick_stride)),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--render",
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.include_private:
        command.append("--include-private")
    for role in args.camera_role or []:
        command.extend(["--camera-role", str(role)])
    for camera_id in args.camera_id or []:
        command.extend(["--camera-id", str(camera_id)])
    for modality in args.modality or []:
        command.extend(["--modality", str(modality)])
    command.extend(["--segmentation-backend", str(args.segmentation_backend)])
    command.extend(["--runtime-uav-control-backend", str(args.runtime_uav_control_backend)])
    if args.semantic_rules_path:
        command.extend(["--semantic-rules-path", str(args.semantic_rules_path)])
    if args.semantic_stencil_audit_only:
        command.append("--semantic-stencil-audit-only")
    for episode in episodes:
        command.extend(["--episode", episode])

    print(json.dumps({"phase": "run_batch", "episodes": episodes, "command": command}, ensure_ascii=False))
    if args.dry_run:
        return

    timeout = None if float(args.batch_timeout_s) <= 0.0 else float(args.batch_timeout_s)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True, timeout=timeout)


def run_chunks_with_single_unreal(args: argparse.Namespace, chunks: list[list[str]]) -> None:
    if args.dry_run:
        for chunk_index, episodes in enumerate(chunks, start=1):
            print(
                json.dumps(
                    {
                        "phase": "chunk_dry_run",
                        "chunk_index": chunk_index,
                        "episodes": episodes,
                    },
                    ensure_ascii=False,
                )
            )
            run_batch(args, episodes)
        return

    if not can_reuse_existing_unreal(args):
        raise RuntimeError(
            "Existing UE PIE/RPC session is required by the capture contract. "
            "Start PIE explicitly, then rerun this wrapper."
        )

    attempts = max(1, int(args.retries) + 1)
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        reused_existing_ue = False
        try:
            print(
                json.dumps(
                    {
                        "phase": "run_start",
                        "attempt": attempt_index + 1,
                        "attempts": attempts,
                        "chunk_count": len(chunks),
                        "reuse_existing_pie": True,
                    },
                    ensure_ascii=False,
                )
            )
            if can_reuse_existing_unreal(args):
                reused_existing_ue = True
                rpc_preflight(args, phase="rpc_preflight_reuse_existing_ue")
                print(
                    json.dumps(
                        {
                            "phase": "reuse_existing_ue",
                            "host": str(args.host),
                            "port": int(args.port),
                            "note": "Using already-running PIE/RPC session; UE will not be closed by this wrapper.",
                        },
                        ensure_ascii=False,
                    )
                )
            for chunk_index, episodes in enumerate(chunks, start=1):
                print(
                    json.dumps(
                        {
                            "phase": "chunk_start",
                            "chunk_index": chunk_index,
                            "attempt": attempt_index + 1,
                            "attempts": attempts,
                            "episodes": episodes,
                        },
                        ensure_ascii=False,
                    )
                )
                run_batch(args, episodes)
                print(json.dumps({"phase": "chunk_done", "chunk_index": chunk_index, "episodes": episodes}, ensure_ascii=False))
            print(json.dumps({"phase": "run_done", "chunk_count": len(chunks)}, ensure_ascii=False))
            return
        except Exception as exc:
            last_error = exc
            print(
                json.dumps(
                    {
                        "phase": "run_failed",
                        "attempt": attempt_index + 1,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            if reused_existing_ue:
                print(
                    json.dumps(
                        {
                            "phase": "kept_existing_ue_open_after_failure",
                            "host": str(args.host),
                            "port": int(args.port),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            if attempt_index + 1 < attempts:
                time.sleep(10.0)
    raise RuntimeError(f"run failed after {attempts} attempts: {last_error}")


def main() -> None:
    args = parse_args()
    validate_single_capture_selection(
        camera_roles=list(args.camera_role or []),
        camera_ids=list(args.camera_id or []),
        modalities=list(args.modality or []),
        segmentation_backend=str(args.segmentation_backend),
        semantic_rules_path=args.semantic_rules_path,
        semantic_stencil_audit_only=bool(args.semantic_stencil_audit_only),
    )
    episodes = select_episodes(args)
    if not episodes:
        raise SystemExit("No source episodes selected.")

    total_chunks = int(math.ceil(len(episodes) / max(1, int(args.chunk_size))))
    print(json.dumps({"phase": "selected", "episode_count": len(episodes), "chunk_count": total_chunks}, ensure_ascii=False))
    chunks = list(chunked(episodes, int(args.chunk_size)))
    run_chunks_with_single_unreal(args, chunks)


if __name__ == "__main__":
    main()
