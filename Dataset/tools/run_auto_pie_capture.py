#!/usr/bin/env python3
"""Launch or reuse UE PIE and run deterministic episode captures.

This wrapper intentionally avoids UI automation. The UE editor module starts PIE
when launched with -AeroAutoPIE and writes a ready sentinel after PostPIEStarted.
For long capture runs, the default is to reuse an existing PIE session on the
AirSim RPC port and to keep UE open after each chunk. Only close UE explicitly
when requested or before a required C++ rebuild.
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
DEFAULT_PROJECT = PROJECT_ROOT / "DynamicCityCreatorEx.uproject"
DEFAULT_UE_ROOT = Path(r"E:\UE_5.2")
DEFAULT_EDITOR_EXE = DEFAULT_UE_ROOT / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe"
DEFAULT_READY_FILE = PROJECT_ROOT / "Saved" / "AutoPIE" / "auto_pie_ready.json"
DEFAULT_BATCH_SCRIPT = Path(__file__).resolve().parent / "batch_render_dataset.py"

if str(SUMO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUMO_SCRIPTS_DIR))

from aero_sim_client import AeroSimClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start UE Editor, auto-enter PIE, then render selected episodes in restartable chunks."
    )
    parser.add_argument("--ue-root", type=Path, default=DEFAULT_UE_ROOT)
    parser.add_argument("--editor-exe", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--map", default="/Game/Maps/donghu")
    parser.add_argument("--ready-file", type=Path, default=DEFAULT_READY_FILE)
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--startup-delay-s", type=float, default=8.0)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--rpc-timeout-s", type=float, default=180.0)
    parser.add_argument("--batch-timeout-s", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--keep-ue-open", action="store_true", default=True)
    parser.add_argument("--close-ue-on-success", action="store_false", dest="keep_ue_open")
    parser.add_argument("--reuse-existing-ue", action="store_true", default=True)
    parser.add_argument("--no-reuse-existing-ue", action="store_false", dest="reuse_existing_ue")
    parser.add_argument(
        "--reuse-ue-per-run",
        action="store_true",
        default=True,
        help="Launch or reuse UE/PIE once and run all selected chunks against that session.",
    )
    parser.add_argument("--restart-ue-per-chunk", action="store_false", dest="reuse_ue_per_run")
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


def remove_ready_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def read_ready_status(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ""
    return str(payload.get("status", ""))


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


def wait_for_auto_pie(
    proc: subprocess.Popen,
    ready_file: Path,
    host: str,
    port: int,
    startup_timeout_s: float,
    rpc_timeout_s: float,
) -> None:
    ready_deadline = time.monotonic() + float(startup_timeout_s)
    accepted_statuses = {"post_pie_started", "play_world_available"}
    while time.monotonic() < ready_deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"UnrealEditor exited before PIE was ready; exit_code={proc.returncode}")
        status = read_ready_status(ready_file)
        if status in accepted_statuses:
            break
        if status == "timeout":
            raise RuntimeError("UE auto-PIE reported timeout")
        time.sleep(2.0)
    else:
        raise RuntimeError(f"Timed out waiting for auto-PIE ready file: {ready_file}")

    rpc_deadline = time.monotonic() + float(rpc_timeout_s)
    while time.monotonic() < rpc_deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"UnrealEditor exited before RPC was ready; exit_code={proc.returncode}")
        if is_port_open(host, port):
            return
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for RPC {host}:{port}")


def launch_unreal(args: argparse.Namespace) -> subprocess.Popen:
    editor_exe = args.editor_exe or (Path(args.ue_root) / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe")
    if not editor_exe.exists():
        raise FileNotFoundError(f"UnrealEditor.exe not found: {editor_exe}")
    if not Path(args.project).exists():
        raise FileNotFoundError(f"uproject not found: {args.project}")

    remove_ready_file(args.ready_file)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(editor_exe),
        str(args.project),
        str(args.map),
        "-AeroAutoPIE",
        f"-AeroPIEMap={args.map}",
        f"-AeroPIEReadyFile={args.ready_file}",
        f"-AeroPIEStartupDelaySeconds={float(args.startup_delay_s):.3f}",
        f"-AeroPIEStartupTimeoutSeconds={float(args.startup_timeout_s):.3f}",
        "-AeroPIEFailOnTimeout",
        "-nosplash",
        "-log",
    ]
    print(json.dumps({"phase": "launch_ue", "command": command}, ensure_ascii=False))
    if args.dry_run:
        raise RuntimeError("dry-run requested before launching UE")
    return subprocess.Popen(command, cwd=str(PROJECT_ROOT))


def can_reuse_existing_unreal(args: argparse.Namespace) -> bool:
    return bool(args.reuse_existing_ue) and is_port_open(str(args.host), int(args.port), timeout_s=2.0)


def terminate_unreal(proc: subprocess.Popen, grace_s: float = 45.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30.0)


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
    for episode in episodes:
        command.extend(["--episode", episode])

    print(json.dumps({"phase": "run_batch", "episodes": episodes, "command": command}, ensure_ascii=False))
    if args.dry_run:
        return

    timeout = None if float(args.batch_timeout_s) <= 0.0 else float(args.batch_timeout_s)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True, timeout=timeout)


def run_chunk_with_retries(args: argparse.Namespace, episodes: list[str], chunk_index: int) -> None:
    if args.dry_run:
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

    attempts = max(1, int(args.retries) + 1)
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        proc: subprocess.Popen | None = None
        reused_existing_ue = False
        try:
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
            else:
                proc = launch_unreal(args)
                wait_for_auto_pie(
                    proc,
                    args.ready_file,
                    str(args.host),
                    int(args.port),
                    float(args.startup_timeout_s),
                    float(args.rpc_timeout_s),
                )
                rpc_preflight(args, phase="rpc_preflight_launched_ue")
            run_batch(args, episodes)
            print(json.dumps({"phase": "chunk_done", "chunk_index": chunk_index, "episodes": episodes}, ensure_ascii=False))
            if proc is not None and not args.keep_ue_open:
                terminate_unreal(proc)
            return
        except Exception as exc:
            last_error = exc
            print(
                json.dumps(
                    {
                        "phase": "chunk_failed",
                        "chunk_index": chunk_index,
                        "attempt": attempt_index + 1,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            if proc is not None and not args.keep_ue_open:
                terminate_unreal(proc)
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
    raise RuntimeError(f"chunk {chunk_index} failed after {attempts} attempts: {last_error}")


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

    attempts = max(1, int(args.retries) + 1)
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        proc: subprocess.Popen | None = None
        reused_existing_ue = False
        try:
            print(
                json.dumps(
                    {
                        "phase": "run_start",
                        "attempt": attempt_index + 1,
                        "attempts": attempts,
                        "chunk_count": len(chunks),
                        "reuse_ue_per_run": True,
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
            else:
                proc = launch_unreal(args)
                wait_for_auto_pie(
                    proc,
                    args.ready_file,
                    str(args.host),
                    int(args.port),
                    float(args.startup_timeout_s),
                    float(args.rpc_timeout_s),
                )
                rpc_preflight(args, phase="rpc_preflight_launched_ue")
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
            if proc is not None and not args.keep_ue_open:
                terminate_unreal(proc)
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
            if proc is not None and not args.keep_ue_open:
                terminate_unreal(proc)
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
    )
    episodes = select_episodes(args)
    if not episodes:
        raise SystemExit("No source episodes selected.")

    total_chunks = int(math.ceil(len(episodes) / max(1, int(args.chunk_size))))
    print(json.dumps({"phase": "selected", "episode_count": len(episodes), "chunk_count": total_chunks}, ensure_ascii=False))
    chunks = list(chunked(episodes, int(args.chunk_size)))
    if args.reuse_ue_per_run:
        run_chunks_with_single_unreal(args, chunks)
    else:
        for chunk_index, chunk in enumerate(chunks, start=1):
            run_chunk_with_retries(args, chunk, chunk_index)


if __name__ == "__main__":
    main()
