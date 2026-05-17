#!/usr/bin/env python3
"""Long-run supervisor for formal event-chain capture.

The supervisor delegates capture work to ``run_semantic_event_chain_every10.py``
but owns the slow, stateful orchestration around UE/PIE readiness. It advances
one high overview or one active UAV entity at a time, lets the runner resume
completed modality/tick outputs, and reuses the current PIE session for
ordinary work. It starts/re-enters PIE only when the existing session is not
available and the user explicitly opts into recovery; runner guard failures are
allowed to fail first and leave UE open.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "Dataset" / "tools"
SCRIPT_DIR = ROOT / "Plugins" / "SumoImporter" / "Scripts"
RUNNER = TOOLS_DIR / "run_semantic_event_chain_every10.py"
DEFAULT_EPISODES_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
DEFAULT_OUTPUT_ROOT = Path("F:/aw_cap")
DEFAULT_SUMMARY = Path("F:/aw_cap_summary.csv")
DEFAULT_UNREAL_EDITOR = Path("E:/UE_5.2/Engine/Binaries/Win64/UnrealEditor.exe")
DEFAULT_UPROJECT = ROOT / "DynamicCityCreatorEx.uproject"
DEFAULT_AIRSIM_SETTINGS = Path("E:/HuaweiMoveData/Users/weizhiwei/Documents/AirSim/settings.json")

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_semantic_event_chain_every10 as runner  # noqa: E402
from editor_hook_client import UnrealEditorRemoteExecution  # noqa: E402


def is_f_drive_path(path: Path) -> bool:
    text = str(path.resolve() if path.is_absolute() else path).replace("\\", "/").lower()
    return text.startswith("f:/")


def ps_single(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_powershell(script: str, *, timeout_s: float = 120.0) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(1.0, float(timeout_s)),
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or f"PowerShell failed rc={completed.returncode}")
    return completed.stdout


def parse_csv_ints(value: str) -> list[int]:
    values: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("fallback chunk sizes must be positive")
        values.append(parsed)
    return values


def episode_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Episode root does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "render_host_config.json").exists())


def selected_episode_indexes(root: Path, names: Sequence[str], start_index: int, limit: int) -> list[tuple[int, Path]]:
    dirs = episode_dirs(root)
    if names:
        wanted = list(names)
        by_name = {path.name: (index, path) for index, path in enumerate(dirs)}
        missing = sorted(name for name in wanted if name not in by_name)
        if missing:
            raise SystemExit(f"Missing capture-filtered episodes: {missing}")
        selected = [by_name[name] for name in wanted]
    else:
        selected = [(index, path) for index, path in enumerate(dirs)]
        selected = selected[max(0, int(start_index)) :]
        if limit > 0:
            selected = selected[:limit]
    return selected


def capture_ticks(args: argparse.Namespace) -> list[int]:
    return list(range(int(args.tick_start), int(args.tick_end) + 1, int(args.tick_step)))


def active_uavs(episode_dir: Path, args: argparse.Namespace) -> list[tuple[str, list[int]]]:
    active = runner.scene_uav_active_ticks_by_entity(episode_dir, capture_ticks(args))
    return [(entity_id, active[entity_id]) for entity_id in sorted(active) if active[entity_id]]


def capture_view_id(entity_id: str, ordinal: int) -> str:
    return f"uav_view_{int(ordinal):03d}__{runner.safe_name(entity_id)}"


def summary_rows(summary: Path) -> list[dict[str, str]]:
    if not summary.exists() or summary.stat().st_size == 0:
        return []
    with summary.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def existing_file(path_text: Any) -> bool:
    text = str(path_text or "").strip()
    return bool(text) and Path(text).exists()


def high_complete(args: argparse.Namespace, episode_index: int, episode_id: str = "") -> bool:
    expected = set(capture_ticks(args))
    rows_by_tick: dict[int, dict[str, str]] = {}
    for row in summary_rows(args.summary):
        row_episode = str(row.get("episode") or "")
        if episode_id:
            if row_episode != str(episode_id):
                continue
        elif str(row.get("index") or "") != str(int(episode_index)):
            continue
        if str(row.get("view") or "") != "high_overview_rgb":
            continue
        if str(row.get("status") or "") != "ok":
            continue
        try:
            tick = int(row.get("tick") or "")
        except Exception:
            continue
        rows_by_tick[tick] = row
    if set(rows_by_tick) != expected:
        return False
    return all(existing_file(row.get("rgb_path")) for row in rows_by_tick.values())


def entity_complete(
    args: argparse.Namespace,
    *,
    episode_index: int,
    episode_id: str = "",
    entity_id: str,
    view_id: str,
    active_ticks: list[int],
) -> bool:
    expected = {int(tick) for tick in active_ticks}
    rows_by_tick: dict[int, dict[str, str]] = {}
    for row in summary_rows(args.summary):
        row_episode = str(row.get("episode") or "")
        if episode_id:
            if row_episode != str(episode_id):
                continue
        elif str(row.get("index") or "") != str(int(episode_index)):
            continue
        if str(row.get("view") or "") != "uav_event_chain":
            continue
        if str(row.get("status") or "") != "ok":
            continue
        if str(row.get("capture_entity_id") or "") != entity_id:
            continue
        if str(row.get("capture_view_id") or "") != view_id:
            continue
        try:
            tick = int(row.get("tick") or "")
        except Exception:
            continue
        rows_by_tick[tick] = row
    if set(rows_by_tick) != expected:
        return False
    required = ("rgb_path", "depth_path", "seg_path", "seg_palette_path")
    return all(all(existing_file(row.get(field)) for field in required) for row in rows_by_tick.values())


def rpc_available(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def wait_for_rpc(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    while time.monotonic() < deadline:
        if rpc_available(host, port, timeout_s=2.0):
            return
        time.sleep(2.0)
    raise RuntimeError(f"AirSim RPC did not become available at {host}:{port}")


def wait_for_remote(project_root: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_error = ""
    while time.monotonic() < deadline:
        remote = UnrealEditorRemoteExecution(project_root=project_root, discovery_timeout_s=5.0)
        try:
            result = remote.run_python("print('UE_REMOTE_READY')", unattended=True, raise_on_failure=False)
            output = json.dumps(result, ensure_ascii=False, default=str)
            if "UE_REMOTE_READY" in output:
                return
        except Exception as exc:
            last_error = str(exc)
        finally:
            remote.close()
        time.sleep(3.0)
    raise RuntimeError(f"UE remote execution did not become ready: {last_error}")


def verify_pie_world(project_root: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_output = ""
    code = """
import unreal
sub = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
print('IN_PIE', sub.is_in_play_in_editor())
worlds = unreal.EditorLevelLibrary.get_pie_worlds(False)
print('PIE_WORLD_COUNT', len(worlds))
"""
    while time.monotonic() < deadline:
        remote = UnrealEditorRemoteExecution(project_root=project_root, discovery_timeout_s=5.0)
        try:
            result = remote.run_python(code, unattended=True, raise_on_failure=False)
            last_output = json.dumps(result, ensure_ascii=False, default=str)
            if "IN_PIE" in last_output and "True" in last_output and "PIE_WORLD_COUNT" in last_output:
                if '"output": "PIE_WORLD_COUNT 0' not in last_output and "PIE_WORLD_COUNT 0" not in last_output:
                    return
        except Exception as exc:
            last_output = str(exc)
        finally:
            remote.close()
        time.sleep(2.0)
    raise RuntimeError(f"PIE world did not become ready: {last_output}")


def stop_unreal_processes() -> None:
    run_powershell(
        """
Get-Process UnrealEditor -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process CrashReportClientEditor -ErrorAction SilentlyContinue | Stop-Process -Force
""",
        timeout_s=60.0,
    )


def start_unreal(args: argparse.Namespace) -> None:
    script = f"""
$ue = {ps_single(args.unreal_editor)}
$uproject = {ps_single(args.uproject)}
$settings = {ps_single(args.airsim_settings)}
$argList = @('"' + $uproject + '"', '-settings="' + $settings + '"')
$proc = Start-Process -FilePath $ue -ArgumentList $argList -PassThru
Write-Output "STARTED_UNREAL_PID=$($proc.Id)"
"""
    output = run_powershell(script, timeout_s=60.0).strip()
    if output:
        print(f"[supervisor] {output}", flush=True)


def send_play_hotkey() -> None:
    script = r"""
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class CodexUeFocus {
  [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd);
}
"@
Add-Type -AssemblyName System.Windows.Forms
$p = Get-Process UnrealEditor -ErrorAction Stop | Select-Object -First 1
$h = [IntPtr]$p.MainWindowHandle
[CodexUeFocus]::ShowWindowAsync($h, 3) | Out-Null
Start-Sleep -Milliseconds 500
[CodexUeFocus]::BringWindowToTop($h) | Out-Null
[CodexUeFocus]::SetForegroundWindow($h) | Out-Null
Start-Sleep -Seconds 2
[System.Windows.Forms.SendKeys]::SendWait('%p')
"""
    run_powershell(script, timeout_s=30.0)


def recover_pie(args: argparse.Namespace) -> None:
    print("[supervisor] restarting UE/PIE", flush=True)
    stop_unreal_processes()
    time.sleep(max(0.0, float(args.unreal_restart_delay_s)))
    start_unreal(args)
    wait_for_remote(ROOT, float(args.ue_remote_timeout_s))
    send_play_hotkey()
    wait_for_rpc(str(args.host), int(args.port), float(args.rpc_timeout_s))
    verify_pie_world(ROOT, float(args.ue_remote_timeout_s))
    print("[supervisor] PIE ready", flush=True)


def try_existing_pie(args: argparse.Namespace) -> bool:
    remote_timeout = min(float(args.ue_remote_timeout_s), float(args.reuse_pie_check_timeout_s))
    rpc_timeout = min(float(args.rpc_timeout_s), float(args.reuse_rpc_check_timeout_s))
    try:
        wait_for_remote(ROOT, remote_timeout)
        wait_for_rpc(str(args.host), int(args.port), rpc_timeout)
        verify_pie_world(ROOT, remote_timeout)
    except Exception as exc:
        print(f"[supervisor] existing UE/PIE not ready: {exc}", flush=True)
        return False
    print("[supervisor] reusing existing UE/PIE", flush=True)
    return True


def ensure_pie(args: argparse.Namespace, *, force_recover: bool) -> None:
    if bool(args.recover_pie_before_attempt) or force_recover:
        recover_pie(args)
        return
    if try_existing_pie(args):
        return
    raise RuntimeError(
        "Existing UE PIE/AirSim RPC is not ready. Refusing to restart UE automatically; "
        "pass --recover-pie-before-attempt only when the user explicitly approves closing/restarting UE."
    )


def base_runner_command(args: argparse.Namespace, *, episode_index: int, chunk_size: int) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--episodes-root",
        str(args.episodes_root),
        "--output-root",
        str(args.output_root),
        "--summary",
        str(args.summary),
        "--rules",
        str(args.rules),
        "--contract",
        str(args.contract),
        "--capture-presets",
        str(args.capture_presets),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--start-index",
        str(episode_index),
        "--limit",
        "1",
        "--tick-start",
        str(args.tick_start),
        "--tick-end",
        str(args.tick_end),
        "--tick-step",
        str(args.tick_step),
        "--simulation-tick-stride",
        str(args.simulation_tick_stride),
        "--capture-ticks-per-host-run",
        str(chunk_size),
        "--max-working-set-gb",
        "18.0",
        "--max-private-memory-gb",
        "18.0",
        "--min-system-free-memory-gb",
        str(args.min_system_free_memory_gb),
        "--uav-capture-backend",
        "editor_hook",
        "--uav-scene-control-backend",
        "truth_frame_scene_sync",
    ]


def runner_command_for_high(args: argparse.Namespace, *, episode_index: int) -> list[str]:
    command = base_runner_command(args, episode_index=episode_index, chunk_size=int(args.high_capture_ticks_per_host_run))
    command.extend(["--skip-uav", "--append-summary", "--resume-completed-ok"])
    return command


def runner_command_for_entity(
    args: argparse.Namespace,
    *,
    episode_index: int,
    entity_id: str,
    chunk_size: int,
) -> list[str]:
    command = base_runner_command(args, episode_index=episode_index, chunk_size=chunk_size)
    command.extend(
        [
            "--skip-high-overview",
            "--append-summary",
            "--resume-completed-ok",
            "--resume-partial-modalities",
            "--airsim-capture-entity",
            entity_id,
        ]
    )
    return command


def run_attempt(args: argparse.Namespace, command: list[str], *, force_recover: bool) -> int:
    ensure_pie(args, force_recover=force_recover)
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


def validate_supervisor_args(args: argparse.Namespace) -> None:
    for path, label in (
        (RUNNER, "formal runner"),
        (args.episodes_root, "episodes root"),
        (args.rules, "semantic rules"),
        (args.contract, "runtime contract"),
        (args.capture_presets, "capture presets"),
        (args.unreal_editor, "UnrealEditor.exe"),
        (args.uproject, "uproject"),
        (args.airsim_settings, "AirSim settings"),
    ):
        if not Path(path).exists():
            raise SystemExit(f"Missing {label}: {path}")
    if not is_f_drive_path(args.output_root):
        raise SystemExit(f"Formal output root must be on F:, got {args.output_root}")
    if not is_f_drive_path(args.summary):
        raise SystemExit(f"Formal summary must be on F:, got {args.summary}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Supervise formal event-chain capture entity by entity.")
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rules", type=Path, default=ROOT / "Config" / "LowAltitude" / "semantic_stencil_rules.json")
    parser.add_argument("--contract", type=Path, default=ROOT / "Config" / "LowAltitude" / "semantic_capture_runtime_contract.json")
    parser.add_argument("--capture-presets", type=Path, default=ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_capture_presets.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--episode", action="append", default=[], help="Capture one named filtered episode; repeatable.")
    parser.add_argument("--tick-start", type=int, default=0)
    parser.add_argument("--tick-end", type=int, default=900)
    parser.add_argument("--tick-step", type=int, default=5)
    parser.add_argument("--simulation-tick-stride", type=int, default=5)
    parser.add_argument("--capture-ticks-per-host-run", type=int, default=32)
    parser.add_argument("--high-capture-ticks-per-host-run", type=int, default=32)
    parser.add_argument("--fallback-capture-ticks-per-host-run", type=parse_csv_ints, default=parse_csv_ints("16,8"))
    parser.add_argument("--min-system-free-memory-gb", type=float, default=1.0)
    parser.add_argument("--max-attempts-per-high", type=int, default=8)
    parser.add_argument("--max-attempts-per-entity", type=int, default=24)
    parser.add_argument("--unreal-editor", type=Path, default=DEFAULT_UNREAL_EDITOR)
    parser.add_argument("--uproject", type=Path, default=DEFAULT_UPROJECT)
    parser.add_argument("--airsim-settings", type=Path, default=DEFAULT_AIRSIM_SETTINGS)
    parser.add_argument("--recover-pie-before-attempt", dest="recover_pie_before_attempt", action="store_true", default=False)
    parser.add_argument("--no-recover-pie-before-attempt", dest="recover_pie_before_attempt", action="store_false")
    parser.add_argument("--reuse-pie-check-timeout-s", type=float, default=45.0)
    parser.add_argument("--reuse-rpc-check-timeout-s", type=float, default=30.0)
    parser.add_argument("--unreal-restart-delay-s", type=float, default=5.0)
    parser.add_argument("--ue-remote-timeout-s", type=float, default=600.0)
    parser.add_argument("--rpc-timeout-s", type=float, default=360.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Print selected episodes and active UAVs without touching UE.")
    args = parser.parse_args(argv)

    args.episodes_root = args.episodes_root.resolve()
    args.output_root = args.output_root.resolve()
    args.summary = args.summary.resolve()
    args.rules = args.rules.resolve()
    args.contract = args.contract.resolve()
    args.capture_presets = args.capture_presets.resolve()
    args.unreal_editor = args.unreal_editor.resolve()
    args.uproject = args.uproject.resolve()
    args.airsim_settings = args.airsim_settings.resolve()
    validate_supervisor_args(args)

    names = [name for group in args.episode for name in str(group).split(",") if name.strip()]
    selected = selected_episode_indexes(args.episodes_root, names, args.start_index, args.limit)
    chunk_sizes = [int(args.capture_ticks_per_host_run), *list(args.fallback_capture_ticks_per_host_run)]
    chunk_sizes = list(dict.fromkeys(size for size in chunk_sizes if size > 0))
    if args.plan_only:
        plan = {
            "episodes_root": str(args.episodes_root),
            "output_root": str(args.output_root),
            "summary": str(args.summary),
            "episode_count": len(selected),
            "chunk_retry_profile": chunk_sizes,
            "episodes": [
                {
                    "index": index,
                    "name": path.name,
                    "high_complete": high_complete(args, index, path.name),
                    "active_uavs": [
                        {
                            "ordinal": ordinal,
                            "entity_id": entity_id,
                            "capture_view_id": capture_view_id(entity_id, ordinal),
                            "active_tick_count": len(ticks),
                            "complete": entity_complete(
                                args,
                                episode_index=index,
                                episode_id=path.name,
                                entity_id=entity_id,
                                view_id=capture_view_id(entity_id, ordinal),
                                active_ticks=ticks,
                            ),
                        }
                        for ordinal, (entity_id, ticks) in enumerate(active_uavs(path, args))
                    ],
                }
                for index, path in selected
            ],
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    print(
        json.dumps(
            {
                "episodes_root": str(args.episodes_root),
                "output_root": str(args.output_root),
                "summary": str(args.summary),
                "episode_count": len(selected),
                "chunk_retry_profile": chunk_sizes,
                "planning_mode": "streaming_per_episode",
                "recover_pie_before_attempt": bool(args.recover_pie_before_attempt),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    failures: list[dict[str, str]] = []
    for episode_ordinal, (episode_index, path) in enumerate(selected, start=1):
        print(f"[supervisor] episode {episode_ordinal}/{len(selected)} e{episode_index:02d} {path.name}", flush=True)
        if high_complete(args, episode_index, path.name):
            print(f"[supervisor] e{episode_index:02d} high_overview_rgb already complete", flush=True)
        else:
            high_ok = False
            for attempt in range(1, int(args.max_attempts_per_high) + 1):
                print(f"[supervisor] e{episode_index:02d} high attempt {attempt}", flush=True)
                rc = run_attempt(
                    args,
                    runner_command_for_high(args, episode_index=episode_index),
                    force_recover=False,
                )
                if rc == 0 and high_complete(args, episode_index, path.name):
                    high_ok = True
                    break
                print(f"[supervisor] e{episode_index:02d} high incomplete rc={rc}", flush=True)
            if not high_ok:
                failure = {"episode": path.name, "stage": "high_overview_rgb", "return_code": "incomplete"}
                failures.append(failure)
                if not args.continue_on_error:
                    print(json.dumps({"ok": False, "failures": failures}, ensure_ascii=False, indent=2), file=sys.stderr)
                    return 1

        for ordinal, (entity_id, ticks) in enumerate(active_uavs(path, args)):
            view_id = capture_view_id(entity_id, ordinal)
            if entity_complete(args, episode_index=episode_index, episode_id=path.name, entity_id=entity_id, view_id=view_id, active_ticks=ticks):
                print(f"[supervisor] e{episode_index:02d} entity already complete: {entity_id}", flush=True)
                continue
            entity_ok = False
            for attempt in range(1, int(args.max_attempts_per_entity) + 1):
                if entity_complete(args, episode_index=episode_index, episode_id=path.name, entity_id=entity_id, view_id=view_id, active_ticks=ticks):
                    entity_ok = True
                    break
                chunk_size = chunk_sizes[min(attempt - 1, len(chunk_sizes) - 1)]
                print(
                    f"[supervisor] e{episode_index:02d} entity {ordinal:03d} {entity_id} "
                    f"attempt {attempt} chunk_size={chunk_size}",
                    flush=True,
                )
                rc = run_attempt(
                    args,
                    runner_command_for_entity(
                        args,
                        episode_index=episode_index,
                        entity_id=entity_id,
                        chunk_size=chunk_size,
                    ),
                    force_recover=False,
                )
                if rc == 0 and entity_complete(
                    args,
                    episode_index=episode_index,
                    episode_id=path.name,
                    entity_id=entity_id,
                    view_id=view_id,
                    active_ticks=ticks,
                ):
                    entity_ok = True
                    break
                print(f"[supervisor] e{episode_index:02d} entity incomplete rc={rc}: {entity_id}", flush=True)
            if not entity_ok:
                failure = {"episode": path.name, "stage": "uav_event_chain", "entity_id": entity_id, "return_code": "incomplete"}
                failures.append(failure)
                if not args.continue_on_error:
                    print(json.dumps({"ok": False, "failures": failures}, ensure_ascii=False, indent=2), file=sys.stderr)
                    return 1

    print(json.dumps({"ok": not failures, "episode_count": len(selected), "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
