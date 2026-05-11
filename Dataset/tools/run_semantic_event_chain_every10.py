from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOST_SCRIPT = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_render_host.py"
DEFAULT_CAPTURE_PRESETS = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_capture_presets.json"
DEFAULT_OUTPUT_ROOT = Path("F:/aw_cap")
DEFAULT_SUMMARY_PATH = Path("F:/aw_cap_summary.csv")
DEFAULT_CONTRACT = PROJECT_ROOT / "Config" / "LowAltitude" / "semantic_capture_runtime_contract.json"
DEFAULT_RULES = PROJECT_ROOT / "Config" / "LowAltitude" / "semantic_stencil_rules.json"
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "Dataset" / "render_ready_episodes"
UAV_MODALITIES = ("rgb", "depth", "seg")
MODALITY_EXT = {"rgb": "png", "depth": "npy", "seg": "png"}
FATAL_RUNTIME_PATTERNS = (
    "Retry connection over the limit",
    "Client is closed, connection could not be set",
    "Unable to discover a matching Unreal Editor remote execution node",
    "AirSim RPC is unavailable",
)


class FatalRuntimeUnavailable(RuntimeError):
    pass


class HostRunTimeout(RuntimeError):
    pass


@dataclass
class GuardStats:
    ue_working_set_gb_peak: float = 0.0
    ue_private_memory_gb_peak: float = 0.0
    child_working_set_gb_peak: float = 0.0
    child_private_memory_gb_peak: float = 0.0
    system_free_memory_gb_min: float = 0.0

    def absorb(self, other: "GuardStats") -> None:
        self.ue_working_set_gb_peak = max(self.ue_working_set_gb_peak, other.ue_working_set_gb_peak)
        self.ue_private_memory_gb_peak = max(self.ue_private_memory_gb_peak, other.ue_private_memory_gb_peak)
        self.child_working_set_gb_peak = max(self.child_working_set_gb_peak, other.child_working_set_gb_peak)
        self.child_private_memory_gb_peak = max(self.child_private_memory_gb_peak, other.child_private_memory_gb_peak)
        if other.system_free_memory_gb_min > 0:
            if self.system_free_memory_gb_min <= 0:
                self.system_free_memory_gb_min = other.system_free_memory_gb_min
            else:
                self.system_free_memory_gb_min = min(self.system_free_memory_gb_min, other.system_free_memory_gb_min)


@dataclass
class HostRunResult:
    log_paths: list[Path] = field(default_factory=list)
    stats: GuardStats = field(default_factory=GuardStats)


@dataclass
class EpisodePlan:
    index: int
    episode_dir: Path
    site_id: str
    high_camera_id: str
    capture_ticks: list[int]
    uav_active_ticks: dict[str, list[int]]

    @property
    def episode_id(self) -> str:
        return self.episode_dir.name


SUMMARY_FIELDS = [
    "index",
    "episode",
    "view",
    "status",
    "tick",
    "tick_start",
    "tick_end",
    "tick_step",
    "capture_tick_count",
    "capture_tick_index",
    "camera_id",
    "capture_entity_id",
    "capture_view_id",
    "logical_event_id",
    "logical_sample_id",
    "batch_id",
    "frame_id",
    "frame_seq",
    "runtime_uav_count",
    "active_uav_tick_count",
    "uav_control_backend",
    "contract_path",
    "contract_version",
    "rgb_path",
    "depth_path",
    "seg_path",
    "seg_palette_path",
    "modality_outputs",
    "alignment_key",
    "alignment_source",
    "output_dir",
    "host_logs",
    "working_set_gb",
    "private_memory_gb",
    "child_working_set_gb_peak",
    "child_private_memory_gb_peak",
    "system_free_memory_gb_min",
    "error",
]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def short_stable_name(value: str, prefix: str) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}{digest}"


def simple_episode_dir_name(index: int) -> str:
    return f"e{int(index):02d}"


def simple_capture_view_dir_name(capture_view_id: str) -> str:
    match = re.search(r"uav_view_(\d+)", str(capture_view_id))
    if match:
        return f"v{int(match.group(1)):03d}"
    return short_stable_name(str(capture_view_id), "v")


def event_chain_output_dir(output_root: Path, index: int, view: str) -> Path:
    normalized = str(view or "").strip().lower()
    if normalized.startswith("high"):
        return output_root / "hi" / simple_episode_dir_name(index)
    return output_root / "uav" / simple_episode_dir_name(index)


def event_chain_uav_output_dir(output_root: Path, index: int, capture_view_id: str) -> Path:
    return event_chain_output_dir(output_root, index, "uav_event_chain") / simple_capture_view_dir_name(capture_view_id)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_cell(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def is_f_drive_path(path: Path) -> bool:
    return str(path.drive).lower() == "f:"


def normalized_abs_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").rstrip("/").lower()


def looks_timestamped_or_versioned(path: Path) -> bool:
    for part in path.parts:
        normalized = str(part).strip().lower()
        if re.match(r"^v\d+$", normalized):
            return True
        if re.match(r"^version[_-]?\d+$", normalized):
            return True
        if re.match(r"^20\d{6}[_-]?\d{4,}", normalized):
            return True
    return False


def assert_relative_to(path: Path, root: Path, label: str) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under {resolved_root}, got {resolved_path}") from exc


def rpc_available(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _run_powershell_json(script: str) -> dict[str, Any]:
    command = ["powershell", "-NoProfile", "-Command", script]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = result.stdout.strip()
    if not text:
        return {}
    payload = json.loads(text)
    if isinstance(payload, list):
        return dict(payload[0]) if payload else {}
    return dict(payload)


def unreal_memory_gb() -> dict[str, float]:
    payload = _run_powershell_json(
        "Get-Process UnrealEditor -ErrorAction SilentlyContinue | "
        "Select-Object -First 1 Id,"
        "@{Name='WorkingSetGB';Expression={[math]::Round($_.WorkingSet64/1GB,3)}},"
        "@{Name='PrivateMemoryGB';Expression={[math]::Round($_.PrivateMemorySize64/1GB,3)}} | "
        "ConvertTo-Json -Compress"
    )
    return {
        "pid": float(payload.get("Id") or 0.0),
        "working_set_gb": float(payload.get("WorkingSetGB") or 0.0),
        "private_memory_gb": float(payload.get("PrivateMemoryGB") or 0.0),
    }


def process_memory_gb(pid: int) -> dict[str, float]:
    if int(pid) <= 0:
        return {"pid": 0.0, "working_set_gb": 0.0, "private_memory_gb": 0.0}
    payload = _run_powershell_json(
        f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue | "
        "Select-Object -First 1 Id,"
        "@{Name='WorkingSetGB';Expression={[math]::Round($_.WorkingSet64/1GB,3)}},"
        "@{Name='PrivateMemoryGB';Expression={[math]::Round($_.PrivateMemorySize64/1GB,3)}} | "
        "ConvertTo-Json -Compress"
    )
    return {
        "pid": float(payload.get("Id") or 0.0),
        "working_set_gb": float(payload.get("WorkingSetGB") or 0.0),
        "private_memory_gb": float(payload.get("PrivateMemoryGB") or 0.0),
    }


def system_memory_gb() -> dict[str, float]:
    payload = _run_powershell_json(
        "Get-CimInstance Win32_OperatingSystem | "
        "Select-Object -First 1 "
        "@{Name='FreePhysicalGB';Expression={[math]::Round($_.FreePhysicalMemory/1MB,3)}},"
        "@{Name='TotalVisibleGB';Expression={[math]::Round($_.TotalVisibleMemorySize/1MB,3)}} | "
        "ConvertTo-Json -Compress"
    )
    return {
        "free_physical_gb": float(payload.get("FreePhysicalGB") or 0.0),
        "total_visible_gb": float(payload.get("TotalVisibleGB") or 0.0),
    }


def assert_memory_under_limit(args: argparse.Namespace, memory: dict[str, float]) -> None:
    if memory["working_set_gb"] > args.max_working_set_gb:
        raise FatalRuntimeUnavailable(f"Unreal working-set memory guard tripped: {memory}")
    if memory["private_memory_gb"] > args.max_private_memory_gb:
        raise FatalRuntimeUnavailable(f"Unreal private-memory guard tripped: {memory}")


def assert_runtime_available(args: argparse.Namespace) -> dict[str, float]:
    memory = unreal_memory_gb()
    if memory["pid"] <= 0:
        raise FatalRuntimeUnavailable("UnrealEditor process is unavailable.")
    assert_memory_under_limit(args, memory)
    if not rpc_available(args.host, args.port):
        raise FatalRuntimeUnavailable(f"AirSim RPC is unavailable at {args.host}:{args.port}")
    return memory


def looks_like_runtime_unavailable(message: str) -> bool:
    return any(pattern in message for pattern in FATAL_RUNTIME_PATTERNS)


def _kill_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.kill()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass


def run_child_with_guards(
    command: list[str],
    *,
    args: argparse.Namespace,
    log_path: Path,
    label: str,
) -> tuple[int, GuardStats]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1.0, float(args.host_run_timeout_s))
    stats = GuardStats()
    env = dict(os.environ)
    env["Process_narration"] = "false"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("[EventChainRunner] command=" + json_cell(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        while True:
            return_code = process.poll()
            ue_memory = unreal_memory_gb()
            stats.ue_working_set_gb_peak = max(stats.ue_working_set_gb_peak, ue_memory["working_set_gb"])
            stats.ue_private_memory_gb_peak = max(stats.ue_private_memory_gb_peak, ue_memory["private_memory_gb"])
            child_memory = process_memory_gb(int(process.pid))
            stats.child_working_set_gb_peak = max(stats.child_working_set_gb_peak, child_memory["working_set_gb"])
            stats.child_private_memory_gb_peak = max(stats.child_private_memory_gb_peak, child_memory["private_memory_gb"])
            sys_memory = system_memory_gb()
            if sys_memory["free_physical_gb"] > 0:
                if stats.system_free_memory_gb_min <= 0:
                    stats.system_free_memory_gb_min = sys_memory["free_physical_gb"]
                else:
                    stats.system_free_memory_gb_min = min(stats.system_free_memory_gb_min, sys_memory["free_physical_gb"])
            if return_code is not None:
                return int(return_code), stats
            try:
                assert_memory_under_limit(args, ue_memory)
            except Exception:
                _kill_process(process)
                log.write(f"\n[RunnerGuard] {label} stopped by Unreal memory guard: {ue_memory}\n")
                raise
            if child_memory["private_memory_gb"] > args.max_child_private_memory_gb:
                _kill_process(process)
                log.write(f"\n[RunnerGuard] {label} stopped by child private-memory guard: {child_memory}\n")
                raise FatalRuntimeUnavailable(f"Host child private-memory guard tripped: {child_memory}")
            if child_memory["working_set_gb"] > args.max_child_working_set_gb:
                _kill_process(process)
                log.write(f"\n[RunnerGuard] {label} stopped by child working-set guard: {child_memory}\n")
                raise FatalRuntimeUnavailable(f"Host child working-set guard tripped: {child_memory}")
            if 0.0 < sys_memory["free_physical_gb"] < args.min_system_free_memory_gb:
                _kill_process(process)
                log.write(f"\n[RunnerGuard] {label} stopped by system free-memory guard: {sys_memory}\n")
                raise FatalRuntimeUnavailable(f"System free-memory guard tripped: {sys_memory}")
            if time.monotonic() >= deadline:
                _kill_process(process)
                log.write(f"\n[RunnerGuard] {label} timed out after {args.host_run_timeout_s:.1f}s.\n")
                raise HostRunTimeout(f"{label} timed out after {args.host_run_timeout_s:.1f}s; log={log_path}")
            time.sleep(max(0.5, float(args.host_guard_poll_s)))


def sorted_truth_ticks(episode_dir: Path) -> list[int]:
    ticks: list[int] = []
    with (episode_dir / "truth_frames.jsonl").open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            ticks.append(int(frame.get("tick", 0)))
    return sorted(set(ticks))


def event_chain_capture_ticks(
    episode_dir: Path,
    *,
    tick_start: int,
    tick_end: int,
    tick_step: int,
    strict: bool,
) -> list[int]:
    if tick_start != 0:
        raise RuntimeError("Event-chain capture must start from tick 0; nonzero start would be a direct jump.")
    if tick_step <= 0:
        raise RuntimeError(f"--tick-step must be positive, got {tick_step}")
    expected = list(range(int(tick_start), int(tick_end) + 1, int(tick_step)))
    available = set(sorted_truth_ticks(episode_dir))
    missing = [tick for tick in expected if tick not in available]
    if missing and strict:
        preview = ",".join(str(value) for value in missing[:12])
        raise RuntimeError(f"Missing required truth capture ticks in {episode_dir.name}: {preview}")
    return [tick for tick in expected if tick in available]


def truth_submission_state(entity: dict[str, Any]) -> str:
    render_presence = dict(entity.get("render_presence") or {})
    return str(render_presence.get("submission_state") or "").strip()


def truth_visibility_state(entity: dict[str, Any]) -> str:
    render_presence = dict(entity.get("render_presence") or {})
    return str(render_presence.get("visibility_state") or "").strip()


def runtime_uav_ids_from_roster(episode_dir: Path) -> set[str]:
    roster = read_json(episode_dir / "global_entity_roster.json")
    return {
        str(entity.get("entity_id") or "").strip()
        for entity in roster.get("entities") or []
        if str(entity.get("entity_category") or "").lower() == "uav"
        and str(entity.get("mode") or "").lower() in {"runtime_multirotor", ""}
        and str(entity.get("entity_id") or "").strip()
    }


def runtime_uav_active_ticks_by_entity(episode_dir: Path, capture_ticks: list[int]) -> dict[str, list[int]]:
    runtime_ids = runtime_uav_ids_from_roster(episode_dir)
    result: dict[str, list[int]] = {entity_id: [] for entity_id in sorted(runtime_ids)}
    if not runtime_ids:
        return result
    wanted_ticks = set(int(tick) for tick in capture_ticks)
    with (episode_dir / "truth_frames.jsonl").open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            tick = int(frame.get("tick", -1))
            if tick not in wanted_ticks:
                continue
            for entity in frame.get("entities") or []:
                entity_id = str(entity.get("entity_id") or "").strip()
                if entity_id not in runtime_ids:
                    continue
                if truth_submission_state(entity) not in {"", "submit_to_ue"}:
                    continue
                if truth_visibility_state(entity) not in {"", "visible"}:
                    continue
                result.setdefault(entity_id, []).append(tick)
    return {entity_id: sorted(set(ticks)) for entity_id, ticks in result.items()}


def infer_site_id(episode_dir: Path) -> str:
    config = read_json(episode_dir / "render_host_config.json")
    sites = [str(value).strip() for value in (dict(config.get("batch_strategy") or {}).get("sites") or []) if str(value).strip()]
    if sites:
        return sites[0]
    scenario_plan = read_json(episode_dir / "scenario_plan.json")
    site_contracts = dict((scenario_plan.get("compiled_plan_summary") or {}).get("site_contracts") or {})
    if site_contracts:
        return sorted(site_contracts.keys())[0]
    with (episode_dir / "truth_frames.jsonl").open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            site_id = str(frame.get("active_site_id") or "").strip()
            if site_id:
                return site_id
    return "default"


def filter_event_chain_capture_presets(source_path: Path) -> dict[str, Any]:
    presets = read_json(source_path)
    ground = dict(presets.get("ground_cameras") or {})
    filtered_ground: dict[str, list[dict[str, Any]]] = {}
    for site_id, cameras in ground.items():
        selected = []
        for camera in cameras or []:
            camera_payload = dict(camera)
            camera_id = str(camera_payload.get("camera_id", camera_payload.get("camera_name", "")))
            if camera_id.lower().endswith("overview_top"):
                camera_payload["modalities"] = ["rgb"]
                selected.append(camera_payload)
        if selected:
            filtered_ground[str(site_id)] = selected
    if not filtered_ground:
        raise RuntimeError(f"No overview_top ground cameras found in {source_path}")
    presets["ground_cameras"] = filtered_ground
    presets["default_modalities"] = list(UAV_MODALITIES)
    uav_cameras = dict(presets.get("uav_cameras") or {})
    default_uav = []
    for camera in uav_cameras.get("default") or []:
        payload = dict(camera)
        payload["modalities"] = list(UAV_MODALITIES)
        default_uav.append(payload)
    uav_cameras["default"] = default_uav
    presets["uav_cameras"] = uav_cameras
    return presets


def write_event_chain_capture_presets(args: argparse.Namespace) -> Path:
    target = args.output_root / "_meta" / "configs" / "event_chain_capture_presets.json"
    presets = filter_event_chain_capture_presets(args.capture_presets)
    write_json(target, presets)
    return target


def high_overview_camera_id_from_presets(presets: dict[str, Any], site_id: str) -> str:
    ground = dict(presets.get("ground_cameras") or {})
    candidates = ground.get(site_id) or ground.get("default") or []
    for camera in candidates:
        camera_id = str(camera.get("camera_id", camera.get("camera_name", "")))
        if camera_id.lower().endswith("overview_top"):
            return camera_id
    raise RuntimeError(f"No overview_top camera for site {site_id}")


def write_guarded_config(args: argparse.Namespace, episode_dir: Path, index: int, presets_path: Path) -> Path:
    source_config = episode_dir / "render_host_config.json"
    config = read_json(source_config)
    config["capture_presets_path"] = str(presets_path)
    batch_strategy = dict(config.get("batch_strategy") or {})
    batch_strategy["tick_window_size"] = 0
    config["batch_strategy"] = batch_strategy
    timeouts = dict(config.get("timeouts") or {})
    if float(timeouts.get("editor_hook_capture_timeout_s", 0.0) or 0.0) < args.editor_hook_capture_timeout_s:
        timeouts["editor_hook_capture_timeout_s"] = float(args.editor_hook_capture_timeout_s)
    config["timeouts"] = timeouts
    runtime_uav = dict(config.get("runtime_uav") or {})
    runtime_uav["control_backend"] = str(args.uav_control_backend)
    if args.uav_control_backend == "pose_sync":
        runtime_uav["editor_hook_fallback_enabled"] = False
        runtime_uav["non_capture_rpc_failure_nonfatal"] = True
    config["runtime_uav"] = runtime_uav
    config["capture_path_contract"] = {
        "output_root_must_be_on_f_drive": True,
        "simple_primary_paths": True,
        "complex_identifiers_live_in_sidecars": True,
        "source_config_path": str(source_config),
        "source_episode_dir": episode_dir.name,
    }
    config["event_chain_runner"] = {
        "runner": "Dataset/tools/run_semantic_event_chain_every10.py",
        "tick_start": int(args.tick_start),
        "tick_end": int(args.tick_end),
        "tick_step": int(args.tick_step),
        "capture_ticks_per_host_run": int(args.capture_ticks_per_host_run),
        "ground_modalities": ["rgb"],
        "uav_modalities": list(args.uav_modalities),
        "write_depth_preview": bool(args.write_depth_preview),
        "oom_policy": "parent_guard_kills_host_child_keeps_ue_open",
    }
    target = args.output_root / "_meta" / "configs" / f"{simple_episode_dir_name(index)}_{short_stable_name(episode_dir.name, 'e')}.json"
    write_json(target, config)
    return target


def build_episode_plan(args: argparse.Namespace, episode_dir: Path, index: int, presets: dict[str, Any]) -> EpisodePlan:
    ticks = event_chain_capture_ticks(
        episode_dir,
        tick_start=args.tick_start,
        tick_end=args.tick_end,
        tick_step=args.tick_step,
        strict=not args.allow_missing_capture_ticks,
    )
    site_id = infer_site_id(episode_dir)
    camera_id = high_overview_camera_id_from_presets(presets, site_id)
    active_by_entity = runtime_uav_active_ticks_by_entity(episode_dir, ticks)
    active_by_entity = {entity_id: tick_list for entity_id, tick_list in active_by_entity.items() if tick_list}
    explicit = str(args.airsim_capture_entity or "").strip()
    if not explicit:
        raise RuntimeError(
            f"{episode_dir.name}: --airsim-capture-entity is required for canonical UAV capture tasks"
        )
    if explicit not in active_by_entity:
        raise RuntimeError(
            f"{episode_dir.name}: explicit --airsim-capture-entity {explicit} has no active capture ticks"
        )
    active_by_entity = {explicit: active_by_entity.get(explicit, [])}
    return EpisodePlan(
        index=index,
        episode_dir=episode_dir,
        site_id=site_id,
        high_camera_id=camera_id,
        capture_ticks=ticks,
        uav_active_ticks=active_by_entity,
    )


def selected_episode_dirs(args: argparse.Namespace) -> list[tuple[int, Path]]:
    episodes = sorted(path for path in args.episodes_root.iterdir() if path.is_dir())
    selected = episodes[max(0, args.start_index) :]
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("No episodes selected.")
    return [(index, path) for index, path in enumerate(selected, start=max(0, args.start_index))]


def chunk_ticks(ticks: list[int], chunk_size: int) -> list[list[int]]:
    if not ticks:
        return []
    if chunk_size <= 0:
        return [list(ticks)]
    return [ticks[index : index + chunk_size] for index in range(0, len(ticks), chunk_size)]


def clear_output_targets(paths: list[Path], root: Path) -> None:
    for path in paths:
        assert_relative_to(path, root, "Capture output cleanup target")
        if path.exists():
            shutil.rmtree(path)


def base_host_command(args: argparse.Namespace, guarded_config: Path, output_dir: Path, site_id: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(HOST_SCRIPT),
        "--config",
        str(guarded_config),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output_dir",
        str(output_dir),
        "--site",
        site_id,
        "--segmentation-backend",
        "ue_custom_stencil",
        "--semantic-rules-path",
        str(args.rules),
        "--simulation_tick_stride",
        "1",
        "--max_batches",
        "1",
        "--preserve_capture_output_dir",
    ]
    if bool(args.write_depth_preview):
        command.append("--write-depth-preview")
    return command


def run_host_chunks(
    args: argparse.Namespace,
    *,
    base_command: list[str],
    capture_ticks: list[int],
    log_dir: Path,
    log_stem: str,
    label: str,
) -> HostRunResult:
    result = HostRunResult()

    def run_tick_chunk(tick_chunk: list[int], log_suffix: str, chunk_label: str) -> None:
        command = list(base_command)
        command.extend(["--start_tick", str(args.tick_start), "--end_tick", str(max(tick_chunk))])
        for tick in tick_chunk:
            command.extend(["--capture_tick", str(int(tick))])
        log_path = log_dir / f"{log_stem}_{log_suffix}.log"
        try:
            return_code, stats = run_child_with_guards(command, args=args, log_path=log_path, label=chunk_label)
        except HostRunTimeout:
            result.log_paths.append(log_path)
            if len(tick_chunk) <= 1:
                raise
            split_at = max(1, len(tick_chunk) // 2)
            left = tick_chunk[:split_at]
            right = tick_chunk[split_at:]
            print(
                f"  {label}: timeout on {log_suffix}; retrying as "
                f"{left[0]}..{left[-1]} and {right[0]}..{right[-1]}",
                flush=True,
            )
            run_tick_chunk(left, f"{log_suffix}_retry_a", f"{chunk_label} retry a")
            run_tick_chunk(right, f"{log_suffix}_retry_b", f"{chunk_label} retry b")
            return
        result.log_paths.append(log_path)
        result.stats.absorb(stats)
        if return_code != 0:
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            if looks_like_runtime_unavailable(log_text):
                raise FatalRuntimeUnavailable(f"episode_render_host lost runtime connectivity; log={log_path}")
            raise RuntimeError(f"episode_render_host failed rc={return_code}; log={log_path}")
        assert_runtime_available(args)

    for chunk_index, tick_chunk in enumerate(chunk_ticks(capture_ticks, int(args.capture_ticks_per_host_run))):
        run_tick_chunk(tick_chunk, f"chunk{chunk_index:03d}", f"{label} chunk {chunk_index}")
    return result


def normalize_histogram(histogram: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in histogram.items():
        try:
            result[str(int(key))] = int(value)
        except Exception:
            continue
    return result


def semantic_histogram(png_path: Path) -> dict[str, int]:
    from PIL import Image
    import numpy as np

    image = Image.open(png_path)
    values = np.asarray(image.convert("L"), dtype=np.uint8)
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(class_id)): int(count) for class_id, count in zip(unique, counts)}


def sidecar_common_payload(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        raise RuntimeError(f"Missing sidecar: {sidecar_path}")
    payload = read_json(sidecar_path)
    payload["_sidecar_path"] = str(sidecar_path)
    return payload


def require_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(f"Invalid integer field {label}: {value!r}") from exc


def validate_uav_modality_outputs(
    *,
    out_dir: Path,
    modality: str,
    ticks: list[int],
    episode_id: str,
    entity_id: str,
    capture_view_id: str,
    verify_seg_pixels: bool,
) -> dict[int, dict[str, Any]]:
    ext = MODALITY_EXT[modality]
    result: dict[int, dict[str, Any]] = {}
    for tick in ticks:
        primary_path = out_dir / modality / f"tick_{int(tick):06d}.{ext}"
        if not primary_path.exists():
            raise RuntimeError(f"Missing UAV {modality} output: {primary_path}")
        sidecar = sidecar_common_payload(primary_path)
        if str(sidecar.get("episode_id") or "") != episode_id:
            raise RuntimeError(f"UAV {modality} sidecar episode mismatch at tick {tick}: {sidecar.get('episode_id')}")
        if require_int(sidecar.get("tick"), f"{modality}.tick") != int(tick):
            raise RuntimeError(f"UAV {modality} sidecar tick mismatch at tick {tick}: {sidecar.get('tick')}")
        if str(sidecar.get("capture_view_id") or "") != capture_view_id:
            raise RuntimeError(f"UAV {modality} sidecar view mismatch at tick {tick}: {sidecar.get('capture_view_id')}")
        source_entity = str(sidecar.get("source_uav_entity_id", sidecar.get("uav_entity_id", "")) or "")
        if source_entity != entity_id:
            raise RuntimeError(f"UAV {modality} sidecar entity mismatch at tick {tick}: {source_entity}")
        if str(sidecar.get("logical_sample_id") or "") != f"{episode_id}:tick{int(tick):06d}:{capture_view_id}":
            raise RuntimeError(f"UAV {modality} logical_sample_id mismatch at tick {tick}: {sidecar.get('logical_sample_id')}")
        payload = {
            "path": str(primary_path),
            "sidecar": str(sidecar.get("_sidecar_path") or ""),
            "alignment_key": sidecar.get("capture_alignment_key", ""),
            "alignment_source": sidecar.get("capture_alignment_source", ""),
            "logical_sample_id": sidecar.get("logical_sample_id", ""),
            "episode_id": sidecar.get("episode_id", ""),
            "batch_id": sidecar.get("batch_id", ""),
            "frame_id": sidecar.get("frame_id", ""),
            "frame_seq": require_int(sidecar.get("frame_seq", sidecar.get("tick")), f"{modality}.frame_seq"),
            "tick": require_int(sidecar.get("tick"), f"{modality}.tick"),
            "capture_view_id": sidecar.get("capture_view_id", ""),
            "source_uav_entity_id": source_entity,
            "output_format": sidecar.get("output_format", ""),
            "width": require_int(sidecar.get("width"), f"{modality}.width"),
            "height": require_int(sidecar.get("height"), f"{modality}.height"),
            "fov_degrees": float(sidecar.get("fov_degrees") or 0.0),
            "camera_name": str(sidecar.get("camera_name") or ""),
        }
        if not payload["alignment_key"]:
            raise RuntimeError(f"UAV {modality} missing capture_alignment_key at tick {tick}")
        if not payload["alignment_source"]:
            raise RuntimeError(f"UAV {modality} missing capture_alignment_source at tick {tick}")
        if payload["width"] <= 0 or payload["height"] <= 0:
            raise RuntimeError(f"UAV {modality} invalid sidecar dimensions at tick {tick}: {payload}")
        if payload["fov_degrees"] <= 0.0:
            raise RuntimeError(f"UAV {modality} invalid fov_degrees at tick {tick}: {payload['fov_degrees']}")
        if modality in {"rgb", "seg"}:
            from PIL import Image

            with Image.open(primary_path) as image:
                if tuple(image.size) != (payload["width"], payload["height"]):
                    raise RuntimeError(
                        f"UAV {modality} image size mismatch at {primary_path}: "
                        f"image={image.size} sidecar={(payload['width'], payload['height'])}"
                    )
        if modality == "depth":
            import numpy as np

            depth = np.load(primary_path, mmap_mode="r")
            if tuple(depth.shape[:2]) != (payload["height"], payload["width"]):
                raise RuntimeError(
                    f"UAV depth shape mismatch at {primary_path}: "
                    f"npy={tuple(depth.shape)} sidecar={(payload['height'], payload['width'])}"
                )
            payload["depth_shape"] = [int(value) for value in depth.shape]
            depth_preview_path = str(sidecar.get("depth_preview_path") or "").strip()
            if depth_preview_path:
                preview_path = Path(depth_preview_path)
                if not preview_path.exists():
                    raise RuntimeError(f"Missing depth preview output: {preview_path}")
                payload["depth_preview_path"] = str(preview_path)
        if modality == "seg":
            palette_path = Path(str(sidecar.get("semantic_palette_preview_path") or primary_path.with_name(f"{primary_path.stem}__palette.png")))
            if not palette_path.exists():
                raise RuntimeError(f"Missing semantic palette preview: {palette_path}")
            payload["palette"] = str(palette_path)
            sidecar_hist = normalize_histogram(dict(sidecar.get("class_histogram") or {}))
            if verify_seg_pixels:
                actual_hist = semantic_histogram(primary_path)
                if sidecar_hist and sidecar_hist != actual_hist:
                    raise RuntimeError(
                        f"Seg histogram mismatch at {primary_path}: sidecar={sidecar_hist} actual={actual_hist}"
                    )
                payload["histogram"] = actual_hist
            else:
                payload["histogram"] = sidecar_hist
        result[int(tick)] = payload
    return result


def validate_uav_event_chain_outputs(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    ticks: list[int],
    episode_id: str,
    entity_id: str,
    capture_view_id: str,
) -> dict[int, dict[str, Any]]:
    modality_payloads = {
        modality: validate_uav_modality_outputs(
            out_dir=out_dir,
            modality=modality,
            ticks=ticks,
            episode_id=episode_id,
            entity_id=entity_id,
            capture_view_id=capture_view_id,
            verify_seg_pixels=bool(args.verify_seg_pixels),
        )
        for modality in args.uav_modalities
    }
    rows: dict[int, dict[str, Any]] = {}
    for tick in ticks:
        outputs = {modality: modality_payloads[modality][int(tick)] for modality in args.uav_modalities}
        for field_name in (
            "alignment_key",
            "alignment_source",
            "logical_sample_id",
            "episode_id",
            "batch_id",
            "frame_id",
            "frame_seq",
            "tick",
            "capture_view_id",
            "source_uav_entity_id",
        ):
            values = {str(payload.get(field_name) or "") for payload in outputs.values()}
            if len(values) != 1:
                raise RuntimeError(f"UAV modality {field_name} mismatch at tick {tick}: {sorted(values)}")
        for field_name in ("width", "height", "fov_degrees", "camera_name"):
            values = {str(payload.get(field_name) or "") for payload in outputs.values()}
            if len(values) != 1:
                raise RuntimeError(f"UAV modality camera geometry {field_name} mismatch at tick {tick}: {sorted(values)}")
        depth_shape = outputs.get("depth", {}).get("depth_shape")
        if isinstance(depth_shape, list) and len(depth_shape) >= 2:
            expected_shape = [int(outputs["rgb"]["height"]), int(outputs["rgb"]["width"])]
            if [int(depth_shape[0]), int(depth_shape[1])] != expected_shape:
                raise RuntimeError(f"UAV depth shape does not match RGB dimensions at tick {tick}: {depth_shape} vs {expected_shape}")
        rows[int(tick)] = {
            "modality_outputs": outputs,
            "rgb_path": str(outputs.get("rgb", {}).get("path") or ""),
            "depth_path": str(outputs.get("depth", {}).get("path") or ""),
            "seg_path": str(outputs.get("seg", {}).get("path") or ""),
            "seg_palette_path": str(outputs.get("seg", {}).get("palette") or ""),
            "alignment_key": str(outputs["rgb"].get("alignment_key") or ""),
            "alignment_source": str(outputs["rgb"].get("alignment_source") or ""),
            "logical_sample_id": str(outputs["rgb"].get("logical_sample_id") or ""),
            "batch_id": str(outputs["rgb"].get("batch_id") or ""),
            "frame_id": str(outputs["rgb"].get("frame_id") or ""),
            "frame_seq": int(outputs["rgb"].get("frame_seq") or 0),
        }
    return rows


def high_rgb_dir(out_dir: Path, site_id: str, camera_id: str) -> Path:
    return out_dir / safe_name(site_id) / safe_name(camera_id) / "rgb"


def validate_high_outputs(
    *,
    out_dir: Path,
    ticks: list[int],
    episode_id: str,
    site_id: str,
    camera_id: str,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    primary_dir = high_rgb_dir(out_dir, site_id, camera_id)
    for tick in ticks:
        path = primary_dir / f"tick_{int(tick):06d}.png"
        if not path.exists():
            matches = [
                candidate
                for candidate in sorted(out_dir.rglob(f"rgb/tick_{int(tick):06d}.png"))
                if "__" not in candidate.name
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one high RGB output for tick {tick} under {out_dir}, found {len(matches)}")
            path = matches[0]
        sidecar = sidecar_common_payload(path)
        if str(sidecar.get("episode_id") or "") != episode_id:
            raise RuntimeError(f"High RGB sidecar episode mismatch at tick {tick}: {sidecar.get('episode_id')}")
        if require_int(sidecar.get("tick"), "high.tick") != int(tick):
            raise RuntimeError(f"High RGB sidecar tick mismatch at tick {tick}: {sidecar.get('tick')}")
        if str(sidecar.get("modality") or "") != "rgb":
            raise RuntimeError(f"High overview must be RGB only, got modality={sidecar.get('modality')}")
        from PIL import Image

        width = require_int(sidecar.get("width"), "high.width")
        height = require_int(sidecar.get("height"), "high.height")
        if width <= 0 or height <= 0:
            raise RuntimeError(f"High overview invalid sidecar dimensions at tick {tick}: {width}x{height}")
        with Image.open(path) as image:
            if tuple(image.size) != (width, height):
                raise RuntimeError(f"High overview image size mismatch at {path}: image={image.size} sidecar={(width, height)}")
        rows[int(tick)] = {
            "rgb_path": str(path),
            "sidecar": str(sidecar.get("_sidecar_path") or ""),
            "batch_id": str(sidecar.get("batch_id") or ""),
            "frame_id": str(sidecar.get("frame_id") or ""),
            "frame_seq": int(sidecar.get("frame_seq") or sidecar.get("tick") or 0),
        }
    return rows


def current_memory_row(args: argparse.Namespace) -> dict[str, float]:
    try:
        memory = assert_runtime_available(args)
        return {
            "working_set_gb": memory["working_set_gb"],
            "private_memory_gb": memory["private_memory_gb"],
        }
    except Exception:
        memory = unreal_memory_gb()
        return {
            "working_set_gb": memory["working_set_gb"],
            "private_memory_gb": memory["private_memory_gb"],
        }


def summary_base(
    args: argparse.Namespace,
    contract: dict[str, Any],
    plan: EpisodePlan,
    *,
    view: str,
    tick: int | str = "",
    camera_id: str = "",
    capture_entity_id: str = "",
    capture_view_id: str = "",
    output_dir: Path | str = "",
) -> dict[str, Any]:
    memory = current_memory_row(args)
    return {
        "index": plan.index,
        "episode": plan.episode_id,
        "view": view,
        "status": "pending",
        "tick": tick,
        "tick_start": args.tick_start,
        "tick_end": args.tick_end,
        "tick_step": args.tick_step,
        "capture_tick_count": len(plan.capture_ticks),
        "capture_tick_index": "",
        "camera_id": camera_id,
        "capture_entity_id": capture_entity_id,
        "capture_view_id": capture_view_id,
        "logical_event_id": plan.episode_id,
        "logical_sample_id": "",
        "batch_id": "",
        "frame_id": "",
        "frame_seq": "",
        "runtime_uav_count": len(plan.uav_active_ticks),
        "active_uav_tick_count": len(plan.uav_active_ticks.get(capture_entity_id, [])) if capture_entity_id else "",
        "uav_control_backend": args.uav_control_backend,
        "contract_path": str(args.contract),
        "contract_version": str(contract.get("contract_version") or ""),
        "rgb_path": "",
        "depth_path": "",
        "seg_path": "",
        "seg_palette_path": "",
        "modality_outputs": "{}",
        "alignment_key": "",
        "alignment_source": "",
        "output_dir": str(output_dir),
        "host_logs": "[]",
        "working_set_gb": memory["working_set_gb"],
        "private_memory_gb": memory["private_memory_gb"],
        "child_working_set_gb_peak": "",
        "child_private_memory_gb_peak": "",
        "system_free_memory_gb_min": "",
        "error": "",
    }


def apply_host_result(row: dict[str, Any], result: HostRunResult) -> None:
    row["host_logs"] = json_cell([str(path) for path in result.log_paths])
    row["child_working_set_gb_peak"] = round(result.stats.child_working_set_gb_peak, 3)
    row["child_private_memory_gb_peak"] = round(result.stats.child_private_memory_gb_peak, 3)
    row["system_free_memory_gb_min"] = round(result.stats.system_free_memory_gb_min, 3)


def write_high_rows(
    args: argparse.Namespace,
    writer: csv.DictWriter,
    handle: Any,
    contract: dict[str, Any],
    plan: EpisodePlan,
    result: HostRunResult,
    validations: dict[int, dict[str, Any]],
    out_dir: Path,
) -> None:
    tick_index = {tick: index for index, tick in enumerate(plan.capture_ticks)}
    for tick in plan.capture_ticks:
        validation = validations[int(tick)]
        row = summary_base(
            args,
            contract,
            plan,
            view="high_overview_rgb",
            tick=tick,
            camera_id=plan.high_camera_id,
            output_dir=out_dir,
        )
        row["status"] = "ok"
        row["capture_tick_index"] = tick_index[int(tick)]
        row["rgb_path"] = validation["rgb_path"]
        row["modality_outputs"] = json_cell({"rgb": validation})
        row["batch_id"] = validation.get("batch_id", "")
        row["frame_id"] = validation.get("frame_id", "")
        row["frame_seq"] = validation.get("frame_seq", "")
        apply_host_result(row, result)
        writer.writerow(row)
    handle.flush()


def write_uav_rows(
    args: argparse.Namespace,
    writer: csv.DictWriter,
    handle: Any,
    contract: dict[str, Any],
    plan: EpisodePlan,
    *,
    entity_id: str,
    capture_view_id: str,
    camera_id: str,
    result: HostRunResult,
    validations: dict[int, dict[str, Any]],
    out_dir: Path,
) -> None:
    active_ticks = plan.uav_active_ticks[entity_id]
    tick_index = {tick: index for index, tick in enumerate(active_ticks)}
    for tick in active_ticks:
        validation = validations[int(tick)]
        row = summary_base(
            args,
            contract,
            plan,
            view="uav_event_chain",
            tick=tick,
            camera_id=camera_id,
            capture_entity_id=entity_id,
            capture_view_id=capture_view_id,
            output_dir=out_dir,
        )
        row["status"] = "ok"
        row["capture_tick_index"] = tick_index[int(tick)]
        row["logical_sample_id"] = validation["logical_sample_id"]
        row["batch_id"] = validation["batch_id"]
        row["frame_id"] = validation["frame_id"]
        row["frame_seq"] = validation["frame_seq"]
        row["rgb_path"] = validation["rgb_path"]
        row["depth_path"] = validation["depth_path"]
        row["seg_path"] = validation["seg_path"]
        row["seg_palette_path"] = validation["seg_palette_path"]
        row["modality_outputs"] = json_cell(validation["modality_outputs"])
        row["alignment_key"] = validation["alignment_key"]
        row["alignment_source"] = validation["alignment_source"]
        apply_host_result(row, result)
        writer.writerow(row)
    handle.flush()


def write_failure_row(
    args: argparse.Namespace,
    writer: csv.DictWriter,
    handle: Any,
    contract: dict[str, Any],
    plan: EpisodePlan,
    *,
    view: str,
    error: Exception,
    output_dir: Path | str = "",
    camera_id: str = "",
    capture_entity_id: str = "",
    capture_view_id: str = "",
    result: HostRunResult | None = None,
) -> None:
    row = summary_base(
        args,
        contract,
        plan,
        view=view,
        camera_id=camera_id,
        capture_entity_id=capture_entity_id,
        capture_view_id=capture_view_id,
        output_dir=output_dir,
    )
    row["status"] = "failed"
    row["error"] = str(error)
    if result is not None:
        apply_host_result(row, result)
    writer.writerow(row)
    handle.flush()


def run_high_overview(
    args: argparse.Namespace,
    writer: csv.DictWriter,
    handle: Any,
    contract: dict[str, Any],
    plan: EpisodePlan,
    guarded_config: Path,
) -> None:
    out_dir = event_chain_output_dir(args.output_root, plan.index, "high_overview_rgb")
    clear_output_targets([high_rgb_dir(out_dir, plan.site_id, plan.high_camera_id)], args.output_root)
    base_command = base_host_command(args, guarded_config, out_dir, plan.site_id)
    base_command.extend(["--camera-role", "ground", "--camera-id", plan.high_camera_id, "--modality", "rgb"])
    log_dir = args.output_root / "_meta" / "logs" / simple_episode_dir_name(plan.index)
    result = run_host_chunks(
        args,
        base_command=base_command,
        capture_ticks=plan.capture_ticks,
        log_dir=log_dir,
        log_stem="high_rgb",
        label=f"high RGB {plan.episode_id}",
    )
    validations = validate_high_outputs(
        out_dir=out_dir,
        ticks=plan.capture_ticks,
        episode_id=plan.episode_id,
        site_id=plan.site_id,
        camera_id=plan.high_camera_id,
    )
    write_high_rows(args, writer, handle, contract, plan, result, validations, out_dir)


def run_uav_entity(
    args: argparse.Namespace,
    writer: csv.DictWriter,
    handle: Any,
    contract: dict[str, Any],
    plan: EpisodePlan,
    guarded_config: Path,
    *,
    entity_id: str,
    ordinal: int,
) -> None:
    capture_view_id = f"uav_view_{int(ordinal):03d}__{safe_name(entity_id)}"
    camera_id = f"{safe_name(entity_id)}__nadir_down"
    out_dir = event_chain_uav_output_dir(args.output_root, plan.index, capture_view_id)
    active_ticks = plan.uav_active_ticks.get(entity_id, [])
    if not active_ticks:
        row = summary_base(
            args,
            contract,
            plan,
            view="uav_event_chain",
            camera_id=camera_id,
            capture_entity_id=entity_id,
            capture_view_id=capture_view_id,
            output_dir=out_dir,
        )
        row["status"] = "skipped_no_active_capture_ticks"
        row["error"] = "UAV is not active at any selected event-chain capture tick."
        writer.writerow(row)
        handle.flush()
        return
    clear_output_targets([out_dir / modality for modality in args.uav_modalities], args.output_root)
    combined_result = HostRunResult()
    log_dir = args.output_root / "_meta" / "logs" / simple_episode_dir_name(plan.index)
    for modality in args.uav_modalities:
        base_command = base_host_command(args, guarded_config, out_dir, plan.site_id)
        base_command.extend(
            [
                "--camera-role",
                "uav",
                "--camera-id",
                camera_id,
                "--modality",
                modality,
                "--runtime-uav-control-backend",
                args.uav_control_backend,
                "--airsim-capture-entity",
                entity_id,
                "--capture-view-id",
                capture_view_id,
                "--airsim-capture-vehicle",
                args.airsim_capture_vehicle,
            ]
        )
        result = run_host_chunks(
            args,
            base_command=base_command,
            capture_ticks=active_ticks,
            log_dir=log_dir,
            log_stem=f"{simple_capture_view_dir_name(capture_view_id)}_{modality}",
            label=f"uav {plan.episode_id} {capture_view_id} {modality}",
        )
        combined_result.log_paths.extend(result.log_paths)
        combined_result.stats.absorb(result.stats)
    validations = validate_uav_event_chain_outputs(
        args=args,
        out_dir=out_dir,
        ticks=active_ticks,
        episode_id=plan.episode_id,
        entity_id=entity_id,
        capture_view_id=capture_view_id,
    )
    write_uav_rows(
        args,
        writer,
        handle,
        contract,
        plan,
        entity_id=entity_id,
        capture_view_id=capture_view_id,
        camera_id=camera_id,
        result=combined_result,
        validations=validations,
        out_dir=out_dir,
    )


def validate_contract(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    defaults = dict(contract.get("defaults") or {})
    must_follow = dict(contract.get("must_follow") or {})
    if str(args.segmentation_backend) != "ue_custom_stencil":
        raise RuntimeError("Formal event-chain runner must use UE CustomStencil segmentation.")
    if list(args.uav_modalities) != list(UAV_MODALITIES):
        raise RuntimeError(f"UAV modalities must be rgb/depth/seg, got {args.uav_modalities}")
    if looks_timestamped_or_versioned(args.output_root):
        raise RuntimeError(f"Output root looks timestamped/versioned, which is forbidden: {args.output_root}")
    if bool(must_follow.get("output_root_must_be_f_drive_root", True)):
        if not args.output_root.is_absolute() or not is_f_drive_path(args.output_root):
            raise RuntimeError(f"Output root must be an absolute F: drive path, got: {args.output_root}")
        if not args.summary.is_absolute() or not is_f_drive_path(args.summary):
            raise RuntimeError(f"Summary path must be an absolute F: drive path, got: {args.summary}")
        expected_output_root = Path(str(defaults.get("output_root") or DEFAULT_OUTPUT_ROOT))
        expected_summary = Path(str(defaults.get("summary") or DEFAULT_SUMMARY_PATH))
        if normalized_abs_path_text(args.output_root) != normalized_abs_path_text(expected_output_root):
            raise RuntimeError(f"Formal event-chain output root must be exactly {expected_output_root}, got: {args.output_root}")
        if normalized_abs_path_text(args.summary) != normalized_abs_path_text(expected_summary):
            raise RuntimeError(f"Formal event-chain summary must be exactly {expected_summary}, got: {args.summary}")
    if args.tick_start != 0:
        raise RuntimeError("Event-chain capture must keep --tick-start 0 to avoid direct tick jumps.")
    if args.tick_step != 10 and not args.allow_nonstandard_tick_step:
        raise RuntimeError("Formal event-chain capture is every 10 ticks; pass --allow-nonstandard-tick-step only for debug.")
    if float(args.max_private_memory_gb) > float(defaults.get("max_private_memory_gb", 20.0)):
        raise RuntimeError("Private memory guard cannot exceed the contract default.")
    if float(args.max_working_set_gb) > float(defaults.get("max_working_set_gb", 20.0)):
        raise RuntimeError("Working-set memory guard cannot exceed the contract default.")
    if float(args.host_run_timeout_s) > float(defaults.get("host_run_timeout_s", 300.0)):
        raise RuntimeError("Host run timeout cannot exceed the contract default; reduce --capture-ticks-per-host-run instead.")
    if int(args.capture_ticks_per_host_run) < 0:
        raise RuntimeError("--capture-ticks-per-host-run must be positive, or 0 with --allow-single-host-full-chain.")
    if int(args.capture_ticks_per_host_run) == 0 and not args.allow_single_host_full_chain:
        raise RuntimeError(
            "Full-chain single host runs are disabled by default for OOM safety; "
            "use a positive --capture-ticks-per-host-run or pass --allow-single-host-full-chain."
        )


def print_plan(plans: list[EpisodePlan], args: argparse.Namespace) -> None:
    total_high = 0
    total_uav = 0
    total_host_runs = 0
    for plan in plans:
        high_files = 0 if args.skip_high_overview else len(plan.capture_ticks)
        uav_files = 0 if args.skip_uav else sum(len(ticks) * len(args.uav_modalities) for ticks in plan.uav_active_ticks.values())
        total_high += high_files
        total_uav += uav_files
        chunk_count_high = 0 if args.skip_high_overview else len(chunk_ticks(plan.capture_ticks, args.capture_ticks_per_host_run))
        chunk_count_uav = 0
        if not args.skip_uav:
            for ticks in plan.uav_active_ticks.values():
                chunk_count_uav += len(args.uav_modalities) * len(chunk_ticks(ticks, args.capture_ticks_per_host_run))
        total_host_runs += chunk_count_high + chunk_count_uav
        print(
            f"{simple_episode_dir_name(plan.index)} {plan.episode_id} "
            f"site={plan.site_id} ticks={len(plan.capture_ticks)} high_files={high_files} "
            f"uav_entities={len(plan.uav_active_ticks)} uav_files={uav_files} "
            f"host_runs={chunk_count_high + chunk_count_uav}",
            flush=True,
        )
    print("[EventChainPlan]", flush=True)
    print(f"  episodes={len(plans)}", flush=True)
    print(f"  tick_start={args.tick_start} tick_end={args.tick_end} tick_step={args.tick_step}", flush=True)
    print(f"  capture_ticks_per_host_run={args.capture_ticks_per_host_run}", flush=True)
    print(f"  host_run_timeout_s={args.host_run_timeout_s}", flush=True)
    print(f"  primary_high_rgb_files={total_high}", flush=True)
    print(f"  primary_uav_files={total_uav}", flush=True)
    print(f"  primary_data_files_total={total_high + total_uav}", flush=True)
    print(f"  estimated_host_processes={total_host_runs}", flush=True)
    print(f"  output_root={args.output_root}", flush=True)
    print(f"  summary={args.summary}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--capture-presets", type=Path, default=DEFAULT_CAPTURE_PRESETS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--tick-start", type=int, default=0)
    parser.add_argument("--tick-end", type=int, default=900)
    parser.add_argument("--tick-step", type=int, default=10)
    parser.add_argument("--capture-ticks-per-host-run", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--editor-hook-capture-timeout-s", type=float, default=90.0)
    parser.add_argument("--host-run-timeout-s", type=float, default=900.0)
    parser.add_argument("--host-guard-poll-s", type=float, default=2.0)
    parser.add_argument("--max-working-set-gb", type=float, default=20.0)
    parser.add_argument("--max-private-memory-gb", type=float, default=20.0)
    parser.add_argument("--max-child-working-set-gb", type=float, default=8.0)
    parser.add_argument("--max-child-private-memory-gb", type=float, default=8.0)
    parser.add_argument("--min-system-free-memory-gb", type=float, default=4.0)
    parser.add_argument("--uav-modalities", nargs="+", default=list(UAV_MODALITIES))
    parser.add_argument("--segmentation-backend", default="ue_custom_stencil")
    parser.add_argument("--uav-control-backend", choices=["pose_sync", "airsim_move"], default="pose_sync")
    parser.add_argument("--airsim-capture-vehicle", default="CaptureUAV_0")
    parser.add_argument("--airsim-capture-entity", default="", help="Optional explicit UAV entity for targeted capture.")
    parser.add_argument("--skip-high-overview", action="store_true")
    parser.add_argument("--skip-uav", action="store_true")
    parser.add_argument("--append-summary", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-missing-capture-ticks", action="store_true")
    parser.add_argument("--allow-nonstandard-tick-step", action="store_true")
    parser.add_argument("--allow-single-host-full-chain", action="store_true")
    parser.add_argument("--write-depth-preview", dest="write_depth_preview", action="store_true", default=True)
    parser.add_argument("--no-write-depth-preview", dest="write_depth_preview", action="store_false")
    parser.add_argument("--no-verify-seg-pixels", dest="verify_seg_pixels", action="store_false", default=True)
    parser.add_argument("--plan-only", action="store_true", help="Print deterministic plan and exit without touching UE.")
    args = parser.parse_args()
    args.uav_modalities = [str(value).strip().lower() for value in args.uav_modalities if str(value).strip()]
    return args


def main() -> int:
    args = parse_args()
    contract = read_json(args.contract)
    validate_contract(args, contract)
    preset_payload = filter_event_chain_capture_presets(args.capture_presets)
    virtual_presets_path = args.output_root / "_meta" / "configs" / "event_chain_capture_presets.json"
    if not args.plan_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        virtual_presets_path = write_event_chain_capture_presets(args)

    plans = [
        build_episode_plan(args, episode_dir, index, preset_payload)
        for index, episode_dir in selected_episode_dirs(args)
    ]
    print_plan(plans, args)
    if args.plan_only:
        return 0

    assert_runtime_available(args)
    write_header = not args.append_summary or not args.summary.exists() or args.summary.stat().st_size == 0
    if args.append_summary and not write_header:
        existing_header = args.summary.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0].split(",")
        if existing_header != SUMMARY_FIELDS:
            raise RuntimeError(f"Existing summary header does not match event-chain schema: {args.summary}")

    mode = "a" if args.append_summary else "w"
    with args.summary.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        for ordinal_plan, plan in enumerate(plans, start=1):
            guarded_config = write_guarded_config(args, plan.episode_dir, plan.index, virtual_presets_path)
            print(
                f"[{ordinal_plan}/{len(plans)}] {simple_episode_dir_name(plan.index)} {plan.episode_id} "
                f"ticks={len(plan.capture_ticks)} uavs={len(plan.uav_active_ticks)}",
                flush=True,
            )
            if not args.skip_high_overview:
                try:
                    run_high_overview(args, writer, handle, contract, plan, guarded_config)
                    print("  high_overview_rgb: ok", flush=True)
                except Exception as exc:
                    write_failure_row(
                        args,
                        writer,
                        handle,
                        contract,
                        plan,
                        view="high_overview_rgb",
                        error=exc,
                        output_dir=event_chain_output_dir(args.output_root, plan.index, "high_overview_rgb"),
                        camera_id=plan.high_camera_id,
                    )
                    print(f"  high_overview_rgb: failed err={str(exc)[:180]}", flush=True)
                    if not args.continue_on_error:
                        raise
            if args.skip_uav:
                continue
            if not plan.uav_active_ticks:
                row = summary_base(args, contract, plan, view="uav_event_chain")
                row["status"] = "skipped_no_runtime_uav"
                row["error"] = "No active runtime UAV at selected event-chain capture ticks."
                writer.writerow(row)
                handle.flush()
                print("  uav_event_chain: skipped_no_runtime_uav", flush=True)
                continue
            for uav_ordinal, entity_id in enumerate(sorted(plan.uav_active_ticks.keys())):
                try:
                    run_uav_entity(
                        args,
                        writer,
                        handle,
                        contract,
                        plan,
                        guarded_config,
                        entity_id=entity_id,
                        ordinal=uav_ordinal,
                    )
                    print(f"  uav_event_chain: ok entity={entity_id}", flush=True)
                except Exception as exc:
                    capture_view_id = f"uav_view_{int(uav_ordinal):03d}__{safe_name(entity_id)}"
                    write_failure_row(
                        args,
                        writer,
                        handle,
                        contract,
                        plan,
                        view="uav_event_chain",
                        error=exc,
                        output_dir=event_chain_uav_output_dir(args.output_root, plan.index, capture_view_id),
                        camera_id=f"{safe_name(entity_id)}__nadir_down",
                        capture_entity_id=entity_id,
                        capture_view_id=capture_view_id,
                    )
                    print(f"  uav_event_chain: failed entity={entity_id} err={str(exc)[:180]}", flush=True)
                    if not args.continue_on_error:
                        raise
    print(f"SUMMARY={args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
