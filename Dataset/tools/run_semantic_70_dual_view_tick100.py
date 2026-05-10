from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMO_SCRIPTS = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scripts"
if str(SUMO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SUMO_SCRIPTS))

from editor_hook_client import FixedWorldCaptureEditorHook  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("F:/aw_cap")
DEFAULT_SUMMARY_PATH = Path("F:/aw_cap_summary.csv")

FATAL_RUNTIME_PATTERNS = (
    "Retry connection over the limit",
    "Client is closed, connection could not be set",
    "Unable to discover a matching Unreal Editor remote execution node",
    "AirSim RPC is unavailable",
)


class FatalRuntimeUnavailable(RuntimeError):
    pass


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def short_stable_name(value: str, prefix: str) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}{digest}"


def simple_capture_view_dir_name(capture_view_id: str) -> str:
    match = re.search(r"uav_view_(\d+)", str(capture_view_id))
    if match:
        return f"v{int(match.group(1)):03d}"
    return short_stable_name(str(capture_view_id), "v")


def simple_episode_dir_name(index: int) -> str:
    return f"e{int(index):02d}"


def is_f_drive_path(path: Path) -> bool:
    return str(path.drive).lower() == "f:"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_int(value: Any, label: str) -> int:
    if value is None:
        raise RuntimeError(f"Missing integer field: {label}")
    if isinstance(value, str) and not value.strip():
        raise RuntimeError(f"Missing integer field: {label}")
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(f"Invalid integer field {label}: {value!r}") from exc


def rpc_available(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def unreal_memory_gb() -> dict[str, float]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-Process UnrealEditor -ErrorAction SilentlyContinue | "
            "Select-Object -First 1 Id,"
            "@{Name='WorkingSetGB';Expression={[math]::Round($_.WorkingSet64/1GB,3)}},"
            "@{Name='PrivateMemoryGB';Expression={[math]::Round($_.PrivateMemorySize64/1GB,3)}} | "
            "ConvertTo-Json -Compress"
        ),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = result.stdout.strip()
    if not text:
        return {"pid": 0.0, "working_set_gb": 0.0, "private_memory_gb": 0.0}
    payload = json.loads(text)
    return {
        "pid": float(payload.get("Id") or 0),
        "working_set_gb": float(payload.get("WorkingSetGB") or 0.0),
        "private_memory_gb": float(payload.get("PrivateMemoryGB") or 0.0),
    }


def looks_like_runtime_unavailable(message: str) -> bool:
    return any(pattern in message for pattern in FATAL_RUNTIME_PATTERNS)


def assert_runtime_available(args: argparse.Namespace) -> dict[str, float]:
    memory = unreal_memory_gb()
    if memory["pid"] <= 0:
        raise FatalRuntimeUnavailable("UnrealEditor process is unavailable.")
    assert_memory_under_limit(args, memory)
    if not rpc_available(args.host, args.port):
        raise FatalRuntimeUnavailable(f"AirSim RPC is unavailable at {args.host}:{args.port}")
    return memory


def assert_memory_under_limit(args: argparse.Namespace, memory: dict[str, float]) -> None:
    if memory["working_set_gb"] > args.max_working_set_gb:
        raise FatalRuntimeUnavailable(f"Working set memory guard tripped: {memory}")
    if memory["private_memory_gb"] > args.max_private_memory_gb:
        raise FatalRuntimeUnavailable(f"Private memory guard tripped: {memory}")


def run_child_with_guards(
    command: list[str],
    *,
    args: argparse.Namespace,
    log_path: Path,
    label: str,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1.0, float(args.host_run_timeout_s))
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while True:
            return_code = process.poll()
            if return_code is not None:
                return int(return_code)
            if time.monotonic() >= deadline:
                process.kill()
                log.write(f"\n[RunnerGuard] {label} timed out after {args.host_run_timeout_s:.1f}s.\n")
                raise RuntimeError(f"{label} timed out after {args.host_run_timeout_s:.1f}s; log={log_path}")
            memory = unreal_memory_gb()
            try:
                assert_memory_under_limit(args, memory)
            except Exception:
                process.kill()
                log.write(f"\n[RunnerGuard] {label} stopped by memory guard: {memory}\n")
                raise
            time.sleep(max(0.5, float(args.host_guard_poll_s)))


def write_palette_preview(image: Image.Image, output_path: Path) -> None:
    values = np.asarray(image.convert("L"), dtype=np.uint8)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    palette[1] = [140, 140, 140]
    palette[2] = [40, 120, 255]
    palette[3] = [30, 180, 80]
    palette[4] = [0, 200, 220]
    palette[5] = [230, 60, 50]
    palette[6] = [220, 60, 220]
    palette[7] = [255, 220, 40]
    palette[8] = [255, 140, 30]
    palette[9] = [150, 80, 255]
    palette[10] = [245, 245, 245]
    palette[11] = [255, 120, 170]
    palette[12] = [80, 255, 210]
    Image.fromarray(palette[values], mode="RGB").save(output_path)


def semantic_histogram(png_path: Path) -> dict[str, Any]:
    image = Image.open(png_path)
    values = np.asarray(image.convert("L"), dtype=np.uint8)
    unique, counts = np.unique(values, return_counts=True)
    histogram = {str(int(class_id)): int(count) for class_id, count in zip(unique, counts)}
    return {
        "semantic_png_mode": str(image.mode),
        "class_histogram": histogram,
        "ignore_pixel_count": int(histogram.get("0", 0)),
        "non_ignore_pixel_count": int(values.size - int(histogram.get("0", 0))),
        "semantic_unique_class_ids": [int(value) for value in unique.tolist()],
    }


def runtime_uav_entity_ids_at_tick(episode_dir: Path, tick: int) -> list[str]:
    roster = read_json(episode_dir / "global_entity_roster.json")
    runtime_ids = {
        str(entity.get("entity_id") or "").strip()
        for entity in roster.get("entities") or []
        if str(entity.get("entity_category") or "").lower() == "uav"
        and str(entity.get("mode") or "").lower() in {"runtime_multirotor", ""}
        and str(entity.get("entity_id") or "").strip()
    }
    if not runtime_ids:
        return []
    truth_path = episode_dir / "truth_frames.jsonl"
    with truth_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            frame = json.loads(line)
            if int(frame.get("tick", -1)) != tick:
                continue
            active: list[str] = []
            for entity in frame.get("entities") or []:
                entity_id = str(entity.get("entity_id") or "").strip()
                if entity_id not in runtime_ids:
                    continue
                render_presence = dict(entity.get("render_presence") or {})
                if str(render_presence.get("submission_state") or "").strip() not in {"", "submit_to_ue"}:
                    continue
                if str(render_presence.get("visibility_state") or "").strip() not in {"", "visible"}:
                    continue
                active.append(entity_id)
            return sorted(set(active))
    return []


def first_runtime_uav_entity_id(episode_dir: Path) -> str:
    roster = read_json(episode_dir / "global_entity_roster.json")
    for entity in roster.get("entities") or []:
        if str(entity.get("entity_category") or "").lower() != "uav":
            continue
        if str(entity.get("mode") or "").lower() not in {"runtime_multirotor", ""}:
            continue
        entity_id = str(entity.get("entity_id") or "").strip()
        if entity_id:
            return entity_id
    raise RuntimeError("no_runtime_uav")


def tick_at_or_before(episode_dir: Path, requested_tick: int) -> tuple[int, str]:
    truth_path = episode_dir / "truth_frames.jsonl"
    selected: int | None = None
    with truth_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tick = int(json.loads(line).get("tick", 0))
            if tick <= requested_tick:
                selected = tick
            else:
                break
    if selected is None:
        raise RuntimeError(f"No truth tick <= {requested_tick} in {truth_path}")
    status = "requested_tick" if selected == requested_tick else "nearest_tick_before_requested"
    return selected, status


def altitude_at_tick(episode_dir: Path, entity_id: str, tick: int) -> float:
    truth_path = episode_dir / "truth_frames.jsonl"
    with truth_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            frame = json.loads(line)
            if int(frame.get("tick", -1)) != tick:
                continue
            for entity in frame.get("entities") or []:
                if str(entity.get("entity_id") or "") != entity_id:
                    continue
                position = entity.get("truth_pose", {}).get("position_enu_m")
                if isinstance(position, list) and len(position) >= 3:
                    return float(position[2])
            break
    return 0.0


def config_with_capture_timeout(
    source_config: Path,
    output_root: Path,
    timeout_s: float,
    *,
    uav_control_backend: str,
) -> Path:
    config = read_json(source_config)
    timeouts = dict(config.get("timeouts") or {})
    if float(timeouts.get("editor_hook_capture_timeout_s", 0.0) or 0.0) < timeout_s:
        timeouts["editor_hook_capture_timeout_s"] = float(timeout_s)
        config["timeouts"] = timeouts
    runtime_uav = dict(config.get("runtime_uav") or {})
    runtime_uav["control_backend"] = str(uav_control_backend)
    if str(uav_control_backend) == "pose_sync":
        runtime_uav["editor_hook_fallback_enabled"] = False
        runtime_uav["non_capture_rpc_failure_nonfatal"] = True
    config["runtime_uav"] = runtime_uav
    config["capture_path_contract"] = {
        "output_root_must_be_on_f_drive": True,
        "simple_primary_paths": True,
        "complex_identifiers_live_in_sidecars": True,
        "source_config_path": str(source_config),
        "source_episode_dir": source_config.parent.name,
    }
    target = output_root / "_meta" / "configs" / f"{short_stable_name(source_config.parent.name, 'e')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def output_dir(output_root: Path, index: int, episode: str, view: str) -> Path:
    del episode
    normalized_view = str(view or "").strip().lower()
    if normalized_view.startswith("high"):
        return output_root / "hi" / simple_episode_dir_name(index)
    return output_root / "uav" / simple_episode_dir_name(index)


def uav_output_dir(output_root: Path, index: int, episode: str, capture_view_id: str) -> Path:
    return output_dir(output_root, index, episode, "uav_tick100") / simple_capture_view_dir_name(capture_view_id)


def logical_sample_id(episode: str, tick: int, capture_view_id: str) -> str:
    return f"{episode}:tick{int(tick):06d}:{capture_view_id}"


def validate_single_uav_modality_output(output_root: Path, modality: str) -> dict[str, Any]:
    modality_root = output_root / "site.intersection_a"
    if not modality_root.exists():
        modality_root = output_root
    modality = str(modality).strip().lower()
    if modality == "depth":
        candidates = sorted(modality_root.rglob("depth/*.npy"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one depth NPY under {output_root}, found {len(candidates)}")
        image_path = candidates[0]
        sidecar_path = image_path.with_suffix(".json")
        if not sidecar_path.exists():
            raise RuntimeError(f"Missing depth sidecar: {sidecar_path}")
        sidecar = read_json(sidecar_path)
        return {
            "path": str(image_path),
            "sidecar": str(sidecar_path),
            "alignment_key": sidecar.get("capture_alignment_key", ""),
            "alignment_source": sidecar.get("capture_alignment_source", ""),
            "logical_sample_id": sidecar.get("logical_sample_id", ""),
            "episode_id": sidecar.get("episode_id", ""),
            "batch_id": sidecar.get("batch_id", ""),
            "frame_id": sidecar.get("frame_id", ""),
            "frame_seq": int(sidecar.get("frame_seq") or sidecar.get("tick") or 0),
            "tick": int(sidecar.get("tick") or 0),
            "capture_view_id": sidecar.get("capture_view_id", ""),
            "source_uav_entity_id": sidecar.get("source_uav_entity_id", sidecar.get("uav_entity_id", "")),
            "output_format": sidecar.get("output_format", ""),
            "depth_shape": sidecar.get("depth_shape", []),
            "depth_min_m": sidecar.get("depth_min_m", 0.0),
            "depth_max_m": sidecar.get("depth_max_m", 0.0),
        }
    if modality == "rgb":
        candidates = [
            path
            for path in sorted(modality_root.rglob("rgb/*.png"))
            if "__palette" not in path.name and "__depth_preview" not in path.name and "__airsim_raw" not in path.name
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one RGB PNG under {output_root}, found {len(candidates)}")
        image_path = candidates[0]
        sidecar_path = image_path.with_suffix(".json")
        if not sidecar_path.exists():
            raise RuntimeError(f"Missing RGB sidecar: {sidecar_path}")
        sidecar = read_json(sidecar_path)
        return {
            "path": str(image_path),
            "sidecar": str(sidecar_path),
            "alignment_key": sidecar.get("capture_alignment_key", ""),
            "alignment_source": sidecar.get("capture_alignment_source", ""),
            "logical_sample_id": sidecar.get("logical_sample_id", ""),
            "episode_id": sidecar.get("episode_id", ""),
            "batch_id": sidecar.get("batch_id", ""),
            "frame_id": sidecar.get("frame_id", ""),
            "frame_seq": int(sidecar.get("frame_seq") or sidecar.get("tick") or 0),
            "tick": int(sidecar.get("tick") or 0),
            "capture_view_id": sidecar.get("capture_view_id", ""),
            "source_uav_entity_id": sidecar.get("source_uav_entity_id", sidecar.get("uav_entity_id", "")),
            "output_format": sidecar.get("output_format", ""),
        }
    if modality != "seg":
        raise RuntimeError(f"Unsupported UAV modality validation: {modality}")
    pngs = [
        path
        for path in sorted(modality_root.rglob("seg/*.png"))
        if "__palette" not in path.name and "__depth_preview" not in path.name and "__airsim_raw" not in path.name
    ]
    if len(pngs) != 1:
        raise RuntimeError(f"Expected one semantic PNG under {output_root}, found {len(pngs)}")
    png_path = pngs[0]
    sidecar_path = png_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise RuntimeError(f"Missing sidecar: {sidecar_path}")
    sidecar = read_json(sidecar_path)
    hist = semantic_histogram(png_path)
    palette_path = png_path.with_name(f"{png_path.stem}__palette.png")
    write_palette_preview(Image.open(png_path), palette_path)
    sidecar["semantic_palette_preview_path"] = str(palette_path)
    sidecar["palette_preview_kind"] = "rgb_visualization_only_not_training_label"
    sidecar["semantic_png_mode"] = hist["semantic_png_mode"]
    sidecar["class_histogram"] = hist["class_histogram"]
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "png": str(png_path),
        "palette": str(palette_path),
        "sidecar": str(sidecar_path),
        "histogram": hist["class_histogram"],
        "mode": hist["semantic_png_mode"],
        "alignment_key": sidecar.get("capture_alignment_key", ""),
        "alignment_source": sidecar.get("capture_alignment_source", ""),
        "logical_sample_id": sidecar.get("logical_sample_id", ""),
        "episode_id": sidecar.get("episode_id", ""),
        "batch_id": sidecar.get("batch_id", ""),
        "frame_id": sidecar.get("frame_id", ""),
        "frame_seq": int(sidecar.get("frame_seq") or sidecar.get("tick") or 0),
        "tick": int(sidecar.get("tick") or 0),
        "capture_view_id": sidecar.get("capture_view_id", ""),
        "source_uav_entity_id": sidecar.get("source_uav_entity_id", sidecar.get("uav_entity_id", "")),
    }


def run_uav_episode(
    args: argparse.Namespace,
    index: int,
    episode_dir: Path,
    tick: int,
    entity_id: str,
    capture_view_id: str,
) -> dict[str, Any]:
    camera_id = f"{safe_name(entity_id)}__nadir_down"
    out_dir = uav_output_dir(args.output_root, index, episode_dir.name, capture_view_id)
    guarded_config = config_with_capture_timeout(
        episode_dir / "render_host_config.json",
        args.output_root,
        args.editor_hook_capture_timeout_s,
        uav_control_backend=args.uav_control_backend,
    )
    base_command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "Plugins/SumoImporter/Scripts/episode_render_host.py"),
        "--config",
        str(guarded_config),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output_dir",
        str(out_dir),
        "--camera-role",
        "uav",
        "--camera-id",
        camera_id,
        "--segmentation-backend",
        "ue_custom_stencil",
        "--runtime-uav-control-backend",
        args.uav_control_backend,
        "--airsim-capture-entity",
        entity_id,
        "--capture-view-id",
        capture_view_id,
        "--airsim-capture-vehicle",
        args.airsim_capture_vehicle,
        "--semantic-rules-path",
        str(args.rules),
        "--start_tick",
        "0",
        "--end_tick",
        str(tick),
        "--capture_tick",
        str(tick),
        "--max_batches",
        "1",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for modality in args.uav_modalities:
        command = list(base_command)
        command.extend(["--modality", modality])
        log_path = out_dir / f"uav_{modality}_host.log"
        return_code = run_child_with_guards(
            command,
            args=args,
            log_path=log_path,
            label=f"uav {episode_dir.name} {capture_view_id} {modality}",
        )
        if return_code != 0:
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            if looks_like_runtime_unavailable(log_text):
                raise FatalRuntimeUnavailable(f"episode_render_host lost runtime connectivity; log={log_path}")
            raise RuntimeError(f"episode_render_host failed modality={modality} rc={return_code}; log={log_path}")
    results: dict[str, Any] = {}
    for modality in args.uav_modalities:
        results[modality] = validate_single_uav_modality_output(out_dir, modality)
    alignment_keys = {str(payload.get("alignment_key") or "") for payload in results.values()}
    if len(alignment_keys) != 1:
        raise RuntimeError(f"UAV modality capture_alignment_key mismatch: {sorted(alignment_keys)}")
    frame_ids = {str(payload.get("frame_id") or "") for payload in results.values()}
    if len(frame_ids) != 1:
        raise RuntimeError(f"UAV modality frame_id mismatch: {sorted(frame_ids)}")
    frame_seqs = {require_int(payload.get("frame_seq"), "frame_seq") for payload in results.values()}
    if len(frame_seqs) != 1:
        raise RuntimeError(f"UAV modality frame_seq mismatch: {sorted(frame_seqs)}")
    batch_ids = {str(payload.get("batch_id") or "") for payload in results.values()}
    if len(batch_ids) != 1:
        raise RuntimeError(f"UAV modality batch_id mismatch: {sorted(batch_ids)}")
    alignment_sources = {str(payload.get("alignment_source") or "") for payload in results.values()}
    if len(alignment_sources) != 1:
        raise RuntimeError(f"UAV modality capture_alignment_source mismatch: {sorted(alignment_sources)}")
    expected_logical_sample_id = logical_sample_id(episode_dir.name, tick, capture_view_id)
    for modality, payload in sorted(results.items()):
        if str(payload.get("episode_id") or "") != episode_dir.name:
            raise RuntimeError(
                f"UAV {modality} sidecar episode mismatch: expected {episode_dir.name}, got {payload.get('episode_id')}"
            )
        if require_int(payload.get("tick"), f"{modality}.tick") != int(tick):
            raise RuntimeError(f"UAV {modality} sidecar tick mismatch: expected {tick}, got {payload.get('tick')}")
        if str(payload.get("capture_view_id") or "") != capture_view_id:
            raise RuntimeError(
                f"UAV {modality} sidecar view mismatch: expected {capture_view_id}, got {payload.get('capture_view_id')}"
            )
        if str(payload.get("source_uav_entity_id") or "") != entity_id:
            raise RuntimeError(
                f"UAV {modality} sidecar entity mismatch: expected {entity_id}, got {payload.get('source_uav_entity_id')}"
            )
        if str(payload.get("logical_sample_id") or "") != expected_logical_sample_id:
            raise RuntimeError(
                f"UAV {modality} logical sample mismatch: expected {expected_logical_sample_id}, "
                f"got {payload.get('logical_sample_id')}"
            )
    seg_validation = dict(results.get("seg") or {})
    validation = {
        "camera_id": camera_id,
        "entity_id": entity_id,
        "output_dir": str(out_dir),
        "modalities": list(args.uav_modalities),
        "capture_entity_id": entity_id,
        "capture_view_id": capture_view_id,
        "logical_event_id": episode_dir.name,
        "logical_sample_id": expected_logical_sample_id,
        "batch_id": next(iter(batch_ids)),
        "frame_id": next(iter(frame_ids)),
        "frame_seq": next(iter(frame_seqs)),
        "modality_outputs": results,
        "rgb_path": str((results.get("rgb") or {}).get("path") or ""),
        "depth_path": str((results.get("depth") or {}).get("path") or ""),
        "seg_path": str(seg_validation.get("png") or seg_validation.get("path") or ""),
        "seg_palette_path": str(seg_validation.get("palette") or ""),
        "histogram": dict(seg_validation.get("histogram") or {}),
        "alignment_key": str(seg_validation.get("alignment_key") or ""),
        "alignment_source": str(seg_validation.get("alignment_source") or ""),
    }
    return validation


def run_high_view_episode(
    args: argparse.Namespace,
    index: int,
    episode_dir: Path,
    tick: int,
    class_by_id: dict[str, str],
) -> dict[str, Any]:
    out_dir = output_dir(args.output_root, index, episode_dir.name, "high_overview")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tick_{tick:06d}"
    png_path = out_dir / f"{stem}.png"
    audit_path = out_dir / f"{stem}__seg_audit.json"
    palette_path = out_dir / f"{stem}__palette.png"
    sidecar_path = out_dir / f"{stem}.json"
    asset_id = f"SemanticHighOverview_{index:02d}"

    hook = FixedWorldCaptureEditorHook(
        project_root=PROJECT_ROOT,
        discovery_timeout_s=90.0,
        capture_timeout_s=args.editor_hook_capture_timeout_s,
    )
    try:
        hook.ensure_fixed_world_camera(
            map_id=args.map_id,
            asset_id=asset_id,
            logical_asset_id="camera.fixed_world_capture.rgb.v1",
            position_enu_m=args.high_position,
            rotation_deg={"pitch": args.high_pitch, "yaw": args.high_yaw, "roll": args.high_roll},
        )
        hook.capture_modality(
            map_id=args.map_id,
            asset_id=asset_id,
            modality="seg",
            output_path=png_path,
            width=args.width,
            height=args.height,
            fov_degrees=args.high_fov,
            semantic_rules_path=args.rules,
            semantic_audit_path=audit_path,
        )
    finally:
        hook.close()
    hist = semantic_histogram(png_path)
    write_palette_preview(Image.open(png_path), palette_path)
    sidecar = {
        "episode_id": episode_dir.name,
        "tick": tick,
        "map_id": args.map_id,
        "capture_backend": "ue_custom_stencil_fixed_world_camera",
        "segmentation_backend": "ue_custom_stencil",
        "segmentation_kind": "ue_custom_stencil_class_id_u8",
        "camera_role": "high_overview",
        "camera_pose_enu_m": args.high_position,
        "camera_rotation_deg": {"pitch": args.high_pitch, "yaw": args.high_yaw, "roll": args.high_roll},
        "fov_degrees": args.high_fov,
        "width": args.width,
        "height": args.height,
        "semantic_rules_path": str(args.rules),
        "semantic_class_by_id": class_by_id,
        "semantic_palette_preview_path": str(palette_path),
        "audit_path": str(audit_path),
        "capture_alignment_key": f"{episode_dir.name}:high_overview:{tick}:SemanticHighOverview",
        "capture_alignment_source": "static_pie_world_fixed_camera_no_episode_replay",
        "palette_preview_kind": "rgb_visualization_only_not_training_label",
        **hist,
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "png": str(png_path),
        "palette": str(palette_path),
        "sidecar": str(sidecar_path),
        "audit": str(audit_path),
        "histogram": hist["class_histogram"],
        "mode": hist["semantic_png_mode"],
        "alignment_key": sidecar["capture_alignment_key"],
        "alignment_source": sidecar["capture_alignment_source"],
        "output_dir": str(out_dir),
    }


def class_by_id(rules_path: Path) -> dict[str, str]:
    root = read_json(rules_path)
    result = {}
    for name, value in dict(root.get("classes") or {}).items():
        result[str(int(value))] = str(name)
    return result


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


def validate_contract(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    defaults = dict(contract.get("defaults") or {})
    must_follow = dict(contract.get("must_follow") or {})
    allowed_policies = set(contract.get("allowed_uav_policies") or [])
    allowed_backends = set(contract.get("allowed_uav_control_backends") or [])
    if str(args.segmentation_backend) != str(defaults.get("segmentation_backend", "ue_custom_stencil")):
        raise RuntimeError("Formal runner must use UE CustomStencil segmentation.")
    if str(args.segmentation_backend) != "ue_custom_stencil":
        raise RuntimeError("AirSim native segmentation is forbidden as a formal output.")
    if args.uav_policy not in allowed_policies:
        raise RuntimeError(f"Unsupported UAV policy '{args.uav_policy}'. Allowed: {sorted(allowed_policies)}")
    if args.uav_control_backend not in allowed_backends:
        raise RuntimeError(
            f"Unsupported UAV control backend '{args.uav_control_backend}'. Allowed: {sorted(allowed_backends)}"
        )
    required_uav_modalities = [str(value).strip().lower() for value in defaults.get("uav_modalities", ["rgb", "depth", "seg"])]
    if list(args.uav_modalities) != required_uav_modalities:
        raise RuntimeError(f"UAV modalities must be rgb/depth/seg for formal capture, got {args.uav_modalities}")
    if not bool(must_follow.get("single_modality_per_uav_host_run", True)):
        raise RuntimeError("Contract must require one image modality per UAV host run.")
    if bool(must_follow.get("forbid_formal_runtime_uav_editor_hook_fallback", True)) and str(args.uav_control_backend) == "pose_sync":
        runtime_defaults = dict(defaults)
        if bool(runtime_defaults.get("runtime_uav_editor_hook_fallback_enabled", False)):
            raise RuntimeError("Formal pose_sync runner must disable runtime UAV editor-hook fallback.")
    if not bool(must_follow.get("coordinate_audit_required", True)):
        raise RuntimeError("Contract must require coordinate audit for spawned/captured semantic entities.")
    if not bool(must_follow.get("require_uav_logical_sample_id_match", True)):
        raise RuntimeError("Contract must require logical sample id matching across UAV modalities.")
    if not bool(must_follow.get("require_same_event_tick_view_entity_across_modalities", True)):
        raise RuntimeError("Contract must require same event/tick/view/entity across UAV modalities.")
    if looks_timestamped_or_versioned(args.output_root):
        raise RuntimeError(f"Output root looks timestamped/versioned, which is forbidden: {args.output_root}")
    if bool(must_follow.get("output_root_must_be_f_drive_root", True)):
        if not is_f_drive_path(args.output_root):
            raise RuntimeError(f"Output root must be an absolute F: drive path, got: {args.output_root}")
        if not is_f_drive_path(args.summary):
            raise RuntimeError(f"Summary path must be an absolute F: drive path, got: {args.summary}")
    if float(args.host_run_timeout_s) > float(defaults.get("host_run_timeout_s", 300.0)):
        raise RuntimeError("Host run timeout cannot exceed the contract default.")
    if float(args.max_private_memory_gb) > float(defaults.get("max_private_memory_gb", 20.0)):
        raise RuntimeError("Private memory guard cannot exceed the contract default.")
    if float(args.max_working_set_gb) > float(defaults.get("max_working_set_gb", 20.0)):
        raise RuntimeError("Working-set memory guard cannot exceed the contract default.")
    if bool(must_follow.get("uav_host_run_requires_exactly_one_capture_entity", True)) and args.skip_uav:
        return


def print_contract_banner(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    version = str(contract.get("contract_version") or "")
    print("[SemanticCaptureContract]", flush=True)
    print(f"  version={version}", flush=True)
    print(f"  contract={args.contract}", flush=True)
    print(f"  output_root={args.output_root}", flush=True)
    print(f"  summary={args.summary}", flush=True)
    print("  path_contract=F_drive_root/simple_names/meta_sidecars", flush=True)
    print(f"  ue_reuse=true close_ue_on_success=false keep_ue_open_on_failure=true", flush=True)
    print(f"  tick_mode=sequential_from_zero_to_{args.requested_tick}", flush=True)
    print(f"  segmentation_backend={args.segmentation_backend}", flush=True)
    print(f"  uav_policy={args.uav_policy}", flush=True)
    print(f"  uav_control_backend={args.uav_control_backend}", flush=True)
    print(f"  runtime_uav_editor_hook_fallback=false", flush=True)
    print(f"  coordinate_audit=required", flush=True)
    print(f"  uav_modalities={','.join(args.uav_modalities)}", flush=True)
    print("  host_run_modality=single_modality_per_host_process", flush=True)
    print(f"  host_run_timeout_s={args.host_run_timeout_s}", flush=True)
    print(
        f"  memory_guard_gb=working_set:{args.max_working_set_gb} private:{args.max_private_memory_gb}",
        flush=True,
    )


def row_base(
    args: argparse.Namespace,
    contract: dict[str, Any],
    *,
    index: int,
    episode: str,
    tick: int,
    tick_selection: str,
    view: str,
    altitude: float,
    capture_entity_id: str = "",
    capture_view_id: str = "",
) -> dict[str, Any]:
    memory = assert_runtime_available(args)
    return {
        "index": index,
        "episode": episode,
        "requested_tick": args.requested_tick,
        "tick": tick,
        "tick_selection": tick_selection,
        "view": view,
        "status": "pending",
        "altitude_m": round(float(altitude), 3),
        "camera_id": "",
        "capture_entity_id": capture_entity_id,
        "capture_view_id": capture_view_id,
        "logical_event_id": episode,
        "logical_sample_id": logical_sample_id(episode, tick, capture_view_id) if capture_view_id else "",
        "batch_id": "",
        "frame_id": "",
        "frame_seq": "",
        "runtime_uav_count": "",
        "uav_policy": args.uav_policy,
        "uav_control_backend": args.uav_control_backend,
        "contract_path": str(args.contract),
        "contract_version": str(contract.get("contract_version") or ""),
        "working_set_gb": memory["working_set_gb"],
        "private_memory_gb": memory["private_memory_gb"],
        "histogram": "{}",
        "rgb_path": "",
        "depth_path": "",
        "seg_path": "",
        "seg_palette_path": "",
        "modality_outputs": "{}",
        "alignment_key": "",
        "alignment_source": "",
        "output_dir": str(output_dir(args.output_root, index, episode, view)),
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", type=Path, default=PROJECT_ROOT / "Dataset/render_ready_episodes")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "Config/LowAltitude/semantic_stencil_rules.json")
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / "Config/LowAltitude/semantic_capture_runtime_contract.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--requested-tick", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--editor-hook-capture-timeout-s", type=float, default=90.0)
    parser.add_argument("--host-run-timeout-s", type=float, default=300.0)
    parser.add_argument("--host-guard-poll-s", type=float, default=2.0)
    parser.add_argument("--max-working-set-gb", type=float, default=20.0)
    parser.add_argument("--max-private-memory-gb", type=float, default=20.0)
    parser.add_argument("--map-id", default="donghu_road_topo")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--high-position", nargs=3, type=float, default=[6972.4485, 6223.144, 260.0])
    parser.add_argument("--high-pitch", type=float, default=-90.0)
    parser.add_argument("--high-yaw", type=float, default=-115.0)
    parser.add_argument("--high-roll", type=float, default=0.0)
    parser.add_argument("--high-fov", type=float, default=70.0)
    parser.add_argument("--uav-modalities", nargs="+", default=["rgb", "depth", "seg"])
    parser.add_argument("--segmentation-backend", default="ue_custom_stencil")
    parser.add_argument("--uav-policy", choices=["one_uav_per_episode", "all_uavs_by_separate_runs"], default="one_uav_per_episode")
    parser.add_argument("--uav-control-backend", choices=["pose_sync", "airsim_move"], default="pose_sync")
    parser.add_argument("--airsim-capture-vehicle", default="CaptureUAV_0")
    parser.add_argument("--airsim-capture-entity", default="", help="Optional explicit UAV entity for targeted single-episode capture.")
    parser.add_argument("--skip-high-overview", action="store_true")
    parser.add_argument("--skip-uav", action="store_true")
    parser.add_argument("--append-summary", action="store_true", help="Append rows to an existing summary instead of replacing it.")
    args = parser.parse_args()
    args.uav_modalities = [str(value).strip().lower() for value in args.uav_modalities if str(value).strip()]
    for modality in args.uav_modalities:
        if modality not in {"rgb", "depth", "seg"}:
            raise RuntimeError(f"Unsupported --uav-modalities value: {modality}")
    contract = read_json(args.contract)
    validate_contract(args, contract)
    print_contract_banner(args, contract)

    episodes = sorted(path for path in args.episodes_root.iterdir() if path.is_dir())
    episodes = episodes[max(0, args.start_index) :]
    if args.limit > 0:
        episodes = episodes[: args.limit]
    if not episodes:
        raise RuntimeError("No episodes selected.")
    assert_runtime_available(args)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    classes = class_by_id(args.rules)
    write_header = not args.append_summary or not args.summary.exists() or args.summary.stat().st_size == 0
    summary_mode = "a" if args.append_summary else "w"
    with args.summary.open(summary_mode, encoding="utf-8", newline="") as handle:
        fields = [
            "index",
            "episode",
            "requested_tick",
            "tick",
            "tick_selection",
            "view",
            "status",
            "altitude_m",
            "camera_id",
            "capture_entity_id",
            "capture_view_id",
            "logical_event_id",
            "logical_sample_id",
            "batch_id",
            "frame_id",
            "frame_seq",
            "runtime_uav_count",
            "uav_policy",
            "uav_control_backend",
            "contract_path",
            "contract_version",
            "working_set_gb",
            "private_memory_gb",
            "histogram",
            "rgb_path",
            "depth_path",
            "seg_path",
            "seg_palette_path",
            "modality_outputs",
            "alignment_key",
            "alignment_source",
            "output_dir",
            "error",
        ]
        if args.append_summary and not write_header:
            existing_header = args.summary.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0].split(",")
            if existing_header != fields:
                raise RuntimeError(
                    "Existing summary header does not match the current runtime contract schema. "
                    f"Use a fresh deterministic summary path or remove the stale file: {args.summary}"
                )
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for index, episode_dir in enumerate(episodes, start=max(0, args.start_index)):
            assert_runtime_available(args)
            tick, tick_selection = tick_at_or_before(episode_dir, args.requested_tick)
            print(f"[{index + 1}/{len(episodes)}] {episode_dir.name} tick={tick}", flush=True)
            active_uavs = runtime_uav_entity_ids_at_tick(episode_dir, tick)
            default_altitude = altitude_at_tick(episode_dir, active_uavs[0], tick) if active_uavs else 0.0

            if not args.skip_high_overview:
                row = row_base(
                    args,
                    contract,
                    index=index,
                    episode=episode_dir.name,
                    tick=tick,
                    tick_selection=tick_selection,
                    view="high_overview",
                    altitude=default_altitude,
                )
                row["runtime_uav_count"] = len(active_uavs)
                try:
                    validation = run_high_view_episode(args, index, episode_dir, tick, classes)
                    row["camera_id"] = "SemanticHighOverview"
                    row["status"] = "ok"
                    row["histogram"] = json.dumps(validation.get("histogram") or {}, sort_keys=True)
                    row["rgb_path"] = str(validation.get("rgb_path") or "")
                    row["depth_path"] = str(validation.get("depth_path") or "")
                    row["seg_path"] = str(validation.get("seg_path") or validation.get("png") or "")
                    row["seg_palette_path"] = str(validation.get("seg_palette_path") or validation.get("palette") or "")
                    row["modality_outputs"] = json.dumps(validation.get("modality_outputs") or {}, sort_keys=True)
                    row["alignment_key"] = str(validation.get("alignment_key") or "")
                    row["alignment_source"] = str(validation.get("alignment_source") or "")
                    row["output_dir"] = str(validation.get("output_dir") or row["output_dir"])
                except FatalRuntimeUnavailable as exc:
                    row["status"] = "fatal_runtime_unavailable"
                    row["error"] = str(exc)
                    memory = unreal_memory_gb()
                    row["working_set_gb"] = memory["working_set_gb"]
                    row["private_memory_gb"] = memory["private_memory_gb"]
                    writer.writerow(row)
                    handle.flush()
                    print(f"  high_overview: {row['status']} mem={memory['working_set_gb']:.2f}/{memory['private_memory_gb']:.2f}GB err={row['error'][:180]}", flush=True)
                    raise
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = str(exc)
                memory = unreal_memory_gb()
                row["working_set_gb"] = memory["working_set_gb"]
                row["private_memory_gb"] = memory["private_memory_gb"]
                writer.writerow(row)
                handle.flush()
                print(f"  high_overview: {row['status']} mem={memory['working_set_gb']:.2f}/{memory['private_memory_gb']:.2f}GB hist={row['histogram']} err={row['error'][:180]}", flush=True)
                assert_memory_under_limit(args, memory)
                if not rpc_available(args.host, args.port):
                    raise FatalRuntimeUnavailable(f"AirSim RPC is unavailable at {args.host}:{args.port}")

            if args.skip_uav:
                continue

            explicit_entity = str(args.airsim_capture_entity or "").strip()
            if explicit_entity:
                selected_uavs = [explicit_entity] if explicit_entity in active_uavs else []
                skipped_uavs = [entity_id for entity_id in active_uavs if entity_id != explicit_entity]
                if not selected_uavs:
                    row = row_base(
                        args,
                        contract,
                        index=index,
                        episode=episode_dir.name,
                        tick=tick,
                        tick_selection=tick_selection,
                        view="uav_tick100",
                        altitude=0.0,
                        capture_entity_id=explicit_entity,
                        capture_view_id=f"uav_view_000__{safe_name(explicit_entity)}",
                    )
                    row["runtime_uav_count"] = len(active_uavs)
                    row["status"] = "skipped_by_explicit_capture_entity"
                    row["error"] = f"Requested capture entity is not active at tick {tick}."
                    writer.writerow(row)
                    handle.flush()
                    print(f"  uav_tick100: {row['status']} entity={explicit_entity}", flush=True)
            elif args.uav_policy == "one_uav_per_episode":
                selected_uavs = active_uavs[:1]
                skipped_uavs = active_uavs[1:]
            else:
                selected_uavs = list(active_uavs)
                skipped_uavs = []

            if not active_uavs:
                row = row_base(
                    args,
                    contract,
                    index=index,
                    episode=episode_dir.name,
                    tick=tick,
                    tick_selection=tick_selection,
                    view="uav_tick100",
                    altitude=0.0,
                )
                row["runtime_uav_count"] = 0
                row["status"] = "skipped_no_runtime_uav"
                row["error"] = f"No active runtime UAV at tick {tick}."
                writer.writerow(row)
                handle.flush()
                print("  uav_tick100: skipped_no_runtime_uav", flush=True)
                continue

            for skipped_entity in skipped_uavs:
                row = row_base(
                    args,
                    contract,
                    index=index,
                    episode=episode_dir.name,
                    tick=tick,
                    tick_selection=tick_selection,
                    view="uav_tick100",
                    altitude=altitude_at_tick(episode_dir, skipped_entity, tick),
                    capture_entity_id=skipped_entity,
                    capture_view_id=f"uav_view_skip__{safe_name(skipped_entity)}",
                )
                row["runtime_uav_count"] = len(active_uavs)
                row["status"] = "skipped_by_one_uav_policy" if not explicit_entity else "skipped_by_explicit_capture_entity"
                row["error"] = "UAV was intentionally not captured by the selected UAV policy."
                writer.writerow(row)
                handle.flush()
                print(f"  uav_tick100: {row['status']} entity={skipped_entity}", flush=True)

            for uav_ordinal, capture_entity in enumerate(selected_uavs):
                capture_view_id = f"uav_view_{uav_ordinal:03d}__{safe_name(capture_entity)}"
                row = row_base(
                    args,
                    contract,
                    index=index,
                    episode=episode_dir.name,
                    tick=tick,
                    tick_selection=tick_selection,
                    view="uav_tick100",
                    altitude=altitude_at_tick(episode_dir, capture_entity, tick),
                    capture_entity_id=capture_entity,
                    capture_view_id=capture_view_id,
                )
                row["runtime_uav_count"] = len(active_uavs)
                row["output_dir"] = str(uav_output_dir(args.output_root, index, episode_dir.name, capture_view_id))
                try:
                    validation = run_uav_episode(args, index, episode_dir, tick, capture_entity, capture_view_id)
                    row["camera_id"] = str(validation.get("camera_id") or "")
                    row["status"] = "ok"
                    row["logical_event_id"] = str(validation.get("logical_event_id") or row["logical_event_id"])
                    row["logical_sample_id"] = str(validation.get("logical_sample_id") or row["logical_sample_id"])
                    row["batch_id"] = str(validation.get("batch_id") or "")
                    row["frame_id"] = str(validation.get("frame_id") or "")
                    row["frame_seq"] = str(validation.get("frame_seq") or "")
                    row["histogram"] = json.dumps(validation.get("histogram") or {}, sort_keys=True)
                    row["rgb_path"] = str(validation.get("rgb_path") or "")
                    row["depth_path"] = str(validation.get("depth_path") or "")
                    row["seg_path"] = str(validation.get("seg_path") or "")
                    row["seg_palette_path"] = str(validation.get("seg_palette_path") or "")
                    row["modality_outputs"] = json.dumps(validation.get("modality_outputs") or {}, sort_keys=True)
                    row["alignment_key"] = str(validation.get("alignment_key") or "")
                    row["alignment_source"] = str(validation.get("alignment_source") or "")
                    row["output_dir"] = str(validation.get("output_dir") or row["output_dir"])
                except FatalRuntimeUnavailable as exc:
                    row["status"] = "fatal_runtime_unavailable"
                    row["error"] = str(exc)
                    memory = unreal_memory_gb()
                    row["working_set_gb"] = memory["working_set_gb"]
                    row["private_memory_gb"] = memory["private_memory_gb"]
                    writer.writerow(row)
                    handle.flush()
                    print(f"  uav_tick100: {row['status']} entity={capture_entity} mem={memory['working_set_gb']:.2f}/{memory['private_memory_gb']:.2f}GB err={row['error'][:180]}", flush=True)
                    raise
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = str(exc)
                memory = unreal_memory_gb()
                row["working_set_gb"] = memory["working_set_gb"]
                row["private_memory_gb"] = memory["private_memory_gb"]
                writer.writerow(row)
                handle.flush()
                print(f"  uav_tick100: {row['status']} entity={capture_entity} mem={memory['working_set_gb']:.2f}/{memory['private_memory_gb']:.2f}GB hist={row['histogram']} err={row['error'][:180]}", flush=True)
                assert_memory_under_limit(args, memory)
                if not rpc_available(args.host, args.port):
                    raise FatalRuntimeUnavailable(f"AirSim RPC is unavailable at {args.host}:{args.port}")
    print(f"SUMMARY={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
