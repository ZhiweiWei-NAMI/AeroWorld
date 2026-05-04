#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
STATUS_PATH = PROJECT_ROOT / "Saved" / "editor_start_pie_status.json"
TARGET_MAP = "/Game/Maps/donghu"
STARTUP_DELAY_S = 5.0
TIMEOUT_S = 90.0

_state: dict[str, object] = {
    "start_time_s": time.time(),
    "tick_handle": None,
    "play_requested": False,
}


def _write_status(status: str, **payload: object) -> None:
    data = {
        "status": status,
        "timestamp_s": time.time(),
        **payload,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup() -> None:
    tick_handle = _state.get("tick_handle")
    if tick_handle is not None:
        unreal.unregister_slate_post_tick_callback(tick_handle)
        _state["tick_handle"] = None


def _tick(delta_seconds: float) -> None:
    del delta_seconds
    try:
        elapsed_s = float(time.time() - float(_state["start_time_s"]))
        level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if level_editor is None:
            _write_status("waiting_for_level_editor", elapsed_s=elapsed_s)
            return

        editor_world = unreal.EditorLevelLibrary.get_editor_world()
        current_map = editor_world.get_path_name() if editor_world is not None else ""
        pie_worlds = unreal.EditorLevelLibrary.get_pie_worlds(False)
        if pie_worlds:
            _write_status(
                "pie_ready",
                elapsed_s=elapsed_s,
                current_map=current_map,
                pie_world_count=len(pie_worlds),
            )
            unreal.log("[editor_start_pie] PIE world is ready.")
            _cleanup()
            return

        if elapsed_s >= TIMEOUT_S:
            _write_status(
                "timeout",
                elapsed_s=elapsed_s,
                current_map=current_map,
                play_requested=bool(_state["play_requested"]),
            )
            unreal.log_error("[editor_start_pie] Timed out waiting for PIE to start.")
            _cleanup()
            return

        if current_map != TARGET_MAP:
            if not level_editor.load_level(TARGET_MAP):
                raise RuntimeError(f"Failed to load target map '{TARGET_MAP}'.")
            _write_status("loading_map", elapsed_s=elapsed_s, current_map=current_map, target_map=TARGET_MAP)
            unreal.log(f"[editor_start_pie] Loading target map {TARGET_MAP}.")
            return

        if elapsed_s < STARTUP_DELAY_S:
            _write_status("waiting_for_editor", elapsed_s=elapsed_s, current_map=current_map)
            return

        if not bool(_state["play_requested"]):
            level_editor.editor_play_simulate()
            _state["play_requested"] = True
            _write_status("requested_play", elapsed_s=elapsed_s, current_map=current_map)
            unreal.log("[editor_start_pie] Requested EditorPlaySimulate.")
            return

        _write_status("waiting_for_pie", elapsed_s=elapsed_s, current_map=current_map)
    except Exception as exc:
        _write_status(
            "error",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        unreal.log_error(f"[editor_start_pie] {exc}")
        _cleanup()
        raise


_write_status("registered", target_map=TARGET_MAP)
_state["tick_handle"] = unreal.register_slate_post_tick_callback(_tick)
unreal.log("[editor_start_pie] Registered post-tick callback.")
