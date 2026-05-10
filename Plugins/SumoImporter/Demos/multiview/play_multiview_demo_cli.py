#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import time
from collections import deque
from pathlib import Path
from typing import Any

SUMO_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Scripts"
if str(SUMO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUMO_SCRIPTS_DIR))

from donghu_core.discovery import default_timeline_path, project_root_from

DEFAULT_TIMELINE_PATH = default_timeline_path(project_root_from(Path(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay multiview UAV demo logs in a terminal.")
    parser.add_argument("--timeline", default=str(DEFAULT_TIMELINE_PATH), help="Timeline JSON path")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--hold-final-s", type=float, default=2.0, help="Hold the last frame before exit")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser.parse_args()


def _ansi(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _clear_screen() -> None:
    if os.name == "nt":
        sys.stdout.write("\x1b[2J\x1b[H")
    else:
        sys.stdout.write("\x1b[2J\x1b[H")


def _format_weather(frame: dict[str, Any], *, color: bool) -> str:
    weather = dict(frame.get("weather") or {})
    condition = str(weather.get("condition") or "unknown").upper()
    rain = float(weather.get("rain") or 0.0)
    fog = float(weather.get("fog_density") or 0.0)
    if condition == "RAIN":
        condition = _ansi(condition, "33;1", enabled=color)
    return f"{condition} rain={rain:.2f} fog={fog:.2f}"


def _format_counts(frame: dict[str, Any]) -> str:
    counts = dict(frame.get("uav_counts") or {})
    return (
        f"uav_total={int(counts.get('total') or 0)}  "
        f"submit_to_ue={int(counts.get('submit_to_ue') or 0)}  "
        f"visible={int(counts.get('visible') or 0)}  "
        f"offstage={int(counts.get('offstage') or 0)}"
    )


def _group_lines(frame: dict[str, Any], width: int) -> list[str]:
    lines: list[str] = []
    for group in frame.get("uav_groups") or []:
        label = str(group.get("activity_type") or "unknown")
        count = int(group.get("count") or 0)
        ids_compact = str(group.get("ids_compact") or "-")
        text = f"{label:<10} ({count:>2}): {ids_compact}"
        lines.extend(textwrap.wrap(text, width=width) or [""])
    return lines


def _tracked_lines(frame: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for row in frame.get("tracked_uavs") or []:
        pos = list(row.get("position_enu_m") or [0.0, 0.0, 0.0])
        lines.append(
            f"{row['short_id']:<4} {row['activity_type']:<10} {row['submission_state']:<16} "
            f"x={pos[0]:>7.2f} y={pos[1]:>7.2f} z={pos[2]:>6.2f} v={float(row.get('speed_mps') or 0.0):>4.1f}"
        )
    return lines


def _all_uav_compact_lines(frame: dict[str, Any], width: int) -> list[str]:
    entries = []
    for row in frame.get("all_uav_briefs") or []:
        pos = list(row.get("position_enu_m") or [0.0, 0.0, 0.0])
        status = str(row.get("activity_type") or "")[:3].upper()
        if str(row.get("visibility_state") or "") == "offstage":
            status = f"{status}*"
        entries.append(f"{row['short_id']}({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) {status}")

    if not entries:
        return []

    cols = 3 if width >= 135 else 2
    col_width = max(24, width // cols)
    lines: list[str] = []
    for start in range(0, len(entries), cols):
        chunk = entries[start : start + cols]
        padded = [item.ljust(col_width) for item in chunk]
        lines.append("".join(padded).rstrip())
    return lines


def _draw_frame(meta: dict[str, Any], frame: dict[str, Any], log_feed: deque[str], *, color: bool) -> None:
    term_width = max(80, shutil.get_terminal_size((120, 40)).columns)
    _clear_screen()
    playback_duration_s = float(meta.get("playback_duration_s") or 0.0)
    frame_index = int(frame.get("frame_index") or 0)
    playback_time_s = float(frame.get("playback_time_s") or 0.0)
    sim_time_s = float(frame.get("sim_time_s") or 0.0)
    tick = int(frame.get("tick") or 0)

    print(_ansi("Dense UAV Rain/Fall Inspection Replay", "36;1", enabled=color))
    print(
        f"playback {playback_time_s:6.2f}/{playback_duration_s:6.2f}s   "
        f"sim {sim_time_s:6.2f}/90.00s   tick {tick:>4}   frame {frame_index + 1:>3}/{int(meta.get('frame_count') or 0):>3}"
    )
    print(f"weather {_format_weather(frame, color=color)}")
    print(_format_counts(frame))
    print(_ansi("-" * min(term_width, 120), "90", enabled=color))
    print(_ansi("Recent Log Feed", "35;1", enabled=color))
    for line in log_feed:
        print(line[:term_width])
    print(_ansi("-" * min(term_width, 120), "90", enabled=color))
    print(_ansi("UAV Groups", "35;1", enabled=color))
    for line in _group_lines(frame, term_width):
        print(line[:term_width])
    print(_ansi("-" * min(term_width, 120), "90", enabled=color))
    print(_ansi("Tracked UAV Positions", "35;1", enabled=color))
    for line in _tracked_lines(frame):
        print(line[:term_width])
    print(_ansi("-" * min(term_width, 120), "90", enabled=color))
    print(_ansi("All UAV Compact", "35;1", enabled=color))
    for line in _all_uav_compact_lines(frame, term_width):
        print(line[:term_width])
    sys.stdout.flush()


def _run_once(timeline: dict[str, Any], *, hold_final_s: float, color: bool) -> None:
    meta = dict(timeline.get("meta") or {})
    frames = list(timeline.get("frames") or [])
    if not frames:
        raise RuntimeError("Timeline contains no frames.")

    frame_interval_s = float(meta.get("frame_interval_s") or 0.25)
    log_feed: deque[str] = deque(maxlen=16)
    start_time = time.perf_counter()

    for index, frame in enumerate(frames):
        for message in frame.get("event_messages") or []:
            stamp = f"[play {float(frame.get('playback_time_s') or 0.0):5.2f}s | t{int(frame.get('tick') or 0):04d}]"
            log_feed.append(f"{stamp} {message}")
        _draw_frame(meta, frame, log_feed, color=color)

        next_deadline = start_time + (index + 1) * frame_interval_s
        sleep_s = next_deadline - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

    if hold_final_s > 0.0:
        time.sleep(hold_final_s)


def main() -> None:
    args = parse_args()
    timeline_path = Path(args.timeline).resolve()
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    color = not bool(args.no_color)
    while True:
        _run_once(timeline, hold_final_s=float(args.hold_final_s), color=color)
        if not args.loop:
            break


if __name__ == "__main__":
    main()
