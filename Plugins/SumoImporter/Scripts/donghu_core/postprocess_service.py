from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .artifact_writer import load_jsonl, write_json


INTERESTING_UAV_IDS = [
    "drone_demo_a_001",
    "drone_demo_a_005",
    "drone_demo_a_007",
    "drone_demo_a_021",
    "drone_demo_a_023",
    "drone_demo_a_024",
]

PLANNING_LOGS: dict[int, list[str]] = {
    0: [
        "planner:init dense patrol roster a001-a020,a022; local monitor a021; reserve inspector a023; transit a024",
        "planner:route_a_dense_patrol nominal over intersection_a; weather=clear; corridor open",
    ],
    100: [
        "plan:update a001,a005,a013 maintain nominal sweep schedule over site.intersection_a",
    ],
    250: [
        "plan:update a021 keeps local monitor orbit; patrol fleet remains staged for route refresh",
    ],
    400: [
        "ops: pre-rain traffic window normal; patrol schedule remains dense and unconstrained",
    ],
    450: [
        "policy: rain-mode enabled; reduce low-altitude exposure and tighten ROI crossings",
    ],
    500: [
        "dispatch: divert order issued to most patrol UAVs; hold corridor clear after rain onset",
    ],
    505: [
        "dispatch: divert groups a001-a008 and a009-a020,a022 acknowledged; residual monitor a021 retained",
    ],
    550: [
        "ops: ROI traffic intentionally thinned after rain; only residual/local monitoring routes remain near scene",
    ],
    600: [
        "incident: fall alert raised for pedestrian_a_005; request airborne inspection support",
    ],
    610: [
        "planner: candidate scoring complete -> a023 selected for inspection mission; route solve in progress",
    ],
    620: [
        "planner: a023 route locked to fall target; divert holds remain active for surrounding patrol UAVs",
    ],
    650: [
        "dispatch: a023 inspection mission active; local monitor a021 remains peripheral; transit a024 unchanged",
    ],
    700: [
        "inspection: a023 holding over incident; patrol throughput through ROI remains reduced under rain policy",
    ],
    800: [
        "ops: rain persists; diverted fleet stays outside ROI, preserving clear airspace above incident point",
    ],
    900: [
        "episode:end rain active; fall incident still monitored by a023; reduced-traffic policy maintained",
    ],
}


@dataclass(frozen=True)
class FrameEntry:
    tick: int
    png_path: Path
    json_path: Path


class PostprocessService:
    def __init__(self, *, interesting_uav_ids: list[str] | None = None) -> None:
        self.interesting_uav_ids = list(interesting_uav_ids or INTERESTING_UAV_IDS)

    @staticmethod
    def sorted_frame_entries(rgb_dir: Path) -> list[FrameEntry]:
        by_stem: dict[str, dict[str, Path]] = defaultdict(dict)
        for path in rgb_dir.iterdir():
            if path.suffix.lower() not in {".png", ".json"}:
                continue
            by_stem[path.stem][path.suffix.lower()] = path

        entries: list[FrameEntry] = []
        tick_pattern = re.compile(r"tick_(\d+)")
        for stem, files in by_stem.items():
            png_path = files.get(".png")
            json_path = files.get(".json")
            if png_path is None or json_path is None:
                continue
            match = tick_pattern.search(stem)
            if not match:
                continue
            entries.append(FrameEntry(tick=int(match.group(1)), png_path=png_path, json_path=json_path))
        entries.sort(key=lambda item: item.tick)
        return entries

    @staticmethod
    def _short_uav_id(entity_id: str) -> str:
        match = re.match(r"drone_demo_a_(\d+)$", entity_id)
        if match:
            return f"a{int(match.group(1)):03d}"
        return entity_id

    @staticmethod
    def _compress_short_ids(short_ids: Iterable[str]) -> str:
        values = sorted(set(short_ids))
        numeric_ids: list[int] = []
        raw_ids: list[str] = []
        for item in values:
            match = re.fullmatch(r"a(\d{3})", item)
            if match:
                numeric_ids.append(int(match.group(1)))
            else:
                raw_ids.append(item)
        parts: list[str] = []
        if numeric_ids:
            start = prev = numeric_ids[0]
            for value in numeric_ids[1:]:
                if value == prev + 1:
                    prev = value
                    continue
                parts.append(f"a{start:03d}" if start == prev else f"a{start:03d}-a{prev:03d}")
                start = prev = value
            parts.append(f"a{start:03d}" if start == prev else f"a{start:03d}-a{prev:03d}")
        parts.extend(raw_ids)
        return ",".join(parts)

    def _activity_group_lines(self, uavs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for row in uavs:
            groups[str(row.get("activity_type") or "unknown")].append(self._short_uav_id(str(row.get("entity_id") or "")))
        output: list[dict[str, Any]] = []
        for activity, short_ids in sorted(groups.items(), key=lambda item: item[0]):
            output.append(
                {
                    "activity_type": activity,
                    "count": len(short_ids),
                    "ids": sorted(short_ids),
                    "ids_compact": self._compress_short_ids(short_ids),
                }
            )
        return output

    def _tracked_uav_rows(self, uavs: list[dict[str, Any]], uav_runtime: dict[str, Any]) -> list[dict[str, Any]]:
        by_id = {str(row.get("entity_id") or ""): row for row in uavs}
        tracked_rows: list[dict[str, Any]] = []
        for entity_id in self.interesting_uav_ids:
            row = by_id.get(entity_id)
            if row is None:
                continue
            runtime = dict(uav_runtime.get(entity_id) or {})
            runtime_move = dict(runtime.get("move") or {})
            runtime_pos = list(runtime_move.get("position_enu_m") or [])
            source_pos = list(row.get("source_position_enu_m") or row.get("position_enu_m") or [0.0, 0.0, 0.0])
            shown_pos = runtime_pos if len(runtime_pos) >= 3 else source_pos
            tracked_rows.append(
                {
                    "entity_id": entity_id,
                    "short_id": self._short_uav_id(entity_id),
                    "activity_type": str(row.get("activity_type") or ""),
                    "submission_state": str(row.get("submission_state") or ""),
                    "visibility_state": str(row.get("visibility_state") or ""),
                    "position_enu_m": [round(float(value), 2) for value in shown_pos[:3]],
                    "source_position_enu_m": [round(float(value), 2) for value in source_pos[:3]],
                    "runtime_position_enu_m": [round(float(value), 2) for value in runtime_pos[:3]] if len(runtime_pos) >= 3 else [],
                    "speed_mps": round(math.sqrt(sum(float(v) ** 2 for v in (row.get("velocity_enu_mps") or [0.0, 0.0, 0.0]))), 2),
                }
            )
        return tracked_rows

    def _all_uav_briefs(self, uavs: list[dict[str, Any]], uav_runtime: dict[str, Any]) -> list[dict[str, Any]]:
        briefs: list[dict[str, Any]] = []
        for row in sorted(uavs, key=lambda item: str(item.get("entity_id") or "")):
            entity_id = str(row.get("entity_id") or "")
            runtime = dict(uav_runtime.get(entity_id) or {})
            runtime_move = dict(runtime.get("move") or {})
            runtime_pos = list(runtime_move.get("position_enu_m") or [])
            source_pos = list(row.get("source_position_enu_m") or row.get("position_enu_m") or [0.0, 0.0, 0.0])
            shown_pos = runtime_pos if len(runtime_pos) >= 3 else source_pos
            briefs.append(
                {
                    "entity_id": entity_id,
                    "short_id": self._short_uav_id(entity_id),
                    "activity_type": str(row.get("activity_type") or ""),
                    "visibility_state": str(row.get("visibility_state") or ""),
                    "position_enu_m": [round(float(value), 1) for value in shown_pos[:3]],
                }
            )
        return briefs

    def _build_logs_for_tick(self, tick: int, actual_events: list[dict[str, Any]], low_frame: dict[str, Any]) -> list[str]:
        messages = list(PLANNING_LOGS.get(tick, []))
        for event in actual_events:
            payload = dict(event.get("payload") or {})
            title = str(payload.get("title") or event.get("topic") or "event")
            severity = str(((event.get("render_hints") or {}).get("severity")) or "info").upper()
            messages.append(f"{severity}: {title}")

        if tick in {500, 550, 650}:
            uavs = [row for row in low_frame.get("entity_records", []) if str(row.get("entity_category") or "") == "uav"]
            groups = {group["activity_type"]: group["ids_compact"] for group in self._activity_group_lines(uavs)}
            if tick == 500 and "diverting" in groups:
                messages.append(f"planner: diverting group now includes {groups['diverting']}")
            if tick == 550:
                reduced_count = sum(
                    1
                    for row in uavs
                    if str(row.get("activity_type") or "") in {"patrolling", "inspection", "transit"}
                )
                messages.append(f"ops: ROI-adjacent UAV traffic reduced to {reduced_count} active mission threads after rain")
            if tick == 650 and "inspection" in groups:
                messages.append(f"dispatch: inspection group active -> {groups['inspection']}")
        return messages

    def build_gif(
        self,
        entries: list[FrameEntry],
        output_path: Path,
        *,
        frame_duration_ms: int,
        final_hold_ms: int,
        max_width: int,
        max_colors: int,
    ) -> dict[str, Any]:
        frames: list[Image.Image] = []
        durations: list[int] = []
        for entry in entries:
            image = Image.open(entry.png_path).convert("RGB")
            if max_width > 0 and image.width > max_width:
                resized_height = round(image.height * (max_width / image.width))
                image = image.resize((max_width, resized_height), Image.Resampling.LANCZOS)
            palette_image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=max_colors)
            frames.append(palette_image)
            durations.append(frame_duration_ms)

        if not frames:
            raise RuntimeError(f"No frames available for GIF output: {output_path}")

        durations[-1] += final_hold_ms
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
            disposal=2,
        )
        return {
            "path": str(output_path),
            "frame_count": len(frames),
            "frame_duration_ms": frame_duration_ms,
            "final_hold_ms": final_hold_ms,
            "size_bytes": output_path.stat().st_size,
        }

    def build_timeline(self, entries: list[FrameEntry], episode_dir: Path, *, playback_duration_s: float, hold_final_s: float) -> dict[str, Any]:
        event_rows = load_jsonl(episode_dir / "event_trace.jsonl")
        manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
        events_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in event_rows:
            events_by_tick[int(row.get("tick") or 0)].append(row)

        frame_interval_s = playback_duration_s / max(1, len(entries))
        timeline_frames: list[dict[str, Any]] = []
        for index, entry in enumerate(entries):
            low_frame = json.loads(entry.json_path.read_text(encoding="utf-8"))
            uavs = [row for row in low_frame.get("entity_records", []) if str(row.get("entity_category") or "") == "uav"]
            frame = {
                "frame_index": index,
                "tick": int(low_frame.get("tick") or entry.tick),
                "sim_time_s": float(low_frame.get("sim_time_s") or 0.0),
                "playback_time_s": round(index * frame_interval_s, 3),
                "weather": {
                    "condition": str(((low_frame.get("weather") or {}).get("condition")) or "unknown"),
                    "rain": float(((low_frame.get("weather") or {}).get("rain")) or 0.0),
                    "wetness": float(((low_frame.get("weather") or {}).get("wetness")) or 0.0),
                    "fog_density": float(((low_frame.get("weather") or {}).get("fog_density")) or 0.0),
                },
                "uav_counts": {
                    "total": len(uavs),
                    "submit_to_ue": sum(1 for row in uavs if str(row.get("submission_state") or "") == "submit_to_ue"),
                    "visible": sum(1 for row in uavs if str(row.get("visibility_state") or "") == "visible"),
                    "offstage": sum(1 for row in uavs if str(row.get("visibility_state") or "") == "offstage"),
                },
                "uav_groups": self._activity_group_lines(uavs),
                "tracked_uavs": self._tracked_uav_rows(uavs, dict(low_frame.get("uav_runtime") or {})),
                "all_uav_briefs": self._all_uav_briefs(uavs, dict(low_frame.get("uav_runtime") or {})),
                "event_messages": self._build_logs_for_tick(entry.tick, events_by_tick.get(entry.tick, []), low_frame),
                "actual_events": [
                    {
                        "tick": int(row.get("tick") or 0),
                        "title": str((row.get("payload") or {}).get("title") or row.get("topic") or ""),
                        "severity": str(((row.get("render_hints") or {}).get("severity")) or "info"),
                    }
                    for row in events_by_tick.get(entry.tick, [])
                ],
            }
            timeline_frames.append(frame)

        return {
            "meta": {
                "episode_id": str(manifest.get("episode_id") or ""),
                "frame_count": len(timeline_frames),
                "playback_duration_s": playback_duration_s,
                "frame_interval_s": frame_interval_s,
                "tick_stride": 5,
                "hold_final_s": hold_final_s,
                "id_legend": "a### == drone_demo_a_###",
                "interesting_uav_ids": self.interesting_uav_ids,
            },
            "frames": timeline_frames,
        }

    def build_assets(
        self,
        *,
        low_dir: Path,
        high_dir: Path,
        episode_dir: Path,
        output_dir: Path,
        playback_duration_s: float,
        gif_max_width: int,
        gif_max_colors: int,
        hold_final_s: float,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        low_entries = self.sorted_frame_entries(low_dir)
        high_entries = self.sorted_frame_entries(high_dir)
        if len(low_entries) != len(high_entries):
            raise RuntimeError(f"low/high frame counts differ: low={len(low_entries)} high={len(high_entries)}")
        if not low_entries:
            raise RuntimeError("No frames found for GIF generation.")

        frame_duration_ms = max(1, round(playback_duration_s * 1000.0 / len(low_entries)))
        final_hold_ms = max(0, round(hold_final_s * 1000.0))

        gif_dir = output_dir / "gifs"
        timeline_dir = output_dir / "timeline"
        gif_dir.mkdir(parents=True, exist_ok=True)
        timeline_dir.mkdir(parents=True, exist_ok=True)

        low_gif = self.build_gif(
            low_entries,
            gif_dir / "low_view_2x.gif",
            frame_duration_ms=frame_duration_ms,
            final_hold_ms=final_hold_ms,
            max_width=int(gif_max_width),
            max_colors=int(gif_max_colors),
        )
        high_gif = self.build_gif(
            high_entries,
            gif_dir / "high_view_2x.gif",
            frame_duration_ms=frame_duration_ms,
            final_hold_ms=final_hold_ms,
            max_width=int(gif_max_width),
            max_colors=int(gif_max_colors),
        )

        timeline = self.build_timeline(low_entries, episode_dir, playback_duration_s=playback_duration_s, hold_final_s=hold_final_s)
        timeline_path = timeline_dir / "multiview_timeline.json"
        write_json(timeline_path, timeline)

        manifest = {
            "output_dir": str(output_dir),
            "low_gif": low_gif,
            "high_gif": high_gif,
            "timeline_path": str(timeline_path),
            "frame_count": len(low_entries),
            "frame_duration_ms": frame_duration_ms,
            "playback_duration_s": playback_duration_s,
        }
        manifest_path = output_dir / "manifest.json"
        write_json(manifest_path, manifest)
        return manifest
