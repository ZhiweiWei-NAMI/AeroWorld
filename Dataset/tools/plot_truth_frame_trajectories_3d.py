"""Plot render-ready truth-frame P/V/U trajectories in x-y-time space.

The visual contract for final dataset review is intentionally simple:
truth-frame positions are shown from a top-down ENU x/y perspective while the
3D z axis is simulation time.  Markers encode actor class:
P=pedestrian, V=vehicle, U=UAV.  Line colors are per-track so visual drift,
static actors, and missing background flow are easy to spot.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes"
DEFAULT_OUTPUT_DIR = Path("F:/AeroWorldTrajectoryPlots/truth_frame_3d_latest")

CATEGORY_STYLE = {
    "P": {"marker": "o", "label": "P pedestrian"},
    "V": {"marker": "s", "label": "V vehicle"},
    "U": {"marker": "^", "label": "U UAV"},
}


@dataclass
class Track:
    entity_id: str
    category: str
    points: list[tuple[float, float, float, int]] = field(default_factory=list)

    def add(self, x: float, y: float, time_s: float, tick: int) -> None:
        self.points.append((x, y, time_s, tick))

    @property
    def xy_span_m(self) -> float:
        if len(self.points) < 2:
            return 0.0
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    @property
    def first_move_tick(self) -> int | None:
        if len(self.points) < 2:
            return None
        x0, y0 = self.points[0][0], self.points[0][1]
        for x, y, _time_s, tick in self.points[1:]:
            if math.hypot(x - x0, y - y0) >= 0.5:
                return tick
        return None

    def first_continuous_move_tick(self, sample_step_ticks: int) -> int | None:
        if len(self.points) < 2:
            return None
        max_gap = max(1, int(sample_step_ticks * 1.5))
        x0, y0 = self.points[0][0], self.points[0][1]
        previous_tick = self.points[0][3]
        for x, y, _time_s, tick in self.points[1:]:
            if tick - previous_tick > max_gap:
                return None
            if math.hypot(x - x0, y - y0) >= 0.5:
                return tick
            previous_tick = tick
        return None

    @property
    def first_tick(self) -> int | None:
        if not self.points:
            return None
        return self.points[0][3]


def _category_from_entity(entity: dict[str, Any]) -> str | None:
    for value in (entity.get("entity_category"), entity.get("label_class")):
        text = str(value or "").lower()
        if text == "pedestrian":
            return "P"
        if text == "vehicle":
            return "V"
        if text == "uav":
            return "U"

    entity_category = str(entity.get("entity_category") or "").lower()
    if entity_category in {"airspace_corridor", "facility", "prop", "trigger", "airspace_constraint"}:
        return None

    typed_values = [entity.get("entity_type"), entity.get("entity_kind")]
    text = " ".join(str(v).lower() for v in typed_values if v)
    if "corridor" in text or "facility" in text or "trigger" in text:
        return None
    if text.startswith("pedestrian") or ".pedestrian" in text:
        return "P"
    if text.startswith("vehicle") or ".vehicle" in text or ".car" in text:
        return "V"
    if text.startswith("uav") or ".uav" in text or text.startswith("drone"):
        return "U"
    return None


def _position_from_entity(entity: dict[str, Any]) -> tuple[float, float] | None:
    pose = entity.get("truth_pose") or entity.get("pose") or {}
    position = pose.get("position_enu_m") or pose.get("position")
    if not isinstance(position, list) or len(position) < 2:
        return None
    try:
        return float(position[0]), float(position[1])
    except (TypeError, ValueError):
        return None


def _iter_sampled_jsonl_lines(path: Path, sample_step_ticks: int):
    step = max(1, int(sample_step_ticks))
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            start = 0
            line_index = 0
            last_index: int | None = None
            last_line: bytes | None = None
            emitted_last_sample = False
            size = len(mapped)
            while start <= size:
                end = mapped.find(b"\n", start)
                if end < 0:
                    end = size
                    is_final = True
                else:
                    is_final = False
                line = bytes(mapped[start:end]).strip()
                if line:
                    last_index = line_index
                    last_line = line
                    if line_index % step == 0:
                        emitted_last_sample = True
                        yield line_index, line.decode("utf-8-sig" if line_index == 0 else "utf-8")
                    else:
                        emitted_last_sample = False
                if is_final:
                    break
                start = end + 1
                line_index += 1
            if last_line is not None and last_index is not None and not emitted_last_sample:
                yield last_index, last_line.decode("utf-8-sig" if last_index == 0 else "utf-8")


def _load_sampled_tracks(episode_dir: Path, sample_step_ticks: int) -> tuple[dict[str, Track], dict[str, Any]]:
    tracks: dict[str, Track] = {}
    truth_path = episode_dir / "truth_frames.jsonl"
    meta: dict[str, Any] = {
        "episode": episode_dir.name,
        "capture_boundary_id": "",
        "uav_crosses_boundary": None,
        "inspect_observes_boundary": None,
        "sampled_frames": 0,
        "first_tick": None,
        "last_tick": None,
    }
    for line_index, line in _iter_sampled_jsonl_lines(truth_path, sample_step_ticks):
        frame = json.loads(line)
        tick = int(frame.get("tick") if frame.get("tick") is not None else frame.get("frame_seq", line_index))
        time_s = float(frame.get("sim_time_s") if frame.get("sim_time_s") is not None else tick / 10.0)
        _read_frame(frame, tracks, time_s, tick)
        _update_meta(meta, frame, tick)

    return tracks, meta


def _read_frame(frame: dict[str, Any], tracks: dict[str, Track], time_s: float, tick: int) -> None:
    for entity in frame.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        category = _category_from_entity(entity)
        if category is None:
            continue
        entity_id = str(entity.get("entity_id") or entity.get("id") or "")
        if not entity_id:
            continue
        position = _position_from_entity(entity)
        if position is None:
            continue
        track = tracks.setdefault(entity_id, Track(entity_id=entity_id, category=category))
        track.add(position[0], position[1], time_s, tick)


def _update_meta(meta: dict[str, Any], frame: dict[str, Any], tick: int) -> None:
    meta["sampled_frames"] += 1
    meta["first_tick"] = tick if meta["first_tick"] is None else min(int(meta["first_tick"]), tick)
    meta["last_tick"] = tick if meta["last_tick"] is None else max(int(meta["last_tick"]), tick)
    if not meta["capture_boundary_id"]:
        meta["capture_boundary_id"] = frame.get("capture_boundary_id") or ""
    if meta["uav_crosses_boundary"] is not True:
        meta["uav_crosses_boundary"] = bool(frame.get("uav_crosses_boundary"))
    if meta["inspect_observes_boundary"] is not True:
        meta["inspect_observes_boundary"] = bool(frame.get("inspect_observes_boundary"))


def _plot_episode(episode_dir: Path, output_path: Path, sample_step_ticks: int) -> dict[str, Any]:
    tracks, meta = _load_sampled_tracks(episode_dir, sample_step_ticks)
    ordered_tracks = sorted(tracks.values(), key=lambda t: (t.category, t.entity_id))

    fig = plt.figure(figsize=(11.5, 8.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        f"{episode_dir.name} | 50 tick samples | z=time",
        fontsize=10,
        pad=12,
    )
    ax.set_xlabel("x ENU m")
    ax.set_ylabel("y ENU m")
    ax.set_zlabel("time s")
    ax.view_init(elev=24, azim=-58)

    cmap = plt.get_cmap("tab20")
    for index, track in enumerate(ordered_tracks):
        if not track.points:
            continue
        color = cmap(index % cmap.N)
        xs = [p[0] for p in track.points]
        ys = [p[1] for p in track.points]
        ts = [p[2] for p in track.points]
        style = CATEGORY_STYLE[track.category]
        ax.plot(xs, ys, ts, color=color, linewidth=1.25, alpha=0.86)
        ax.scatter(
            xs,
            ys,
            ts,
            color=color,
            marker=style["marker"],
            s=16 if track.category != "U" else 22,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.9,
        )
        ax.scatter([xs[0]], [ys[0]], [ts[0]], color=color, marker="x", s=28, linewidths=1.0)

    counts = defaultdict(int)
    static_tracks: list[dict[str, Any]] = []
    delayed_tracks: list[dict[str, Any]] = []
    for track in ordered_tracks:
        counts[track.category] += 1
        span = track.xy_span_m
        first_move_tick = track.first_continuous_move_tick(sample_step_ticks)
        first_tick = track.first_tick
        if len(track.points) >= 3 and span < 0.75:
            static_tracks.append(
                {
                    "entity": track.entity_id,
                    "category": track.category,
                    "xy_span_m": round(span, 3),
                    "point_count": len(track.points),
                }
            )
        if (
            first_move_tick is not None
            and first_tick is not None
            and first_move_tick - first_tick > sample_step_ticks * 2
        ):
            delayed_tracks.append(
                {
                    "entity": track.entity_id,
                    "category": track.category,
                    "xy_span_m": round(span, 3),
                    "first_tick": first_tick,
                    "first_move_tick": first_move_tick,
                    "stationary_ticks_after_appearance": first_move_tick - first_tick,
                }
            )

    legend_handles = [
        Line2D([0], [0], marker=style["marker"], color="black", label=style["label"], linestyle="None")
        for style in CATEGORY_STYLE.values()
    ]
    legend_handles.append(Line2D([0], [0], marker="x", color="black", label="track start", linestyle="None"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    footer = (
        f"P={counts['P']} V={counts['V']} U={counts['U']} | "
        f"boundary={meta['capture_boundary_id'] or 'none'} | "
        f"uav_cross={meta['uav_crosses_boundary']} inspect_fov={meta['inspect_observes_boundary']}"
    )
    fig.text(0.02, 0.02, footer, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    return {
        **meta,
        "plot": str(output_path),
        "track_count": len(ordered_tracks),
        "pedestrian_tracks": counts["P"],
        "vehicle_tracks": counts["V"],
        "uav_tracks": counts["U"],
        "static_track_count": len(static_tracks),
        "static_tracks_first_20": static_tracks[:20],
        "delayed_motion_track_count": len(delayed_tracks),
        "delayed_motion_tracks_first_20": delayed_tracks[:20],
    }


def _make_contact_sheets(
    episode_summaries: list[dict[str, Any]],
    output_dir: Path,
    images_per_sheet: int = 30,
    columns: int = 5,
    filename_prefix: str = "contact_sheet",
) -> list[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []

    contact_sheets: list[str] = []
    thumb_w, thumb_h = 360, 270
    rows = math.ceil(images_per_sheet / columns)
    for sheet_index, start in enumerate(range(0, len(episode_summaries), images_per_sheet), start=1):
        chunk = episode_summaries[start : start + images_per_sheet]
        sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 28)), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, summary in enumerate(chunk):
            path = Path(summary["plot"])
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (idx % columns) * thumb_w
            y = (idx // columns) * (thumb_h + 28)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb_h + 4), str(summary["episode"])[:52], fill=(0, 0, 0))
        sheet_path = output_dir / f"{filename_prefix}_{sheet_index:02d}.png"
        sheet.save(sheet_path)
        contact_sheets.append(str(sheet_path))
    return contact_sheets


def _make_key_sheet(episode_summaries: list[dict[str, Any]], output_dir: Path) -> str | None:
    key_prefixes = (
        "L1-1_v1",
        "L2-4_v1",
        "L3-1_v1",
        "L4-3_v2",
        "L4-4_v1",
        "L4-9_v1",
        "L5-1_v1",
        "X1_rain",
        "X3_pedestrian",
        "X4_fog",
        "X5_comm",
        "X6_crowd",
    )
    selected = [
        item
        for item in episode_summaries
        if any(str(item["episode"]).startswith(prefix) for prefix in key_prefixes)
    ]
    if not selected:
        return None
    sheets = _make_contact_sheets(
        selected,
        output_dir,
        images_per_sheet=36,
        columns=6,
        filename_prefix="key_episode_contact_sheet_part",
    )
    if not sheets:
        return None
    key_path = output_dir / "key_episode_contact_sheet.png"
    shutil.copyfile(sheets[0], key_path)
    for sheet in sheets:
        Path(sheet).unlink(missing_ok=True)
    return str(key_path)


def _parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for name in value.split(","):
            stripped = name.strip()
            if stripped:
                names.append(stripped)
    return names


def _select_episode_dirs(
    input_root: Path,
    episode_names: list[str],
    start_index: int,
    limit: int | None,
) -> list[Path]:
    if start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if limit is not None and limit < 0:
        raise SystemExit("--limit must be non-negative")

    all_episode_dirs = sorted(path for path in input_root.iterdir() if (path / "truth_frames.jsonl").exists())
    if episode_names:
        episode_dirs_by_name = {path.name: path for path in all_episode_dirs}
        missing_names = [name for name in episode_names if name not in episode_dirs_by_name]
        if missing_names:
            raise SystemExit(f"Unknown --episode value(s): {', '.join(missing_names)}")
        selected_episode_dirs = [episode_dirs_by_name[name] for name in episode_names]
    else:
        selected_episode_dirs = all_episode_dirs

    selected_episode_dirs = selected_episode_dirs[start_index:]
    if limit is not None:
        selected_episode_dirs = selected_episode_dirs[:limit]
    if not selected_episode_dirs:
        raise SystemExit("No episodes selected")
    return selected_episode_dirs


def _remove_stale_contact_sheets(output_dir: Path) -> None:
    for pattern in (
        "contact_sheet_*.png",
        "key_episode_contact_sheet.png",
        "key_episode_contact_sheet_part_*.png",
    ):
        for stale in output_dir.glob(pattern):
            stale.unlink()


def plot_all(
    input_root: Path,
    output_dir: Path,
    sample_step_ticks: int,
    episode_names: list[str] | None = None,
    start_index: int = 0,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    requested_episode_names = episode_names or []
    episode_dirs = _select_episode_dirs(input_root, requested_episode_names, start_index, limit)
    episode_plot_dir = output_dir / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_plot_dir.mkdir(parents=True, exist_ok=True)
    for stale in episode_plot_dir.glob("*.png"):
        stale.unlink()
    _remove_stale_contact_sheets(output_dir)

    summaries: list[dict[str, Any]] = []
    worker_count = max(1, int(workers or 1))
    if worker_count == 1 or len(episode_dirs) <= 1:
        for index, episode_dir in enumerate(episode_dirs, start=1):
            output_path = episode_plot_dir / f"{episode_dir.name}.png"
            summary = _plot_episode(episode_dir, output_path, sample_step_ticks)
            summary["index"] = index
            summaries.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(_plot_episode, episode_dir, episode_plot_dir / f"{episode_dir.name}.png", sample_step_ticks): index
                for index, episode_dir in enumerate(episode_dirs, start=1)
            }
            for future in as_completed(future_to_index):
                summary = future.result()
                summary["index"] = future_to_index[future]
                summaries.append(summary)
        summaries.sort(key=lambda item: int(item["index"]))

    contact_sheets = _make_contact_sheets(summaries, output_dir)
    key_sheet = _make_key_sheet(summaries, output_dir)
    all_plans_true = all(
        item["uav_crosses_boundary"] is True and item["inspect_observes_boundary"] is True
        for item in summaries
    )
    summary = {
        "source": str(input_root),
        "output": str(output_dir),
        "episode_plot_dir": str(episode_plot_dir),
        "episode_count": len(summaries),
        "sample_step_ticks": sample_step_ticks,
        "episode_filter": requested_episode_names,
        "start_index": start_index,
        "limit": limit,
        "workers": worker_count,
        "selected_episodes": [path.name for path in episode_dirs],
        "contact_sheets": contact_sheets,
        "key_contact_sheet": key_sheet,
        "all_uav_boundary_and_inspect_flags_true": all_plans_true,
        "total_tracks": sum(int(item["track_count"]) for item in summaries),
        "total_pedestrian_tracks": sum(int(item["pedestrian_tracks"]) for item in summaries),
        "total_vehicle_tracks": sum(int(item["vehicle_tracks"]) for item in summaries),
        "total_uav_tracks": sum(int(item["uav_tracks"]) for item in summaries),
        "static_track_episode_count": sum(1 for item in summaries if item["static_track_count"] > 0),
        "delayed_motion_episode_count": sum(1 for item in summaries if item["delayed_motion_track_count"] > 0),
        "episodes": summaries,
    }
    summary_path = output_dir / "truth_frame_3d_plot_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 3D x-y-time P/V/U trajectories from truth frames.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-step-ticks", type=int, default=50)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Episode name to plot; may be repeated or comma-separated. Defaults to all episodes.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based start offset after optional --episode filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of selected episodes to plot.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel plotting workers. Keep modest because each worker owns a matplotlib process.",
    )
    args = parser.parse_args()

    if args.sample_step_ticks <= 0:
        raise SystemExit("--sample-step-ticks must be positive")
    summary = plot_all(
        args.input_root,
        args.output_dir,
        args.sample_step_ticks,
        episode_names=_parse_episode_names(args.episode),
        start_index=args.start_index,
        limit=args.limit,
        workers=args.workers,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "episodes"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
