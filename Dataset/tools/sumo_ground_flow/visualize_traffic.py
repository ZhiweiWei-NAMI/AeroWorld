"""Visualize SUMO topology and sampled vehicle positions as PNG frames and GIF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

from .incident_plan import DEFAULT_SUMO_NET_XML, vehicle_class_for_edge
from .planner import SumoGroundFlowPlanner
from .run_traffic import DEFAULT_OUTPUT_DIR

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_EPISODE_TICK_START = 0
EXPECTED_EPISODE_TICK_END = 900
EXPECTED_EPISODE_TICK_STRIDE = 5
EXPECTED_EPISODE_GIF_FRAME_COUNT = 181


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _bbox_from_list(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, xmax, ymin, ymax


def _episode_spatial_bounds(
    *,
    manifest: dict[str, Any],
    output_dir: Path,
    mode: str,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    explicit_plan = dict(manifest.get("explicit_vehicle_plan") or {})
    plan_path = Path(str(explicit_plan.get("path") or ""))
    if not plan_path.is_absolute():
        candidates = [ROOT / plan_path, output_dir / plan_path]
        plan_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not plan_path.exists():
        raise FileNotFoundError(f"Explicit vehicle plan not found for ROI bounds: {plan_path}")
    plan = _load_json(plan_path)
    spatial_scope = dict(plan.get("spatial_scope") or {})
    raw_bbox_key = "expanded_bbox_enu_m" if mode == "episode-expanded-roi" else "bbox_enu_m"
    bounds = _bbox_from_list(spatial_scope.get(raw_bbox_key))
    if bounds is None:
        raise ValueError(f"{plan_path} spatial_scope.{raw_bbox_key} is missing or invalid")
    overlay = {
        "source_plan": str(plan_path),
        "roi_id": spatial_scope.get("roi_id"),
        "bbox_enu_m": spatial_scope.get("bbox_enu_m"),
        "expanded_bbox_enu_m": spatial_scope.get("expanded_bbox_enu_m"),
        "expanded_boundary_padding_m": spatial_scope.get("expanded_boundary_padding_m"),
    }
    return bounds, overlay


def _iter_frames(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _selected_frames(
    path: Path,
    frame_stride: int,
    *,
    min_sim_time_s: float | None = None,
    max_sim_time_s: float | None = None,
) -> list[dict[str, Any]]:
    stride = max(1, int(frame_stride))
    rows = [
        frame
        for frame in _iter_frames(path)
        if (min_sim_time_s is None or float(frame.get("sim_time_s") or 0.0) >= min_sim_time_s - 1e-9)
        and (max_sim_time_s is None or float(frame.get("sim_time_s") or 0.0) <= max_sim_time_s + 1e-9)
    ]
    selected = [frame for index, frame in enumerate(rows) if index % stride == 0]
    if rows and selected[-1].get("sim_time_s") != rows[-1].get("sim_time_s"):
        selected.append(rows[-1])
    return selected


def _episode_tick(frame: dict[str, Any], *, capture_start_s: float, tick_hz: float) -> int:
    return int(round((float(frame.get("sim_time_s") or 0.0) - float(capture_start_s)) * float(tick_hz)))


def _semantic_metadata_text(vehicle: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "semantic_role",
        "semantic_type",
        "semantic_vehicle_type",
        "semantic_event_type",
        "vehicle_type",
        "type",
        "semantic_metadata",
        "metadata",
    ):
        value = vehicle.get(key)
        if isinstance(value, (dict, list)):
            values.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        elif value is not None:
            values.append(str(value))
    return " ".join(values).lower()


def _has_emergency_intent(vehicle: dict[str, Any]) -> bool:
    semantic_text = _semantic_metadata_text(vehicle)
    return "ambulance" in semantic_text or "emergency" in semantic_text


def _vehicle_topology_segments(planner: SumoGroundFlowPlanner) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    for edge in planner.edges.values():
        if not vehicle_class_for_edge(edge):
            continue
        if len(edge.shape_xy) >= 2:
            segments.append([(float(x), float(y)) for x, y in edge.shape_xy])
    return segments


def _plot_bounds(
    topo_segments: Sequence[Sequence[tuple[float, float]]],
    frames: Sequence[dict[str, Any]],
    margin_m: float = 40.0,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for segment in topo_segments:
        for x, y in segment:
            xs.append(float(x))
            ys.append(float(y))
    for frame in frames:
        for vehicle in frame.get("vehicles") or []:
            pos = vehicle.get("truth_position_enu_m") or []
            if len(pos) >= 2:
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
    if not xs or not ys:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs) - margin_m, max(xs) + margin_m, min(ys) - margin_m, max(ys) + margin_m


def _draw_frame(
    *,
    frame: dict[str, Any],
    topo_segments: Sequence[Sequence[tuple[float, float]]],
    bounds: tuple[float, float, float, float],
    overlay: dict[str, Any] | None,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.set_facecolor("#f8f8f4")
    ax.add_collection(LineCollection(topo_segments, colors="#9b9b95", linewidths=0.35, alpha=0.75, zorder=1))

    bg_x: list[float] = []
    bg_y: list[float] = []
    controlled_x: list[float] = []
    controlled_y: list[float] = []
    emergency_x: list[float] = []
    emergency_y: list[float] = []
    signal_x: list[float] = []
    signal_y: list[float] = []
    stopped_x: list[float] = []
    stopped_y: list[float] = []
    semantic_x: list[float] = []
    semantic_y: list[float] = []
    semantic_labels: list[str] = []
    for vehicle in frame.get("vehicles") or []:
        pos = vehicle.get("truth_position_enu_m") or []
        if len(pos) < 2:
            continue
        x = float(pos[0])
        y = float(pos[1])
        signals = int(vehicle.get("signals") or 0)
        speed = float(vehicle.get("speed_mps") or 0.0)
        role = str(vehicle.get("control_role") or "")
        emergency_intent = _has_emergency_intent(vehicle)
        if role == "semantic_event" or bool(vehicle.get("semantic_vehicle")):
            semantic_x.append(x)
            semantic_y.append(y)
            semantic_labels.append(str(vehicle.get("source_entity_id") or vehicle.get("vehicle_id") or "semantic"))
        elif emergency_intent:
            emergency_x.append(x)
            emergency_y.append(y)
        elif signals and signals != 0:
            signal_x.append(x)
            signal_y.append(y)
        elif role == "incident_controlled":
            controlled_x.append(x)
            controlled_y.append(y)
        elif speed <= 0.2:
            stopped_x.append(x)
            stopped_y.append(y)
        else:
            bg_x.append(x)
            bg_y.append(y)

    if bg_x:
        ax.scatter(bg_x, bg_y, s=14, marker="o", c="#1f77b4", edgecolors="none", alpha=0.82, label="background vehicle", zorder=4)
    if controlled_x:
        ax.scatter(controlled_x, controlled_y, s=36, marker="^", c="#d62728", edgecolors="white", linewidths=0.35, label="incident vehicle", zorder=5)
    if emergency_x:
        ax.scatter(emergency_x, emergency_y, s=46, marker="*", c="#ff8c00", edgecolors="#5a2c00", linewidths=0.35, label="emergency vehicle", zorder=6)
    if signal_x:
        ax.scatter(signal_x, signal_y, s=20, marker="x", c="#ff8c00", linewidths=0.55, label="turn/brake signal", zorder=5)
    if stopped_x:
        ax.scatter(stopped_x, stopped_y, s=24, marker="s", c="#4d4d4d", edgecolors="white", linewidths=0.3, label="stopped", zorder=5)
    if semantic_x:
        ax.scatter(semantic_x, semantic_y, s=70, marker="D", c="#e31a1c", edgecolors="white", linewidths=0.7, label="semantic vehicle", zorder=8)
        for x, y, label in zip(semantic_x, semantic_y, semantic_labels):
            short_label = label.replace("_l4_10_v2", "").replace("__seed02", "")
            ax.annotate(
                short_label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                color="#7f0000",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#e31a1c", "lw": 0.35, "alpha": 0.82},
                zorder=9,
            )

    active = frame.get("active_incidents") or []
    if active:
        ax.scatter(
            [float(item["anchor"]["truth_position_enu_m"][0]) for item in active if item.get("anchor")],
            [float(item["anchor"]["truth_position_enu_m"][1]) for item in active if item.get("anchor")],
            s=55,
            marker="x",
            c="#7b3294",
            linewidths=1.2,
            label="active incident anchor",
            zorder=7,
        )

    if overlay:
        bbox = _bbox_from_list(overlay.get("bbox_enu_m"))
        expanded_bbox = _bbox_from_list(overlay.get("expanded_bbox_enu_m"))
        for item, color, label, linewidth in (
            (expanded_bbox, "#6a3d9a", "extended boundary", 1.0),
            (bbox, "#33a02c", "ROI boundary", 1.25),
        ):
            if item is None:
                continue
            bxmin, bxmax, bymin, bymax = item
            ax.plot(
                [bxmin, bxmax, bxmax, bxmin, bxmin],
                [bymin, bymin, bymax, bymax, bymin],
                color=color,
                linewidth=linewidth,
                linestyle="--" if label.startswith("extended") else "-",
                label=label,
                zorder=3,
            )

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UE truth X / m")
    ax.set_ylabel("UE truth Y / m")
    sim_time_s = float(frame.get("sim_time_s") or 0.0)
    title_tick = f"t={sim_time_s:.1f}s"
    if overlay:
        capture_start_s = overlay.get("capture_start_s")
        tick_hz = overlay.get("tick_hz")
        if capture_start_s is not None and tick_hz:
            episode_tick = int(round((sim_time_s - float(capture_start_s)) * float(tick_hz)))
            title_tick = f"episode_tick={episode_tick}"
    title = f"SUMO Donghu Traffic | {title_tick} | vehicles={len(frame.get('vehicles') or [])}"
    if active:
        title += f" | active incidents={len(active)}"
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="#d8d8d0", linewidth=0.35, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=7, frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_gif(frame_paths: Sequence[Path], gif_path: Path, duration_ms: int) -> None:
    if not frame_paths:
        raise ValueError("No frame paths to write GIF")
    images = []
    for path in frame_paths:
        last_error: OSError | None = None
        for _ in range(10):
            try:
                images.append(Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE))
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)
        if last_error is not None:
            raise last_error
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=True,
    )
    for image in images:
        image.close()


def visualize_traffic(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    net_xml: Path = DEFAULT_SUMO_NET_XML,
    frame_stride: int = 5,
    dpi: int = 120,
    gif_duration_ms: int = 120,
    bounds_mode: str = "full",
    output_suffix: str = "",
    episode_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = _load_json(output_dir / "sumo_traffic_manifest.json")
    frames_path = output_dir / "sumo_traffic_frames.jsonl"
    capture_start_s = float(manifest.get("capture_start_s") or manifest.get("warmup_s") or 0.0)
    capture_duration_s = float(manifest.get("capture_duration_s") or 0.0)
    tick_hz = float(manifest.get("tick_hz") or 0.0)
    selected = _selected_frames(
        frames_path,
        frame_stride,
        min_sim_time_s=capture_start_s if episode_only else None,
        max_sim_time_s=(capture_start_s + capture_duration_s) if episode_only and capture_duration_s > 0.0 else None,
    )
    episode_ticks: list[int] = []
    if episode_only:
        if tick_hz <= 0.0:
            raise ValueError(f"tick_hz must be positive for episode-only GIFs, got {tick_hz}")
        episode_ticks = [
            _episode_tick(frame, capture_start_s=capture_start_s, tick_hz=tick_hz)
            for frame in selected
        ]
        expected_ticks = list(
            range(
                EXPECTED_EPISODE_TICK_START,
                EXPECTED_EPISODE_TICK_END + 1,
                EXPECTED_EPISODE_TICK_STRIDE,
            )
        )
        if episode_ticks != expected_ticks:
            raise ValueError(
                "episode-only GIF must contain formal ticks 0..900 every 5 ticks "
                f"({EXPECTED_EPISODE_GIF_FRAME_COUNT} frames); got count={len(episode_ticks)}, "
                f"first={episode_ticks[0] if episode_ticks else None}, "
                f"last={episode_ticks[-1] if episode_ticks else None}, "
                f"stride={int(frame_stride)}"
            )
    planner = SumoGroundFlowPlanner(net_xml)
    topo_segments = _vehicle_topology_segments(planner)
    overlay: dict[str, Any] | None = None
    if bounds_mode == "full":
        bounds = _plot_bounds(topo_segments, selected)
    elif bounds_mode in {"episode-roi", "episode-expanded-roi"}:
        bounds, overlay = _episode_spatial_bounds(
            manifest=manifest,
            output_dir=output_dir,
            mode=bounds_mode,
        )
        if episode_only:
            overlay["capture_start_s"] = capture_start_s
            overlay["capture_duration_s"] = capture_duration_s
            overlay["tick_hz"] = tick_hz
    else:
        raise ValueError(f"Unsupported bounds mode: {bounds_mode}")
    suffix = str(output_suffix or "").strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    frames_dir = output_dir / f"visualization_frames{suffix}"
    if overwrite and frames_dir.exists():
        for path in frames_dir.glob("sumo_topo_vehicles_*.png"):
            path.unlink()
    frame_paths: list[Path] = []
    for index, frame in enumerate(selected):
        path = frames_dir / f"sumo_topo_vehicles_{index:04d}.png"
        _draw_frame(frame=frame, topo_segments=topo_segments, bounds=bounds, overlay=overlay, output_path=path, dpi=dpi)
        frame_paths.append(path)
    gif_path = output_dir / f"sumo_topo_vehicles{suffix}.gif"
    if overwrite and gif_path.exists():
        gif_path.unlink()
    _write_gif(frame_paths, gif_path, gif_duration_ms)
    summary = {
        "schema_name": "sumo_traffic_visualization",
        "schema_version": "v1",
        "source_manifest": str(output_dir / "sumo_traffic_manifest.json"),
        "source_frames": str(frames_path),
        "source_sample_period_s": manifest.get("sample_period_s"),
        "visualized_frame_stride": int(frame_stride),
        "visualized_time_step_s": round(float(manifest.get("sample_period_s") or 0.0) * int(frame_stride), 6),
        "episode_only": bool(episode_only),
        "episode_time_filter_s": [capture_start_s, capture_start_s + capture_duration_s] if episode_only else [],
        "episode_tick_first": episode_ticks[0] if episode_ticks else None,
        "episode_tick_last": episode_ticks[-1] if episode_ticks else None,
        "episode_tick_stride": int(frame_stride) if episode_only else None,
        "episode_tick_count": len(episode_ticks) if episode_only else 0,
        "expected_episode_tick_first": EXPECTED_EPISODE_TICK_START if episode_only else None,
        "expected_episode_tick_last": EXPECTED_EPISODE_TICK_END if episode_only else None,
        "expected_episode_tick_stride": EXPECTED_EPISODE_TICK_STRIDE if episode_only else None,
        "expected_episode_gif_frame_count": EXPECTED_EPISODE_GIF_FRAME_COUNT if episode_only else None,
        "bounds_mode": bounds_mode,
        "bounds_enu_m": [round(float(value), 6) for value in bounds],
        "overlay": overlay or {},
        "png_frame_count": len(frame_paths),
        "gif_path": str(gif_path),
        "frames_dir": str(frames_dir),
        "topology_segment_count": len(topo_segments),
    }
    summary_path = output_dir / f"sumo_traffic_visualization{suffix}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SUMO topology and vehicle markers as PNG frames plus GIF.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--net-xml", type=Path, default=DEFAULT_SUMO_NET_XML)
    parser.add_argument("--frame-stride", type=int, default=5, help="Use every N exported 0.5s SUMO frame.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument(
        "--bounds-mode",
        choices=("full", "episode-roi", "episode-expanded-roi"),
        default="full",
        help="Visualization bounds. episode-expanded-roi uses spatial_scope.expanded_bbox_enu_m from the explicit plan.",
    )
    parser.add_argument("--output-suffix", default="", help="Suffix for frames directory, GIF, and summary JSON.")
    parser.add_argument("--episode-only", action="store_true", help="Visualize only the formal post-warmup episode window.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = visualize_traffic(
        output_dir=args.output_dir,
        net_xml=args.net_xml,
        frame_stride=args.frame_stride,
        dpi=args.dpi,
        gif_duration_ms=args.gif_duration_ms,
        bounds_mode=args.bounds_mode,
        output_suffix=args.output_suffix,
        episode_only=bool(args.episode_only),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
