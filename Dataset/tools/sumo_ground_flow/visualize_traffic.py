"""Visualize SUMO topology and sampled vehicle positions as PNG frames and GIF."""

from __future__ import annotations

import argparse
import json
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _iter_frames(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _selected_frames(path: Path, frame_stride: int) -> list[dict[str, Any]]:
    stride = max(1, int(frame_stride))
    rows = list(_iter_frames(path))
    selected = [frame for index, frame in enumerate(rows) if index % stride == 0]
    if rows and selected[-1].get("sim_time_s") != rows[-1].get("sim_time_s"):
        selected.append(rows[-1])
    return selected


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
    stopped_x: list[float] = []
    stopped_y: list[float] = []
    for vehicle in frame.get("vehicles") or []:
        pos = vehicle.get("truth_position_enu_m") or []
        if len(pos) < 2:
            continue
        x = float(pos[0])
        y = float(pos[1])
        signals = int(vehicle.get("signals") or 0)
        speed = float(vehicle.get("speed_mps") or 0.0)
        role = str(vehicle.get("control_role") or "")
        if signals and signals != 0:
            emergency_x.append(x)
            emergency_y.append(y)
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
        ax.scatter(emergency_x, emergency_y, s=46, marker="*", c="#ff8c00", edgecolors="#5a2c00", linewidths=0.35, label="signal/emergency", zorder=6)
    if stopped_x:
        ax.scatter(stopped_x, stopped_y, s=24, marker="s", c="#4d4d4d", edgecolors="white", linewidths=0.3, label="stopped", zorder=5)

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

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UE truth X / m")
    ax.set_ylabel("UE truth Y / m")
    title = f"SUMO Donghu Traffic | t={float(frame.get('sim_time_s') or 0.0):.1f}s | vehicles={len(frame.get('vehicles') or [])}"
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
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
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
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = _load_json(output_dir / "sumo_traffic_manifest.json")
    frames_path = output_dir / "sumo_traffic_frames.jsonl"
    selected = _selected_frames(frames_path, frame_stride)
    planner = SumoGroundFlowPlanner(net_xml)
    topo_segments = _vehicle_topology_segments(planner)
    bounds = _plot_bounds(topo_segments, selected)
    frames_dir = output_dir / "visualization_frames"
    if overwrite and frames_dir.exists():
        for path in frames_dir.glob("sumo_topo_vehicles_*.png"):
            path.unlink()
    frame_paths: list[Path] = []
    for index, frame in enumerate(selected):
        path = frames_dir / f"sumo_topo_vehicles_{index:04d}.png"
        _draw_frame(frame=frame, topo_segments=topo_segments, bounds=bounds, output_path=path, dpi=dpi)
        frame_paths.append(path)
    gif_path = output_dir / "sumo_topo_vehicles.gif"
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
        "png_frame_count": len(frame_paths),
        "gif_path": str(gif_path),
        "frames_dir": str(frames_dir),
        "topology_segment_count": len(topo_segments),
    }
    (output_dir / "sumo_traffic_visualization.json").write_text(
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
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
