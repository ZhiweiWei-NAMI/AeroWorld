"""Render Donghu UAV global flow with SUMO topology and vehicle context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "Dataset" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

from Dataset.tools.sumo_ground_flow.incident_plan import DEFAULT_SUMO_NET_XML, vehicle_class_for_edge  # noqa: E402
from Dataset.tools.sumo_ground_flow.planner import SumoGroundFlowPlanner  # noqa: E402
from Dataset.tools.uav_global_flow.generate_uav_flow import DEFAULT_OUTPUT_DIR, DEFAULT_SUMO_OUTPUT_DIR  # noqa: E402


MISSION_COLORS = {
    "pad_patrol": "#2ca02c",
    "intersection_inspect": "#9467bd",
    "edge_compute_relay": "#17becf",
    "logistics_delivery": "#ff7f0e",
    "infrastructure_inspection": "#8c564b",
    "incident_response_inspection": "#d62728",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _selected_frames(path: Path, frame_stride: int) -> list[dict[str, Any]]:
    rows = list(_iter_jsonl(path))
    stride = max(1, int(frame_stride))
    selected = [frame for index, frame in enumerate(rows) if index % stride == 0]
    if rows and selected[-1].get("tick") != rows[-1].get("tick"):
        selected.append(rows[-1])
    return selected


def _topology_segments(planner: SumoGroundFlowPlanner) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    for edge in planner.edges.values():
        if not vehicle_class_for_edge(edge):
            continue
        if len(edge.shape_xy) >= 2:
            segments.append([(float(x), float(y)) for x, y in edge.shape_xy])
    return segments


def _sumo_frames_by_tick(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {int(frame.get("tick") or 0): frame for frame in _iter_jsonl(path)}


def _plot_bounds(
    topo_segments: Sequence[Sequence[tuple[float, float]]],
    uav_frames: Sequence[dict[str, Any]],
    sumo_frames: dict[int, dict[str, Any]],
    margin_m: float = 70.0,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for segment in topo_segments:
        for x, y in segment:
            xs.append(float(x))
            ys.append(float(y))
    for frame in uav_frames:
        for uav in frame.get("uavs") or []:
            pos = uav.get("position_enu_m") or []
            if len(pos) >= 2:
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
        for vehicle in (sumo_frames.get(int(frame.get("tick") or 0), {}).get("vehicles") or []):
            pos = vehicle.get("truth_position_enu_m") or []
            if len(pos) >= 2:
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
    return min(xs) - margin_m, max(xs) + margin_m, min(ys) - margin_m, max(ys) + margin_m


def _draw_2d_frame(
    *,
    frame: dict[str, Any],
    sumo_frame: dict[str, Any] | None,
    topo_segments: Sequence[Sequence[tuple[float, float]]],
    pads: Sequence[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 9.0), dpi=dpi)
    ax.set_facecolor("#f7f7f1")
    ax.add_collection(LineCollection(topo_segments, colors="#9a9a94", linewidths=0.32, alpha=0.62, zorder=1))

    if sumo_frame:
        veh_x: list[float] = []
        veh_y: list[float] = []
        incident_x: list[float] = []
        incident_y: list[float] = []
        for vehicle in sumo_frame.get("vehicles") or []:
            pos = vehicle.get("truth_position_enu_m") or []
            if len(pos) < 2:
                continue
            if str(vehicle.get("control_role") or "") == "incident_controlled" or int(vehicle.get("signals") or 0) != 0:
                incident_x.append(float(pos[0]))
                incident_y.append(float(pos[1]))
            else:
                veh_x.append(float(pos[0]))
                veh_y.append(float(pos[1]))
        if veh_x:
            ax.scatter(veh_x, veh_y, s=9, marker="s", c="#4f79a8", alpha=0.45, edgecolors="none", label="SUMO V", zorder=3)
        if incident_x:
            ax.scatter(incident_x, incident_y, s=26, marker="s", c="#d62728", alpha=0.85, edgecolors="white", linewidths=0.25, label="incident V", zorder=4)
        active = sumo_frame.get("active_incidents") or []
        if active:
            ax.scatter(
                [float(item["anchor"]["truth_position_enu_m"][0]) for item in active if item.get("anchor")],
                [float(item["anchor"]["truth_position_enu_m"][1]) for item in active if item.get("anchor")],
                s=48,
                marker="x",
                c="#7b3294",
                linewidths=1.15,
                label="incident anchor",
                zorder=7,
            )

    if pads:
        ax.scatter(
            [float(pad["position_enu_m"][0]) for pad in pads],
            [float(pad["position_enu_m"][1]) for pad in pads],
            s=36,
            marker="D",
            c="#111111",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
            label="pad",
            zorder=6,
        )

    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for uav in frame.get("uavs") or []:
        pos = uav.get("position_enu_m") or []
        if len(pos) < 3:
            continue
        grouped.setdefault(str(uav.get("mission_type") or "unknown"), []).append((float(pos[0]), float(pos[1]), float(pos[2])))
    for mission_type, items in sorted(grouped.items()):
        xs = [item[0] for item in items]
        ys = [item[1] for item in items]
        sizes = [18.0 + min(28.0, item[2] * 0.12) for item in items]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            marker="^",
            c=MISSION_COLORS.get(mission_type, "#555555"),
            edgecolors="black",
            linewidths=0.2,
            alpha=0.86,
            label=f"U {mission_type}",
            zorder=8,
        )

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UE truth X / m")
    ax.set_ylabel("UE truth Y / m")
    ax.set_title(
        f"Donghu SUMO + UAV Flow | t={float(frame.get('sim_time_s') or 0.0):.1f}s | "
        f"UAV={len(frame.get('uavs') or [])}",
        fontsize=10,
    )
    ax.grid(True, color="#d9d9d0", linewidth=0.28, alpha=0.55)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        deduped: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if label not in deduped:
                deduped[label] = handle
        ax.legend(deduped.values(), deduped.keys(), loc="upper right", fontsize=6.2, frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _draw_3d_frame(
    *,
    frame: dict[str, Any],
    bounds: tuple[float, float, float, float],
    output_path: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(8.5, 7.2), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for uav in frame.get("uavs") or []:
        pos = uav.get("position_enu_m") or []
        if len(pos) >= 3:
            grouped.setdefault(str(uav.get("mission_type") or "unknown"), []).append((float(pos[0]), float(pos[1]), float(pos[2])))
    for mission_type, items in sorted(grouped.items()):
        ax.scatter(
            [item[0] for item in items],
            [item[1] for item in items],
            [item[2] for item in items],
            marker="^",
            s=18,
            c=MISSION_COLORS.get(mission_type, "#555555"),
            edgecolors="black",
            linewidths=0.18,
            alpha=0.85,
            label=mission_type,
        )
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0, 140)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("UAV altitude / m")
    ax.view_init(elev=28, azim=-58)
    ax.set_title(f"UAV altitude layers | t={float(frame.get('sim_time_s') or 0.0):.1f}s | UAV={len(frame.get('uavs') or [])}", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_gif(frame_paths: Sequence[Path], gif_path: Path, duration_ms: int) -> None:
    if not frame_paths:
        raise ValueError("No PNG frames to encode")
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


def visualize(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sumo_output_dir: Path = DEFAULT_SUMO_OUTPUT_DIR,
    frame_stride: int = 4,
    dpi: int = 110,
    gif_duration_ms: int = 120,
    overwrite: bool = False,
    make_3d_gif: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    task_plan = _load_json(output_dir / "uav_task_plan.json")
    frames_path = output_dir / "uav_traffic_frames.jsonl"
    selected = _selected_frames(frames_path, frame_stride)
    planner = SumoGroundFlowPlanner(DEFAULT_SUMO_NET_XML)
    topo_segments = _topology_segments(planner)
    sumo_frames = _sumo_frames_by_tick(Path(sumo_output_dir) / "sumo_traffic_frames.jsonl")
    bounds = _plot_bounds(topo_segments, selected, sumo_frames)
    pads = list(task_plan.get("pads") or [])

    frames_dir = output_dir / "visualization_frames"
    if overwrite and frames_dir.exists():
        for stale in frames_dir.glob("*.png"):
            stale.unlink()
    frame_paths: list[Path] = []
    for index, frame in enumerate(selected):
        path = frames_dir / f"uav_sumo_flow_{index:04d}.png"
        _draw_2d_frame(
            frame=frame,
            sumo_frame=sumo_frames.get(int(frame.get("tick") or 0)),
            topo_segments=topo_segments,
            pads=pads,
            bounds=bounds,
            output_path=path,
            dpi=dpi,
        )
        frame_paths.append(path)
    gif_path = output_dir / "uav_sumo_flow.gif"
    if overwrite and gif_path.exists():
        gif_path.unlink()
    _write_gif(frame_paths, gif_path, gif_duration_ms)

    gif_3d_path: Path | None = None
    frame_paths_3d: list[Path] = []
    if make_3d_gif:
        frames_3d_dir = output_dir / "visualization_frames_3d"
        if overwrite and frames_3d_dir.exists():
            for stale in frames_3d_dir.glob("*.png"):
                stale.unlink()
        for index, frame in enumerate(selected):
            path = frames_3d_dir / f"uav_altitude_3d_{index:04d}.png"
            _draw_3d_frame(frame=frame, bounds=bounds, output_path=path, dpi=dpi)
            frame_paths_3d.append(path)
        gif_3d_path = output_dir / "uav_altitude_3d.gif"
        if overwrite and gif_3d_path.exists():
            gif_3d_path.unlink()
        _write_gif(frame_paths_3d, gif_3d_path, gif_duration_ms)

    summary = {
        "schema_name": "donghu_uav_flow_visualization",
        "schema_version": "v1",
        "source_frames": str(frames_path),
        "source_task_plan": str(output_dir / "uav_task_plan.json"),
        "sumo_frames": str(Path(sumo_output_dir) / "sumo_traffic_frames.jsonl"),
        "frame_stride": int(frame_stride),
        "visualized_frame_count": len(frame_paths),
        "gif_path": str(gif_path),
        "gif_3d_path": str(gif_3d_path) if gif_3d_path else "",
        "frames_dir": str(frames_dir),
        "frames_3d_dir": str(output_dir / "visualization_frames_3d") if make_3d_gif else "",
        "topology_segment_count": len(topo_segments),
    }
    (output_dir / "uav_flow_visualization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Donghu UAV flow GIFs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sumo-output-dir", type=Path, default=DEFAULT_SUMO_OUTPUT_DIR)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-3d-gif", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = visualize(
        output_dir=args.output_dir,
        sumo_output_dir=args.sumo_output_dir,
        frame_stride=args.frame_stride,
        dpi=args.dpi,
        gif_duration_ms=args.gif_duration_ms,
        overwrite=args.overwrite,
        make_3d_gif=not args.skip_3d_gif,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

