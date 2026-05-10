from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUMO_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Scripts"
if str(SUMO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUMO_SCRIPTS_DIR))

from donghu_core.discovery import (
    DEFAULT_EPISODE_ID,
    DEFAULT_SCENARIO_DIR,
    default_presentation_dir,
    latest_capture_rgb_dir,
    project_root_from,
    resolve_scenario_package,
)
from donghu_core.postprocess_service import PostprocessService


def _default_capture_dir(project_root: Path, camera_id: str) -> Path:
    try:
        return latest_capture_rgb_dir(project_root, episode_id=DEFAULT_EPISODE_ID, camera_id=camera_id)
    except FileNotFoundError:
        return (
            project_root
            / "Saved"
            / "AirSim"
            / "episode_render_host"
            / f"{DEFAULT_EPISODE_ID}_placeholder"
            / "site.intersection_a"
            / camera_id
            / "rgb"
        )


def parse_args() -> argparse.Namespace:
    project_root = project_root_from(Path(__file__))
    parser = argparse.ArgumentParser(description="Build GIF + timeline assets for the multiview demo replay.")
    parser.add_argument("--scenario-dir", default=DEFAULT_SCENARIO_DIR, help="Canonical scenario directory name under Plugins/SumoImporter/Scenarios")
    parser.add_argument(
        "--low-dir",
        default=str(_default_capture_dir(project_root, "demo_low_ground_8m")),
        help="Low-view RGB directory",
    )
    parser.add_argument(
        "--high-dir",
        default=str(_default_capture_dir(project_root, "demo_high_overview")),
        help="High-view RGB directory",
    )
    parser.add_argument("--episode-dir", default="", help="Episode artifact directory; defaults to the canonical path for --scenario-dir")
    parser.add_argument(
        "--output-dir",
        default=str(default_presentation_dir(project_root, episode_id=DEFAULT_EPISODE_ID)),
        help="Output directory",
    )
    parser.add_argument("--playback-duration-s", type=float, default=45.0, help="Target playback duration for each GIF")
    parser.add_argument("--gif-max-width", type=int, default=960, help="Resize GIF frames to this width (0 keeps original)")
    parser.add_argument("--gif-max-colors", type=int, default=192, help="Palette size for GIF quantization")
    parser.add_argument("--hold-final-s", type=float, default=1.0, help="Extra hold duration on the last frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = project_root_from(Path(__file__))
    service = PostprocessService()
    scenario_package = resolve_scenario_package(
        project_root,
        scenario_dir=str(args.scenario_dir),
        episode_id=DEFAULT_EPISODE_ID,
    )
    manifest = service.build_assets(
        low_dir=Path(args.low_dir).resolve(),
        high_dir=Path(args.high_dir).resolve(),
        episode_dir=Path(args.episode_dir).resolve() if str(args.episode_dir).strip() else scenario_package.root_dir,
        output_dir=Path(args.output_dir).resolve(),
        playback_duration_s=float(args.playback_duration_s),
        gif_max_width=int(args.gif_max_width),
        gif_max_colors=int(args.gif_max_colors),
        hold_final_s=float(args.hold_final_s),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
