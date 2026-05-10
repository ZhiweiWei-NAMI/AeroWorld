from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SUMO_SCRIPTS_DIR = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scripts"
DEFAULT_BASE_CONFIG = SUMO_SCRIPTS_DIR / "episode_render_host_config.json"
DEFAULT_RENDER_HOST = SUMO_SCRIPTS_DIR / "episode_render_host.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_to_render_ready import (  # noqa: E402
    DEFAULT_MAP_ID,
    DEFAULT_ROI_ID,
    DEFAULT_SITE_ID,
    DEFAULT_TICK_HZ,
    convert_episode,
    default_output_dir,
    load_json,
    write_json,
)


def validate_single_capture_selection(
    *,
    camera_roles: list[str],
    camera_ids: list[str],
    modalities: list[str],
    segmentation_backend: str,
    semantic_rules_path: Path,
    semantic_stencil_audit_only: bool,
) -> None:
    roles = [str(value).strip().lower() for value in camera_roles if str(value).strip()]
    ids = [str(value).strip() for value in camera_ids if str(value).strip()]
    mods = [str(value).strip() for value in modalities if str(value).strip()]
    if len(ids) != 1:
        raise RuntimeError(
            "Rendering requires exactly one --camera-id per process to keep capture memory bounded "
            f"and modality alignment auditable; got {len(ids)}."
        )
    if len(mods) != 1:
        raise RuntimeError(
            "Rendering requires exactly one --modality per process; "
            f"got {len(mods)}."
        )
    if len(roles) > 1 or "all" in roles:
        raise RuntimeError(
            "Rendering must select at most one concrete camera role per process; "
            f"got {roles or '<none>'}."
        )
    if str(segmentation_backend) != "ue_custom_stencil":
        raise RuntimeError("Formal rendering only supports --segmentation-backend ue_custom_stencil.")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def episode_id_from_manifest(render_episode_dir: Path) -> str:
    manifest_path = render_episode_dir / "episode_manifest.json"
    if not manifest_path.exists():
        return render_episode_dir.name
    manifest = load_json(manifest_path)
    return str(manifest.get("episode_id") or render_episode_dir.name)


def render_config_for_episode(
    *,
    render_episode_dir: Path,
    base_config_path: Path,
    map_id: str,
    site_id: str,
    output_root: Path,
) -> dict[str, Any]:
    config = load_json(base_config_path)
    manifest_path = render_episode_dir / "episode_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    episode_id = str(manifest.get("episode_id") or episode_id_from_manifest(render_episode_dir))
    config["episode_dir"] = repo_relative(render_episode_dir)
    config["map_id"] = map_id
    config["output_dir"] = repo_relative(output_root / episode_id)
    config["template_resolver_path"] = "Plugins/SumoImporter/Scripts/episode_template_resolver.json"
    config["capture_presets_path"] = "Plugins/SumoImporter/Scripts/episode_capture_presets.json"
    config["truth_frame_coordinate_space"] = "map_enu"
    config["batch_strategy"] = {
        "sites": [site_id],
        "tick_window_size": 0,
    }
    source_event_script_path = str(manifest.get("source_event_script_path") or "").strip()
    if source_event_script_path:
        config["event_script_path"] = source_event_script_path
    else:
        config.pop("event_script_path", None)
    config.pop("event_script_parameters", None)
    if _is_dataset_render_package(render_episode_dir, manifest):
        pedestrian_projection = dict(config.get("pedestrian_roadside_projection") or {})
        pedestrian_projection["enabled"] = False
        pedestrian_projection["disabled_reason"] = "Dataset pedestrian coordinates are prevalidated upstream"
        config["pedestrian_roadside_projection"] = pedestrian_projection
    return config


def _is_dataset_render_package(render_episode_dir: Path, manifest: dict[str, Any]) -> bool:
    dataset_root = PROJECT_ROOT / "Dataset"
    try:
        render_episode_dir.resolve().relative_to(dataset_root.resolve())
        return True
    except ValueError:
        pass
    source_path = str(manifest.get("source_event_script_path") or manifest.get("source_scene_setup_path") or "")
    return source_path.replace("\\", "/").startswith("Dataset/")


def discover_source_episodes(episodes_root: Path, episode_names: list[str], *, include_private: bool) -> list[Path]:
    if episode_names:
        return [episodes_root / name for name in episode_names]
    return sorted(
        path
        for path in episodes_root.iterdir()
        if path.is_dir() and (include_private or not path.name.startswith("_"))
    )


def write_render_config(
    *,
    render_episode_dir: Path,
    base_config_path: Path,
    map_id: str,
    site_id: str,
    output_root: Path,
) -> Path:
    config = render_config_for_episode(
        render_episode_dir=render_episode_dir,
        base_config_path=base_config_path,
        map_id=map_id,
        site_id=site_id,
        output_root=output_root,
    )
    config_path = render_episode_dir / "render_host_config.json"
    write_json(config_path, config)
    return config_path


def render_episode(
    config_path: Path,
    *,
    tick_stride: int,
    host: str,
    port: int,
    camera_roles: list[str],
    camera_ids: list[str],
    modalities: list[str],
    segmentation_backend: str,
    semantic_rules_path: Path,
    semantic_stencil_audit_only: bool,
    runtime_uav_control_backend: str,
) -> None:
    command = [
        sys.executable,
        str(DEFAULT_RENDER_HOST),
        "--config",
        str(config_path),
        "--host",
        host,
        "--port",
        str(port),
        "--tick_stride",
        str(max(1, tick_stride)),
    ]
    for role in camera_roles:
        command.extend(["--camera-role", str(role)])
    for camera_id in camera_ids:
        command.extend(["--camera-id", str(camera_id)])
    for modality in modalities:
        command.extend(["--modality", str(modality)])
    command.extend(["--segmentation-backend", str(segmentation_backend)])
    command.extend(["--runtime-uav-control-backend", str(runtime_uav_control_backend)])
    if semantic_rules_path:
        command.extend(["--semantic-rules-path", str(semantic_rules_path)])
    if semantic_stencil_audit_only:
        command.append("--semantic-stencil-audit-only")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Dataset episodes to render-ready packages and optionally render them through episode_render_host.py."
        )
    )
    parser.add_argument("--episodes-root", type=Path, default=Path("Dataset/episodes"))
    parser.add_argument("--render-ready-root", type=Path, default=Path("Dataset/render_ready_episodes"))
    parser.add_argument("--output-root", type=Path, default=Path("F:/aw_render"))
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--episode", action="append", default=[], help="Episode directory name under --episodes-root. Repeatable.")
    parser.add_argument("--start", type=int, default=0, help="Start index after sorting episodes")
    parser.add_argument("--end", type=int, default=None, help="End index after sorting episodes")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--map-id", default=DEFAULT_MAP_ID)
    parser.add_argument("--site-id", default=DEFAULT_SITE_ID)
    parser.add_argument("--roi-id", default=DEFAULT_ROI_ID)
    parser.add_argument("--tick-hz", type=int, default=DEFAULT_TICK_HZ)
    parser.add_argument("--tick-stride", type=int, default=1)
    parser.add_argument("--camera-role", action="append", default=[], choices=["all", "ground", "uav"])
    parser.add_argument("--camera-id", action="append", default=[])
    parser.add_argument("--modality", action="append", default=[])
    parser.add_argument("--segmentation-backend", choices=["ue_custom_stencil"], default="ue_custom_stencil")
    parser.add_argument("--runtime-uav-control-backend", choices=["airsim_move", "pose_sync"], default="airsim_move")
    parser.add_argument("--semantic-rules-path", type=Path, default=Path("Config/LowAltitude/semantic_stencil_rules.json"))
    parser.add_argument("--semantic-stencil-audit-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-private", action="store_true", help="Include '_' prefixed scratch episode directories.")
    parser.add_argument("--render", action="store_true", help="Run episode_render_host.py after conversion. Requires UE PIE.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing or rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render:
        validate_single_capture_selection(
            camera_roles=list(args.camera_role or []),
            camera_ids=list(args.camera_id or []),
            modalities=list(args.modality or []),
            segmentation_backend=str(args.segmentation_backend),
            semantic_rules_path=args.semantic_rules_path,
            semantic_stencil_audit_only=bool(args.semantic_stencil_audit_only),
        )
    source_episodes = discover_source_episodes(
        args.episodes_root,
        list(args.episode or []),
        include_private=bool(args.include_private),
    )
    source_episodes = source_episodes[int(args.start): args.end]
    if args.max_episodes > 0:
        source_episodes = source_episodes[: int(args.max_episodes)]
    if not source_episodes:
        raise SystemExit("No source episodes selected.")

    print(f"[batch_render_dataset] selected episodes: {len(source_episodes)}")
    for source_episode_dir in source_episodes:
        render_episode_dir = default_output_dir(source_episode_dir, args.render_ready_root)
        config_path = render_episode_dir / "render_host_config.json"
        if args.dry_run:
            print(f"[batch_render_dataset] would convert {source_episode_dir} -> {render_episode_dir}")
            print(f"[batch_render_dataset] would write {config_path}")
            if args.render:
                print(f"[batch_render_dataset] would render {config_path}")
            continue

        result = convert_episode(
            source_episode_dir,
            render_episode_dir,
            project_root=PROJECT_ROOT,
            map_id=str(args.map_id),
            site_id=str(args.site_id),
            roi_id=str(args.roi_id),
            tick_hz=max(1, int(args.tick_hz)),
            overwrite=bool(args.overwrite),
        )
        config_path = write_render_config(
            render_episode_dir=render_episode_dir,
            base_config_path=args.base_config,
            map_id=str(args.map_id),
            site_id=str(args.site_id),
            output_root=args.output_root,
        )
        print(
            json.dumps(
                {
                    "episode": source_episode_dir.name,
                    "status": "skipped" if result.get("skipped") else "converted",
                    "render_episode_dir": str(render_episode_dir),
                    "render_config": str(config_path),
                },
                ensure_ascii=False,
            )
        )
        if args.render:
            render_episode(
                config_path,
                tick_stride=int(args.tick_stride),
                host=str(args.host),
                port=int(args.port),
                camera_roles=list(args.camera_role or []),
                camera_ids=list(args.camera_id or []),
                modalities=list(args.modality or []),
                segmentation_backend=str(args.segmentation_backend),
                semantic_rules_path=args.semantic_rules_path,
                semantic_stencil_audit_only=bool(args.semantic_stencil_audit_only),
                runtime_uav_control_backend=str(args.runtime_uav_control_backend),
            )

    if not args.render:
        print(
            "[batch_render_dataset] conversion complete. Start UE PIE, then rerun with --render "
            "or call episode_render_host.py with an episode render_host_config.json."
        )


if __name__ == "__main__":
    main()
