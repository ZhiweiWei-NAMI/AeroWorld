from __future__ import annotations

from pathlib import Path

from .interfaces import MapPackage, ScenarioPackage


DEFAULT_MAP_ID = "donghu_road_topo"
DEFAULT_SCENARIO_DIR = "donghu_dense_uav_rain_fall"
DEFAULT_SCENARIO_ID = "scenario.donghu_demo_dense_uav_rain_fall.v1"
DEFAULT_EPISODE_ID = "episode_demo_dense_uav_rain_fall_90s"


def project_root_from(path: Path | None = None) -> Path:
    current = (path or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if candidate.is_dir() and (candidate / "DynamicCityCreatorEx.uproject").exists():
            return candidate
    raise FileNotFoundError("Unable to resolve project root from current path.")


def resolve_map_package(project_root: Path, map_id: str = DEFAULT_MAP_ID) -> MapPackage:
    map_root = project_root / "Config" / "LowAltitude" / "Maps" / map_id
    source_geojson_root = project_root / "Content" / "Maps" / map_id
    return MapPackage(
        map_id=map_id,
        root_dir=map_root,
        map_context_path=map_root / "map_context.json",
        traffic_bundle_dir=map_root / "traffic_bundle",
        ped_nav_bundle_path=map_root / "ped_nav_semantic.bundle.json",
        asset_catalog_path=project_root / "Config" / "LowAltitude" / "asset_catalog.json",
        weather_profiles_path=project_root / "Config" / "LowAltitude" / "weather_render_profiles.json",
        scenario_objects_runtime_path=map_root / "scenario_objects.runtime.json",
        scenario_objects_source_path=map_root / "scenario_objects.json",
        source_geojson={
            "road": source_geojson_root / "road" / "road.geojson",
            "building": source_geojson_root / "building" / "building.geojson",
            "water": source_geojson_root / "water" / "water.geojson",
            "green": source_geojson_root / "green" / "green.geojson",
            "block": source_geojson_root / "block" / "block.geojson",
            "bounds": source_geojson_root / "bounds" / "bounds.geojson",
        },
    )


def resolve_scenario_root(project_root: Path, scenario_dir: str = DEFAULT_SCENARIO_DIR) -> Path:
    return project_root / "Plugins" / "SumoImporter" / "Scenarios" / scenario_dir


def resolve_seed_package_root(project_root: Path) -> Path:
    return project_root / "Plugins" / "SumoImporter" / "Scenarios" / "_seed_packages"


def resolve_scenario_package(
    project_root: Path,
    *,
    scenario_dir: str = DEFAULT_SCENARIO_DIR,
    episode_id: str = DEFAULT_EPISODE_ID,
    scenario_id: str = DEFAULT_SCENARIO_ID,
) -> ScenarioPackage:
    scenario_root = resolve_scenario_root(project_root, scenario_dir=scenario_dir)
    root_dir = scenario_root / "artifacts" / "episodes" / episode_id
    return ScenarioPackage(
        scenario_id=scenario_id,
        episode_id=episode_id,
        root_dir=root_dir,
        truth_frames_path=root_dir / "truth_frames.jsonl",
        weather_meta_path=root_dir / "weather_meta.jsonl",
        scenario_plan_path=root_dir / "scenario_plan.json",
        capture_plan_path=scenario_root / "artifacts" / "capture" / "demo_capture_plan.json",
        episode_manifest_path=root_dir / "episode_manifest.json",
    )


def latest_capture_rgb_dir(project_root: Path, *, episode_id: str, camera_id: str) -> Path:
    root = project_root / "Saved" / "AirSim" / "episode_render_host"
    candidates = sorted(root.glob(f"{episode_id}*/site.intersection_a/{camera_id}/rgb"))
    if not candidates:
        raise FileNotFoundError(f"Unable to locate RGB capture directory for episode='{episode_id}' camera='{camera_id}'.")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def default_presentation_dir(project_root: Path, episode_id: str = DEFAULT_EPISODE_ID) -> Path:
    return project_root / "Saved" / "Presentation" / f"{episode_id}_multiview"


def default_timeline_path(project_root: Path, episode_id: str = DEFAULT_EPISODE_ID) -> Path:
    return default_presentation_dir(project_root, episode_id=episode_id) / "timeline" / "multiview_timeline.json"
