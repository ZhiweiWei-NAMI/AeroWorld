"""Resolve episode-local source authority from the generated manifest."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def source_episode_root(episode_root: Path) -> Path | None:
    """Optional offstage scenario poses; formal physical inputs remain required.

    Absent original poses are not replaced by another episode or by old labels.
    Their declared location/status is published by source_episode_descriptor.
    """
    manifest = json.loads((episode_root / 'episode_manifest.json').read_text())
    generation = manifest['generation']
    if (generation.get('generator') == 'Dataset/tools/repair_l25_signal_truth.py'
            and 'source_episode_dir' not in generation):
        return None
    ref = generation['source_episode_dir']
    if not isinstance(ref, str) or not ref:
        raise ValueError('generation.source_episode_dir must be a nonempty path')
    source = (PROJECT_ROOT / ref).resolve()
    source.relative_to(PROJECT_ROOT)
    if source.name != episode_root.name:
        raise ValueError(f'episode source identity/path mismatch: {source}')
    if not source.exists():
        return None
    if not source.is_dir():
        raise ValueError(f'episode source is not a directory: {source}')
    for name in ('global_entity_roster.json', 'trajectories.jsonl'):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    return source


def source_episode_descriptor(episode_root: Path) -> dict:
    manifest = json.loads((episode_root / 'episode_manifest.json').read_text())
    ref = manifest['generation'].get('source_episode_dir')
    source = source_episode_root(episode_root)
    return {
        'path': ref,
        'status': 'present' if source is not None else 'source_missing' if ref else 'not_declared',
        'role': 'offstage_scenario_roster_and_trajectories',
        'missing_policy': 'retain_unknown_preflight_and_absent_pose_evidence',
    }


def source_sumo_frames(episode_root: Path) -> Path:
    manifest = json.loads((episode_root / 'episode_manifest.json').read_text())
    ref = manifest['sumo_traffic']['source']['frames']
    if not isinstance(ref, str) or not ref:
        raise ValueError('sumo_traffic.source.frames must be a nonempty path')
    source = (PROJECT_ROOT / ref).resolve()
    source.relative_to(PROJECT_ROOT)
    if not source.is_file() or source.parent.name != episode_root.name:
        raise ValueError(f'episode SUMO source identity/path mismatch: {source}')
    return source
