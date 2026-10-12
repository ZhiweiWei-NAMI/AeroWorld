"""Read declared branch layouts without treating a delta as a full episode."""

from Dataset.semantic_truth.provenance import open_artifact_text
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELTA_SCOPES = frozenset({
    "parent_reference_plus_local_delta",
    "parent_prefix_ref_plus_intervention_delta_window",
})


class BranchSourceError(ValueError):
    """An explicit branch artifact contract cannot be resolved."""


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise BranchSourceError(f"expected JSON object: {path}")
    return value


def iter_objects(path: Path) -> Iterator[dict[str, Any]]:
    with open_artifact_text(path, encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BranchSourceError(f"expected JSON object: {path}:{number}")
            yield value


def project_path(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise BranchSourceError(f"source outside project: {value}") from exc
    return path


@dataclass(frozen=True)
class BranchSource:
    root: Path
    raw: Path
    manifest_path: Path
    manifest: dict[str, Any]
    parent: Path
    fork: int
    end: int

    def raw_file(self, name: str) -> Path:
        path = self.raw / name
        if not path.is_file():
            raise BranchSourceError(f"missing branch artifact: {path}")
        return path

    def objective_file(self, name: str) -> Path:
        path = self.root / "objective" / name
        if not path.is_file():
            raise BranchSourceError(f"missing objective artifact: {path}")
        return path


def resolve_branch_source(
    group: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> BranchSource:
    value = arm.get("branch_root")
    if not isinstance(value, str) or not value:
        raise BranchSourceError("missing branch_root")
    root = project_path(value, project_root)
    candidates = [root / "raw" / "branch_manifest.json", root / "branch_manifest.json"]
    manifests = [path for path in candidates if path.is_file()]
    if len(manifests) != 1:
        raise BranchSourceError(f"expected exactly one declared branch layout: {root}")
    manifest_path = manifests[0]
    manifest = read_object(manifest_path)
    if not manifest.get("parent_script_ref"):
        refs = manifest.get("source_refs")
        script_ref = refs.get("event_script") if isinstance(refs, dict) else None
        if isinstance(script_ref, str) and script_ref:
            manifest = dict(manifest)
            manifest["parent_script_ref"] = script_ref
    expected = {
        "episode_id": group["episode_id"],
        "profile_id": group["profile_id"],
        "seed_id": group["seed_id"],
        "group_id": group["group_id"],
        "arm_id": arm["arm_id"],
        "arm_kind": arm["arm_kind"],
        "branch_key": arm["branch_key"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise BranchSourceError(f"branch identity mismatch ({key}): {manifest_path}")
    if manifest.get("delta_only") is not True or manifest.get("artifact_scope") not in DELTA_SCOPES:
        raise BranchSourceError(f"unsupported branch trajectory scope: {manifest_path}")
    parent_ref = f"aw_data/render_ready_episodes_capture_filtered/{group['episode_id']}"
    if manifest.get("parent_episode_ref") != parent_ref:
        raise BranchSourceError(f"branch parent reference mismatch: {manifest_path}")
    fork, end = manifest.get("fork_tick"), manifest.get("window_end_tick")
    if type(fork) is not int or type(end) is not int or not 0 <= fork <= end <= 900:
        raise BranchSourceError(f"invalid formal branch window: {manifest_path}")
    prefix = manifest.get("parent_prefix_ref")
    if prefix is not None:
        if not isinstance(prefix, dict) or (
            prefix.get("capture_episode") != parent_ref
            or prefix.get("fork_tick") != fork
            or prefix.get("window_end_tick") != end
        ):
            raise BranchSourceError(f"parent prefix reference mismatch: {manifest_path}")
    if manifest.get("lifecycle_delta_file") != "entity_lifecycle_delta.jsonl":
        raise BranchSourceError(f"missing declared lifecycle delta: {manifest_path}")
    source = BranchSource(root, manifest_path.parent, manifest_path, manifest,
                          project_path(parent_ref, project_root), fork, end)
    source.raw_file("trajectories.jsonl")
    source.raw_file("entity_lifecycle_delta.jsonl")
    if not (source.parent / "trajectories.jsonl").is_file():
        raise BranchSourceError(f"missing parent trajectory: {source.parent}")
    return source
