"""Source-checked sample identity and source-record scope for graph projections."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping

from Dataset.world_model.graph.contract import REPO, read_json, source_path


_SEED = re.compile(r"seed[0-9]{2}\Z")
_FORMAL_RENDER = REPO / "aw_data/render_ready_episodes_capture_filtered"
_FORMAL_OBJECTIVE = REPO / "aw_data/objective_semantic_truth"
_FORMAL_DOMAIN = REPO / "aw_data/domain_state_supplement"
_FORMAL_COMPUTE = REPO / "aw_data/compute_comm_supplement"
_FORMAL_CHAIN = REPO / "aw_data/event_chains"
_ARM_ROOT = REPO / "arms"
_SCENARIOS = REPO / "Dataset/scenarios"
_DESIGN = REPO / "design"
_INTERVENTIONS = REPO / "interventions"
_RULES = REPO / "Dataset/semantic_rules"

_FORMAL_RENDER_FILES = {
    "formal_episode_manifest": "episode_manifest.json",
    "formal_roster": "global_entity_roster.json",
    "formal_truth_frame": "truth_frames.jsonl",
    "formal_event_realization": "event_realization.jsonl",
    "formal_weather_meta": "weather_meta.jsonl",
}
_FORMAL_COMPUTE_FILES = {
    "compute_state": "compute_state.jsonl",
    "communication_state": "communication_state.jsonl",
    "predicate_truth": "predicate_truth.jsonl",
    "predicate_truth_matrix": "predicate_truth_matrix.jsonl",
    "events": "events.jsonl",
}
_SHARED_FAMILIES = frozenset({
    "formal_source_script", "formal_source_scene_setup", "script_category_source",
    "arm_scene_setup", "rule_contract",
})
_ARM_ROW_FAMILIES = frozenset({
    "arm_manifest", "arm_states", "arm_trajectories", "arm_actions",
    "arm_predicate_truth", "arm_predicate_truth_ticks", "arm_domain_state_ticks",
    "arm_predicate_transitions", "arm_local_business_truth",
    "arm_event_occurrences", "arm_event_outcomes", "arm_chain_relations",
    "arm_chain_nodes", "arm_chain_events", "arm_initial_events",
    "arm_semantic_history",
})
_ARM_FILES = {
    "arm_manifest": "manifest.json",
    "arm_states": "raw/states.jsonl",
    "arm_trajectories": "raw/trajectories.jsonl",
    "arm_actions": "raw/actions.jsonl",
    "arm_predicate_truth": "objective/predicate_truth.jsonl",
    "arm_predicate_truth_ticks": "objective/predicate_truth_ticks.jsonl",
    "arm_domain_state_ticks": "objective/predicate_truth_ticks.jsonl",
    "arm_predicate_transitions": "objective/predicate_transitions.jsonl",
    "arm_local_business_truth": "objective/local_business_truth.jsonl",
    "arm_event_occurrences": "objective/event_occurrences.jsonl",
    "arm_event_outcomes": "objective/event_outcomes.jsonl",
    "arm_chain_relations": "chain/relations.jsonl",
    "arm_chain_nodes": "chain/nodes.jsonl",
    "arm_chain_events": "chain/events.jsonl",
    "arm_initial_events": "objective/initial_events.json",
    "arm_branch_roster": "ue/global_entity_roster.json",
    "arm_script_plan": "ue/event_script.json",
}
_L2_CHAIN_EVENT_FIELDS = frozenset({
    "node_id", "family", "bindings", "trigger_tick", "trigger_predicate_id",
})


def _raw_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    raise TypeError(type(value))


def manifest_reference(manifest: Mapping[str, Any], name: str) -> str | None:
    sources, refs = manifest.get("sources", {}), manifest.get("source_refs", {})
    if not isinstance(sources, Mapping) or not isinstance(refs, Mapping):
        raise ValueError(f"manifest source containers have invalid types: {name}")
    if name in sources and name in refs and sources[name] != refs[name]:
        raise ValueError(f"manifest {name} references conflict")
    value = sources.get(name, refs.get(name))
    if value is None:
        return None
    reference = value if isinstance(value, str) else value.get("path") if isinstance(value, Mapping) else None
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"manifest {name} reference has no path")
    return reference


def _source_fields(manifest: Mapping[str, Any], path: Path,
                   names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"field": name, "raw_value": manifest[name],
             "source_type": _raw_type(manifest[name]),
             "source_file": source_path(path), "source_tokens": [name]}
            for name in names if name in manifest]


def _episode_parts(episode_id: str) -> tuple[str, str]:
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError(f"formal episode ID is absent: {episode_id!r}")
    profile, separator, seed = episode_id.rpartition("__")
    if not separator or not profile or _SEED.fullmatch(seed) is None:
        raise ValueError(f"formal episode ID lacks profile/seedXX: {episode_id!r}")
    return profile, seed


def _seed_from_raw(value: Any, expected: str, *, source: str,
                   allow_string: bool) -> str:
    if type(value) is int and value >= 0:
        normalized = f"seed{value:02d}"
    elif allow_string and type(value) is str and _SEED.fullmatch(value):
        normalized = value
    else:
        raise ValueError(f"sample seed has invalid source type/value: {source}: {value!r}")
    if normalized != expected:
        raise ValueError(f"sample seed differs from directory identity: {source}: {value!r} != {expected}")
    return normalized


def read_formal_identity(episode_id: str) -> dict[str, Any]:
    """Use only the declared episode directory and its identity manifest."""
    profile, seed = _episode_parts(episode_id)
    manifest_path = _FORMAL_RENDER / episode_id / "episode_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("episode_id") != episode_id:
        raise ValueError(f"formal episode manifest identity differs: {manifest_path}")
    if "seed" not in manifest:
        raise ValueError(f"formal episode manifest seed is absent: {manifest_path}")
    _seed_from_raw(manifest["seed"], seed, source=source_path(manifest_path),
                   allow_string=False)
    return {"sample_key": "formal:" + episode_id, "source_kind": "formal",
            "episode_id": episode_id, "profile": profile, "seed": seed,
            "group": None, "branch": None, "arm_key": None,
            "identity_sources": {
                "directory": {"source_path": source_path(manifest_path.parent),
                              "segments": [{"segment_index": 0,
                                            "role": "episode_id",
                                            "raw_value": episode_id,
                                            "source_type": "string"}]},
                "fields": _source_fields(manifest, manifest_path,
                                          ("episode_id", "seed"))}}


@lru_cache(maxsize=210)
def formal_identity(episode_id: str) -> dict[str, Any]:
    return read_formal_identity(episode_id)


def formal_scope(episode_id: str, *, source_root: Path,
                 start_tick: int, end_tick: int) -> dict[str, Any]:
    identity = formal_identity(episode_id)
    if source_root.resolve() != _FORMAL_OBJECTIVE / episode_id:
        raise ValueError(f"formal source root differs from sample: {source_root}")
    return {**identity, "source_root": source_path(source_root),
            "start_tick": start_tick, "end_tick": end_tick}


def arm_identity(branch_root: Path, manifest: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """Bind one four-part ARM path to the fields actually present in its manifest."""
    branch_root = (REPO / branch_root).resolve() if not branch_root.is_absolute() else branch_root.resolve()
    if not branch_root.is_relative_to(_ARM_ROOT):
        raise ValueError(f"ARM branch is outside arms/: {branch_root}")
    relative = branch_root.relative_to(_ARM_ROOT)
    if len(relative.parts) != 4:
        raise ValueError(f"ARM branch is not four-part: {branch_root}")
    profile, seed, group, branch = relative.parts
    if _SEED.fullmatch(seed) is None or not profile or not group or not branch:
        raise ValueError(f"ARM branch path lacks profile/seedXX/group/branch: {branch_root}")
    manifest_path = branch_root / "manifest.json"
    if manifest is None:
        manifest = read_json(manifest_path)
    base = f"{profile}__{seed}"
    arm_key = relative.as_posix()
    expected = {"episode_id": base, "parent_episode_id": base,
                "profile": profile,
                "group_key": f"{profile}/{seed}/{group}",
                "arm_id": branch, "arm_key": arm_key}
    if manifest.get("episode_id") != base or manifest.get("arm_id") != branch:
        raise ValueError(f"ARM manifest episode/branch identity differs: {manifest_path}")
    for name, value in expected.items():
        if name in manifest and (type(manifest[name]) is not str or manifest[name] != value):
            raise ValueError(f"ARM manifest {name} differs from branch path: {manifest_path}")
    if "seed" in manifest:
        _seed_from_raw(manifest["seed"], seed,
                       source=f"{source_path(manifest_path)}:seed", allow_string=True)
    return {"sample_key": "arm:" + arm_key, "source_kind": "arm",
            "episode_id": base, "profile": profile, "seed": seed,
            "group": group, "branch": branch, "arm_key": arm_key,
            "source_root": source_path(branch_root),
            "identity_sources": {
                "directory": {"source_path": source_path(branch_root),
                              "segments": [{"segment_index": index, "role": role,
                                            "raw_value": value,
                                            "source_type": "string"}
                                           for index, (role, value) in enumerate((
                                               ("profile", profile), ("seed", seed),
                                               ("group", group), ("branch", branch)))]},
                "fields": _source_fields(manifest, manifest_path,
                    ("episode_id", "parent_episode_id", "profile", "seed",
                     "group_key", "arm_id", "arm_key"))}}


@lru_cache(maxsize=690)
def _arm_identity_from_relative(relative: str) -> dict[str, Any]:
    return arm_identity(_ARM_ROOT / relative)


def arm_scope(branch_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = arm_identity(branch_root, manifest)
    window = manifest["window"]
    start_tick, end_tick = window["start_tick"], window["end_tick"]
    if type(start_tick) is not int or type(end_tick) is not int or start_tick > end_tick:
        raise ValueError(f"ARM window ticks differ: {branch_root}/manifest.json")
    return {**identity, "source_root": source_path(branch_root),
            "start_tick": start_tick, "end_tick": end_tick}


def validate_scope_identity(scope: Mapping[str, Any]) -> None:
    kind = scope["source_kind"]
    profile, seed = _episode_parts(scope["episode_id"])
    if scope["profile"] != profile or scope["seed"] != seed:
        raise ValueError(f"sample scope profile/seed differs: {scope['sample_key']}")
    if kind == "formal":
        if (scope["sample_key"] != "formal:" + scope["episode_id"]
                or any(scope.get(name) is not None for name in ("group", "branch", "arm_key"))):
            raise ValueError(f"formal sample scope differs: {scope['sample_key']}")
        expected_root = _FORMAL_OBJECTIVE / scope["episode_id"]
    elif kind == "arm":
        arm_key = f"{profile}/{seed}/{scope['group']}/{scope['branch']}"
        if (scope.get("arm_key") != arm_key or scope["sample_key"] != "arm:" + arm_key):
            raise ValueError(f"ARM sample scope differs: {scope['sample_key']}")
        expected_root = _ARM_ROOT / arm_key
    else:
        raise ValueError(f"unknown sample source kind: {kind}")
    if "source_root" in scope and REPO / scope["source_root"] != expected_root:
        raise ValueError(f"sample source root differs: {scope['sample_key']}")


def identity_envelope(scope: Mapping[str, Any]) -> dict[str, Any]:
    validate_scope_identity(scope)
    return {"identity_status": "sample", "sample_key": scope["sample_key"],
            "source_kind": scope["source_kind"], "episode_id": scope["episode_id"],
            "profile": scope["profile"], "seed": scope["seed"],
            "group": scope.get("group"), "branch": scope.get("branch"),
            "arm_key": scope.get("arm_key")}


def _identity_fields_for_family(source_family: str) -> tuple[str, ...]:
    if source_family == "arm_manifest":
        return ("episode_id", "parent_episode_id", "profile", "seed",
                "group_key", "arm_id", "arm_key")
    if source_family == "formal_episode_manifest":
        return ("episode_id", "seed")
    if (source_family in _ARM_ROW_FAMILIES or source_family.startswith("arm_chain_nodes:")
            or source_family.startswith("arm_chain_events:")):
        return ("episode_id", "parent_episode_id", "arm_key")
    if (_formal_root_for_family(source_family) is not None
            or source_family == "formal_objective_manifest"):
        return ("episode_id",)
    return ()


def normalize_identity_field(scope: Mapping[str, Any], source_family: str,
                             field_path: str, raw_value: Any, *,
                             source_file: str, source_line: int | None = None
                             ) -> tuple[str, str] | None:
    """Normalize only a top-level source field with a declared sample role."""
    if field_path not in _identity_fields_for_family(source_family):
        return None
    where = f"{source_file}:{source_line}" if source_line is not None else source_file
    validate_scope_identity(scope)
    base = scope["episode_id"]
    if field_path == "seed":
        _seed_from_raw(raw_value, scope["seed"], source=f"{where}:seed",
                       allow_string=source_family == "arm_manifest")
        return "structural:sample:seed", scope["seed"]
    if type(raw_value) is not str:
        raise ValueError(f"sample identity field has invalid type: {where}:{field_path}")
    if field_path == "episode_id":
        allowed = ({base, base + "::" + scope["arm_key"]}
                   if scope["source_kind"] == "arm" and source_family != "arm_manifest"
                   else {base})
        if raw_value not in allowed:
            raise ValueError(f"source episode ID differs from sample: {where}:{raw_value!r}")
        return "structural:sample:episode_id", base
    if field_path == "parent_episode_id":
        if raw_value != base:
            raise ValueError(f"source parent episode differs from sample: {where}:{raw_value!r}")
        return "structural:sample:episode_id", base
    expected = {"arm_key": scope.get("arm_key"), "profile": scope["profile"],
                "group_key": f"{scope['profile']}/{scope['seed']}/{scope['group']}",
                "arm_id": scope.get("branch")}[field_path]
    if raw_value != expected:
        raise ValueError(f"source {field_path} differs from sample: {where}:{raw_value!r}")
    normalized_id = {"arm_key": "structural:sample:arm_key",
                     "profile": "structural:sample:profile",
                     "group_key": "structural:sample:group_key",
                     "arm_id": "structural:sample:branch"}[field_path]
    return normalized_id, expected


def validate_record_identity(record: Mapping[str, Any], scope: Mapping[str, Any], *,
                             source_family: str, source_file: str,
                             source_line: int | None = None) -> dict[str, tuple[str, str]]:
    """Check identity fields present in a real source row; absence stays absent."""
    return {name: normalized for name in _identity_fields_for_family(source_family)
            if name in record
            if (normalized := normalize_identity_field(scope, source_family, name,
                record[name], source_file=source_file, source_line=source_line)) is not None}


def _formal_root_for_family(source_family: str) -> tuple[Path, str] | None:
    if source_family in _FORMAL_RENDER_FILES:
        return _FORMAL_RENDER, _FORMAL_RENDER_FILES[source_family]
    if source_family == "formal_objective_manifest":
        return _FORMAL_OBJECTIVE, "manifest.json"
    if source_family.startswith("formal_") and source_family not in _SHARED_FAMILIES:
        stem = source_family.removeprefix("formal_")
        if stem in {"world_truth_graph_base", "world_truth_graph_deltas",
                    "predicate_truth", "predicate_transitions", "predicate_continuity_breaks",
                    "event_occurrences", "event_outcomes"}:
            suffix = ".json" if stem == "world_truth_graph_base" else ".jsonl"
            return _FORMAL_OBJECTIVE, stem + suffix
    if source_family == "domain_state":
        return _FORMAL_OBJECTIVE, "domain_state_observations.jsonl"
    if source_family in {"compute_state", "communication_state"}:
        return _FORMAL_OBJECTIVE, _FORMAL_COMPUTE_FILES[source_family]
    if source_family in _FORMAL_COMPUTE_FILES:
        return _FORMAL_OBJECTIVE, {"predicate_truth": "compute_predicate_truth.jsonl",
            "predicate_truth_matrix": "compute_predicate_matrix.jsonl", "events": "compute_events.jsonl"}[source_family]
    if source_family == "formal_event_chain_relations":
        return _FORMAL_CHAIN, "event_chain_relations.jsonl"
    return None


def formal_source_family(path: Path, episode_id: str) -> str:
    """Name only a declared formal source path for row identity validation."""
    path = path.resolve()
    for family in (*_FORMAL_RENDER_FILES, "formal_objective_manifest",
                   "formal_world_truth_graph_base", "formal_world_truth_graph_deltas",
                   "formal_predicate_truth", "formal_predicate_transitions",
                   "formal_predicate_continuity_breaks", "formal_event_occurrences",
                   "formal_event_outcomes", "domain_state", *_FORMAL_COMPUTE_FILES,
                   "formal_event_chain_relations"):
        root_file = _formal_root_for_family(family)
        if root_file is not None and path == root_file[0] / episode_id / root_file[1]:
            return family
    raise ValueError(f"formal source path is not declared for {episode_id}: {path}")


def arm_chain_row_family(source_family: str, row: Mapping[str, Any]) -> str:
    """Name a chain node or event from its source row discriminator."""
    if source_family not in {"arm_chain_nodes", "arm_chain_events"}:
        raise ValueError(f"source family is not an ARM chain row: {source_family}")
    if not isinstance(row, Mapping):
        raise TypeError(f"ARM chain row is not an object: {source_family}")
    if (source_family == "arm_chain_events" and "kind" not in row
            and row.keys() == _L2_CHAIN_EVENT_FIELDS):
        return "arm_chain_events:l2_event_occurrence"
    kind = row.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"ARM chain row has no kind: {source_family}")
    return f"{source_family}:{kind}"


def arm_source_family(branch_root: Path, path: Path, row: Mapping[str, Any]) -> str:
    """Name a branch-local source path, including chain row discriminators."""
    path = path.resolve()
    branch_root = branch_root.resolve()
    if not path.is_relative_to(branch_root):
        raise ValueError(f"ARM source is outside branch: {path}")
    relative = path.relative_to(branch_root).as_posix()
    if relative == "chain/nodes.jsonl" or relative == "chain/events.jsonl":
        return arm_chain_row_family("arm_chain_" + path.stem, row)
    for family, suffix in _ARM_FILES.items():
        if relative == suffix:
            return family
    raise ValueError(f"ARM source path is not declared: {path}")


@lru_cache(maxsize=4096)
def _canonical_source_path(source_file: str) -> Path:
    """Resolve each immutable release source once within a worker run."""
    path = (REPO / source_file).resolve()
    if source_path(path) != source_file:
        raise ValueError(f"source file is not a canonical repository path: {source_file}")
    return path


def source_scope_for_record(source_family: str, source_file: str,
                            record: Mapping[str, Any], owner_scope: Mapping[str, Any]
                            ) -> dict[str, Any]:
    """Resolve a source's own scope; shared files keep no invented sample ID."""
    validate_scope_identity(owner_scope)
    path = _canonical_source_path(source_file)
    if source_family in _SHARED_FAMILIES or (
            source_family == "arm_script_plan" and not source_file.startswith("arms/")):
        if source_family == "rule_contract":
            valid = path.is_relative_to(_RULES)
        elif source_family == "script_category_source":
            valid = (path.is_relative_to(_SCENARIOS) or path.is_relative_to(_DESIGN))
        elif source_family == "arm_scene_setup":
            if owner_scope["source_kind"] != "arm":
                raise ValueError(f"ARM scene has no ARM owner: {source_file}")
            manifest = read_json(REPO / owner_scope["source_root"] / "manifest.json")
            valid = (manifest_reference(manifest, "scene") == source_file and
                     any(path.is_relative_to(root) for root in
                         (_SCENARIOS, _DESIGN, _INTERVENTIONS)))
        elif source_family in {"formal_source_script", "formal_source_scene_setup"} or (
                source_family == "arm_script_plan" and owner_scope["source_kind"] == "formal"):
            if owner_scope["source_kind"] != "formal":
                raise ValueError(f"formal script/scene has no formal owner: {source_file}")
            manifest = read_json(_FORMAL_RENDER / owner_scope["episode_id"] /
                                 "episode_manifest.json")
            key = ("source_scene_setup_path" if source_family == "formal_source_scene_setup"
                   else "source_event_script_path")
            valid = manifest[key] == source_file and path.is_relative_to(_SCENARIOS)
        elif source_family == "arm_script_plan" and owner_scope["source_kind"] == "arm":
            manifest = read_json(REPO / owner_scope["source_root"] / "manifest.json")
            valid = (manifest_reference(manifest, "script") == source_file and
                     any(path.is_relative_to(root) for root in (_SCENARIOS, _DESIGN)))
        else:
            valid = False
        if not valid:
            raise ValueError(f"shared source family/path differs: {source_family}:{source_file}")
        return {"identity_status": "not_applicable", "sample_key": None,
                "source_kind": None, "episode_id": None, "profile": None,
                "seed": None, "group": None, "branch": None, "arm_key": None,
                "source_file": source_file, "relation": "shared_reference"}
    formal_root = _formal_root_for_family(source_family)
    if formal_root is not None:
        root, filename = formal_root
        if path.name != filename or path.parent.parent != root:
            raise ValueError(f"formal source family/path differs: {source_family}:{source_file}")
        scope = formal_identity(path.parent.name)
    elif source_family == "arm_semantic_history":
        if owner_scope["source_kind"] != "arm":
            raise ValueError(f"L6 history has no ARM owner: {source_file}")
        history_ref = manifest_reference(
            read_json(REPO / owner_scope["source_root"] / "manifest.json"),
            "semantic_history")
        if (history_ref != source_file or
                path.parent != _DESIGN / owner_scope["profile"] /
                owner_scope["seed"] / owner_scope["group"] / "history" or
                path.suffixes[-2:] != [".json", ".gz"]):
            raise ValueError(f"L6 history reference differs from ARM owner: {source_file}")
        scope = _arm_identity_from_relative(owner_scope["arm_key"])
    elif source_family in _ARM_ROW_FAMILIES or source_family in {"arm_branch_roster", "arm_script_plan"} or (
            source_family.startswith("arm_chain_nodes:") or
            source_family.startswith("arm_chain_events:")):
        if not path.is_relative_to(_ARM_ROOT):
            raise ValueError(f"ARM source family/path differs: {source_family}:{source_file}")
        parts = path.relative_to(_ARM_ROOT).parts
        if len(parts) < 5:
            raise ValueError(f"ARM source path lacks four-part branch: {source_file}")
        relative = "/".join(parts[:4])
        branch_root = _ARM_ROOT / relative
        expected_family = arm_source_family(branch_root, path, record)
        if source_family != expected_family and not (
                {source_family, expected_family} == {
                    "arm_predicate_truth_ticks", "arm_domain_state_ticks"}):
            raise ValueError(f"ARM source family/path differs: {source_family}:{source_file}")
        scope = _arm_identity_from_relative(relative)
    else:
        raise ValueError(f"source family has no identity scope rule: {source_family}:{source_file}")
    validate_record_identity(record, scope, source_family=source_family,
                             source_file=source_file)
    envelope = identity_envelope(scope)
    envelope["source_file"] = source_file
    if envelope["sample_key"] == owner_scope["sample_key"]:
        envelope["relation"] = "same_sample"
    elif (owner_scope["source_kind"] == "arm" and envelope["source_kind"] == "formal"
          and envelope["episode_id"] == owner_scope["episode_id"]
          and source_family in {"formal_roster", "formal_truth_frame"}):
        manifest = read_json(REPO / owner_scope["source_root"] / "manifest.json")
        if source_family == "formal_roster":
            if manifest_reference(manifest, "roster") != source_file:
                raise ValueError(f"ARM parent roster reference differs: {source_file}")
        else:
            manifest_scope = manifest.get("scope")
            if (not isinstance(manifest_scope, Mapping) or
                    manifest_scope.get("background_policy") !=
                    "frozen_parent_reference_no_feedback_claim"):
                raise ValueError(f"ARM parent truth frame lacks frozen-reference policy: {source_file}")
            trajectory_ref = manifest_reference(manifest, "trajectories")
            if trajectory_ref is None:
                raise ValueError(f"ARM parent trajectory reference is absent: {source_file}")
            trajectory_path = (REPO / trajectory_ref).resolve()
            if (trajectory_path.name != "trajectories.jsonl" or
                    source_path(trajectory_path.with_name("truth_frames.jsonl")) != source_file):
                raise ValueError(f"ARM parent truth frame reference differs: {source_file}")
        envelope["relation"] = "parent_formal"
    else:
        raise ValueError(f"source sample has no declared relation to owner: {source_file}")
    return envelope


def sample_identity_for_key(sample_key: str) -> dict[str, Any]:
    """Resolve an owner from its declared sample path and current manifest."""
    if not isinstance(sample_key, str):
        raise ValueError(f"sample key has invalid type: {sample_key!r}")
    if sample_key.startswith("formal:"):
        scope = formal_identity(sample_key.removeprefix("formal:"))
    elif sample_key.startswith("arm:"):
        scope = _arm_identity_from_relative(sample_key.removeprefix("arm:"))
    else:
        raise ValueError(f"sample key has no declared identity: {sample_key!r}")
    if scope["sample_key"] != sample_key:
        raise ValueError(f"sample key differs from declared identity: {sample_key}")
    return scope


def node_source_scope(node: Mapping[str, Any], source_family: str | None,
                      owner_scope: Mapping[str, Any] | None = None, *,
                      require_stored: bool = False
                      ) -> dict[str, Any] | None:
    """Recheck a node's owner and saved source envelope against current sources."""
    owner = owner_scope if owner_scope is not None else sample_identity_for_key(node["sample_key"])
    validate_scope_identity(owner)
    if node["sample_key"] != owner["sample_key"]:
        raise ValueError(f"node owner sample differs: {node['node_id']}")
    attrs = node["attrs"]
    raw = attrs.get("raw")
    stored = attrs.get("source_scope")
    stored_present = "source_scope" in attrs
    if raw is None:
        if stored_present:
            raise ValueError(f"node has source scope without raw source: {node['node_id']}")
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"node raw source is not an object: {node['node_id']}")
    if source_family is None and node["source_file"].endswith(".json.gz") and owner["source_kind"] == "arm":
        source_family = "arm_semantic_history"
    if source_family is None:
        if stored_present:
            raise ValueError(f"node has unresolvable source scope: {node['node_id']}")
        return None
    actual = source_scope_for_record(source_family, node["source_file"], raw, owner)
    if require_stored and not stored_present:
        raise ValueError(f"node source scope is missing: {node['node_id']}")
    if stored_present and stored != actual:
        raise ValueError(f"node source scope differs from source: {node['node_id']}")
    return actual


def inventory_scope(source_family: str, path: Path, sample: str
                    ) -> dict[str, Any] | None:
    """Return a sample only for its own declared source file, never for shared refs."""
    if source_family in _SHARED_FAMILIES or source_family in {
            "formal_source_script", "formal_source_scene_setup", "script_category_source"}:
        return None
    if _formal_root_for_family(source_family) is not None:
        scope = formal_identity(sample)
    elif source_family in _ARM_ROW_FAMILIES or source_family in {"arm_branch_roster", "arm_script_plan"} or (
            source_family.startswith("arm_chain_nodes:") or
            source_family.startswith("arm_chain_events:")):
        if not sample.startswith("arms/"):
            raise ValueError(f"ARM inventory sample is not a branch: {sample}")
        scope = _arm_identity_from_relative(sample.removeprefix("arms/"))
    else:
        raise ValueError(f"source inventory family has no identity rule: {source_family}:{path}")
    if source_family == "arm_semantic_history":
        history_ref = manifest_reference(
            read_json(REPO / scope["source_root"] / "manifest.json"),
            "semantic_history")
        if (history_ref != source_path(path) or
                path.parent != REPO / "design" / scope["profile"] /
                scope["seed"] / scope["group"] / "history"):
            raise ValueError(f"ARM history inventory reference differs: {path}")
    elif source_family in {"arm_chain_nodes", "arm_chain_events"}:
        expected = REPO / scope["source_root"] / _ARM_FILES[source_family]
        if path.resolve() != expected.resolve():
            raise ValueError(f"ARM inventory family/path differs: {source_family}:{path}")
    else:
        resolved = source_scope_for_record(source_family, source_path(path), {}, scope)
        if resolved["sample_key"] != scope["sample_key"]:
            raise ValueError(f"source inventory sample/path differs: {source_family}:{path}")
    return scope
