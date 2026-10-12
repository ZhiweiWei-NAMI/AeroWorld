"""Read-only review data interface for the current ARM delivery.

Public API (consumed later by an HTTP layer):
- ``ArmReviewError``: raised on any missing, malformed, or inconsistent input.
- ``catalog(root)``: scan ``arms/index.jsonl``, cross-check against
  ``design/ue_arm_delivery/delivery_manifest.json`` and the package
  directories, and return ``{"sources", "counts", "delivery_cross_check",
  "groups", "arms"}``. Family breakdown lives in ``counts["by_family"]``.
  Any structural problem (missing directory,
  missing configuration/manifest, key/path/window/configuration mismatch,
  delivery-manifest gap or duplicate) raises ``ArmReviewError``.
- ``group_detail(root, profile, seed, group, *, catalog_data=None)``: full load
  of every arm in the group. ``catalog_data`` (keyword-only) accepts a
  previously validated ``catalog(root)`` result so batch callers scan the
  directory once; by default the catalog is recomputed. Every row carries
  ``evidence = {"path", "line", "raw"}`` with a repository-relative path and
  1-based JSONL line numbers (``line`` is null for JSON files).

Normalization kept for the UI (no invented content):
- transitions use ``to_tick`` + ``predicate_id`` (both required; missing or
  non-integer ``to_tick`` and missing ``predicate_id`` raise ``ArmReviewError``);
- X occurrences use ``tick`` + ``predicate_id``;
- L1/L2/L3/L5/L6 occurrences use ``trigger_tick`` + ``trigger_predicate_id``
  (both required);
- L4 occurrences use ``trigger_tick`` and keep ``predicate_key`` null when the
  row carries no predicate id (no fabricated ``None|...`` keys);
- ``bindings`` are preserved verbatim;
- outcomes keep ``terminal_tick`` null for pending and right-censored rows; the field itself must
  be present (a missing key raises ``ArmReviewError``).
- relation endpoints resolve against chain nodes and objective ids; endpoints
  that reference an event phase whose tick lies before the arm's window
  ``start_tick`` are classified as outside-window references (reported in
  ``errors`` with kind ``relation_endpoint_outside_window`` and counted in
  ``outside_window_references``), any other unresolved endpoint is a real
  error (kind ``relation_endpoint``); nothing is fabricated.
- configuration YAML must parse to a mapping; empty files, lists, or scalars
  raise ``ArmReviewError``. Manifest audit fields present in the manifest are
  passed through verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

from Dataset.semantic_truth.window_outcome import (
    validate_arm_window_outcome,
    validate_l4_business_outcome,
)

INDEX_PATH = Path("arms/index.jsonl")
DELIVERY_MANIFEST_PATH = Path("design/ue_arm_delivery/delivery_manifest.json")
INTERVENTIONS_PREFIX = "interventions/"
ARM_MODES = ("factual_observation", "intervention")
NODE_SOURCE_BY_FAMILY = {
    "L1": "chain/events.jsonl",
    "L2": "chain/events.jsonl",
    "L4": "chain/events.jsonl",
    "L5": "chain/events.jsonl",
    "L3": "chain/nodes.jsonl",
    "L6": "chain/nodes.jsonl",
    "X": "chain/nodes.jsonl",
}


class ArmReviewError(ValueError):
    pass


def _repo_path(root: Path, relative: str) -> Path:
    """Resolve a repo-relative declared path, refusing escapes from root."""
    if not isinstance(relative, str) or not relative:
        raise ArmReviewError(f"Invalid declared path: {relative!r}")
    if Path(relative).is_absolute():
        raise ArmReviewError(f"Declared path must be relative: {relative}")
    parts = Path(relative).parts
    if ".." in parts or "." in parts:
        raise ArmReviewError(f"Declared path must not contain '..' or '.': {relative}")
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ArmReviewError(f"Declared path escapes repository root: {relative}")
    return candidate


def _read_json(root: Path, relative: str) -> tuple[dict, dict]:
    """Return (content, evidence) for a single JSON file under root."""
    path = _repo_path(root, relative)
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            content = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArmReviewError(f"Cannot read JSON {relative}: {exc}") from exc
    if not isinstance(content, dict):
        raise ArmReviewError(f"Expected JSON object: {relative}")
    evidence = {"path": relative, "line": None, "raw": content}
    return content, evidence


def _read_jsonl(root: Path, relative: str) -> list[dict]:
    """Read required JSONL rows, each wrapped with evidence {path, line, raw}."""
    path = _repo_path(root, relative)
    if not path.is_file():
        raise ArmReviewError(f"Missing required file: {relative}")
    rows = []
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ArmReviewError(
                        f"Malformed JSON: {relative}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ArmReviewError(f"Expected JSON object: {relative}:{line_number}")
                rows.append({
                    "evidence": {"path": relative, "line": line_number, "raw": row},
                })
    except OSError as exc:
        raise ArmReviewError(f"Cannot read {relative}: {exc}") from exc
    return rows


def _predicate_key(predicate_id, bindings) -> str | None:
    """Stable key: predicate id + canonical bindings serialization; null id -> null."""
    if predicate_id is None:
        return None
    bindings_part = json.dumps(bindings, sort_keys=True, separators=(",", ":"))
    return f"{predicate_id}|{bindings_part}"


def _require_tick(raw: dict, field: str, relative: str, line: int) -> int:
    """A required tick field must be present and an integer; never silently null."""
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArmReviewError(
            f"{relative}:{line}: required tick field {field!r} must be an integer, "
            f"got {value!r}")
    return value


def _normalized_transition_row(row: dict, relative: str) -> dict:
    raw = row["evidence"]["raw"]
    line = row["evidence"]["line"]
    predicate_id = raw.get("predicate_id")
    if not isinstance(predicate_id, str) or not predicate_id:
        raise ArmReviewError(
            f"{relative}:{line}: transition row lacks predicate_id: {json.dumps(raw)[:200]}")
    _require_tick(raw, "to_tick", relative, line)
    return {
        "tick": raw["to_tick"],
        "predicate_key": _predicate_key(predicate_id, raw.get("bindings")),
        "row": row,
        "evidence": row["evidence"],
    }


def _normalized_occurrence_row(row: dict, family: str, relative: str) -> dict:
    raw = row["evidence"]["raw"]
    line = row["evidence"]["line"]
    if family == "X":
        tick_field = "tick"
        predicate_field = "predicate_id"
    else:
        tick_field = "trigger_tick"
        predicate_field = "trigger_predicate_id"
    _require_tick(raw, tick_field, relative, line)
    predicate_id = raw.get(predicate_field)
    if family == "X" and not (isinstance(predicate_id, str) and predicate_id):
        raise ArmReviewError(
            f"{relative}:{line}: X occurrence row lacks {predicate_field!r}: "
            f"{json.dumps(raw)[:200]}")
    # L4 rows may legitimately lack a predicate id; everything else must carry one.
    if family != "L4" and family != "X" and not (isinstance(predicate_id, str) and predicate_id):
        raise ArmReviewError(
            f"{relative}:{line}: occurrence row lacks {predicate_field!r}: "
            f"{json.dumps(raw)[:200]}")
    return {
        "tick": raw[tick_field],
        "predicate_key": _predicate_key(predicate_id, raw.get("bindings")),
        "row": row,
    }


def _normalize_relation(raw: dict) -> dict:
    """Normalize L2 from/to/relation, L1-L6 source/target/relation_type, X source/target/type."""
    if "source" in raw and "target" in raw:
        source, target = raw["source"], raw["target"]
    elif "from" in raw and "to" in raw:
        source, target = raw["from"], raw["to"]
    else:
        raise ArmReviewError(f"Relation row lacks endpoints: {json.dumps(raw)[:200]}")
    rel_type = raw.get("type", raw.get("relation_type", raw.get("relation")))
    if rel_type is None:
        raise ArmReviewError(f"Relation row lacks relation type: {json.dumps(raw)[:200]}")
    return {"source": source, "target": target, "type": rel_type, "raw": raw}


def _validate_path_segment(value, label: str) -> str:
    """One profile/seed/group segment must be a single path component."""
    if not isinstance(value, str) or not value or "/" in value or value in (".", ".."):
        raise ArmReviewError(f"Invalid {label} segment: {value!r}")
    return value


def _validate_index_row(row: dict, line_number: int, index_display: str) -> dict:
    for field in ("key", "path", "family", "profile", "seed", "group", "arm_id",
                  "window", "configuration"):
        if field not in row:
            raise ArmReviewError(f"Index row missing {field!r}: {index_display}:{line_number}")
    profile = _validate_path_segment(row["profile"], "profile")
    seed = _validate_path_segment(row["seed"], "seed")
    group = _validate_path_segment(row["group"], "group")
    expected_key = f"{profile}/{seed}/{group}/{row['arm_id']}"
    if row["key"] != expected_key:
        raise ArmReviewError(
            f"Index key {row['key']!r} != profile/seed/group/arm_id {expected_key!r}: "
            f"{index_display}:{line_number}")
    expected_path = f"arms/{row['key']}"
    if row["path"] != expected_path:
        raise ArmReviewError(
            f"Index path {row['path']!r} != 'arms/' + key ({expected_path!r}): "
            f"{index_display}:{line_number}")
    configuration = row["configuration"]
    if not isinstance(configuration, str) or not configuration.startswith(INTERVENTIONS_PREFIX):
        raise ArmReviewError(
            f"Configuration must live under {INTERVENTIONS_PREFIX}*: {configuration!r}: "
            f"{index_display}:{line_number}")
    config_suffix = ".json" if row["family"] == "L6" else ".yaml"
    expected_config = f"{INTERVENTIONS_PREFIX}{profile}/{seed}/{group}/{row['arm_id']}{config_suffix}"
    if configuration != expected_config:
        raise ArmReviewError(
            f"Configuration {configuration!r} != expected {expected_config!r}: "
            f"{index_display}:{line_number}")
    return row


def _load_index(root: Path) -> list[dict]:
    index_path = _repo_path(root, INDEX_PATH.as_posix())
    if not index_path.is_file():
        raise ArmReviewError(f"Missing ARM index: {INDEX_PATH.as_posix()}")
    entries = []
    seen_keys: dict[str, int] = {}
    with index_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ArmReviewError(
                    f"Malformed index row: {INDEX_PATH}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ArmReviewError(f"Expected JSON object: {INDEX_PATH}:{line_number}")
            _validate_index_row(row, line_number, INDEX_PATH.as_posix())
            if row["key"] in seen_keys:
                raise ArmReviewError(
                    f"Duplicate index key {row['key']!r}: {INDEX_PATH}:{line_number} "
                    f"and line {seen_keys[row['key']]}")
            seen_keys[row["key"]] = line_number
            entries.append({"line": line_number, "row": row})
    if not entries:
        raise ArmReviewError(f"Empty ARM index: {INDEX_PATH}")
    return entries


def _load_delivery(root: Path) -> dict:
    delivery, _ = _read_json(root, DELIVERY_MANIFEST_PATH.as_posix())
    packages = delivery.get("packages")
    if not isinstance(packages, list):
        raise ArmReviewError("delivery_manifest.json missing a real 'packages' list")
    by_key: dict[str, dict] = {}
    for index, package in enumerate(packages):
        if not isinstance(package, dict) or "key" not in package:
            raise ArmReviewError(f"Delivery package #{index} lacks a key")
        key = package["key"]
        if key in by_key:
            raise ArmReviewError(f"Duplicate delivery package key: {key}")
        by_key[key] = package
    return {"delivery": delivery, "packages_by_key": by_key}


def _arm_entry_from_package(row: dict, line: int, mode: str | None) -> dict:
    return {
        "arm_key": row["key"],
        "family": row["family"],
        "profile": row["profile"],
        "seed": row["seed"],
        "group": row["group"],
        "arm_id": row["arm_id"],
        "path": row["path"],
        "window": row["window"],
        "configuration": row["configuration"],
        "mode": mode,
        "index_line": line,
        "in_delivery_manifest": True,
    }


def catalog(root: Path, *, allow_partial: bool = False) -> dict:
    """Validate delivery; browsing may expose complete groups plus explicit errors."""
    root = Path(root)
    entries = _load_index(root)
    delivery_info = _load_delivery(root)
    delivery = delivery_info["delivery"]
    packages_by_key = delivery_info["packages_by_key"]

    counts = {"arms": 0, "groups": 0}
    family_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    groups: dict[tuple, dict] = {}
    arms: dict[str, dict] = {}
    mismatches: list[dict] = []

    for entry in entries:
        row = entry["row"]
        line = entry["line"]
        group_key = (row["profile"], row["seed"], row["group"])
        group = groups.setdefault(group_key, {
            "family": row["family"], "profile": row["profile"], "seed": row["seed"],
            "group": row["group"], "arms": [], "modes": {}, "windows": {},
        })
        arm_dir = _repo_path(root, row["path"])
        issues = []
        if not arm_dir.is_dir():
            issues.append(f"directory missing: {row['path']}")
        if not (arm_dir / "manifest.json").is_file():
            issues.append("manifest.json missing")
        if not _repo_path(root, row["configuration"]).is_file():
            issues.append(f"configuration missing: {row['configuration']}")
        package = packages_by_key.get(row["key"])
        if package is None:
            issues.append("key absent from delivery_manifest.json packages")
        else:
            if package.get("path") != row["path"]:
                issues.append(
                    f"delivery path {package.get('path')!r} != index path {row['path']!r}")
            if package.get("configuration") != row["configuration"]:
                issues.append("delivery configuration differs from index")
            if package.get("window") != row["window"]:
                issues.append("delivery window differs from index")
            # Index/delivery windows carry start/end; the manifest window is the
            # superset (adds fork_tick, phase, ...), so cross-check those two
            # after the manifest is loaded below. Record intent here.
            window = row["window"]
            check_manifest_window = (arm_dir / "manifest.json").is_file()
        if issues:
            mismatches.append({"key": row["key"], "index_line": line, "issues": issues})
            continue

        try:
            manifest, _ = _read_json(root, f"{row['path']}/manifest.json")
            mode = manifest.get("mode")
            if mode not in ARM_MODES:
                raise ArmReviewError(
                    f"Unknown manifest mode {mode!r} for {row['key']} "
                    f"({row['path']}/manifest.json)")
            if manifest.get("arm_id") != row["arm_id"]:
                raise ArmReviewError(
                    f"Manifest arm_id {manifest.get('arm_id')!r} != index arm_id "
                    f"{row['arm_id']!r} for {row['key']}")
            manifest_window = manifest.get("window")
            if not isinstance(manifest_window, dict):
                raise ArmReviewError(
                    f"Manifest window must be an object for {row['key']}: {manifest_window!r}")
            for field in ("start_tick", "end_tick"):
                if manifest_window.get(field) != window.get(field):
                    raise ArmReviewError(
                        f"Manifest {field} {manifest_window.get(field)!r} != index "
                        f"{window.get(field)!r} for {row['key']}")
        except ArmReviewError as exc:
            if not allow_partial:
                raise
            mismatches.append({"key": row["key"], "index_line": line, "issues": [str(exc)]})
            continue

        counts["arms"] += 1
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        group["modes"][mode] = group["modes"].get(mode, 0) + 1
        group["windows"][str(row["window"])] = group["windows"].get(str(row["window"]), 0) + 1
        arm_entry = _arm_entry_from_package(row, line, mode)
        group["arms"].append(arm_entry)
        arms[row["key"]] = arm_entry

    counts["groups"] = len(groups)
    missing_from_index = sorted(set(packages_by_key) - {e["row"]["key"] for e in entries})
    if missing_from_index:
        mismatches.append({
            "key": None, "index_line": None,
            "issues": [f"{len(missing_from_index)} delivery keys absent from index",
                       *missing_from_index[:10]],
        })
    if mismatches and not allow_partial:
        raise ArmReviewError(f"ARM delivery cross-check failed: {mismatches[:5]}")
    if not allow_partial and counts["arms"] != delivery.get("arm_count"):
        raise ArmReviewError(
            f"Arm count {counts['arms']} != delivery arm_count {delivery.get('arm_count')}")
    if not allow_partial and counts["groups"] != delivery.get("group_count"):
        raise ArmReviewError(
            f"Group count {counts['groups']} != delivery group_count "
            f"{delivery.get('group_count')}")

    unavailable_groups = []
    if allow_partial:
        bad_groups = {m["key"].rsplit("/", 1)[0] for m in mismatches if m["key"]}
        bad_groups.update(key.rsplit("/", 1)[0] for key in missing_from_index)
        for key, group in list(groups.items()):
            group_path = "/".join(key)
            if group_path in bad_groups:
                unavailable_groups.append(group_path)
                del groups[key]
        arms = {a["arm_key"]: a for g in groups.values() for a in g["arms"]}
        counts = {"arms": len(arms), "groups": len(groups)}
        family_counts, mode_counts = {}, {}
        for arm in arms.values():
            family_counts[arm["family"]] = family_counts.get(arm["family"], 0) + 1
            mode_counts[arm["mode"]] = mode_counts.get(arm["mode"], 0) + 1
        if not mismatches and (counts["arms"] != delivery.get("arm_count") or counts["groups"] != delivery.get("group_count")):
            mismatches.append({"key": None, "index_line": None, "issues": ["delivery counts differ from available catalog"]})

    return {
        "sources": {
            "index": INDEX_PATH.as_posix(),
            "delivery_manifest": DELIVERY_MANIFEST_PATH.as_posix(),
        },
        "counts": {
            **counts,
            "by_family": family_counts,
            "by_mode": mode_counts,
        },
        "delivery_cross_check": {
            "delivery_arm_count": delivery.get("arm_count"),
            "delivery_group_count": delivery.get("group_count"),
            "consistent": not mismatches,
            "mismatches": mismatches,
            "unavailable_groups": sorted(unavailable_groups),
        },
        "groups": sorted(groups.values(), key=lambda g: (g["profile"], g["seed"], g["group"])),
        "arms": arms,
    }


def _validated_outcome_row(row: dict, family: str, window: dict) -> dict:
    """Validate a saved outcome before exposing its terminal status to the UI."""
    raw = row["evidence"]["raw"]
    relative = row["evidence"]["path"]
    line = row["evidence"]["line"]
    if not isinstance(window, dict):
        raise ArmReviewError(f"{relative}:{line}: ARM window is missing")
    start_tick, end_tick = window.get("start_tick"), window.get("end_tick")
    if (type(start_tick) is not int or type(end_tick) is not int
            or start_tick < 0 or start_tick > end_tick):
        raise ArmReviewError(f"{relative}:{line}: ARM window bounds are invalid")
    if family == "X":
        status, tick = raw.get("status"), raw.get("terminal_tick")
        if (type(raw.get("event_id")) is not str or not raw["event_id"]
                or (status == "recovered" and not (
                    set(raw) == {"event_id", "hold_samples", "status", "terminal_tick"}
                    and type(raw["hold_samples"]) is int and raw["hold_samples"] > 0
                    and type(tick) is int and start_tick <= tick <= end_tick))
                or (status == "right_censored" and not (
                    set(raw) == {"event_id", "status", "terminal_tick"} and tick is None))
                or status not in {"recovered", "right_censored"}):
            raise ArmReviewError(f"{relative}:{line}: invalid X window outcome")
    elif raw.get("schema_name") == "arm_window_event_outcome":
        try:
            validate_arm_window_outcome(raw, start_tick=start_tick, end_tick=end_tick)
        except ValueError as exc:
            raise ArmReviewError(f"{relative}:{line}: {exc}") from exc
    elif raw.get("schema_name") == "l4_business_event_outcome" and family == "L4":
        try:
            validate_l4_business_outcome(raw, start_tick=start_tick, end_tick=end_tick)
        except ValueError as exc:
            raise ArmReviewError(f"{relative}:{line}: {exc}") from exc
    else:
        raise ArmReviewError(
            f"{relative}:{line}: unsupported outcome schema {raw.get('schema_name')!r}")
    return {
        "evidence": row["evidence"],
        "event_id": raw.get("event_id"),
        "outcome_id": raw.get("outcome_id"),
        "terminal_tick": raw.get("terminal_tick"),
        "status": raw.get("status"),
    }


def _resolve_index(rows: list[dict]) -> dict[str, dict]:
    """Index by node/transition/event/outcome id from raw rows."""
    index: dict[str, dict] = {}
    for row in rows:
        raw = row["evidence"]["raw"]
        for key_field in ("node_id", "transition_id", "event_id", "outcome_id"):
            value = raw.get(key_field)
            if isinstance(value, str):
                index.setdefault(value, row)
    return index


def _endpoint_evidence(endpoint, node_index: dict[str, dict],
                       transitions: list[dict], occurrences: list[dict],
                       outcomes: list[dict]) -> dict | None:
    """Return the evidence of the real row behind a relation endpoint."""
    if not isinstance(endpoint, str):
        return None
    if endpoint in node_index:
        return node_index[endpoint]["evidence"]
    for rows in (transitions, occurrences, outcomes):
        for row in rows:
            raw = row["evidence"]["raw"]
            if raw.get("transition_id") == endpoint or raw.get("event_id") == endpoint \
                    or raw.get("outcome_id") == endpoint:
                return row["evidence"]
    return None


def _outside_window_phase_tick(endpoint, occurrences: list[dict],
                               window_start) -> int | None:
    """Return the phase tick when ``endpoint`` is referenced by an event phase
    whose tick lies strictly before the arm's ``start_tick``; else None."""
    if not isinstance(endpoint, str) or not isinstance(window_start, int):
        return None
    for row in occurrences:
        raw = row["evidence"]["raw"]
        phases = raw.get("event_phases")
        if not isinstance(phases, list):
            continue
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            if phase.get("transition_id") != endpoint:
                continue
            tick = phase.get("tick")
            if isinstance(tick, int) and not isinstance(tick, bool) and tick < window_start:
                return tick
    return None


def _load_configuration(root: Path, relative: str) -> dict:
    path = _repo_path(root, relative)
    if not path.is_file():
        raise ArmReviewError(f"Missing configuration file: {relative}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ArmReviewError(f"Cannot read configuration {relative}: {exc}") from exc
    try:
        import yaml
        content = yaml.safe_load(text)
    except ImportError as exc:
        raise ArmReviewError("PyYAML is required to parse ARM configurations") from exc
    except yaml.YAMLError as exc:
        raise ArmReviewError(f"Cannot parse configuration {relative}: {exc}") from exc
    if not isinstance(content, dict):
        raise ArmReviewError(
            f"Configuration must parse to a mapping: {relative} "
            f"(got {type(content).__name__})")
    return {"path": relative, "text": text, "content": content}


def group_detail(root: Path, profile: str, seed: str, group: str, *,
                 catalog_data: dict | None = None) -> dict:
    """Fully load one group: every arm's real artifacts with per-row evidence.

    ``catalog_data`` accepts a previously validated ``catalog(root)`` result so
    batch callers scan the directory once; by default the catalog is recomputed.
    """
    root = Path(root)
    _validate_path_segment(profile, "profile")
    _validate_path_segment(seed, "seed")
    _validate_path_segment(group, "group")
    cat = catalog_data if catalog_data is not None else catalog(root)
    matches = [g for g in cat["groups"]
               if (g["profile"], g["seed"], g["group"]) == (profile, seed, group)]
    if not matches:
        raise ArmReviewError(f"Group not found in index: {profile}/{seed}/{group}")
    group_entry = matches[0]

    arms_out = []
    errors: list[dict] = []
    unresolved_endpoints = 0
    for arm_entry in group_entry["arms"]:
        arm_dir_rel = arm_entry["path"]
        manifest, manifest_evidence = _read_json(root, f"{arm_dir_rel}/manifest.json")
        window_status_rel = f"{arm_dir_rel}/objective/window_status.json"
        window_status, window_status_evidence = _read_json(root, window_status_rel)
        configuration = _load_configuration(root, arm_entry["configuration"])

        actions = _read_jsonl(root, f"{arm_dir_rel}/raw/actions.jsonl")
        transitions = _read_jsonl(
            root, f"{arm_dir_rel}/objective/predicate_transitions.jsonl")
        occurrences = _read_jsonl(
            root, f"{arm_dir_rel}/objective/event_occurrences.jsonl")
        outcomes = _read_jsonl(
            root, f"{arm_dir_rel}/objective/event_outcomes.jsonl")

        family = arm_entry["family"]
        node_source = NODE_SOURCE_BY_FAMILY[family]
        nodes = _read_jsonl(root, f"{arm_dir_rel}/{node_source}")
        relations_rows = _read_jsonl(root, f"{arm_dir_rel}/chain/relations.jsonl")

        node_index = _resolve_index(nodes)
        window_start = manifest.get("window", {}).get("start_tick")
        relations = []
        for row in relations_rows:
            try:
                rel = _normalize_relation(row["evidence"]["raw"])
            except ArmReviewError as exc:
                errors.append({"arm": arm_entry["arm_key"], "kind": "relation",
                               "message": str(exc),
                               "evidence": row["evidence"]})
                continue
            source_ev = _endpoint_evidence(
                rel["source"], node_index, transitions, occurrences, outcomes)
            target_ev = _endpoint_evidence(
                rel["target"], node_index, transitions, occurrences, outcomes)
            for side, endpoint, ev in (
                ("source", rel["source"], source_ev),
                ("target", rel["target"], target_ev),
            ):
                if ev is not None:
                    continue
                phase_tick = _outside_window_phase_tick(
                    endpoint, occurrences, window_start)
                if phase_tick is not None:
                    errors.append({
                        "arm": arm_entry["arm_key"],
                        "kind": "relation_endpoint_outside_window",
                        "message": (
                            f"outside-window reference on {side} endpoint "
                            f"{endpoint!r}: event phase at tick {phase_tick} "
                            f"before window start_tick {window_start}"),
                        "endpoint": endpoint,
                        "side": side,
                        "phase_tick": phase_tick,
                        "evidence": row["evidence"],
                    })
                else:
                    unresolved_endpoints += 1
                    errors.append({
                        "arm": arm_entry["arm_key"], "kind": "relation_endpoint",
                        "message": f"unresolved {side} endpoint {endpoint!r}",
                        "endpoint": endpoint, "side": side,
                        "evidence": row["evidence"],
                    })
            relations.append({**rel,
                              "source_evidence": source_ev,
                              "target_evidence": target_ev,
                              "evidence": row["evidence"]})

        arm_detail = {
            "arm_key": arm_entry["arm_key"],
            "family": family,
            "arm_id": arm_entry["arm_id"],
            "path": arm_dir_rel,
            # Real per-arm mode/window from the manifest, never inferred from arm_id.
            "mode": manifest.get("mode"),
            "episode_id": manifest.get("episode_id"),
            "window": manifest.get("window"),
            "index_window": arm_entry["window"],
            "configuration": configuration,
            "manifest": {"evidence": manifest_evidence, "content": manifest},
            "window_status": {"evidence": window_status_evidence, "content": window_status},
            # Audit fields present in the manifest, verbatim; absent fields stay absent.
            "audit": {field: manifest[field] for field in
                      ("certification", "review",
                       "baseline", "limitations") if field in manifest},
            "raw_actions": actions,
            "objective": {
                "transitions": [_normalized_transition_row(r, f"{arm_dir_rel}/objective/predicate_transitions.jsonl")
                                for r in transitions],
                "occurrences": [_normalized_occurrence_row(r, family, f"{arm_dir_rel}/objective/event_occurrences.jsonl")
                                | {"evidence": r["evidence"]} for r in occurrences],
                "outcomes": [_validated_outcome_row(r, family, manifest.get("window")) for r in outcomes],
            },
            "chain": {
                "node_source": node_source,
                "nodes": nodes,
                "relations": relations,
            },
        }
        arms_out.append(arm_detail)

    return {
        "profile": profile,
        "seed": seed,
        "group": group,
        "family": group_entry["family"],
        "arms": arms_out,
        "errors": errors,
        "unresolved_relation_endpoints": unresolved_endpoints,
        "outside_window_references": sum(
            1 for e in errors if e["kind"] == "relation_endpoint_outside_window"),
    }
