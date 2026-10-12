#!/usr/bin/env python3
"""Build the read-only, preliminary Arm review catalog.

The formal Arm release remains validator-gated.  This catalog is deliberately
different from that release boundary: it accounts for every candidate Group
and all five authored Arm kinds, while exposing only artifacts that were
actually persisted by a worker.  A skipped Arm is represented by its audit
status and finding codes; no trajectory or plot is invented for it.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY_ROOT = PROJECT_ROOT / "artifacts/branches/aero210_full_chain_action_branches"
DEFAULT_AUDIT = PROJECT_ROOT / "artifacts/audits/aero210_release_audit_v31_20260904" / "arm_delta_release_audit.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/audits/aero210_action_branch_review_20260904"
DEFAULT_CATALOG_NAME = "preliminary_catalog.json"
REQUIRED_ARM_KINDS = ("factual", "no_op_hold", "delay_recover", "avoid", "trade_off")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CatalogError(RuntimeError):
    """Raised when the review boundary cannot be built fail-closed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogError(f"JSON artifact root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CatalogError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read JSONL artifact {path}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CatalogError(f"cannot hash provenance artifact {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError as exc:
        raise CatalogError(f"path is outside project root: {path}") from exc


def _safe_project_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        return None
    candidate = PROJECT_ROOT / raw
    if candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve()
        resolved.relative_to(PROJECT_ROOT.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _canonical_key(row: Mapping[str, Any]) -> str:
    value = row.get("branch_key")
    if not isinstance(value, str) or value.count("/") != 4:
        raise CatalogError(f"audit row has malformed branch_key: {value!r}")
    return value


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", "..."):
            return value
    return None


def _artifact_file(root: Path, name: str) -> Path | None:
    for parent in (root / "raw", root, root / "objective"):
        for suffix in ("", ".gz"):
            path = parent / f"{name}{suffix}"
            if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
                return path
    return None


def _objective_file(root: Path, name: str) -> Path | None:
    for suffix in ("", ".gz"):
        path = root / "objective" / f"{name}{suffix}"
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
            return path
    return None


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
                value = json.load(stream)
        else:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"malformed optional JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogError(f"optional JSON artifact root is not an object: {path}")
    return value


def _load_candidate_records(
    inventory_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return candidate Arms and Groups keyed by the canonical branch key."""

    candidates: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    paths = sorted(inventory_root.rglob("group_manifest.json"))
    if not paths:
        raise CatalogError(f"candidate inventory has no group manifests: {inventory_root}")
    for manifest_path in paths:
        group = _read_json(manifest_path)
        group_dir = manifest_path.parent
        episode = str(group.get("episode_id") or "")
        profile = str(group.get("scenario_id") or "")
        seed = str(group.get("seed_id") or "")
        if not episode or not profile or not seed:
            raise CatalogError(f"candidate Group identity is incomplete: {manifest_path}")
        group_id = group_dir.name
        group_key = "/".join((profile, episode, seed, group_id))
        if group_key in groups:
            raise CatalogError(f"duplicate candidate Group: {group_key}")
        arm_paths = sorted(group_dir.glob("arm_*.json"))
        arms: dict[str, dict[str, Any]] = {}
        for arm_path in arm_paths:
            arm = _read_json(arm_path)
            arm_id = str(arm.get("arm_id") or arm_path.stem)
            arm_kind = str(
                arm.get("arm_kind")
                or (arm.get("decision_arm") or {}).get("arm_kind")
                or ""
            )
            key = "/".join((profile, episode, seed, group_id, arm_id))
            if key in candidates:
                raise CatalogError(f"duplicate candidate Arm: {key}")
            if arm_kind in arms:
                raise CatalogError(f"duplicate Arm kind in Group {group_key}: {arm_kind}")
            arms[arm_kind] = {
                "arm_id": arm_id,
                "arm_kind": arm_kind,
                "path": _relative(arm_path),
                "record": arm,
                "key": key,
            }
            candidates[key] = arms[arm_kind]
        if set(arms) != set(REQUIRED_ARM_KINDS):
            raise CatalogError(
                f"Group {group_key} has Arm kinds {sorted(arms)!r}; "
                f"expected {list(REQUIRED_ARM_KINDS)!r}"
            )
        groups[group_key] = {
            "profile_id": profile,
            "episode_id": episode,
            "seed_id": seed,
            "group_id": group_id,
            "branch_group_id": str(group.get("branch_group_id") or ""),
            "family": str(group.get("event_family_id") or group.get("family") or ""),
            "anchor_event_id": str(group.get("anchor_event_id") or ""),
            "fork_tick": _int(group.get("fork_tick")),
            "window_end_tick": _int(group.get("window_end_tick")),
            "target_predicate_id": str(group.get("target_predicate_id") or ""),
            "rule_id": str(group.get("rule_id") or ""),
            "manifest_path": _relative(manifest_path),
            "manifest": group,
            "arms": arms,
        }
    return candidates, groups


def _load_audit_rows(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    for path in paths:
        for row in _read_jsonl(path):
            key = _canonical_key(row)
            if key in rows:
                duplicate_keys.append(key)
            rows[key] = row
    if duplicate_keys:
        raise CatalogError(
            "audit inputs contain duplicate canonical Arms; use one independent "
            f"audit manifest or an explicit replacement audit: {sorted(set(duplicate_keys))[:3]}"
        )
    return rows


def _target_for(
    group: Mapping[str, Any],
    arm: Mapping[str, Any],
    artifact_root: Path | None,
) -> dict[str, Any]:
    candidate = arm.get("record") if isinstance(arm.get("record"), Mapping) else {}
    goal = candidate.get("semantic_goal") if isinstance(candidate.get("semantic_goal"), Mapping) else {}
    primary = goal.get("primary_predicate") if isinstance(goal.get("primary_predicate"), Mapping) else {}
    objective = goal.get("objective_event") if isinstance(goal.get("objective_event"), Mapping) else {}
    objective_manifest = _read_optional_json(_objective_file(artifact_root, "manifest.json") if artifact_root else None)
    evidence = {}
    audit_evidence = arm.get("audit_evidence")
    if isinstance(audit_evidence, Mapping):
        evidence = audit_evidence
    semantics = evidence.get("window_semantics") if isinstance(evidence.get("window_semantics"), Mapping) else {}
    actors = semantics.get("actor_ids")
    if not isinstance(actors, list):
        actors = []
    return {
        "predicate_id": str(_first(
            objective_manifest.get("predicate_id"),
            primary.get("predicate_id"),
            group.get("target_predicate_id"),
        ) or ""),
        "objective_event_family_id": str(_first(
            objective_manifest.get("event_family_id"),
            objective.get("event_family_id"),
            group.get("family"),
        ) or ""),
        "objective_event_type_id": str(_first(
            objective_manifest.get("event_type_id"),
            objective.get("event_type_id"),
        ) or ""),
        "rule_id": str(_first(
            objective_manifest.get("rule_id"),
            objective.get("rule_id"),
            group.get("rule_id"),
        ) or ""),
        "bound_entity_ids": sorted({str(item) for item in actors if isinstance(item, str) and item}),
    }


def _artifact_state(row: Mapping[str, Any]) -> tuple[str, Path | None, str | None]:
    value = row.get("artifact_root") or row.get("branch_root")
    root = _safe_project_path(value)
    if root is None or not root.is_dir():
        return "SKIPPED", None, None
    plot = root / "branch_plot.png"
    plot_digest: str | None = None
    if plot.is_file() and not plot.is_symlink():
        try:
            data = plot.read_bytes()
        except OSError:
            data = b""
        if data.startswith(PNG_SIGNATURE):
            plot_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    semantic = any(
        _artifact_file(root, name) is not None
        for name in (
            "branch_manifest.json",
            "branch_event_ledger.jsonl",
            "manifest.json",
            "predicate_truth.jsonl",
            "predicate_transitions.jsonl",
            "event_occurrences.jsonl",
        )
    )
    if plot_digest and semantic and str(row.get("execution_status")) == "EXECUTED":
        return "AVAILABLE", root, plot_digest
    if root.is_dir() and (plot_digest or semantic):
        return "PARTIAL", root, plot_digest
    return "SKIPPED", root, plot_digest


def _finding_summary(row: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    codes: list[str] = []
    messages: list[str] = []
    for finding in row.get("findings") or []:
        if not isinstance(finding, Mapping):
            continue
        code = str(finding.get("code") or "")
        message = str(finding.get("message") or "")
        if code and code not in codes:
            codes.append(code)
        if message and message not in messages:
            messages.append(message)
    return codes[:12], messages[:8]


def _compact_comparison(value: Any) -> dict[str, Any] | None:
    """Keep the measured comparison headline without embedding long tick lists."""

    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {
        key: value[key]
        for key in (
            "schema_name",
            "schema_version",
            "status",
            "classification",
            "reason",
            "arm_kind",
            "baseline_branch_key",
        )
        if key in value
    }
    for label in ("factual", "alternative"):
        measurement = value.get(label)
        if not isinstance(measurement, Mapping):
            continue
        compact = {
            key: measurement[key]
            for key in ("event_observed", "event_window_state", "rising_tick", "unknown_count")
            if key in measurement
        }
        ticks = measurement.get("true_ticks")
        if isinstance(ticks, list):
            compact["true_tick_count"] = len(ticks)
        result[label] = compact
    return result


def _arm_entry(
    group: Mapping[str, Any],
    arm: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = dict(audit or {})
    state, root, plot_digest = _artifact_state(row)
    codes, messages = _finding_summary(row)
    window = row.get("window") if isinstance(row.get("window"), Mapping) else {}
    fork = _int(window.get("fork_tick"))
    end = _int(window.get("window_end_tick"))
    if fork is None:
        fork = _int(group.get("fork_tick"))
    if end is None:
        end = _int(group.get("window_end_tick"))
    target = _target_for(group, {**arm, "audit_evidence": row.get("evidence")}, root)
    entry: dict[str, Any] = {
        "arm_id": str(arm.get("arm_id") or ""),
        "arm_kind": str(arm.get("arm_kind") or ""),
        "branch_key": str(row.get("branch_key") or arm.get("key") or ""),
        "repair_round_id": "arm_delta_v31_20260904",
        "status": state,
        "displayable": state in {"AVAILABLE", "PARTIAL"} and bool(root and plot_digest),
        "execution_status": str(row.get("execution_status") or row.get("source_status") or "MISSING"),
        "real_execution_status": str(row.get("real_execution_status") or "MISSING"),
        "repair_attempted": str(row.get("execution_status") or row.get("source_status") or "MISSING") not in {"", "MISSING", "NOT_EXECUTED"},
        "release_status": str(row.get("release_status") or "NO-GO"),
        # The review catalog is globally outside the training release, even
        # when one Arm independently passes its local checks.  Preserve the
        # local validator bit under a separate name for audit context.
        "training_eligible": False,
        "arm_audit_training_eligible": bool(row.get("training_eligible")),
        "reason": (messages[0] if messages else ("no persisted artifact" if state == "SKIPPED" else "artifact requires manual review")),
        "finding_codes": codes,
        "finding_messages": messages,
        "window": {"fork_tick": fork, "window_end_tick": end},
        "target": target,
        "candidate_path": arm.get("path"),
    }
    if root is not None:
        entry["branch_root"] = _relative(root)
    if plot_digest is not None:
        entry["plot_path"] = _relative(root / "branch_plot.png") if root else None
        entry["plot_sha256"] = plot_digest
    comparison = _compact_comparison(row.get("intent_comparison"))
    if comparison is not None:
        entry["comparison"] = comparison
    return entry


def _write_overview_plot(path: Path, family_counts: Mapping[str, Mapping[str, int]]) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment contract
        raise CatalogError(f"matplotlib is required for the measured overview plot: {exc}") from exc
    families = sorted(family_counts)
    available = [family_counts[item].get("displayable", 0) for item in families]
    skipped = [family_counts[item].get("skipped", 0) for item in families]
    audited_pass = [family_counts[item].get("audit_pass", 0) for item in families]
    figure, axis = plt.subplots(figsize=(18, 8), dpi=130)
    positions = list(range(len(families)))
    width = 0.27
    axis.bar([p - width for p in positions], available, width, label="displayable artifacts", color="#16758d")
    axis.bar(positions, skipped, width, label="skipped / unavailable", color="#b9c4c8")
    axis.bar([p + width for p in positions], audited_pass, width, label="independent audit PASS", color="#5b8f62")
    axis.set_title("AERO_WORLD preliminary Arm review coverage (measured artifacts)")
    axis.set_ylabel("Arm count")
    axis.set_xticks(positions)
    axis.set_xticklabels(families, rotation=55, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(path, format="png")
    plt.close(figure)
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _artifact_run_root(relative_path: str) -> str:
    """Keep distinct run ownership below the organized artifact categories."""
    path = Path(relative_path)
    if (path.is_absolute() or len(path.parts) < 3 or path.parts[0] != "artifacts"
            or path.parts[1] not in {"branches", "datasets", "experiments", "audits"}
            or ".." in path.parts):
        raise CatalogError(f"branch has no declared artifact run root: {relative_path}")
    return Path(*path.parts[:3]).as_posix()


def build_catalog(
    *,
    inventory_root: Path,
    audit_paths: Sequence[Path],
    output_root: Path,
    catalog_name: str = DEFAULT_CATALOG_NAME,
) -> dict[str, Any]:
    inventory_root = inventory_root.resolve()
    candidates, groups = _load_candidate_records(inventory_root)
    audits = _load_audit_rows([path.resolve() for path in audit_paths])
    missing = sorted(set(candidates) - set(audits))
    unknown = sorted(set(audits) - set(candidates))
    if missing or unknown:
        raise CatalogError(
            f"audit coverage does not match candidate inventory (missing={len(missing)}, unknown={len(unknown)})"
        )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    group_rows: list[dict[str, Any]] = []
    family_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    source_roots: set[str] = set()
    all_arms = 0
    for group_key in sorted(groups):
        group = groups[group_key]
        arm_rows: list[dict[str, Any]] = []
        for kind in REQUIRED_ARM_KINDS:
            arm = group["arms"][kind]
            entry = _arm_entry(group, arm, audits.get(arm["key"]))
            arm_rows.append(entry)
            all_arms += 1
            family = str(group.get("family") or "unknown")
            family_counts[family]["arms"] += 1
            family_counts[family][entry["status"].lower()] += 1
            if entry["displayable"]:
                family_counts[family]["displayable"] += 1
            else:
                family_counts[family]["not_displayable"] += 1
            if entry["release_status"] == "PASS":
                family_counts[family]["audit_pass"] += 1
            if entry["execution_status"] == "EXECUTED":
                family_counts[family]["executed"] += 1
            if entry.get("branch_root"):
                source_roots.add(_artifact_run_root(str(entry["branch_root"])))
        group_rows.append(
            {
                "profile_id": group["profile_id"],
                "episode_id": group["episode_id"],
                "seed_id": group["seed_id"],
                "group_id": group["group_id"],
                "branch_group_id": group["branch_group_id"],
                "family": group["family"],
                "anchor_event_id": group["anchor_event_id"],
                "fork_tick": group["fork_tick"],
                "window_end_tick": group["window_end_tick"],
                "target": {
                    "predicate_id": group["target_predicate_id"],
                    "rule_id": group["rule_id"],
                },
                "candidate_manifest": group["manifest_path"],
                "arms": arm_rows,
            }
        )
    overview_path = output_root / "overview_by_family.png"
    overview_digest = _write_overview_plot(overview_path, family_counts)
    source_roots.add(_relative(output_root))
    summary_path = audit_paths[-1].resolve().parent / "arm_delta_release_summary.json"
    audit_summary = _read_optional_json(summary_path if summary_path.is_file() else None)
    catalog: dict[str, Any] = {
        "schema_name": "aeroworld_action_branch_preliminary_catalog",
        "schema_version": "1.0.0",
        "release_status": "REVIEW_ONLY",
        "training_release_status": "NO-GO",
        "review_only": True,
        "strict_fail_closed": True,
        "repair_round_id": "arm_delta_v31_20260904",
        "repair_policy": {
            "max_attempts_per_round": 1,
            "unresolved_action": "SKIPPED",
            "source_status_is_not_evidence": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inventory_root": _relative(inventory_root),
        "source_audit": _relative(audit_paths[-1].resolve()),
        "source_audit_sha256": _sha256(audit_paths[-1].resolve()),
        "source_audit_summary": _relative(summary_path) if summary_path.is_file() else None,
        "source_audit_summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
        "source_roots": sorted(source_roots),
        "overview_plot_path": _relative(overview_path),
        "overview_plot_sha256": overview_digest,
        "counts": {
            "groups": len(group_rows),
            "event_clusters": len(group_rows),
            "unique_anchor_event_ids": len({
                item["anchor_event_id"] for item in group_rows if item["anchor_event_id"]
            }),
            "arms": all_arms,
            "available_arms": sum(item["status"] == "AVAILABLE" for group in group_rows for item in group["arms"]),
            "partial_arms": sum(item["status"] == "PARTIAL" for group in group_rows for item in group["arms"]),
            "skipped_arms": sum(item["status"] == "SKIPPED" for group in group_rows for item in group["arms"]),
            "displayable_arms": sum(item["displayable"] for group in group_rows for item in group["arms"]),
            "executed_arms": sum(item["execution_status"] == "EXECUTED" for group in group_rows for item in group["arms"]),
            "repair_attempted_arms": sum(item["repair_attempted"] for group in group_rows for item in group["arms"]),
            "repair_unattempted_arms": sum(not item["repair_attempted"] for group in group_rows for item in group["arms"]),
            "audit_pass_arms": sum(item["release_status"] == "PASS" for group in group_rows for item in group["arms"]),
            "audit_no_go_arms": sum(item["release_status"] != "PASS" for group in group_rows for item in group["arms"]),
        },
        "family_counts": {family: dict(sorted(counter.items())) for family, counter in sorted(family_counts.items())},
        "event_cluster_coverage": {
            "event_clusters": len(group_rows),
            "groups_with_five_authored_arms": sum(
                len(item["arms"]) == 5 for item in group_rows
            ),
            "groups_with_anchor_event": sum(
                bool(item["anchor_event_id"]) for item in group_rows
            ),
            "required_arm_kinds": list(REQUIRED_ARM_KINDS),
        },
        "audit_counts": audit_summary.get("counts", {}) if audit_summary else {},
        "groups": group_rows,
    }
    catalog_relative = Path(catalog_name)
    if (
        catalog_relative.is_absolute()
        or any(part in {"", ".", ".."} for part in catalog_relative.parts)
        or len(catalog_relative.parts) != 1
    ):
        raise CatalogError("catalog name must be one safe filename")
    catalog_path = output_root / catalog_relative
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"catalog": _relative(catalog_path), "overview_plot": _relative(overview_path), "counts": catalog["counts"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-root", type=Path, default=DEFAULT_INVENTORY_ROOT)
    parser.add_argument("--audit", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--catalog-name", default=DEFAULT_CATALOG_NAME)
    args = parser.parse_args(argv)
    audit_paths = args.audit if args.audit else [DEFAULT_AUDIT]
    result = build_catalog(
        inventory_root=args.inventory_root,
        audit_paths=audit_paths,
        output_root=args.output_root,
        catalog_name=args.catalog_name,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CatalogError, OSError, ValueError, KeyError) as exc:
        print(f"preliminary catalog build failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
