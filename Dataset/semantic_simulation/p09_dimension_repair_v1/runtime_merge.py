"""P09 dimension repair v1 — bounded actor/tick runtime merge (first input-merge checkpoint).

For each requested tick this module emits one JSONL row that retains the
complete original world truth frame of episode L2-1_v2__seed00 and overlays the
recorded runtime state of every adopted actor record at that tick.

Contract enforced here:
- Original entities stay verbatim except explicitly recorded numeric overrides
  (adopted pos_enu/vel_mps/yaw_deg mapped onto the original numeric carrier)
  and adopted runtime-family overlays; every change cites its source path.
- Adopted trajectory rows are the runtime authority, including stationary
  pad/tower facilities.
- A runtime family absent from an adopted row is preserved as an explicit
  absence; the original value is retained and no default is invented.
- No action replay, no event-script interpretation, no omitted-action
  inference, no future runtime values (only rows at requested ticks are read).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "p09.dimension_repair_v1.actor_tick_runtime_merge/1"
EPISODE_ID = "L2-1_v2__seed00"
REPO_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_TRUTH_FRAMES = (
    REPO_ROOT / "aw_data" / "render_ready_episodes_capture_filtered" / EPISODE_ID / "truth_frames.jsonl"
)
ADOPTED_TRAJECTORIES = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v13_metadata_artifacts"
) / EPISODE_ID / "trajectories.jsonl"
V13_REVIEW_DIR = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v13_metadata_review"
) / EPISODE_ID
V12_ADOPTED_DIR = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v12_remaining"
) / EPISODE_ID / "adopted"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "design" / "p09" / "mechanism_completion_v1" / "state_predicate_rebuild"
)

CONTEXT_INPUTS_NOT_CONSUMED = (
    {
        "path": str(V13_REVIEW_DIR / "scene_setup.json"),
        "reason": "authored scene context; never interpreted as actual runtime state",
    },
    {
        "path": str(V13_REVIEW_DIR / "event_script.json"),
        "reason": "authored future intent; never executed or admitted in this checkpoint",
    },
    {
        "path": str(V12_ADOPTED_DIR / "actions.json"),
        "reason": "no action replay in this checkpoint; trajectory rows are the recorded runtime authority",
    },
    {
        "path": str(V12_ADOPTED_DIR / "weather.jsonl"),
        "reason": "weather is outside the actor/tick runtime merge scope",
    },
)

IDENTITY_KEYS = {"tick", "entity_id"}
CATEGORY_KEYS = ("category", "entity_category")
MOTION_KEYS = ("pos_enu", "vel_mps", "yaw_deg")
# Explicit numeric mapping: adopted flat field -> candidate paths in the
# original truth entity, tried in order. The first present carrier wins and is
# recorded per entity; nothing is guessed beyond this declared list.
NUMERIC_CANDIDATES = {
    "pos_enu": ("pos_enu", "truth_pose.position_enu", "truth_pose.position_enu_m", "truth_pose.position"),
    "vel_mps": ("vel_mps", "truth_pose.vel_mps", "truth_pose.velocity_mps", "truth_pose.velocity_enu_mps"),
    "yaw_deg": ("yaw_deg", "truth_pose.yaw_deg", "truth_pose.heading_deg", "truth_pose.rotation_deg.yaw_deg"),
}
MISSION_SAMPLE_KEYS = ("pos_enu", "control_state", "communication_state")
STATIONARY_SAMPLE_KEYS = ("communication_state", "facility_state")


class RuntimeMergeError(RuntimeError):
    pass


FAILURE_CONTEXT = {
    "stage": "not_started",
    "last_entity": None,
    "stages_completed": [],
    "command": None,
    "inputs": {},
    "traceback": None,
}


def _resolve_numeric_path(entity: dict, field: str):
    """Return (dotted_path, current_value) for the first declared carrier, else (None, None)."""
    for dotted in NUMERIC_CANDIDATES[field]:
        node = entity
        found = True
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                found = False
                break
        if found:
            return dotted, node
    return None, None


def _set_dotted(obj: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = obj
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def merge_actor_entity(original_entity: dict, adopted_row: dict, tick: int, source_refs: dict) -> dict:
    entity_id = original_entity["entity_id"]
    if adopted_row.get("entity_id") != entity_id:
        raise RuntimeMergeError(
            f"adopted row entity_id {adopted_row.get('entity_id')!r} does not match "
            f"original {entity_id!r} at tick {tick}"
        )
    adopted_category = adopted_row.get("category", adopted_row.get("entity_category"))
    original_category = original_entity.get("category", original_entity.get("entity_category"))
    if (
        adopted_category is not None
        and original_category is not None
        and adopted_category != original_category
    ):
        raise RuntimeMergeError(
            f"category divergence for {entity_id} at tick {tick}: "
            f"adopted={adopted_category!r} original={original_category!r}"
        )

    merged = copy.deepcopy(original_entity)
    numeric_override_roots: set[str] = set()
    prov = {
        "authority": "adopted_trajectory_recorded_runtime_state",
        "adopted_source": {
            "path": source_refs["adopted_trajectories"],
            "tick": tick,
            "entity_id": entity_id,
        },
        "original_source": {"path": source_refs["original_truth_frames"], "tick": tick},
        "adopted_row_keys": sorted(adopted_row.keys()),
        "original_top_level_keys": sorted(original_entity.keys()),
        "families_overlaid": [],
        "original_dict_families_retained_without_adopted_counterpart": [],
        "numeric_overrides": [],
        "numeric_overrides_not_applied": [],
    }

    for field in MOTION_KEYS:
        if field not in adopted_row:
            prov["numeric_overrides_not_applied"].append(
                {"adopted_field": field, "reason": "absent_in_adopted_row"}
            )
            continue
        adopted_value = copy.deepcopy(adopted_row[field])
        path, current = _resolve_numeric_path(merged, field)
        if path is None:
            if field == "pos_enu" and isinstance(merged.get("truth_pose"), dict):
                raise RuntimeMergeError(
                    f"no position carrier on {entity_id} at tick {tick}; "
                    f"truth_pose keys: {sorted(merged['truth_pose'])}"
                )
            entry = {
                "adopted_field": field,
                "reason": "no_numeric_carrier_in_original_entity",
                "adopted_value": adopted_value,
            }
            if isinstance(merged.get("truth_pose"), dict):
                entry["truth_pose_keys"] = sorted(merged["truth_pose"])
            prov["numeric_overrides_not_applied"].append(entry)
            continue
        entry = {
            "adopted_field": field,
            "original_path": path,
            "adopted_value": adopted_value,
        }
        if current != adopted_value:
            _set_dotted(merged, path, adopted_value)
            entry["before"] = current
            entry["changed"] = True
        else:
            entry["changed"] = False
        prov["numeric_overrides"].append(entry)
        numeric_override_roots.add(path.split(".", 1)[0])

    overlaid_keys: set[str] = set()
    for key, value in adopted_row.items():
        if key in IDENTITY_KEYS or key in MOTION_KEYS or key in CATEGORY_KEYS:
            continue
        overlaid_keys.add(key)
        entry = {"family": key, "original_present": key in merged}
        if key in merged and merged[key] != value:
            entry["replaced_original"] = merged[key]
        merged[key] = copy.deepcopy(value)
        prov["families_overlaid"].append(entry)
    if adopted_category is not None and "category" not in merged:
        merged["category"] = adopted_category
        prov["families_overlaid"].append({"family": "category", "original_present": False})

    for key, value in original_entity.items():
        if key in overlaid_keys or key in numeric_override_roots or key == "runtime_merge":
            continue
        if isinstance(value, dict):
            prov["original_dict_families_retained_without_adopted_counterpart"].append(key)

    merged["runtime_merge"] = prov
    return merged


def adopted_only_entity(adopted_row: dict, tick: int, source_refs: dict) -> dict:
    entity = copy.deepcopy(adopted_row)
    entity["runtime_merge"] = {
        "authority": "adopted_trajectory_entity_absent_from_original_truth_frame",
        "adopted_source": {
            "path": source_refs["adopted_trajectories"],
            "tick": tick,
            "entity_id": adopted_row.get("entity_id"),
        },
        "original_source": {
            "path": source_refs["original_truth_frames"],
            "tick": tick,
            "entity_present": False,
        },
        "adopted_row_keys": sorted(adopted_row.keys()),
        "families_overlaid": [],
        "numeric_overrides": [],
        "numeric_overrides_not_applied": [],
        "original_dict_families_retained_without_adopted_counterpart": [],
        "note": "full recorded adopted row retained as-is; no original counterpart entity exists at this tick",
    }
    return entity


def _actor_samples(adopted_row: dict, keys) -> dict:
    samples = {}
    for key in keys:
        if key in adopted_row:
            samples[key] = copy.deepcopy(adopted_row[key])
        else:
            samples[key] = "absent_in_adopted_row"
    return samples


def validate_segment(
    segment_path: Path,
    ticks: list,
    original_frames: dict,
    adopted_rows: dict,
    mission_actor: str,
    stationary_actor: str,
) -> dict:
    with segment_path.open(encoding="utf-8") as handle:
        written = [json.loads(line) for line in handle if line.strip()]
    if [row["tick"] for row in written] != ticks:
        raise RuntimeMergeError("written segment ticks do not match requested ticks")

    result = {"actors": {}, "background_preservation": {}, "coverage": {}}
    sample_keys_by_label = {
        "mission_actor": MISSION_SAMPLE_KEYS,
        "stationary_actor": STATIONARY_SAMPLE_KEYS,
    }
    for label, actor in (("mission_actor", mission_actor), ("stationary_actor", stationary_actor)):
        per_tick = {}
        for row in written:
            tick = row["tick"]
            matches = [e for e in row["entities"] if e.get("entity_id") == actor]
            if len(matches) != 1:
                raise RuntimeMergeError(
                    f"{label} {actor} appears {len(matches)} times at tick {tick}"
                )
            prov = matches[0].get("runtime_merge")
            if not isinstance(prov, dict) or "authority" not in prov:
                raise RuntimeMergeError(
                    f"{label} {actor} at tick {tick} lacks adopted runtime_merge provenance"
                )
            if actor not in adopted_rows[tick]:
                raise RuntimeMergeError(
                    f"{label} {actor} has no adopted record at tick {tick}"
                )
            per_tick[tick] = {
                "authority": prov["authority"],
                "families_overlaid": [entry["family"] for entry in prov["families_overlaid"]],
                "numeric_overrides": len(prov["numeric_overrides"]),
                "recorded_adopted_samples": _actor_samples(
                    adopted_rows[tick][actor], sample_keys_by_label[label]
                ),
            }
        result["actors"][label] = {
            "entity_id": actor,
            "present_at_all_requested_ticks": True,
            "per_tick": per_tick,
        }

    for row in written:
        tick = row["tick"]
        frame = original_frames[tick]
        original_only = [eid for eid in frame["by_id"] if eid not in adopted_rows[tick]]
        preferred = ["global_pad_08"] if "global_pad_08" in original_only else []
        ordered = preferred + sorted(eid for eid in original_only if eid != "global_pad_08")
        checked = []
        for entity_id in ordered[:2]:
            original_entity = frame["by_id"][entity_id]
            merged_entity = next(
                e for e in row["entities"] if e.get("entity_id") == entity_id
            )
            if merged_entity != original_entity:
                raise RuntimeMergeError(
                    f"original background entity {entity_id} changed at tick {tick}"
                )
            checked.append(entity_id)
        result["background_preservation"][tick] = {
            "original_only_entity_count": len(original_only),
            "checked_verbatim": checked,
        }
        counts = row["counts"]
        if counts["overlaid_from_adopted"] + counts["adopted_only_appended"] != counts["adopted_actor_records"]:
            raise RuntimeMergeError(f"adopted actor coverage mismatch at tick {tick}")
        if counts["merged_entities"] != counts["original_entities"] + counts["adopted_only_appended"]:
            raise RuntimeMergeError(f"merged entity count mismatch at tick {tick}")
        result["coverage"][tick] = dict(counts)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge recorded adopted runtime state onto the original world truth at requested ticks."
    )
    parser.add_argument("--ticks", type=int, nargs="+", default=[260, 261],
                        help="ticks to merge (bounded segment)")
    parser.add_argument("--mission-actor", default="uav_l2_1_v2")
    parser.add_argument("--stationary-actor", default="tower_l2_1_v2")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    ticks = sorted(set(int(t) for t in args.ticks))
    output_dir = Path(args.output_dir)
    FAILURE_CONTEXT["command"] = " ".join(sys.argv)
    FAILURE_CONTEXT["stage"] = "input_checks"
    for path in (ORIGINAL_TRUTH_FRAMES, ADOPTED_TRAJECTORIES):
        if not path.is_file():
            raise RuntimeMergeError(f"required input missing: {path}")
    FAILURE_CONTEXT["inputs"] = {
        "original_truth_frames": str(ORIGINAL_TRUTH_FRAMES),
        "adopted_trajectories": str(ADOPTED_TRAJECTORIES),
        "ticks": ticks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = output_dir / "l2_actor_tick_segment.jsonl"
    receipt_path = output_dir / "segment_receipt.json"

    FAILURE_CONTEXT["stage"] = "scan_original_frames"
    original_frames: dict = {}
    original_frame_count = 0
    with ORIGINAL_TRUTH_FRAMES.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            original_frame_count += 1
            row = json.loads(line)
            tick = row.get("tick")
            if isinstance(tick, int) and tick in ticks:
                if tick in original_frames:
                    raise RuntimeMergeError(f"{ORIGINAL_TRUTH_FRAMES}: duplicate frame for tick {tick}")
                entities = row.get("entities")
                if not isinstance(entities, list):
                    raise RuntimeMergeError(f"{ORIGINAL_TRUTH_FRAMES}: frame {tick} lacks an entities array")
                indexed = {}
                for entity in entities:
                    entity_id = entity.get("entity_id") if isinstance(entity, dict) else None
                    if not isinstance(entity_id, str) or not entity_id:
                        raise RuntimeMergeError(
                            f"{ORIGINAL_TRUTH_FRAMES}: frame {tick} has an entity without entity_id"
                        )
                    if entity_id in indexed:
                        raise RuntimeMergeError(
                            f"{ORIGINAL_TRUTH_FRAMES}: frame {tick} duplicates entity {entity_id}"
                        )
                    indexed[entity_id] = entity
                original_frames[tick] = {"row": row, "by_id": indexed}
    missing = [tick for tick in ticks if tick not in original_frames]
    if missing:
        raise RuntimeMergeError(f"original truth frames missing requested ticks: {missing}")

    FAILURE_CONTEXT["stage"] = "scan_adopted_rows"
    adopted_rows: dict = {tick: {} for tick in ticks}
    adopted_line_count = 0
    with ADOPTED_TRAJECTORIES.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            adopted_line_count += 1
            row = json.loads(line)
            tick = row.get("tick")
            if tick in adopted_rows:
                entity_id = row.get("entity_id")
                if not isinstance(entity_id, str) or not entity_id:
                    raise RuntimeMergeError(f"{ADOPTED_TRAJECTORIES}: row lacks entity_id at tick {tick}")
                if entity_id in adopted_rows[tick]:
                    raise RuntimeMergeError(
                        f"{ADOPTED_TRAJECTORIES}: duplicate adopted row for {entity_id} at tick {tick}"
                    )
                adopted_rows[tick][entity_id] = row
    for tick in ticks:
        if not adopted_rows[tick]:
            raise RuntimeMergeError(f"no adopted actor rows at requested tick {tick}")

    source_refs = {
        "original_truth_frames": str(ORIGINAL_TRUTH_FRAMES),
        "adopted_trajectories": str(ADOPTED_TRAJECTORIES),
    }

    FAILURE_CONTEXT["stage"] = "merge_ticks"
    segment_rows = []
    for tick in ticks:
        FAILURE_CONTEXT["stage"] = f"merge_tick_{tick}"
        frame = original_frames[tick]
        merged_entities = []
        overlaid_ids = []
        appended_ids = []
        for entity in frame["row"]["entities"]:
            entity_id = entity["entity_id"]
            FAILURE_CONTEXT["last_entity"] = {"tick": tick, "entity_id": entity_id}
            adopted_row = adopted_rows[tick].get(entity_id)
            if adopted_row is None:
                merged_entities.append(entity)
            else:
                merged_entities.append(merge_actor_entity(entity, adopted_row, tick, source_refs))
                overlaid_ids.append(entity_id)
        for entity_id, adopted_row in adopted_rows[tick].items():
            if entity_id in frame["by_id"]:
                continue
            FAILURE_CONTEXT["last_entity"] = {"tick": tick, "entity_id": entity_id}
            merged_entities.append(adopted_only_entity(adopted_row, tick, source_refs))
            appended_ids.append(entity_id)
        segment_rows.append({
            "schema": SCHEMA,
            "episode_id": EPISODE_ID,
            "tick": tick,
            "sources": dict(source_refs),
            "counts": {
                "original_entities": len(frame["row"]["entities"]),
                "adopted_actor_records": len(adopted_rows[tick]),
                "overlaid_from_adopted": len(overlaid_ids),
                "adopted_only_appended": len(appended_ids),
                "merged_entities": len(merged_entities),
            },
            "overlaid_entity_ids": overlaid_ids,
            "adopted_only_entity_ids": appended_ids,
            "entities": merged_entities,
        })
    FAILURE_CONTEXT["stages_completed"].append("merge_ticks")

    FAILURE_CONTEXT["stage"] = "write_output"
    with segment_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in segment_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    FAILURE_CONTEXT["stages_completed"].append("write_output")

    FAILURE_CONTEXT["stage"] = "validate"
    validation = validate_segment(
        segment_path, ticks, original_frames, adopted_rows,
        args.mission_actor, args.stationary_actor,
    )
    FAILURE_CONTEXT["stages_completed"].append("validate")

    per_tick = {
        row["tick"]: {
            "counts": row["counts"],
            "overlaid_entity_ids": row["overlaid_entity_ids"],
            "adopted_only_entity_ids": row["adopted_only_entity_ids"],
        }
        for row in segment_rows
    }

    FAILURE_CONTEXT["stage"] = "write_receipt"
    receipt = {
        "schema": "p09.dimension_repair_v1.actor_tick_segment_receipt/1",
        "module": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episode_id": EPISODE_ID,
        "ticks": ticks,
        "inputs": {
            "original_truth_frames": {
                "path": str(ORIGINAL_TRUTH_FRAMES),
                "bytes": os.path.getsize(ORIGINAL_TRUTH_FRAMES),
                "frames_scanned": original_frame_count,
                "frames_matched": ticks,
            },
            "adopted_trajectories": {
                "path": str(ADOPTED_TRAJECTORIES),
                "bytes": os.path.getsize(ADOPTED_TRAJECTORIES),
                "lines_scanned": adopted_line_count,
            },
        },
        "context_inputs_not_consumed": [dict(item) for item in CONTEXT_INPUTS_NOT_CONSUMED],
        "output": {
            "path": str(segment_path),
            "rows": len(segment_rows),
            "bytes": os.path.getsize(segment_path),
        },
        "per_tick": per_tick,
        "validation": validation,
        "merge_rules": {
            "original_world_retained": (
                "original entities verbatim except recorded numeric overrides and adopted family overlays"
            ),
            "numeric_override_mapping": (
                "adopted pos_enu/vel_mps/yaw_deg override the original numeric carrier; "
                "each override records original_path, before/after and the adopted source"
            ),
            "runtime_authority": (
                "adopted trajectory rows are authoritative for runtime families, including stationary pad/tower"
            ),
            "absence_preserved": (
                "families absent from an adopted row keep the original values; no defaults are invented"
            ),
            "no_action_replay": (
                "actions.json is not consumed; omitted or future actions are not updates"
            ),
            "no_script_interpretation": "event_script.json and scene_setup.json are context only",
            "tick_bounded": "only rows at requested ticks are consumed; no future runtime values",
        },
        "scope": (
            "first_input_merge_checkpoint; not a geometry/domain/predicate generator "
            "and not a LOG_ONLY certification"
        ),
    }
    with receipt_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    FAILURE_CONTEXT["stages_completed"].append("write_receipt")

    print(json.dumps({
        "segment_path": str(segment_path),
        "receipt_path": str(receipt_path),
        "segment_rows": len(segment_rows),
        "per_tick_counts": {str(tick): per_tick[tick]["counts"] for tick in ticks},
        "validation_actors": {
            label: data["entity_id"] for label, data in validation["actors"].items()
        },
        "background_preservation": validation["background_preservation"],
    }, ensure_ascii=False, indent=2))
    return 0


def _write_failure_file() -> None:
    try:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = DEFAULT_OUTPUT_DIR / "segment_failure.txt"
        lines = [
            f"command: {FAILURE_CONTEXT.get('command')}",
            f"working_directory: {Path.cwd()}",
            f"stage: {FAILURE_CONTEXT.get('stage')}",
            f"last_entity: {json.dumps(FAILURE_CONTEXT.get('last_entity'), ensure_ascii=False)}",
            f"stages_completed: {FAILURE_CONTEXT.get('stages_completed')}",
            f"inputs: {json.dumps(FAILURE_CONTEXT.get('inputs'), ensure_ascii=False)}",
            "traceback:",
            FAILURE_CONTEXT.get("traceback") or "<no traceback captured>",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"failure context saved: {path}", file=sys.stderr)
    except Exception as inner:  # diagnostics must not mask the original failure
        print(f"could not save failure context: {inner}", file=sys.stderr)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        FAILURE_CONTEXT["traceback"] = traceback.format_exc()
        _write_failure_file()
        traceback.print_exc()
        sys.exit(1)
