#!/usr/bin/env python
"""Approved thin objective-pipeline caller for episode L2-1_v2__seed00.

Glue only: it assembles the declared adopted-input episode root, merges the
recorded adopted runtime onto the FULL original world truth with the existing
runtime_merge callables, then executes the existing objective builder and its
exact persist/check contract.  No pipeline, guard, numerical, or geometry logic
is implemented or modified here; every computed artifact comes from
Dataset/semantic_truth/objective_pipeline.py running with its four default
profile paths, the strict input guard enabled, and the default manifest
episode root.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Dataset.semantic_truth.objective_pipeline import (  # noqa: E402
    DEFAULT_COMPUTE_PROFILE_PATH,
    DEFAULT_CONTRACT_PROFILE_PATH,
    DEFAULT_DOMAIN_PROFILE_PATH,
    DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH,
    build_episode_objective_artifacts,
    check_episode_outputs,
    write_episode_outputs,
)
from Dataset.semantic_simulation.p09_dimension_repair_v1.runtime_merge import (  # noqa: E402
    adopted_only_entity,
    merge_actor_entity,
)

EPISODE_ID = "L2-1_v2__seed00"

ORIGINAL_EPISODE_ROOT = REPO_ROOT / "aw_data" / "render_ready_episodes" / EPISODE_ID
ADOPTED_TRAJECTORIES = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v13_metadata_artifacts"
) / EPISODE_ID / "trajectories.jsonl"
ADOPTED_SCENE_SETUP = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v13_metadata_review"
) / EPISODE_ID / "scene_setup.json"
ADOPTED_EVENT_SCRIPT = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v13_metadata_review"
) / EPISODE_ID / "event_script.json"
ADOPTED_WEATHER = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v12_remaining"
) / EPISODE_ID / "adopted" / "weather.jsonl"

SESSION_ROOT = (
    REPO_ROOT
    / "design"
    / "p09"
    / "mechanism_completion_v1"
    / "numeric_mapping_checkpoint"
    / "builder_short_session_v1"
)
STAGING_ROOT = SESSION_ROOT / "staging_v1" / EPISODE_ID
OBJECTIVE_OUTPUT_ROOT = SESSION_ROOT / "objective_v1"
OBJECTIVE_OUTPUT = OBJECTIVE_OUTPUT_ROOT / EPISODE_ID
CALLER_RECEIPT_PATH = OBJECTIVE_OUTPUT_ROOT / "caller_receipt.json"

MANIFEST_BOUND_INPUTS = (
    "episode_manifest.json",
    "global_entity_roster.json",
    "trajectories.jsonl",
    "weather_meta.jsonl",
)
MANIFEST_OPTIONAL_INPUTS = (
    "scene_occupancy_manifest.json",
    "semantic_static_geometry.json",
    "scene_setup.json",
)

CONTEXT_INPUTS_NOT_CONSUMED = (
    {
        "path": str(ADOPTED_EVENT_SCRIPT),
        "reason": "authored future intent; never executed or interpreted by this caller",
    },
    {
        "path": str(
            Path(
                "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/linked_native_v12_remaining"
            )
            / EPISODE_ID
            / "adopted"
            / "actions.json"
        ),
        "reason": "no action replay; adopted trajectory rows and receipts are the recorded runtime authority",
    },
)

FAILURE_CONTEXT = {
    "stage": "not_started",
    "command": None,
    "inputs": {},
    "traceback": None,
}


class ObjectiveRebuildError(RuntimeError):
    """Raised when caller-level input assembly or binding fails closed."""


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectiveRebuildError(f"cannot read JSON input {path}: {exc}") from exc


def _iter_jsonl(path: Path):
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ObjectiveRebuildError(f"cannot open JSONL input {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ObjectiveRebuildError(
                    f"{path}:{line_number}: invalid JSON row: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ObjectiveRebuildError(
                    f"{path}:{line_number}: row must be a JSON object"
                )
            yield row


def _jsonl_text(rows) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _require_declared_inputs() -> None:
    for path in (
        ORIGINAL_EPISODE_ROOT,
        ORIGINAL_EPISODE_ROOT / "episode_manifest.json",
        ORIGINAL_EPISODE_ROOT / "global_entity_roster.json",
        ORIGINAL_EPISODE_ROOT / "trajectories.jsonl",
        ORIGINAL_EPISODE_ROOT / "truth_frames.jsonl",
        ADOPTED_TRAJECTORIES,
        ADOPTED_SCENE_SETUP,
        ADOPTED_EVENT_SCRIPT,
        ADOPTED_WEATHER,
    ):
        if not path.exists():
            raise ObjectiveRebuildError(f"declared input missing: {path}")


def _merge_full_truth_frames() -> tuple[list, dict, dict]:
    """Merge every recorded adopted actor onto the FULL original truth frames.

    One single pass over each source.  Original frame fields and entity
    ordering are retained verbatim; per-entity recorded runtime is merged via
    merge_actor_entity, uncovered original background entities stay untouched,
    and adopted-only records are appended via adopted_only_entity.  The
    cross-tick roster union is never expanded into per-frame entities.
    """
    adopted_by_tick: dict[int, dict[str, dict]] = {}
    adopted_line_count = 0
    for row in _iter_jsonl(ADOPTED_TRAJECTORIES):
        tick = row.get("tick")
        entity_id = row.get("entity_id")
        if not isinstance(tick, int) or not isinstance(entity_id, str) or not entity_id:
            raise ObjectiveRebuildError(
                f"{ADOPTED_TRAJECTORIES}: row lacks integer tick or entity_id"
            )
        per_tick = adopted_by_tick.setdefault(tick, {})
        if entity_id in per_tick:
            raise ObjectiveRebuildError(
                f"{ADOPTED_TRAJECTORIES}: duplicate adopted row for {entity_id} at tick {tick}"
            )
        per_tick[entity_id] = row
        adopted_line_count += 1
    if not adopted_by_tick:
        raise ObjectiveRebuildError(f"{ADOPTED_TRAJECTORIES}: no adopted rows")

    source_refs = {
        "original_truth_frames": str(
            ORIGINAL_EPISODE_ROOT / "truth_frames.jsonl"
        ),
        "adopted_trajectories": str(ADOPTED_TRAJECTORIES),
    }
    merged_rows: list[dict] = []
    original_frame_count = 0
    totals = {
        "original_entities": 0,
        "adopted_actor_records": 0,
        "overlaid_from_adopted": 0,
        "adopted_only_appended": 0,
        "merged_entities": 0,
    }
    ticks_with_adopted_only: list[int] = []
    consumed_adopted_ticks: set[int] = set()
    for frame in _iter_jsonl(ORIGINAL_EPISODE_ROOT / "truth_frames.jsonl"):
        tick = frame.get("tick")
        if not isinstance(tick, int):
            raise ObjectiveRebuildError(
                f"{ORIGINAL_EPISODE_ROOT / 'truth_frames.jsonl'}: frame lacks integer tick"
            )
        original_frame_count += 1
        entities = frame.get("entities")
        if not isinstance(entities, list):
            raise ObjectiveRebuildError(f"original truth frame {tick} lacks entities array")
        adopted_rows = adopted_by_tick.get(tick, {})
        merged_entities = []
        overlaid_ids = []
        appended_ids = []
        original_ids = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise ObjectiveRebuildError(f"original truth frame {tick}: non-object entity")
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id:
                raise ObjectiveRebuildError(
                    f"original truth frame {tick}: entity lacks entity_id"
                )
            original_ids.append(entity_id)
            adopted_row = adopted_rows.get(entity_id)
            if adopted_row is None:
                # Uncovered original background entity retained verbatim.
                merged_entities.append(entity)
            else:
                merged_entities.append(
                    merge_actor_entity(entity, adopted_row, tick, source_refs)
                )
                overlaid_ids.append(entity_id)
        for entity_id in adopted_rows:
            if entity_id in original_ids:
                continue
            merged_entities.append(
                adopted_only_entity(adopted_rows[entity_id], tick, source_refs)
            )
            appended_ids.append(entity_id)
        if adopted_rows:
            consumed_adopted_ticks.add(tick)
        if appended_ids:
            ticks_with_adopted_only.append(tick)
        # The merged frame retains every original top-level field verbatim and
        # replaces only the entities array; original entity ordering is kept.
        merged_frame = dict(frame)
        merged_frame["entities"] = merged_entities
        merged_rows.append(merged_frame)
        totals["original_entities"] += len(entities)
        totals["adopted_actor_records"] += len(adopted_rows)
        totals["overlaid_from_adopted"] += len(overlaid_ids)
        totals["adopted_only_appended"] += len(appended_ids)
        totals["merged_entities"] += len(merged_entities)

    if original_frame_count != len(merged_rows):
        raise ObjectiveRebuildError(
            f"merged frame count {len(merged_rows)} differs from original {original_frame_count}"
        )
    uncovered_adopted_ticks = sorted(set(adopted_by_tick) - consumed_adopted_ticks)
    if uncovered_adopted_ticks:
        raise ObjectiveRebuildError(
            f"adopted ticks with no original truth frame: {uncovered_adopted_ticks[:10]}..."
        )
    stats = {
        "original_truth_frames": original_frame_count,
        "merged_truth_frames": len(merged_rows),
        "adopted_lines": adopted_line_count,
        "adopted_ticks": len(adopted_by_tick),
        "adopted_entities_per_tick": sorted(
            {len(rows) for rows in adopted_by_tick.values()}
        ),
        "adopted_ticks_without_original_frame": 0,
        "ticks_with_adopted_only_appended": len(ticks_with_adopted_only),
        "totals": dict(totals),
        "source_refs": source_refs,
    }
    return merged_rows, stats, adopted_by_tick


def _declared_adopted_only_entities(
    adopted_by_tick: dict[int, dict[str, dict]],
    world_roster_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """Declare adopted-runtime entities absent from every roster authority.

    The builder's own contract places realized vehicle identities in the world
    roster (scenario-side background-vehicle rows are skipped as plan
    templates), and compute_comm requires every uav/vehicle in the truth
    frames to match the union roster category.  Declarations copy only fields
    recorded in the adopted trajectory rows; no category, identity, or motion
    value is invented.  Returns (roster_entries, trajectory_rows).
    """
    per_entity: dict[str, list[dict]] = {}
    for tick in sorted(adopted_by_tick):
        for entity_id, row in adopted_by_tick[tick].items():
            if entity_id not in world_roster_ids:
                per_entity.setdefault(entity_id, []).append(row)
    if not per_entity:
        return [], []
    roster_entries = []
    trajectory_rows = []
    for entity_id in sorted(per_entity):
        rows = per_entity[entity_id]
        first = rows[0]
        if "category" not in first:
            raise ObjectiveRebuildError(
                f"{ADOPTED_TRAJECTORIES}: adopted row for {entity_id} lacks category; "
                "the roster declaration requires the recorded category"
            )
        declared = {
            key: copy.deepcopy(first[key])
            for key in (
                "entity_id",
                "category",
                "label_class",
                "asset_id",
                "activation_tick",
                "role",
                "background_role",
                "background_vehicle",
                "ground_flow_contract",
            )
            if key in first
        }
        # The roster slot requires the key name `entity_category`; the value is
        # the recorded adopted category, unmodified.
        declared["entity_category"] = first["category"]
        declared["adopted_runtime_source"] = str(ADOPTED_TRAJECTORIES)
        declared["adopted_runtime_tick_range"] = [rows[0]["tick"], rows[-1]["tick"]]
        declared["declaration_note"] = (
            "adopted-runtime entity absent from every roster authority; declared "
            "from the recorded adopted trajectory row, with entity_category "
            "carrying the recorded adopted category value under the roster key name"
        )
        roster_entries.append(declared)
    return roster_entries


def _world_trajectory_gap_rows(
    adopted_by_tick: dict[int, dict[str, dict]],
) -> list[dict]:
    """Recorded adopted rows for (tick, entity) pairs the world trajectory lacks.

    The original world trajectory begins each entity at its activation tick
    (uav_l2_1_v2 at tick 32, uav_observer_l2_1_v2_3 at tick 2, vehicles at
    their recorded ends), while the merged truth frames expose the recorded
    adopted rows from tick 0.  The geometric computer emits the aircraft
    contract tuple from trajectory positions only, and the semantic engine
    requires that tuple at every tick the aircraft is visible in the truth
    frames, so every visible (tick, entity) needs a trajectory row.  This is
    the same adopted-only rule as the truth-frame append: recorded adopted
    rows, verbatim, for exactly the uncovered pairs.
    """
    covered: set[tuple[int, str]] = set()
    world_path = ORIGINAL_EPISODE_ROOT / "trajectories.jsonl"
    with open(world_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            covered.add((int(row["tick"]), str(row["entity_id"])))
    gap_rows = []
    for tick in sorted(adopted_by_tick):
        for entity_id, row in adopted_by_tick[tick].items():
            if (tick, entity_id) not in covered:
                gap_rows.append(copy.deepcopy(row))
    return gap_rows


def _isolated_old_current_view(row: dict) -> dict:
    """Move the stale SUMO current-view into an original-source reference.

    For an actor whose motion the adopted runtime covers (a ``sumo_vehicle``
    family member or a row with the old ``annotations.speed_mps``), the stale
    pre-adoption current-view fields must not remain as current truth: the
    domain and communication readers still bind to ``sumo_vehicle.speed_mps``
    and ``annotations.speed_mps``.  The entire old family is moved verbatim
    into ``original_source_reference.sumo_vehicle_current_view`` (the immutable
    original input is unchanged; nothing is deleted from the artifact other
    than by this recorded relocation) and ``speed_mps`` is removed from the
    old annotations.  The new ``runtime_merge`` block carries the relocation
    receipt, and the current-view fields are bound from the actual adopted
    trajectory row in ``_bind_adopted_motion_current_view``.
    """
    families = ("sumo_vehicle", "sumo_segment", "sumo_visibility")
    has_family = any(k in row for k in families)
    annotations = row.get("annotations")
    old_speed = (
        isinstance(annotations, dict) and isinstance(annotations.get("speed_mps"), (int, float))
    )
    if not has_family and not old_speed:
        return {}
    ref_view: dict = {"policy": "original_source_reference"}
    ref_view["fields"] = sorted(
        [k for k in families if k in row]
        + (["annotations.speed_mps"] if old_speed else [])
    )
    if has_family:
        ref_view["sumo_vehicle_current_view"] = copy.deepcopy(
            {k: row[k] for k in families if k in row}
        )
        for k in families:
            row.pop(k, None)
    if old_speed:
        ref_view["annotations_speed_mps_original"] = copy.deepcopy(
            annotations["speed_mps"]
        )
        ref_view["annotations_without_speed_mps"] = copy.deepcopy(annotations)
        annotations.pop("speed_mps", None)
    row["original_source_reference"] = ref_view
    row["runtime_merge"]["stale_current_view_isolation"] = {
        "policy": "old sumo_vehicle family and annotations.speed_mps are not current truth",
        "original_source_reference": True,
        "current_truth_source": "recorded adopted trajectory row",
    }
    return ref_view


def _bind_adopted_motion_current_view(row: dict, adopted_row: dict) -> None:
    """Bind the current view from the recorded adopted trajectory row.

    Only recorded adopted values are used: ``annotations.speed_mps`` is the
    Euclidean norm of the recorded adopted ``vel_mps`` and ``truth_pose`` is
    projected from the recorded adopted ``pos_enu``/``vel_mps``.  The adopted
    contract is ENU meters per tick-second, so the norm needs no scale factor.
    Nothing is invented: absent adopted fields leave the binding absent, and
    no old lane/accel/signal value is reintroduced as current truth.
    """
    velocity = adopted_row.get("vel_mps")
    if isinstance(velocity, list) and len(velocity) == 3 and all(
        isinstance(v, (int, float)) for v in velocity
    ):
        annotations = row.get("annotations")
        if not isinstance(annotations, dict):
            annotations = {}
            row["annotations"] = annotations
        annotations["speed_mps"] = math.sqrt(
            velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2
        )
        annotations["speed_source"] = "adopted_trajectory_vel_mps_norm"
    position = adopted_row.get("pos_enu")
    if isinstance(position, list) and len(position) == 3 and all(
        isinstance(v, (int, float)) for v in position
    ):
        truth_pose = {
            "authority_mode": "adopted_runtime_authority",
            "authority_owner": "p09_adopted_runtime",
            "coordinate_contract_id": "coord.external_enu_m.v1",
            "position_enu_m": [float(v) for v in position],
        }
        if isinstance(velocity, list) and len(velocity) == 3 and all(
            isinstance(v, (int, float)) for v in velocity
        ):
            truth_pose["velocity_enu_mps"] = [float(v) for v in velocity]
        yaw = adopted_row.get("yaw_deg")
        if isinstance(yaw, (int, float)):
            truth_pose["rotation_deg"] = {"yaw_deg": float(yaw)}
        row["truth_pose"] = truth_pose


def _bind_adopted_only_identity(
    merged_rows: list,
    identity_by_id: dict,
) -> dict:
    """Bind roster identity onto adopted-only truth-frame appends.

    Truth-frame entities are consumed with roster-key identity fields (domain
    state requires ``entity_category``).  Adopted trajectory rows carry the
    recorded ``category`` value but not the roster key name, so each
    adopted-only append receives the missing identity fields from its union
    roster entry (world roster member) or assembled declaration.  Recorded
    adopted values are never overridden; only absent keys are bound.
    """
    bound_ticks: dict[str, int] = {}
    for frame in merged_rows:
        for entity in frame["entities"]:
            provenance = entity.get("runtime_merge")
            if (
                not isinstance(provenance, dict)
                or provenance.get("authority")
                != "adopted_trajectory_entity_absent_from_original_truth_frame"
            ):
                continue
            entity_id = entity["entity_id"]
            identity_entry, identity_source = identity_by_id[entity_id]
            if "entity_category" in entity:
                bound_ticks[entity_id] = bound_ticks.get(entity_id, 0)
                continue
            category = identity_entry.get("entity_category")
            if not isinstance(category, str) or not category:
                raise ObjectiveRebuildError(
                    f"identity source {identity_source} lacks entity_category for "
                    f"adopted-only entity {entity_id}"
                )
            entity["entity_category"] = category
            entity["adopted_identity_binding"] = {
                "roster_entry_source": identity_source,
                "entity_category": category,
                "note": (
                    "identity key bound from the union roster entry; the recorded "
                    "adopted category value is unchanged and no recorded adopted "
                    "field is overridden"
                ),
            }
            bound_ticks[entity_id] = bound_ticks.get(entity_id, 0) + 1
    return bound_ticks


SCENE_BINDING_CONSUMERS = (
    {
        "reader": "Dataset/semantic_truth/charging_supplement.py:_resolve_scene_setup",
        "rule": "episode_root/scene_setup.json is consumed when present",
    },
    {
        "reader": "Dataset/semantic_simulation/domain_state.py:_load_scene_setup",
        "rule": "episode_root/scene_setup.json is consumed when present",
    },
    {
        "reader": "Dataset/semantic_truth/objective_pipeline.py:_objective_strict_episode_root",
        "rule": "episode_root/scene_setup.json is sanitized and copied into the strict scratch root",
    },
)


def _adopted_kinematic_current_view(entity, adopted_row, *, adopted_path):
    """Project one already-merged entity; no source mutation or reader change."""
    import copy
    import math

    if entity.get("entity_id") != adopted_row.get("entity_id"):
        raise ObjectiveRebuildError("current-view/adopted entity identity mismatch")
    category = adopted_row.get("entity_category") or adopted_row.get("category")
    if not isinstance(category, str) or not category:
        raise ObjectiveRebuildError("adopted row has no declared category")
    for key in ("entity_category", "category"):
        if key in entity and entity[key] != category:
            raise ObjectiveRebuildError(f"current-view category conflict: {key}")
    velocity = adopted_row.get("vel_mps")
    if (not isinstance(velocity, (list, tuple)) or len(velocity) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for x in velocity)):
        raise ObjectiveRebuildError("adopted velocity must be a finite numeric vec3")
    pose = entity.get("truth_pose")
    if (not isinstance(pose, dict)
            or pose.get("velocity_enu_mps") != list(velocity)):
        raise ObjectiveRebuildError("already-merged truth velocity differs from adopted row")
    if not isinstance(entity.get("sumo_vehicle"), dict):
        raise ObjectiveRebuildError("expected the recorded stale-SUMO source boundary")
    annotations = entity.get("annotations")
    if not isinstance(annotations, dict):
        raise ObjectiveRebuildError("current entity annotations must be an object")

    original_reference = copy.deepcopy(entity)
    current = copy.deepcopy(entity)
    current.pop("sumo_vehicle")
    current["annotations"].pop("speed_mps", None)
    current["annotations"]["speed_mps"] = math.sqrt(sum(x * x for x in velocity))
    current["entity_category"] = category
    current["source"] = "p09_adopted_kinematic_trajectory"
    current["current_view_source"] = {
        "path": str(adopted_path),
        "tick": adopted_row["tick"],
        "entity_id": adopted_row["entity_id"],
        "speed_definition": "euclidean_norm_of_recorded_adopted_vel_mps",
        "retired_current_fields": ["sumo_vehicle", "annotations.speed_mps", "source"],
    }
    return current, original_reference


def _write_stale_sumo_tick0_checkpoint(merged_rows, *, domain_speed, compute_speed):
    """Bounded evidence only. Does not run the builder or change merged_rows."""
    import copy
    tick = 0
    entity_id = "bg_vehicle_l2_1_v2_01"
    frames = [row for row in merged_rows if row.get("tick") == tick]
    if len(frames) != 1 or not isinstance(frames[0].get("entities"), list):
        raise ObjectiveRebuildError("expected exactly one actual merged tick0 frame")
    matches = [row for row in frames[0]["entities"] if row.get("entity_id") == entity_id]
    adopted = [row for row in _iter_jsonl(ADOPTED_TRAJECTORIES)
               if (row.get("tick"), row.get("entity_id")) == (tick, entity_id)]
    if len(matches) != 1 or len(adopted) != 1:
        raise ObjectiveRebuildError("tick0 current/adopted target must each occur once")
    before = copy.deepcopy(matches[0])
    current, reference = _adopted_kinematic_current_view(
        matches[0], adopted[0], adopted_path=ADOPTED_TRAJECTORIES)
    domain_value = domain_speed(current)
    compute_value = compute_speed(current)
    if adopted[0]["vel_mps"] != [0.0, 0.0, 0.0] or domain_value != 0.0 or compute_value != 0.0:
        raise ObjectiveRebuildError(
            f"actual reader mismatch: domain={domain_value!r}, compute={compute_value!r}")
    if matches[0] != before or reference != before:
        raise ObjectiveRebuildError("source record changed during current-view projection")
    # Outside staging_v1, so a later staging rebuild cannot erase this evidence.
    out = STAGING_ROOT.parents[1] / "author_adopted_input_checkpoint_v3"
    out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "stale_sumo_tick0.original.json": reference,
        "stale_sumo_tick0.current.json": current,
        "stale_sumo_tick0.readers.json": {
            "tick": tick, "entity_id": entity_id,
            "domain_reader": domain_speed.__module__ + "." + domain_speed.__name__,
            "compute_reader": compute_speed.__module__ + "." + compute_speed.__name__,
            "domain_speed_mps": domain_value, "compute_speed_mps": compute_value,
            "adopted_velocity_mps": adopted[0]["vel_mps"],
            "original_record_unchanged": True,
            "scope": "one real merged actor/tick; builder not run",
        },
    }
    for name, payload in payloads.items():
        path = out / name
        if path.exists() and _read_json(path) != payload:
            raise ObjectiveRebuildError(f"checkpoint already exists with different contents: {path}")
    for name, payload in payloads.items():
        path = out / name
        if not path.exists():
            _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return payloads["stale_sumo_tick0.readers.json"]


FORMAL_UE_TRUTH = Path(
    "/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/ue_input_overlay_v14"
    "/capture_filtered_updates"
) / EPISODE_ID / "truth_frames.jsonl"


L2_V14_NUMERIC_TARGETS = frozenset({
    "uav_l2_1_v2", "tower_l2_1_v2",
    "corridor_l2_1_v2_uav_l2_1_v2_03",
    "corridor_l2_1_v2_uav_l2_1_v2_04",
})


def _unique_records(rows, key, label):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ObjectiveRebuildError(f"duplicate {label}: {k!r}")
        result[k] = row
    return result


def _formal_v14_world_rows(full_rows, ue_rows, adopted_rows, *, runtime_families):
    """Keep formal v14 numeric/presence scope; add only untouched full-world background."""
    import copy
    families = tuple(runtime_families)
    forbidden = {"entity_id", "tick", "truth_pose", "render_presence", "annotations",
                 "source", "sumo_vehicle", "category", "entity_category", "state",
                 "pos_enu", "vel_mps", "yaw_deg", "route_waypoints_enu_m"}
    if set(families) & forbidden:
        raise ObjectiveRebuildError("runtime family selection contains numeric/presence/structural fields")
    full = _unique_records(full_rows, lambda r: r["tick"], "full-world tick")
    ue = _unique_records(ue_rows, lambda r: r["tick"], "formal UE tick")
    if set(full) != set(range(901)) or set(ue) != set(full):
        raise ObjectiveRebuildError("pilot must use both complete actual 0..900 inputs")
    adopted = _unique_records(adopted_rows, lambda r: (r["tick"], r["entity_id"]), "adopted row")
    result, runtime_updates = [], []
    ue_numeric_checked = 0
    for tick in sorted(full):
        base, formal = full[tick], ue[tick]
        for clock_key in ("sim_time_s", "sim_time_ns"):
            if clock_key in base and clock_key in formal and base[clock_key] != formal[clock_key]:
                raise ObjectiveRebuildError(f"clock mismatch at {tick}: {clock_key}")
        b = _unique_records(base["entities"], lambda r: r["entity_id"], "full entity")
        u = _unique_records(formal["entities"], lambda r: r["entity_id"], "UE entity")
        if (set(b) - set(u)) & L2_V14_NUMERIC_TARGETS:
            raise ObjectiveRebuildError(f"full-only affected actor needs explicit offstage binding at {tick}")
        if (set(u) - set(b)) - L2_V14_NUMERIC_TARGETS:
            raise ObjectiveRebuildError(f"UE contains unexpected new background at {tick}")
        entities = []
        for entity_id in list(b) + [eid for eid in u if eid not in b]:
            if entity_id in L2_V14_NUMERIC_TARGETS and entity_id in u:
                current = copy.deepcopy(u[entity_id])
            else:
                current = copy.deepcopy(b[entity_id])
                if entity_id in u and current.get("truth_pose") != u[entity_id].get("truth_pose"):
                    raise ObjectiveRebuildError(f"unaffected background numeric mismatch: {(tick, entity_id)}")
            recorded = adopted.get((tick, entity_id))
            changed = []
            if recorded is not None:
                for family in families:
                    if family in recorded:
                        if not isinstance(recorded[family], dict):
                            raise ObjectiveRebuildError(f"runtime family is not an object: {family}")
                        current[family] = copy.deepcopy(recorded[family])
                        changed.append(family)
            if changed:
                runtime_updates.append({"tick": tick, "entity_id": entity_id, "families": changed})
            if entity_id in u:
                if current.get("truth_pose") != u[entity_id].get("truth_pose"):
                    raise ObjectiveRebuildError(f"runtime overlay changed formal UE numeric input: {(tick, entity_id)}")
                ue_numeric_checked += 1
            entities.append(current)
        frame = copy.deepcopy(base)
        frame["entities"] = entities
        if set(r["entity_id"] for r in entities) != set(b) | set(u):
            raise ObjectiveRebuildError(f"unexpected world membership at {tick}")
        result.append(frame)
    return result, {"numeric_targets": sorted(L2_V14_NUMERIC_TARGETS),
                    "formal_ue_entity_numeric_checks": ue_numeric_checked,
                    "runtime_updates": runtime_updates,
                    "source_rule": "full world + actual formal v14 target records; recorded runtime families only"}


def _formal_v14_world_trajectories(original_rows, adopted_rows, rebuilt_frames, *, runtime_families):
    """No world-trajectory gap synthesis; no unspawned/background source-template insertion."""
    import copy
    original = _unique_records(original_rows, lambda r: (r["tick"], r["entity_id"]), "original trajectory")
    adopted = _unique_records(adopted_rows, lambda r: (r["tick"], r["entity_id"]), "adopted trajectory")
    world = {(f["tick"], e["entity_id"]): e for f in rebuilt_frames for e in f["entities"]}
    if set(original) - set(world):
        raise ObjectiveRebuildError("original trajectory has offstage keys; preserve separately and review binding, not implicit gap-fill")
    rows = []
    for key, entity in sorted(world.items()):
        if key[1] in L2_V14_NUMERIC_TARGETS:
            if key not in adopted:
                raise ObjectiveRebuildError(f"formal target lacks actual adopted trajectory: {key}")
            row = copy.deepcopy(adopted[key])
            pose = entity.get("truth_pose") or {}
            if (row.get("pos_enu") != pose.get("position_enu_m")
                    or row.get("vel_mps") != pose.get("velocity_enu_mps")
                    or row.get("yaw_deg") != (pose.get("rotation_deg") or {}).get("yaw_deg")):
                raise ObjectiveRebuildError(f"formal UE/selected trajectory numeric mismatch: {key}")
            category = entity.get("entity_category")
            if not isinstance(category, str) or not category:
                raise ObjectiveRebuildError(f"formal entity category missing: {key}")
            row["entity_category"] = category
        else:
            if key not in original:
                raise ObjectiveRebuildError(f"unaffected world entity lacks original trajectory: {key}")
            row = copy.deepcopy(original[key])
        recorded = adopted.get(key)
        if recorded is not None:
            for family in runtime_families:
                if family in recorded:
                    row[family] = copy.deepcopy(recorded[family])
        rows.append(row)
    return rows


def _curate_controller_service_runtime(adopted_rows, *, runtime_families, runtime_sources_at):
    """Select the current adopted controller/service state source for each row.

    runtime_sources_at[(tick, entity_id)] is {family: actual_source_reference}.
    This mapping is mechanically expanded from the already adopted run/profile
    and controller/service snapshot rule, not a per-tick approval ledger. Real
    initialized state (including False/0) is valid without an action receipt.
    Exclude unadopted templates and renderer defaults; do not infer authority
    merely from membership in the 17-actor raw trajectory. Uncertain fields are
    local source-binding gaps, not a requirement to re-review every family.
    """
    import copy
    families = set(runtime_families)
    structural = {
        "entity_id", "tick", "truth_pose", "render_presence", "annotations",
        "source", "sumo_vehicle", "sumo_segment", "sumo_visibility", "background_vehicle",
        "category", "entity_category", "state", "state_sequence", "activation_tick",
        "pos_enu", "vel_mps", "yaw_deg", "route_waypoints_enu_m",
        "planned_route_waypoints_enu_m", "ground_flow_contract", "motion_contract",
        "lifecycle", "route_metadata_migration",
    }
    if families & structural:
        raise ObjectiveRebuildError("runtime contract selection contains structural/plan/source fields")
    keys = {(r["tick"], r["entity_id"]) for r in adopted_rows}
    if set(runtime_sources_at) - keys:
        raise ObjectiveRebuildError("runtime authority refers to an absent actual recorded row")
    result, audit = [], []
    for raw in adopted_rows:
        key = (raw["tick"], raw["entity_id"])
        row = copy.deepcopy(raw)
        for family in families:
            row.pop(family, None)
        for family, source_ref in runtime_sources_at.get(key, {}).items():
            if family not in families or not isinstance(source_ref, str) or not source_ref.strip():
                raise ObjectiveRebuildError(f"invalid runtime source binding: {key}, {family}")
            if not isinstance(raw.get(family), dict):
                raise ObjectiveRebuildError(f"bound runtime family missing from recorded row: {key}, {family}")
            row[family] = copy.deepcopy(raw[family])
            audit.append({"tick": key[0], "entity_id": key[1], "family": family, "source_ref": source_ref})
        result.append(row)
    return result, audit


def _prepare_formal_v14_inputs(*, runtime_families, runtime_sources_at):
    """Prepare formal v14 caller inputs; the builder is not invoked here.

    Loads the complete original world truth and trajectory files, the pinned
    actual formal UE capture, and the recorded adopted trajectory rows, then
    applies the supplied native-approved helpers in order: controller/service
    runtime curation, formal world-row selection, and formal world-trajectory
    selection.  runtime_families and runtime_sources_at are explicit caller
    parameters naming the current controller/service families and the
    (tick, entity_id) -> {family: source_ref} binding; no family is guessed
    and an empty authority map is rejected instead of hiding a missing
    current binding.  Returns (merged_rows, selected_trajectories,
    source_selection, runtime_authority_audit).
    """
    families = tuple(runtime_families)
    if not families or not all(
        isinstance(family, str) and family.strip() for family in families
    ):
        raise ObjectiveRebuildError(
            "formal v14 requires explicit current runtime families from the caller"
        )
    if not isinstance(runtime_sources_at, dict) or not runtime_sources_at:
        raise ObjectiveRebuildError(
            "formal v14 requires an explicit runtime_sources_at binding map; "
            "an empty map would hide missing current controller/service bindings"
        )
    inputs = {
        "original_truth_frames": ORIGINAL_EPISODE_ROOT / "truth_frames.jsonl",
        "original_trajectories": ORIGINAL_EPISODE_ROOT / "trajectories.jsonl",
        "formal_ue_truth_frames": FORMAL_UE_TRUTH,
        "adopted_trajectories": ADOPTED_TRAJECTORIES,
    }
    for label, path in inputs.items():
        if not path.exists():
            raise ObjectiveRebuildError(
                f"declared formal v14 input missing ({label}): {path}"
            )
    full_rows = list(_iter_jsonl(inputs["original_truth_frames"]))
    original_trajectory_rows = list(_iter_jsonl(inputs["original_trajectories"]))
    ue_rows = list(_iter_jsonl(inputs["formal_ue_truth_frames"]))
    adopted_rows = list(_iter_jsonl(inputs["adopted_trajectories"]))
    curated_rows, runtime_authority_audit = _curate_controller_service_runtime(
        adopted_rows,
        runtime_families=families,
        runtime_sources_at=runtime_sources_at,
    )
    merged_rows, selection = _formal_v14_world_rows(
        full_rows,
        ue_rows,
        curated_rows,
        runtime_families=families,
    )
    selected_trajectories = _formal_v14_world_trajectories(
        original_trajectory_rows,
        curated_rows,
        merged_rows,
        runtime_families=families,
    )
    source_selection = dict(selection)
    source_selection["inputs"] = {label: str(path) for label, path in inputs.items()}
    return merged_rows, selected_trajectories, source_selection, runtime_authority_audit


FORMAL_PRECHECK_CHECKPOINT = SESSION_ROOT / "author_formal_precheck_checkpoint_v7"
EXPECTED_FORMAL_FRAMES = 901
EXPECTED_FORMAL_UE_ENTITY_NUMERIC_CHECKS = 35158
EXPECTED_BG01_TICK0_SPEED_MPS = 0.016473


def _write_formal_precheck_checkpoint(payload: dict) -> Path:
    path = FORMAL_PRECHECK_CHECKPOINT / "formal_source_precheck.json"
    if path.exists():
        raise ObjectiveRebuildError(
            f"formal precheck checkpoint already exists: {path}"
        )
    FORMAL_PRECHECK_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return path


def _write_formal_precheck_failure(payload: dict) -> Path:
    FORMAL_PRECHECK_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    path = FORMAL_PRECHECK_CHECKPOINT / "formal_source_precheck_failure.txt"
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return path


def _formal_source_precheck_body() -> dict:
    """Execute the formal-v14 source precheck and return the evidence payload.

    Staging assembly and the objective builder are not executed here.  The
    runtime family rule is the structured-runtime family tuple declared by the
    existing domain-state reader; the (tick, entity_id) -> {family: source_ref}
    map is that rule expanded mechanically over the recorded adopted rows.
    """
    from Dataset.semantic_simulation import compute_comm, domain_state

    adopted_rows = list(_iter_jsonl(ADOPTED_TRAJECTORIES))
    runtime_families = tuple(domain_state.OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES)
    FAILURE_CONTEXT["stage"] = "formal_source_precheck"
    FAILURE_CONTEXT["inputs"] = {
        "original_episode_root": str(ORIGINAL_EPISODE_ROOT),
        "formal_ue_truth_frames": str(FORMAL_UE_TRUTH),
        "adopted_trajectories": str(ADOPTED_TRAJECTORIES),
        "runtime_families": list(runtime_families),
    }
    input_paths = {
        "original_truth_frames": ORIGINAL_EPISODE_ROOT / "truth_frames.jsonl",
        "original_trajectories": ORIGINAL_EPISODE_ROOT / "trajectories.jsonl",
        "formal_ue_truth_frames": FORMAL_UE_TRUTH,
        "adopted_trajectories": ADOPTED_TRAJECTORIES,
    }
    inputs_before = {
        label: (path.stat().st_size, path.stat().st_mtime_ns)
        for label, path in input_paths.items()
    }

    runtime_sources_at: dict[tuple[int, str], dict[str, str]] = {}
    family_rows = 0
    family_totals: dict[str, int] = {}
    for row in adopted_rows:
        present = [family for family in runtime_families if family in row]
        if not present:
            continue
        key = (row["tick"], row["entity_id"])
        runtime_sources_at[key] = {
            family: str(ADOPTED_TRAJECTORIES) for family in present
        }
        family_rows += 1
        for family in present:
            family_totals[family] = family_totals.get(family, 0) + 1
    if not runtime_sources_at:
        raise ObjectiveRebuildError("no recorded adopted controller/service family rows")

    (
        merged_rows,
        selected_trajectories,
        source_selection,
        runtime_authority_audit,
    ) = _prepare_formal_v14_inputs(
        runtime_families=runtime_families,
        runtime_sources_at=runtime_sources_at,
    )

    payload: dict = {
        "stage": "formal_source_precheck",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "runtime_families": list(runtime_families),
        "runtime_family_rule_source": (
            "Dataset/semantic_simulation/domain_state.py:"
            "OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES"
        ),
        "runtime_source_rule": (
            "owner+family rule expanded mechanically over the recorded adopted "
            "rows; every family dict already recorded on a row keeps its "
            "recorded initialized state (False/0/empty included, no action "
            "receipt), with the adopted trajectory artifact as source_ref"
        ),
        "runtime_sources_at_rows": family_rows,
        "runtime_family_rows": dict(sorted(family_totals.items())),
        "runtime_authority_audit_rows": len(runtime_authority_audit),
        "runtime_authority_audit_sample": runtime_authority_audit[:2],
        "expected": {
            "merged_truth_frames": EXPECTED_FORMAL_FRAMES,
            "formal_ue_entity_numeric_checks": (
                EXPECTED_FORMAL_UE_ENTITY_NUMERIC_CHECKS
            ),
            "bg01_tick0_speed_mps": EXPECTED_BG01_TICK0_SPEED_MPS,
        },
    }
    payload["source_selection"] = source_selection
    payload["selected_trajectory_rows"] = len(selected_trajectories)
    payload["selected_rows_sample"] = selected_trajectories[:2]

    if len(merged_rows) != EXPECTED_FORMAL_FRAMES:
        raise ObjectiveRebuildError(
            f"expected {EXPECTED_FORMAL_FRAMES} formal merged frames: {len(merged_rows)}"
        )
    numeric_checks = source_selection["formal_ue_entity_numeric_checks"]
    if numeric_checks != EXPECTED_FORMAL_UE_ENTITY_NUMERIC_CHECKS:
        raise ObjectiveRebuildError(
            f"expected {EXPECTED_FORMAL_UE_ENTITY_NUMERIC_CHECKS} formal UE entity "
            f"numeric checks: {numeric_checks}"
        )
    payload["merged_truth_frames"] = len(merged_rows)
    payload["formal_ue_entity_numeric_checks"] = numeric_checks

    runtime_updates = source_selection["runtime_updates"]
    updates_by_family: dict[str, int] = {}
    updates_by_entity: dict[str, int] = {}
    for update in runtime_updates:
        for family in update["families"]:
            updates_by_family[family] = updates_by_family.get(family, 0) + 1
        updates_by_entity[update["entity_id"]] = (
            updates_by_entity.get(update["entity_id"], 0) + 1
        )
    payload["runtime_updates"] = len(runtime_updates)
    payload["runtime_updates_by_family"] = dict(sorted(updates_by_family.items()))
    payload["runtime_updates_by_entity"] = dict(sorted(updates_by_entity.items()))

    def _frame_entities(tick: int) -> dict:
        frames = [frame for frame in merged_rows if frame["tick"] == tick]
        if len(frames) != 1 or not isinstance(frames[0].get("entities"), list):
            raise ObjectiveRebuildError(
                f"expected exactly one merged frame at tick {tick}"
            )
        return {row["entity_id"]: row for row in frames[0]["entities"]}

    tick0 = _frame_entities(0)
    absent_expected = (
        "uav_l2_1_v2",
        "bg_vehicle_l2_1_v2_02",
        "bg_vehicle_l2_1_v2_03",
    )
    present_absent = [key for key in absent_expected if key in tick0]
    if present_absent:
        raise ObjectiveRebuildError(f"unexpected tick0 membership: {present_absent}")
    payload["tick0"] = {
        "uav_l2_1_v2_absent": True,
        "bg_vehicle_l2_1_v2_02_absent": True,
        "bg_vehicle_l2_1_v2_03_absent": True,
        "entity_count": len(tick0),
    }

    bg01 = tick0.get("bg_vehicle_l2_1_v2_01")
    if bg01 is None:
        raise ObjectiveRebuildError("tick0 bg_vehicle_l2_1_v2_01 missing")
    domain_value = domain_state._speed(bg01)
    compute_value = compute_comm._entity_speed(bg01)
    if (
        domain_value != EXPECTED_BG01_TICK0_SPEED_MPS
        or compute_value != EXPECTED_BG01_TICK0_SPEED_MPS
    ):
        raise ObjectiveRebuildError(
            f"formal bg01 tick0 reader mismatch: domain={domain_value!r}, "
            f"compute={compute_value!r}"
        )
    payload["tick0"]["bg01_sumo_retained_speed"] = {
        "domain_state_reader": "Dataset.semantic_simulation.domain_state._speed",
        "compute_comm_reader": "Dataset.semantic_simulation.compute_comm._entity_speed",
        "domain_speed_mps": domain_value,
        "compute_speed_mps": compute_value,
    }

    entities301 = _frame_entities(301)
    uav301 = entities301.get("uav_l2_1_v2")
    if uav301 is None:
        raise ObjectiveRebuildError("tick 301 uav_l2_1_v2 missing")
    adopted301 = [
        row
        for row in adopted_rows
        if (row["tick"], row["entity_id"]) == (301, "uav_l2_1_v2")
    ]
    if len(adopted301) != 1:
        raise ObjectiveRebuildError("expected one adopted uav row at tick 301")
    pose301 = uav301.get("truth_pose") or {}
    comparisons = {
        "position_enu_m": (
            pose301.get("position_enu_m"),
            adopted301[0].get("pos_enu"),
        ),
        "velocity_enu_mps": (
            pose301.get("velocity_enu_mps"),
            adopted301[0].get("vel_mps"),
        ),
        "yaw_deg": (
            (pose301.get("rotation_deg") or {}).get("yaw_deg"),
            adopted301[0].get("yaw_deg"),
        ),
    }
    for label, (from_frame, recorded) in comparisons.items():
        if from_frame != recorded:
            raise ObjectiveRebuildError(
                f"uav_l2_1_v2@301 {label} differs from formal v14: "
                f"{from_frame!r} != {recorded!r}"
            )
    payload["uav_l2_1_v2_301"] = {
        "position_enu_m": pose301.get("position_enu_m"),
        "velocity_enu_mps": pose301.get("velocity_enu_mps"),
        "yaw_deg": (pose301.get("rotation_deg") or {}).get("yaw_deg"),
        "matches_formal_v14_row": True,
        "communication_state": uav301.get("communication_state"),
        "control_state": uav301.get("control_state"),
    }

    tower_states: dict[str, dict] = {}
    for tick in (260, 261):
        entity = _frame_entities(tick).get("tower_l2_1_v2")
        if entity is None:
            raise ObjectiveRebuildError(f"tick {tick} tower_l2_1_v2 missing")
        communication = entity.get("communication_state")
        if not isinstance(communication, dict):
            raise ObjectiveRebuildError(
                f"tower_l2_1_v2@{tick} communication_state is not an object"
            )
        tower_states[str(tick)] = {
            "station_unavailable": communication.get("station_unavailable"),
            "backup_link_active": communication.get("backup_link_active"),
        }
    if (
        tower_states["260"]["station_unavailable"] is not False
        or tower_states["261"]["station_unavailable"] is not True
    ):
        raise ObjectiveRebuildError(
            f"tower 260/261 station_unavailable boundary mismatch: {tower_states}"
        )
    payload["tower_l2_1_v2"] = tower_states

    roster_path = ORIGINAL_EPISODE_ROOT / "global_entity_roster.json"
    world_roster = json.loads(roster_path.read_text(encoding="utf-8-sig"))
    roster_ids = {row["entity_id"] for row in world_roster["entities"]}
    world_entity_count = 0
    for frame in merged_rows:
        for entity in frame["entities"]:
            if entity["entity_id"] not in roster_ids:
                raise ObjectiveRebuildError(
                    f"formal world entity outside the original roster: "
                    f"{(frame['tick'], entity['entity_id'])}"
                )
            world_entity_count += 1
    trajectory_ids = {row["entity_id"] for row in selected_trajectories}
    outside = trajectory_ids - roster_ids
    if outside:
        raise ObjectiveRebuildError(
            f"selected trajectory entity outside the original roster: {sorted(outside)}"
        )
    payload["roster"] = {
        "path": str(roster_path),
        "entries": len(world_roster["entities"]),
        "world_entities_sourced_from_original_roster": world_entity_count,
        "selected_trajectory_entities": len(trajectory_ids),
        "roster_additions": 0,
    }

    inputs_after = {
        label: (path.stat().st_size, path.stat().st_mtime_ns)
        for label, path in input_paths.items()
    }
    if inputs_after != inputs_before:
        changed = [
            label
            for label in inputs_before
            if inputs_before[label] != inputs_after[label]
        ]
        raise ObjectiveRebuildError(
            f"formal source inputs changed during the precheck: {changed}"
        )
    payload["source_inputs_unchanged"] = True
    payload["source_inputs"] = {
        label: {"path": str(path), "bytes": inputs_after[label][0]}
        for label, path in input_paths.items()
    }
    payload["scope"] = (
        "formal v14 source selection precheck only; no staging write, no objective "
        "builder, no persist, and no downstream claim in this run"
    )
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    return payload


def _run_formal_source_precheck() -> int:
    """Save the formal precheck evidence; never write outside the checkpoint."""
    try:
        payload = _formal_source_precheck_body()
    except BaseException as exc:
        failure = {
            "stage": "formal_source_precheck",
            "command": " ".join(sys.argv),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "inputs": FAILURE_CONTEXT.get("inputs"),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        saved = _write_formal_precheck_failure(failure)
        traceback.print_exc()
        print(f"formal precheck failure evidence saved: {saved}", file=sys.stderr)
        return 1
    saved = _write_formal_precheck_checkpoint(payload)
    print(
        json.dumps(
            {
                "stage": "formal_source_precheck",
                "checkpoint": str(saved),
                "merged_truth_frames": payload["merged_truth_frames"],
                "formal_ue_entity_numeric_checks": payload[
                    "formal_ue_entity_numeric_checks"
                ],
                "runtime_updates": payload["runtime_updates"],
                "runtime_updates_by_entity": payload["runtime_updates_by_entity"],
                "tick0_bg01_speed_mps": {
                    "domain_state": payload["tick0"]["bg01_sumo_retained_speed"][
                        "domain_speed_mps"
                    ],
                    "compute_comm": payload["tick0"]["bg01_sumo_retained_speed"][
                        "compute_speed_mps"
                    ],
                },
                "uav_301_matches_formal_v14": True,
                "tower_260_261_station_unavailable": [
                    payload["tower_l2_1_v2"]["260"]["station_unavailable"],
                    payload["tower_l2_1_v2"]["261"]["station_unavailable"],
                ],
                "selected_trajectory_rows": payload["selected_trajectory_rows"],
                "source_inputs_unchanged": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


FORMAL_OBJECTIVE_CHECKPOINT = SESSION_ROOT / "author_formal_objective_checkpoint_v8"
FORMAL_OBJECTIVE_STAGING = FORMAL_OBJECTIVE_CHECKPOINT / "staging" / EPISODE_ID
FORMAL_OBJECTIVE_OUTPUT = FORMAL_OBJECTIVE_CHECKPOINT / "objective" / EPISODE_ID
# P09-M06-L2 class-A rerun roots: the original staging/ and objective/ trees of
# the successful run are preserved untouched; the class-A weather-dust staging
# wiring rerun writes only into staging_r2/ and objective_r2/.
FORMAL_OBJECTIVE_STAGING_R2 = FORMAL_OBJECTIVE_CHECKPOINT / "staging_r2" / EPISODE_ID
FORMAL_OBJECTIVE_OUTPUT_R2 = FORMAL_OBJECTIVE_CHECKPOINT / "objective_r2" / EPISODE_ID
FORMAL_OBJECTIVE_R2 = os.environ.get("P09_FORMAL_OBJECTIVE_R2") == "1"
if FORMAL_OBJECTIVE_R2:
    FORMAL_OBJECTIVE_STAGING = FORMAL_OBJECTIVE_STAGING_R2
    FORMAL_OBJECTIVE_OUTPUT = FORMAL_OBJECTIVE_OUTPUT_R2


def _persisted_output_facts(output_dir: Path) -> dict:
    """Read back persisted objective outputs and collect the reported facts.

    Report-only single pass; no new checker, no acceptance decision.  Files are
    parsed as JSONL or JSON, every record carrying (tick, entity_id) is
    collected, and only the acceptance-report values are extracted.
    """
    tokens = ("dispatch", "landing", "touchdown")
    facts = {
        "files_scanned": [],
        "tower_l2_1_v2_station_unavailable": {},
        "bg_vehicle_l2_1_v2_01_tick0": [],
        "uav_l2_1_v2": {"ticks_seen": [], "token_hits": []},
    }

    def _walk(node, records):
        if len(records) > 400000:
            return
        if isinstance(node, dict):
            if isinstance(node.get("tick"), int) and isinstance(
                node.get("entity_id"), str
            ):
                records.append(node)
            for value in node.values():
                _walk(value, records)
        elif isinstance(node, list):
            for value in node:
                _walk(value, records)

    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        facts["files_scanned"].append(path.name)
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".jsonl":
            parsed = [
                json.loads(line) for line in text.splitlines() if line.strip()
            ]
        elif path.suffix == ".json":
            parsed = json.loads(text)
        else:
            continue
        records: list = []
        _walk(parsed, records)
        for record in records:
            tick = record["tick"]
            entity_id = record["entity_id"]
            if entity_id == "tower_l2_1_v2" and tick in (260, 261):
                communication = record.get("communication_state")
                if isinstance(communication, dict) and (
                    "station_unavailable" in communication
                ):
                    slot = facts["tower_l2_1_v2_station_unavailable"].setdefault(
                        str(tick), []
                    )
                    if len(slot) < 4:
                        slot.append(
                            {
                                "file": path.name,
                                "value": communication["station_unavailable"],
                            }
                        )
            elif entity_id == "bg_vehicle_l2_1_v2_01" and tick == 0:
                fields = {}
                for key in ("speed_mps", "speed", "domain_speed_mps"):
                    if key in record:
                        fields[key] = record[key]
                annotations = record.get("annotations")
                if isinstance(annotations, dict) and "speed_mps" in annotations:
                    fields["annotations.speed_mps"] = annotations["speed_mps"]
                sumo = record.get("sumo_vehicle")
                if isinstance(sumo, dict) and "speed_mps" in sumo:
                    fields["sumo_vehicle.speed_mps"] = sumo["speed_mps"]
                truth = record.get("truth_pose")
                if isinstance(truth, dict) and "velocity_enu_mps" in truth:
                    fields["truth_pose.velocity_enu_mps"] = truth[
                        "velocity_enu_mps"
                    ]
                if fields:
                    facts["bg_vehicle_l2_1_v2_01_tick0"].append(
                        {"file": path.name, **fields}
                    )
            elif entity_id == "uav_l2_1_v2":
                seen = set(facts["uav_l2_1_v2"]["ticks_seen"])
                seen.add(tick)
                facts["uav_l2_1_v2"]["ticks_seen"] = sorted(seen)
                blob = json.dumps(record, ensure_ascii=False)
                hits = [token for token in tokens if token in blob]
                if hits and len(facts["uav_l2_1_v2"]["token_hits"]) < 60:
                    facts["uav_l2_1_v2"]["token_hits"].append(
                        {"file": path.name, "tick": tick, "tokens": hits}
                    )
    return facts


def _formal_v14_objective_body() -> dict:
    """Run the existing builder once over the formal v14 selection.

    Inputs are exactly the accepted precheck run3 selection:
    _prepare_formal_v14_inputs with the structured-runtime family tuple from
    the existing domain-state reader and the mechanical owner+family expansion
    over the recorded adopted rows.  No gap rows, no roster additions, no
    stale-SUMO current view, no all17 merge.  The builder, its output check,
    and its persist contract run once with the four default profile paths and
    the strict input guard enabled.
    """
    from Dataset.semantic_simulation import compute_comm, domain_state
    from Dataset.semantic_truth.objective_pipeline import _jsonl_text

    pipeline_jsonl_text = _jsonl_text
    start_utc = datetime.now(timezone.utc)
    stage_log = []

    def _stage(name: str) -> None:
        FAILURE_CONTEXT["stage"] = name
        stage_log.append(
            {
                "stage": name,
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

    _stage("formal_objective_input_checks")
    if FORMAL_OBJECTIVE_STAGING.exists():
        raise ObjectiveRebuildError(
            f"formal staging root already exists: {FORMAL_OBJECTIVE_STAGING}"
        )
    if FORMAL_OBJECTIVE_OUTPUT.exists():
        raise ObjectiveRebuildError(
            f"formal objective output already exists: {FORMAL_OBJECTIVE_OUTPUT}"
        )
    precheck_path = FORMAL_PRECHECK_CHECKPOINT / "formal_source_precheck.json"
    if not precheck_path.exists():
        raise ObjectiveRebuildError(
            f"accepted formal precheck evidence missing: {precheck_path}"
        )
    precheck_payload = json.loads(precheck_path.read_text(encoding="utf-8-sig"))
    precheck_sizes = {
        label: spec["bytes"]
        for label, spec in precheck_payload["source_inputs"].items()
    }

    input_paths = {
        "original_truth_frames": ORIGINAL_EPISODE_ROOT / "truth_frames.jsonl",
        "original_trajectories": ORIGINAL_EPISODE_ROOT / "trajectories.jsonl",
        "formal_ue_truth_frames": FORMAL_UE_TRUTH,
        "adopted_trajectories": ADOPTED_TRAJECTORIES,
    }
    inputs_before = {
        label: (path.stat().st_size, path.stat().st_mtime_ns)
        for label, path in input_paths.items()
    }
    size_mismatch = {
        label
        for label, (size, _) in inputs_before.items()
        if precheck_sizes.get(label) != size
    }
    if size_mismatch:
        raise ObjectiveRebuildError(
            f"source input sizes differ from accepted precheck run3: "
            f"{sorted(size_mismatch)}"
        )
    FAILURE_CONTEXT["inputs"] = {
        "original_episode_root": str(ORIGINAL_EPISODE_ROOT),
        "formal_ue_truth_frames": str(FORMAL_UE_TRUTH),
        "adopted_trajectories": str(ADOPTED_TRAJECTORIES),
        "runtime_families_source": (
            "Dataset/semantic_simulation/domain_state.py:"
            "OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES"
        ),
    }

    _stage("formal_source_selection")
    adopted_rows = list(_iter_jsonl(ADOPTED_TRAJECTORIES))
    runtime_families = tuple(domain_state.OBJECTIVE_STRUCTURED_RUNTIME_FAMILIES)
    runtime_sources_at: dict[tuple[int, str], dict[str, str]] = {}
    for row in adopted_rows:
        present = [family for family in runtime_families if family in row]
        if present:
            runtime_sources_at[(row["tick"], row["entity_id"])] = {
                family: str(ADOPTED_TRAJECTORIES) for family in present
            }
    if not runtime_sources_at:
        raise ObjectiveRebuildError("no recorded adopted controller/service family rows")

    (
        merged_rows,
        selected_trajectories,
        source_selection,
        runtime_authority_audit,
    ) = _prepare_formal_v14_inputs(
        runtime_families=runtime_families,
        runtime_sources_at=runtime_sources_at,
    )
    if len(merged_rows) != EXPECTED_FORMAL_FRAMES:
        raise ObjectiveRebuildError(
            f"expected {EXPECTED_FORMAL_FRAMES} formal merged frames: "
            f"{len(merged_rows)}"
        )
    numeric_checks = source_selection["formal_ue_entity_numeric_checks"]
    if numeric_checks != EXPECTED_FORMAL_UE_ENTITY_NUMERIC_CHECKS:
        raise ObjectiveRebuildError(
            f"expected {EXPECTED_FORMAL_UE_ENTITY_NUMERIC_CHECKS} formal UE entity "
            f"numeric checks: {numeric_checks}"
        )
    FAILURE_CONTEXT["inputs"]["prepared_counts"] = {
        "merged_truth_frames": len(merged_rows),
        "selected_trajectory_rows": len(selected_trajectories),
        "formal_ue_entity_numeric_checks": numeric_checks,
        "runtime_authority_audit_rows": len(runtime_authority_audit),
    }

    entity_rows = 0
    target_rows = 0
    for frame in merged_rows:
        for entity in frame["entities"]:
            entity_rows += 1
            if entity["entity_id"] in L2_V14_NUMERIC_TARGETS:
                target_rows += 1
    background_rows = entity_rows - target_rows

    def _frame_entities(tick: int) -> dict:
        frames = [frame for frame in merged_rows if frame["tick"] == tick]
        if len(frames) != 1 or not isinstance(frames[0].get("entities"), list):
            raise ObjectiveRebuildError(
                f"expected exactly one merged frame at tick {tick}"
            )
        return {row["entity_id"]: row for row in frames[0]["entities"]}

    tick0 = _frame_entities(0)
    for absent in (
        "uav_l2_1_v2",
        "bg_vehicle_l2_1_v2_02",
        "bg_vehicle_l2_1_v2_03",
    ):
        if absent in tick0:
            raise ObjectiveRebuildError(f"unexpected tick0 membership: {absent}")
    bg01 = tick0.get("bg_vehicle_l2_1_v2_01")
    if bg01 is None:
        raise ObjectiveRebuildError("tick0 bg_vehicle_l2_1_v2_01 missing")
    domain_value = domain_state._speed(bg01)
    compute_value = compute_comm._entity_speed(bg01)
    if (
        domain_value != EXPECTED_BG01_TICK0_SPEED_MPS
        or compute_value != EXPECTED_BG01_TICK0_SPEED_MPS
    ):
        raise ObjectiveRebuildError(
            f"formal bg01 tick0 reader mismatch: domain={domain_value!r}, "
            f"compute={compute_value!r}"
        )
    prepared_tower = {}
    for tick in (260, 261):
        tower = _frame_entities(tick).get("tower_l2_1_v2")
        if tower is None:
            raise ObjectiveRebuildError(f"tick {tick} tower_l2_1_v2 missing")
        communication = tower.get("communication_state")
        if not isinstance(communication, dict):
            raise ObjectiveRebuildError(
                f"tower_l2_1_v2@{tick} communication_state is not an object"
            )
        prepared_tower[str(tick)] = {
            "station_unavailable": communication.get("station_unavailable"),
            "backup_link_active": communication.get("backup_link_active"),
        }
    if (
        prepared_tower["260"]["station_unavailable"] is not False
        or prepared_tower["261"]["station_unavailable"] is not True
    ):
        raise ObjectiveRebuildError(
            f"tower 260/261 station_unavailable boundary mismatch: {prepared_tower}"
        )

    _stage("assemble_formal_staging")
    manifest_text = (ORIGINAL_EPISODE_ROOT / "episode_manifest.json").read_text(
        encoding="utf-8-sig"
    )
    manifest = json.loads(manifest_text)
    if manifest.get("episode_id") != EPISODE_ID:
        raise ObjectiveRebuildError(
            f"original manifest episode_id mismatch: {manifest.get('episode_id')!r}"
        )
    roster_text = (
        ORIGINAL_EPISODE_ROOT / "global_entity_roster.json"
    ).read_text(encoding="utf-8-sig")
    roster = json.loads(roster_text)
    adopted_scene_text = ADOPTED_SCENE_SETUP.read_text(encoding="utf-8-sig")
    authority_scene_path = REPO_ROOT / manifest["source_scene_setup_path"]
    if authority_scene_path.name != "scene_setup.json" or (
        not authority_scene_path.is_file()
    ):
        raise ObjectiveRebuildError(
            f"manifest source_scene_setup_path does not resolve to an existing "
            f"scene_setup.json: {authority_scene_path}"
        )
    authority_scene_text = authority_scene_path.read_text(encoding="utf-8-sig")
    adopted_scene = json.loads(adopted_scene_text)
    authority_scene = json.loads(authority_scene_text)
    staging_files = {
        "episode_manifest.json": manifest_text,
        "global_entity_roster.json": roster_text,
        "trajectories.jsonl": pipeline_jsonl_text(selected_trajectories),
        "truth_frames.jsonl": pipeline_jsonl_text(merged_rows),
        "scene_setup.json": adopted_scene_text,
    }
    adopted_weather_rows = list(_iter_jsonl(ADOPTED_WEATHER))
    declared_weather_count = manifest.get("record_counts", {}).get("weather_meta")
    if declared_weather_count is not None and len(adopted_weather_rows) != int(
        declared_weather_count
    ):
        raise ObjectiveRebuildError(
            f"adopted weather rows {len(adopted_weather_rows)} do not match manifest "
            f"record_counts.weather_meta {declared_weather_count}"
        )
    # P09-M06-L2 class-A staging wiring: the communication producer
    # (Dataset/semantic_simulation/compute_comm.py:1256-1258) requires the
    # weather_meta.dust field, which the adopted v12 weather rows lack on every
    # row. Bind the recorded original-episode dust value verbatim per tick
    # where the adopted row lacks the key; no value is invented and no other
    # row field is added or changed. Fail closed on a tick gap or a missing
    # original authority value.
    original_weather_rows = list(
        _iter_jsonl(ORIGINAL_EPISODE_ROOT / "weather_meta.jsonl")
    )
    dust_by_tick = {}
    for row in original_weather_rows:
        tick = row.get("tick")
        if not isinstance(tick, int) or "dust" not in row:
            raise ObjectiveRebuildError(
                "original weather authority lacks an integer tick or dust: "
                f"{ {k: row.get(k) for k in ('tick', 'dust')} }"
            )
        if tick in dust_by_tick:
            raise ObjectiveRebuildError(f"duplicate original weather tick {tick}")
        dust_by_tick[tick] = row["dust"]
    dust_bound_rows = 0
    for row in adopted_weather_rows:
        tick = row.get("tick")
        if not isinstance(tick, int) or "dust" in row:
            continue
        if tick not in dust_by_tick:
            raise ObjectiveRebuildError(
                f"adopted weather tick {tick} lacks dust and the original "
                "authority has no recorded value for that tick"
            )
        row["dust"] = dust_by_tick[tick]
        dust_bound_rows += 1
    weather_binding = {
        "path": str(ADOPTED_WEATHER),
        "rows": len(adopted_weather_rows),
        "dust_rows_bound_from_original_authority": dust_bound_rows,
        "dust_source": str(ORIGINAL_EPISODE_ROOT / "weather_meta.jsonl"),
        "reason": (
            "P09-M06-L2 class-A staging wiring: compute_comm requires "
            "weather_meta.dust; the adopted weather rows lack the key, so the "
            "recorded original dust value is bound verbatim per tick"
        ),
    }
    staging_files["weather_meta.jsonl"] = pipeline_jsonl_text(adopted_weather_rows)
    copied_optional = {}
    for name in MANIFEST_OPTIONAL_INPUTS:
        if name == "scene_setup.json":
            continue
        source = ORIGINAL_EPISODE_ROOT / name
        if source.is_file():
            staging_files[name] = source.read_text(encoding="utf-8-sig")
            copied_optional[name] = {
                "path": str(source),
                "bytes": source.stat().st_size,
            }
    binding = {
        "manifest_policy": "copied verbatim; no field rewritten",
        "roster_policy": (
            "copied verbatim; "
            f"{len(roster['entities'])} entries; no additions, no declarations"
        ),
        "trajectories_policy": (
            "formal v14 selected world trajectory rows only; no gap synthesis"
        ),
        "copied_optional_inputs": copied_optional,
        "scene_setup": {
            "staged_source": str(ADOPTED_SCENE_SETUP),
            "bytes": ADOPTED_SCENE_SETUP.stat().st_size,
            "authority_scene_path": manifest["source_scene_setup_path"],
            "identical_to_authority_scene": (
                adopted_scene_text == authority_scene_text
            ),
            "entity_ids_equal_to_authority": (
                sorted(e.get("entity_id") for e in adopted_scene.get("entities", []))
                == sorted(e.get("entity_id") for e in authority_scene.get("entities", []))
            ),
            "charging_service_plan_identical_to_authority": (
                adopted_scene.get("simulation_plans", {}).get("charging_service_plan")
                == authority_scene.get("simulation_plans", {}).get(
                    "charging_service_plan"
                )
            ),
        },
        "event_script": {
            "path": str(ADOPTED_EVENT_SCRIPT),
            "consumed": False,
            "reason": (
                "context only; no builder reader resolves event_script.json "
                "from the episode root"
            ),
        },
        "weather": weather_binding,
    }
    for name, text in sorted(staging_files.items()):
        _atomic_write_text(FORMAL_OBJECTIVE_STAGING / name, text)

    _stage("build_artifacts")
    artifacts = build_episode_objective_artifacts(
        FORMAL_OBJECTIVE_STAGING,
        FORMAL_OBJECTIVE_OUTPUT,
        domain_profile_path=DEFAULT_DOMAIN_PROFILE_PATH,
        compute_profile_path=DEFAULT_COMPUTE_PROFILE_PATH,
        contract_profile_path=DEFAULT_CONTRACT_PROFILE_PATH,
        stage_acceptance_profile_path=DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH,
    )
    artifact_report = _describe_artifacts(artifacts)

    _stage("check_outputs")
    pre_persist_mismatches = check_episode_outputs(artifacts)

    _stage("persist_outputs")
    write_episode_outputs(artifacts)
    post_persist_mismatches = check_episode_outputs(artifacts)
    if post_persist_mismatches:
        raise ObjectiveRebuildError(
            f"persisted objective outputs drift from artifacts: "
            f"{post_persist_mismatches[:10]}"
        )
    output_files = sorted(
        p.name for p in FORMAL_OBJECTIVE_OUTPUT.iterdir() if p.is_file()
    )

    _stage("persisted_facts")
    persisted_facts = _persisted_output_facts(FORMAL_OBJECTIVE_OUTPUT)

    _stage("source_unchanged")
    inputs_after = {
        label: (path.stat().st_size, path.stat().st_mtime_ns)
        for label, path in input_paths.items()
    }
    if inputs_after != inputs_before:
        changed = [
            label
            for label in inputs_before
            if inputs_before[label] != inputs_after[label]
        ]
        raise ObjectiveRebuildError(
            f"formal source inputs changed during the objective run: {changed}"
        )

    updates_by_family: dict[str, int] = {}
    updates_by_entity: dict[str, int] = {}
    for update in source_selection["runtime_updates"]:
        for family in update["families"]:
            updates_by_family[family] = updates_by_family.get(family, 0) + 1
        updates_by_entity[update["entity_id"]] = (
            updates_by_entity.get(update["entity_id"], 0) + 1
        )

    payload = {
        "stage": "formal_v14_objective",
        "started_at_utc": start_utc.isoformat(timespec="seconds"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": (datetime.now(timezone.utc) - start_utc).total_seconds(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "episode_id": EPISODE_ID,
        "builder": {
            "module": "Dataset/semantic_truth/objective_pipeline.py",
            "entry": "build_episode_objective_artifacts",
            "persist": "write_episode_outputs",
            "check": "check_episode_outputs",
            "profile_paths": {
                "domain_profile_path": str(DEFAULT_DOMAIN_PROFILE_PATH),
                "compute_profile_path": str(DEFAULT_COMPUTE_PROFILE_PATH),
                "contract_profile_path": str(DEFAULT_CONTRACT_PROFILE_PATH),
                "stage_acceptance_profile_path": str(
                    DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH
                ),
            },
            "strict_input_guard": True,
            "manifest_episode_root": "default (formal staging episode root)",
        },
        "inputs": {label: str(path) for label, path in input_paths.items()},
        "source_selection_counts": {
            "merged_truth_frames": len(merged_rows),
            "selected_trajectory_rows": len(selected_trajectories),
            "formal_ue_entity_numeric_checks": numeric_checks,
            "runtime_authority_audit_rows": len(runtime_authority_audit),
            "runtime_sources_at_rows": len(runtime_sources_at),
            "numeric_target_rows": target_rows,
            "untouched_background_rows": background_rows,
            "entity_rows": entity_rows,
        },
        "runtime_updates": {
            "total": len(source_selection["runtime_updates"]),
            "by_family": dict(sorted(updates_by_family.items())),
            "by_entity": dict(sorted(updates_by_entity.items())),
        },
        "prepared_acceptance_values": {
            "tick0": {
                "uav_l2_1_v2_absent": "uav_l2_1_v2" not in tick0,
                "bg_vehicle_l2_1_v2_02_absent": "bg_vehicle_l2_1_v2_02" not in tick0,
                "bg_vehicle_l2_1_v2_03_absent": "bg_vehicle_l2_1_v2_03" not in tick0,
                "bg01_domain_speed_mps": domain_value,
                "bg01_compute_speed_mps": compute_value,
                "entity_count": len(tick0),
            },
            "tower_l2_1_v2": prepared_tower,
        },
        "input_binding": binding,
        "staging": {
            "episode_root": str(FORMAL_OBJECTIVE_STAGING),
            "files": sorted(staging_files),
        },
        "artifacts": artifact_report,
        "persist": {
            "output_dir": str(FORMAL_OBJECTIVE_OUTPUT),
            "output_file_count": len(output_files),
            "output_files": output_files,
            "pre_persist_mismatch_count": len(pre_persist_mismatches),
            "post_persist_mismatch_count": len(post_persist_mismatches),
        },
        "persisted_facts": persisted_facts,
        "source_inputs_unchanged": {
            "sizes_match_accepted_precheck": True,
            "inputs": {
                label: {"path": str(path), "bytes": inputs_after[label][0]}
                for label, path in input_paths.items()
            },
        },
        "stage_log": stage_log,
        "scope": (
            "P09-M01-B formal selection with the P09-M06-L2 class-A staging "
            "wiring (weather_meta.dust bound verbatim from the original "
            "authority); existing builder run once, strict input guard; "
            "LOG_ONLY, no motion/ns-3/SUMO/UE/GPU and no roster change; "
            "adoption gated on P09-M01-A item A1"
        ),
    }
    return payload


def _write_formal_objective_failure(payload: dict) -> Path:
    FORMAL_OBJECTIVE_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    # P09-M06-L2: the class-A rerun never overwrites the first run's evidence.
    name = (
        "formal_objective_failure_r2.txt"
        if FORMAL_OBJECTIVE_R2
        else "formal_objective_failure.txt"
    )
    path = FORMAL_OBJECTIVE_CHECKPOINT / name
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return path


def _run_formal_v14_objective() -> int:
    """Execute the formal objective branch; evidence stays inside v8."""
    try:
        payload = _formal_v14_objective_body()
    except BaseException as exc:
        failure = {
            "stage": FAILURE_CONTEXT.get("stage"),
            "command": " ".join(sys.argv),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "inputs": FAILURE_CONTEXT.get("inputs"),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        saved = _write_formal_objective_failure(failure)
        traceback.print_exc()
        print(f"formal objective failure evidence saved: {saved}", file=sys.stderr)
        return 1
    # P09-M06-L2: the class-A rerun writes its own receipt, never overwriting
    # formal_objective_receipt.json from the first successful run.
    receipt_path = FORMAL_OBJECTIVE_CHECKPOINT / (
        "formal_objective_receipt_r2.json"
        if FORMAL_OBJECTIVE_R2
        else "formal_objective_receipt.json"
    )
    _atomic_write_text(
        receipt_path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {
                "stage": "formal_v14_objective",
                "receipt": str(receipt_path),
                "objective_output": str(FORMAL_OBJECTIVE_OUTPUT),
                "elapsed_s": payload["elapsed_s"],
                "artifact_count": payload["artifacts"]["artifact_count"],
                "closure_status": payload["artifacts"]["closure_status"],
                "merged_truth_frames": payload["source_selection_counts"][
                    "merged_truth_frames"
                ],
                "formal_ue_entity_numeric_checks": payload["source_selection_counts"][
                    "formal_ue_entity_numeric_checks"
                ],
                "numeric_target_rows": payload["source_selection_counts"][
                    "numeric_target_rows"
                ],
                "untouched_background_rows": payload["source_selection_counts"][
                    "untouched_background_rows"
                ],
                "runtime_updates_by_entity": payload["runtime_updates"]["by_entity"],
                "tick0": payload["prepared_acceptance_values"]["tick0"],
                "tower_l2_1_v2": payload["prepared_acceptance_values"][
                    "tower_l2_1_v2"
                ],
                "persisted_tower_station_unavailable": payload["persisted_facts"][
                    "tower_l2_1_v2_station_unavailable"
                ],
                "persisted_bg01_tick0": payload["persisted_facts"][
                    "bg_vehicle_l2_1_v2_01_tick0"
                ],
                "persisted_uav_token_hit_count": len(
                    payload["persisted_facts"]["uav_l2_1_v2"]["token_hits"]
                ),
                "source_inputs_unchanged": True,
                "post_persist_mismatch_count": payload["persist"][
                    "post_persist_mismatch_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _assemble_staging_files(
    merged_rows: list,
    roster_declarations: list[dict],
    world_trajectory_gap_rows: list[dict],
) -> tuple[dict, dict]:
    manifest_text = (ORIGINAL_EPISODE_ROOT / "episode_manifest.json").read_text(
        encoding="utf-8-sig"
    )
    manifest = json.loads(manifest_text)
    if manifest.get("episode_id") != EPISODE_ID:
        raise ObjectiveRebuildError(
            f"original manifest episode_id mismatch: {manifest.get('episode_id')!r}"
        )
    # The manifest is copied verbatim.  The builder requires source_scene_setup_path
    # and source_event_script_path to name the unique scenario authority directory,
    # so no manifest field is rewritten here.
    authority_files = {}
    for field, filename in (
        ("source_scene_setup_path", "scene_setup.json"),
        ("source_event_script_path", "event_script.json"),
    ):
        declared = manifest.get(field)
        if not isinstance(declared, str) or not declared:
            raise ObjectiveRebuildError(f"manifest lacks {field}")
        path = REPO_ROOT / declared
        if path.name != filename or not path.is_file():
            raise ObjectiveRebuildError(
                f"manifest {field} does not resolve to an existing {filename}: {path}"
            )
        authority_files[filename] = path

    files: dict[str, str] = {}
    copied_inputs = {}
    for name in MANIFEST_BOUND_INPUTS:
        source = ORIGINAL_EPISODE_ROOT / name
        text = source.read_text(encoding="utf-8-sig")
        if name == "global_entity_roster.json":
            roster = json.loads(text)
            declared_ids = {row["entity_id"] for row in roster_declarations}
            existing_ids = {row.get("entity_id") for row in roster["entities"]}
            if declared_ids & existing_ids:
                raise ObjectiveRebuildError(
                    f"declared adopted-only entities already in world roster: "
                    f"{sorted(declared_ids & existing_ids)}"
                )
            roster["entities"].extend(roster_declarations)
            text = json.dumps(
                roster, ensure_ascii=False, allow_nan=False, sort_keys=True
            ) + "\n"
        files[name] = text
        copied_inputs[name] = {
            "path": str(source),
            "bytes": source.stat().st_size,
            **(
                {
                    "declared_adopted_only_entities": [
                        row["entity_id"] for row in roster_declarations
                    ]
                }
                if name == "global_entity_roster.json"
                else {}
            ),
        }
    for name in MANIFEST_OPTIONAL_INPUTS:
        source = ORIGINAL_EPISODE_ROOT / name
        if source.is_file():
            files[name] = source.read_text(encoding="utf-8-sig")
            copied_inputs[name] = {"path": str(source), "bytes": source.stat().st_size}

    world_trajectory_text = files["trajectories.jsonl"]
    files["trajectories.jsonl"] = world_trajectory_text + "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in world_trajectory_gap_rows
    )

    adopted_weather_rows = list(_iter_jsonl(ADOPTED_WEATHER))
    declared_weather_count = manifest.get("record_counts", {}).get("weather_meta")
    if declared_weather_count is not None and len(adopted_weather_rows) != int(
        declared_weather_count
    ):
        raise ObjectiveRebuildError(
            f"adopted weather rows {len(adopted_weather_rows)} do not match manifest "
            f"record_counts.weather_meta {declared_weather_count}"
        )
    files["weather_meta.jsonl"] = _jsonl_text(adopted_weather_rows)

    # The adopted scene is bound as the local scene_setup.json.  Scene-consuming
    # readers take the local file first; manifest-locked geometry keeps the
    # scenario authority scene.  The adopted event script stays a record: no
    # builder reader resolves event_script.json from the episode root.
    adopted_scene_text = ADOPTED_SCENE_SETUP.read_text(encoding="utf-8-sig")
    files["scene_setup.json"] = adopted_scene_text
    adopted_scene = json.loads(adopted_scene_text)
    authority_scene = json.loads(
        authority_files["scene_setup.json"].read_text(encoding="utf-8-sig")
    )
    adopted_entity_ids = sorted(
        e.get("entity_id") for e in adopted_scene.get("entities", [])
    )
    authority_entity_ids = sorted(
        e.get("entity_id") for e in authority_scene.get("entities", [])
    )
    scene_binding = {
        "adopted_scene_setup": {
            "path": str(ADOPTED_SCENE_SETUP),
            "bytes": ADOPTED_SCENE_SETUP.stat().st_size,
        },
        "staged_as": "scene_setup.json at the assembled episode root",
        "local_root_consumers": SCENE_BINDING_CONSUMERS,
        "authority_scene_path": manifest["source_scene_setup_path"],
        "identical_to_authority_scene": (
            adopted_scene_text
            == authority_files["scene_setup.json"].read_text(encoding="utf-8-sig")
        ),
        "entity_ids_equal_to_authority": (
            adopted_entity_ids == authority_entity_ids
        ),
        "charging_service_plan_identical_to_authority": (
            adopted_scene.get("simulation_plans", {}).get("charging_service_plan")
            == authority_scene.get("simulation_plans", {}).get("charging_service_plan")
        ),
    }
    binding = {
        "copied_original_inputs": copied_inputs,
        "adopted_inputs": {
            "trajectories": {
                "path": str(ADOPTED_TRAJECTORIES),
                "bytes": ADOPTED_TRAJECTORIES.stat().st_size,
            },
            "scene_setup": scene_binding,
            "event_script": {
                "path": str(ADOPTED_EVENT_SCRIPT),
                "bytes": ADOPTED_EVENT_SCRIPT.stat().st_size,
                "consumed": False,
                "reason": (
                    "context only; no builder reader resolves event_script.json "
                    "from the episode root, and the manifest field must keep "
                    "naming the scenario authority"
                ),
            },
            "weather": {
                "path": str(ADOPTED_WEATHER),
                "bytes": ADOPTED_WEATHER.stat().st_size,
                "rows": len(adopted_weather_rows),
            },
        },
        "adopted_only_runtime_declarations": {
            "policy": (
                "adopted-runtime entities absent from every roster authority are "
                "declared in the assembled world roster with their recorded "
                "trajectory rows appended to the assembled world trajectory file; "
                "no value outside the adopted rows is invented"
            ),
            "entities": sorted({row["entity_id"] for row in roster_declarations}),
            "roster_entries": len(roster_declarations),
            "trajectory_rows_appended": len(world_trajectory_gap_rows),
            "trajectory_row_policy": (
                "recorded adopted rows appended for every (tick, entity) pair the "
                "world trajectory lacks, adopted-only truth appends included"
            ),
            "reason": (
                "compute_comm requires every uav/vehicle in the truth frames to "
                "match the union roster category, and the geometric computer "
                "consumes positions from the trajectory file"
            ),
        },
        "manifest_updates": {},
        "manifest_policy": "copied verbatim; no field rewritten",
        "unchanged_manifest_bindings": {
            "sumo_traffic.source.frames": manifest["sumo_traffic"]["source"]["frames"],
            "generation.source_episode_dir": manifest["generation"][
                "source_episode_dir"
            ],
            "source_scene_setup_path": manifest["source_scene_setup_path"],
            "source_event_script_path": manifest["source_event_script_path"],
        },
    }
    files["episode_manifest.json"] = manifest_text
    files["truth_frames.jsonl"] = _jsonl_text(merged_rows)
    return files, binding


def _describe_artifacts(artifacts) -> dict:
    names = sorted(artifacts.files)
    highlighted = {
        name: {
            "bytes": len(artifacts.files[name].encode("utf-8")),
            "jsonl_rows": (
                sum(1 for line in artifacts.files[name].splitlines() if line.strip())
                if name.endswith(".jsonl")
                else None
            ),
        }
        for name in names
        if any(token in name for token in ("predicate", "geometry", "geometric", "domain", "communication", "semantic", "closure"))
    }
    return {
        "artifact_count": len(names),
        "artifact_names": names,
        "highlighted_materializations": highlighted,
        "closure_status": artifacts.closure.get("status"),
        "manifest_record_counts": artifacts.manifest.get("record_counts"),
    }


def _write_caller_failure(stage: str, exc: BaseException) -> Path:
    path = OBJECTIVE_OUTPUT_ROOT / "caller_failure.txt"
    lines = [
        f"command: {FAILURE_CONTEXT.get('command')}",
        f"working_directory: {Path.cwd()}",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.split()[0]}",
        f"stage: {stage}",
        f"inputs: {json.dumps(FAILURE_CONTEXT.get('inputs'), ensure_ascii=False)}",
        "traceback:",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ]
    try:
        OBJECTIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as inner:
        print(f"could not save caller failure context: {inner}", file=sys.stderr)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Approved thin objective-pipeline caller for L2-1_v2__seed00: "
            "merge adopted runtime onto full original truth and run the existing "
            "objective builder with its exact persist contract."
        )
    )
    parser.add_argument(
        "--formal-v14-source-precheck",
        dest="formal_v14_source_precheck",
        action="store_true",
        help=(
            "opt-in formal v14 source precheck: prepared formal inputs plus the "
            "recorded runtime authority audit, saved to the precheck checkpoint; "
            "no staging write, builder, or persist"
        ),
    )
    parser.add_argument(
        "--formal-v14-objective",
        dest="formal_v14_objective",
        action="store_true",
        help=(
            "opt-in formal v14 objective build: run the existing objective "
            "builder once over the formal selection accepted by the source "
            "precheck, persist outputs into the formal objective checkpoint; "
            "no motion, ns-3, SUMO, UE, or GPU execution"
        ),
    )
    args = parser.parse_args(argv)

    if args.formal_v14_source_precheck:
        return _run_formal_source_precheck()
    if args.formal_v14_objective:
        return _run_formal_v14_objective()

    FAILURE_CONTEXT["command"] = " ".join(sys.argv)
    FAILURE_CONTEXT["stage"] = "input_checks"
    _require_declared_inputs()
    if OBJECTIVE_OUTPUT.exists():
        raise ObjectiveRebuildError(
            "objective output already exists; clear it manually before a rerun, "
            f"the caller never deletes outputs: {OBJECTIVE_OUTPUT}"
        )
    FAILURE_CONTEXT["inputs"] = {
        "original_episode_root": str(ORIGINAL_EPISODE_ROOT),
        "adopted_trajectories": str(ADOPTED_TRAJECTORIES),
        "adopted_scene_setup": str(ADOPTED_SCENE_SETUP),
        "adopted_event_script": str(ADOPTED_EVENT_SCRIPT),
        "adopted_weather": str(ADOPTED_WEATHER),
    }

    FAILURE_CONTEXT["stage"] = "merge_full_truth"
    merged_rows, merge_stats, adopted_by_tick = _merge_full_truth_frames()
    if os.environ.get('P09_STALE_VIEW_CHECKPOINT_ONLY') == '1':
        from Dataset.semantic_simulation import domain_state, compute_comm
        receipt = _write_stale_sumo_tick0_checkpoint(merged_rows, domain_speed=domain_state._speed, compute_speed=compute_comm._entity_speed)
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
        return 0
    print(
        json.dumps(
            {
                "stage": "merge_full_truth",
                "merged_truth_frames": merge_stats["merged_truth_frames"],
                "totals": merge_stats["totals"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    FAILURE_CONTEXT["stage"] = "assemble_staging"
    world_roster = json.loads(
        (ORIGINAL_EPISODE_ROOT / "global_entity_roster.json").read_text(
            encoding="utf-8-sig"
        )
    )
    world_roster_ids = {
        row.get("entity_id") for row in world_roster.get("entities", [])
    }
    roster_declarations = _declared_adopted_only_entities(
        adopted_by_tick,
        world_roster_ids,
    )

    FAILURE_CONTEXT["stage"] = "adopted_trajectory_gap_rows"
    world_trajectory_gap_rows = _world_trajectory_gap_rows(adopted_by_tick)
    print(
        json.dumps(
            {
                "stage": "adopted_trajectory_gap_rows",
                "gap_rows": len(world_trajectory_gap_rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    FAILURE_CONTEXT["stage"] = "bind_adopted_only_identity"
    identity_by_id = {
        row.get("entity_id"): (row, "world_roster")
        for row in world_roster.get("entities", [])
    }
    for row in roster_declarations:
        identity_by_id.setdefault(
            row["entity_id"], (row, "adopted_runtime_declaration")
        )
    bound_ticks = _bind_adopted_only_identity(merged_rows, identity_by_id)
    print(
        json.dumps(
            {"stage": "bind_adopted_only_identity", "bound_appends": bound_ticks},
            ensure_ascii=False,
        ),
        flush=True,
    )

    staging_files, binding = _assemble_staging_files(
        merged_rows,
        roster_declarations,
        world_trajectory_gap_rows=world_trajectory_gap_rows,
    )
    if STAGING_ROOT.exists():
        raise ObjectiveRebuildError(
            f"staging episode root already exists: {STAGING_ROOT}"
        )
    for name, text in sorted(staging_files.items()):
        _atomic_write_text(STAGING_ROOT / name, text)
    print(
        json.dumps(
            {
                "stage": "assemble_staging",
                "staging_root": str(STAGING_ROOT),
                "staging_files": sorted(staging_files),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    FAILURE_CONTEXT["stage"] = "build_artifacts"
    artifacts = build_episode_objective_artifacts(
        STAGING_ROOT,
        OBJECTIVE_OUTPUT,
        domain_profile_path=DEFAULT_DOMAIN_PROFILE_PATH,
        compute_profile_path=DEFAULT_COMPUTE_PROFILE_PATH,
        contract_profile_path=DEFAULT_CONTRACT_PROFILE_PATH,
        stage_acceptance_profile_path=DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH,
    )
    artifact_report = _describe_artifacts(artifacts)

    FAILURE_CONTEXT["stage"] = "persist_outputs"
    pre_persist_mismatches = check_episode_outputs(artifacts)
    write_episode_outputs(artifacts)
    post_persist_mismatches = check_episode_outputs(artifacts)
    if post_persist_mismatches:
        raise ObjectiveRebuildError(
            f"persisted objective outputs drift from artifacts: {post_persist_mismatches[:10]}"
        )
    output_files = sorted(p.name for p in OBJECTIVE_OUTPUT.iterdir() if p.is_file())

    receipt = {
        "schema": "p09.dimension_repair_v1.objective_rebuild_caller_receipt/1",
        "module": str(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episode_id": EPISODE_ID,
        "builder": {
            "module": "Dataset/semantic_truth/objective_pipeline.py",
            "entry": "build_episode_objective_artifacts",
            "persist": "write_episode_outputs",
            "check": "check_episode_outputs",
            "profile_paths": {
                "domain_profile_path": str(DEFAULT_DOMAIN_PROFILE_PATH),
                "compute_profile_path": str(DEFAULT_COMPUTE_PROFILE_PATH),
                "contract_profile_path": str(DEFAULT_CONTRACT_PROFILE_PATH),
                "stage_acceptance_profile_path": str(
                    DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH
                ),
            },
            "strict_input_guard": True,
            "manifest_episode_root": "default (assembled staging episode root)",
        },
        "inputs": FAILURE_CONTEXT["inputs"],
        "context_inputs_not_consumed": [
            dict(item) for item in CONTEXT_INPUTS_NOT_CONSUMED
        ],
        "input_binding": binding,
        "merge": merge_stats,
        "staging": {
            "episode_root": str(STAGING_ROOT),
            "files": sorted(staging_files),
        },
        "artifacts": artifact_report,
        "persist": {
            "output_dir": str(OBJECTIVE_OUTPUT),
            "output_file_count": len(output_files),
            "output_files": output_files,
            "pre_persist_mismatch_count": len(pre_persist_mismatches),
            "pre_persist_mismatches_sample": pre_persist_mismatches[:5],
            "post_persist_mismatch_count": len(post_persist_mismatches),
            "post_persist_mismatches": post_persist_mismatches,
        },
        "scope": (
            "thin caller only: existing builder + persist + check executed once on the "
            "assembled adopted-input episode root; no new pipeline, guard, or numerical logic"
        ),
    }
    _atomic_write_text(CALLER_RECEIPT_PATH, json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(
        json.dumps(
            {
                "caller_receipt": str(CALLER_RECEIPT_PATH),
                "objective_output": str(OBJECTIVE_OUTPUT),
                "artifact_count": artifact_report["artifact_count"],
                "closure_status": artifact_report["closure_status"],
                "record_counts": artifact_report["manifest_record_counts"],
                "post_persist_mismatches": post_persist_mismatches,
                "merge_totals": merge_stats["totals"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        FAILURE_CONTEXT["traceback"] = traceback.format_exc()
        path = _write_caller_failure(FAILURE_CONTEXT["stage"], exc)
        traceback.print_exc()
        print(f"caller failure context saved: {path}", file=sys.stderr)
        sys.exit(1)
