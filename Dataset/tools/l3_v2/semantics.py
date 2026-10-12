"""Authoritative semantics for one L3 arm window.

The window's event content is computed by the repository's objective pipeline
(`build_episode_objective_artifacts`) from a self-consistent episode root that
this module materializes:

  trajectories.jsonl  the ARM's own engine run rows
  truth_frames.jsonl  the capture frames with every engine-produced entity pose,
                      state, activity and typed runtime family replaced by the
                      ARM's own engine state; frames keep their producer-only
                      fields (sumo incidents, segments, traffic lights)
  global_entity_roster.json
                      the engine's own roster unioned with any capture roster
                      entity the engine does not model, so every trajectory key
                      resolves and no entity class is invented

Semantics is never re-implemented here: predicates, transitions, occurrences and
outcomes all come from the pipeline.
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from Dataset.tools.l3_v2.common import (
    CAPTURE, ROOT, RUNTIME_STATE_FAMILIES, load, rows, write, write_rows,
)
from Dataset.tools.batch_generate import EpisodeStateEngine, preserved_fields_from


def _capture_dir(episode_id: str) -> Path:
    path = CAPTURE / episode_id
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _entity_views(frame: Mapping[str, Any]) -> dict[str, dict]:
    out = {}
    for entity in frame.get("entities") or []:
        if isinstance(entity, Mapping) and isinstance(entity.get("entity_id"), str):
            out[entity["entity_id"]] = entity
    return out


def _new_entity_view(source_view: Mapping[str, Any] | None, row: Mapping[str, Any],
                     roster_entry: Mapping[str, Any]) -> dict:
    """One truth-frame entity built from engine state; identity comes from the roster.

    A scene-authored entity is not SUMO traffic.  If the capture frame carried a
    `sumo_vehicle` block for it, that block is producer residue for an entity the
    engine now owns, and keeping it makes the objective pipeline treat the entity
    as render-only SUMO traffic contradicted by the episode's SUMO authority.
    """
    view = copy.deepcopy(dict(source_view)) if isinstance(source_view, Mapping) else {}
    view.pop("sumo_vehicle", None)
    view["entity_id"] = row["entity_id"]
    view.setdefault("label_class", roster_entry.get("label_class"))
    view.setdefault("logical_asset_id", roster_entry.get("asset_id"))
    view.setdefault("entity_category", roster_entry.get("entity_category") or roster_entry.get("category"))
    view.setdefault("entity_kind", roster_entry.get("entity_kind"))
    view.setdefault("entity_type", roster_entry.get("entity_type"))
    pose = view.setdefault("truth_pose", {})
    pose["position_enu_m"] = [float(v) for v in row["pos_enu"]]
    pose["velocity_enu_mps"] = [float(v) for v in row["vel_mps"]]
    if row.get("yaw_deg") is not None:
        pose.setdefault("rotation_deg", {})["yaw_deg"] = float(row["yaw_deg"])
    # The strict objective input accepts typed runtime families only.  Authored
    # `state`, `annotations.activity_type` and their siblings are forbidden there,
    # so the engine's own discrete state is carried in the typed families and the
    # forbidden keys are removed rather than copied.
    for family in RUNTIME_STATE_FAMILIES:
        if family in row and row[family] is not None:
            view[family] = copy.deepcopy(row[family])
    for key in ("state", "activity_state", "activity_type", "posture",
                "semantic_role", "task_id", "active_event_ids", "active_event_id",
                "active_event_label", "activity_label", "dynamic_label"):
        view.pop(key, None)
    annotations = view.get("annotations")
    if isinstance(annotations, Mapping):
        annotations = copy.deepcopy(dict(annotations))
        for key in ("activity_type", "state_facets", "posture", "social_state"):
            annotations.pop(key, None)
        if annotations:
            view["annotations"] = annotations
        else:
            view.pop("annotations", None)
    return view


def _carry_forward_view(source_view: Mapping[str, Any], *, drop_sumo_identity: bool = False) -> dict:
    """A frame entity the engine does not model: keep typed runtime state only.

    `drop_sumo_identity` is set for entities this layer's scene declares.  A scene
    entity is not SUMO traffic, and the strict SUMO authority removes any render
    vehicle whose SUMO id the authoritative tick does not contain; keeping the
    producer's `sumo_vehicle` block would fail that check for every tick before
    the realized SUMO identity first appears.
    """
    view = copy.deepcopy(dict(source_view))
    if drop_sumo_identity:
        view.pop("sumo_vehicle", None)
    for key in ("state", "activity_state", "activity_type", "posture",
                "semantic_role", "task_id", "active_event_ids", "active_event_id",
                "active_event_label", "activity_label", "dynamic_label"):
        view.pop(key, None)
    annotations = view.get("annotations")
    if isinstance(annotations, Mapping):
        annotations = copy.deepcopy(dict(annotations))
        for key in ("activity_type", "state_facets", "posture", "social_state"):
            annotations.pop(key, None)
        if annotations:
            view["annotations"] = annotations
        else:
            view.pop("annotations", None)
    return view


def stage_arm_root(episode_id: str, engine: EpisodeStateEngine, trajectories: list[dict],
                   roster: Mapping[str, Mapping[str, Any]], target: Path,
                   *, sumo_frames: Path | None = None) -> Path:
    """Materialize the ARM's own self-consistent episode root."""
    capture = _capture_dir(episode_id)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("episode_manifest.json", "weather_meta.jsonl",
                 "scene_occupancy_manifest.json", "scenario_package.json",
                 "render_host_config.json"):
        source = capture / name
        if source.is_file():
            shutil.copy2(source, target / name)
    availability = ROOT / f"aw_data/objective_semantic_truth/{episode_id}/l0_predicate_source_availability.json"
    if not availability.is_file():
        raise FileNotFoundError(availability)
    shutil.copy2(availability, target / "l0_predicate_source_availability.json")

    manifest = load(target / "episode_manifest.json")
    manifest["generation"]["source_episode_dir"] = str(target.relative_to(ROOT))
    manifest["duration_ticks"] = max(int(r["tick"]) for r in trajectories)
    if sumo_frames is not None:
        manifest["sumo_traffic"]["source"]["frames"] = str(sumo_frames.relative_to(ROOT))
    write(target / "episode_manifest.json", manifest)
    write_rows(target / "weather_meta.jsonl", engine.weather_rows)
    # Roster: engine entries are authoritative for the entities the engine models;
    # capture-only entries are preserved because the capture authority binds them
    # into events (pads, facilities, corridors, background pedestrians).
    merged: dict[str, dict] = {}
    for entity_id, entry in roster.items():
        # Roster entries are built the way the engine builds its own roster: the
        # declared identity plus the preserved author fields.  Copying the live
        # engine entity would leak runtime keys (`state`, `pos`, `schedules`) into
        # the strict objective input, which rejects authored state by contract.
        entity = dict(engine.entities.get(entity_id, entry))
        merged[entity_id] = {
            "entity_id": entity_id,
            "label_class": entity.get("label_class"),
            "asset_id": entity.get("asset_id"),
            "initial_yaw_deg": round(float(entity.get("initial_yaw_deg") or 0.0), 6),
            **preserved_fields_from(entity, context=f"l3_arm_roster[{entity_id}]"),
        }
        for key, value in list(merged[entity_id].items()):
            if value is None or value == "":
                merged[entity_id].pop(key)
        # The semantic producers resolve a static entity's position from
        # `initial_position_enu_m`.
        if isinstance(entity.get("pos"), list):
            merged[entity_id]["initial_position_enu_m"] = [float(v) for v in entity["pos"]]
        if str(entity.get("spawn_policy") or "") == "event_script_only":
            merged[entity_id]["activation_tick"] = int(entity.get("active_from") or 0)
        if entity.get("inactive_from") is not None:
            merged[entity_id]["deactivation_tick"] = int(entity["inactive_from"])
    capture_roster = load(capture / "global_entity_roster.json")["entities"]
    capture_roster_by_id = {e["entity_id"]: e for e in capture_roster}
    for entry in capture_roster:
        merged.setdefault(entry["entity_id"], copy.deepcopy(dict(entry)))
    # The semantic producers require `entity_category`; the engine roster carries
    # the same value under `category`.  Fill it from the capture authority when the
    # two disagree only by field name, never by value.
    for entity_id, entry in merged.items():
        if entry.get("entity_category") in (None, ""):
            if entry.get("category"):
                entry["entity_category"] = entry["category"]
            elif entry.get("contract_facility") is not None:
                entry["entity_category"] = "facility"
            else:
                raise ValueError(f"{entity_id}: roster entry has no entity_category or category")
    write(target / "global_entity_roster.json", {"entities": [merged[k] for k in sorted(merged)]})

    # Trajectories: the engine's own rows for the entities in scope.
    write_rows(target / "trajectories.jsonl", trajectories)

    # Truth frames: engine state replaces producer state per entity per tick, for
    # the scene-authored entities the engine actually owns.  SUMO background traffic
    # keeps the episode's authoritative SUMO stream: its identity (`sumo_vehicle_id`)
    # and pose come from the SUMO authority, and the objective pipeline removes any
    # render-only SUMO vehicle that the authoritative tick does not contain.  A
    # scene-authored entity that the roster registers as SUMO background traffic is
    # therefore NOT replayed from engine state.
    # An entity authored by this layer's scene is owned by the engine.  The capture
    # roster may label it SUMO background traffic (the historical producer let its
    # realized SUMO identity win), but the scene declaration is the authority for
    # what this layer re-runs.  Only entities the scene does NOT declare keep the
    # capture's SUMO identity untouched.
    scene_declared_ids = {
        str(entity.get("entity_id"))
        for entity in load(ROOT / f"Dataset/scenarios/L3_dynamic_constraints/interaction/{episode_id.split('__')[0]}/scene_setup.json").get("entities", [])
        if isinstance(entity, Mapping) and entity.get("entity_id")
    }
    background_sumo_ids = {
        entity_id for entity_id in capture_roster_by_id
        if isinstance(capture_roster_by_id[entity_id].get("background_vehicle"), Mapping)
        and capture_roster_by_id[entity_id]["background_vehicle"].get("policy") == "sumo_truth_frame_background_v1"
        and entity_id not in scene_declared_ids
    }
    by_tick: dict[int, dict[str, dict]] = {}
    for row in trajectories:
        if str(row["entity_id"]) in background_sumo_ids:
            continue
        by_tick.setdefault(int(row["tick"]), {})[str(row["entity_id"])] = row
    overlay = None
    if sumo_frames is not None:
        from Dataset.tools.l3_v2.sumo_branch import FrameOverlay
        overlay = FrameOverlay(episode_id, sumo_frames, scene_declared_ids)
    with (capture / "truth_frames.jsonl").open(encoding="utf-8-sig") as source, \
            (target / "truth_frames.jsonl").open("w", encoding="utf-8", newline="\n") as out, \
            (target / "render_background_frames.jsonl").open("w", encoding="utf-8", newline="\n") as render_out:
        for line in source:
            if not line.strip():
                continue
            frame = json.loads(line)
            if overlay is not None:
                frame = overlay.apply(frame)
            tick = int(frame["tick"])
            if tick > manifest["duration_ticks"]:
                break
            render_out.write(json.dumps(frame, ensure_ascii=False, sort_keys=True) + "\n")
            engine_rows = by_tick.get(tick, {})
            existing = _entity_views(frame)
            views = []
            for entity_id, source_view in existing.items():
                row = engine_rows.get(entity_id)
                # Every frame entity reaches the strict objective input, so every
                # entity has its forbidden authored fields removed, whether or not
                # the engine produces a row for it.
                if row is None:
                    if entity_id in roster:
                        continue
                    views.append(_carry_forward_view(
                        source_view, drop_sumo_identity=entity_id in scene_declared_ids))
                    continue
                views.append(_new_entity_view(source_view, row, merged.get(entity_id, {})))
            for entity_id, row in sorted(engine_rows.items()):
                if entity_id in existing:
                    continue
                views.append(_new_entity_view(None, row, merged.get(entity_id, {})))
            frame["entities"] = views
            out.write(json.dumps(frame, ensure_ascii=False, sort_keys=True) + "\n")
    return target


EVENT_ASSETS = ("event_occurrences", "event_outcomes", "predicate_transitions",
                "predicate_truth", "predicate_truth_ticks")


def measured_semantic_rows(root: Path, workdir: Path) -> dict:
    """Reuse the objective state's producers and event engine without release graphs."""
    from Dataset.semantic_truth import objective_pipeline as p
    p._assert_no_forbidden_episode_inputs(root)
    with p._objective_strict_episode_root(root) as source:
        profile = p.load_domain_state_profile(p.DEFAULT_DOMAIN_PROFILE_PATH)
        stage_contract = load(p.DEFAULT_STAGE_ACCEPTANCE_PROFILE_PATH)
        episode_id = root.name
        epi_id, _ = p.parse_episode_identity(episode_id)
        p._assert_authoritative_ticks(source / "truth_frames.jsonl", profile)
        context_id, contexts, _, context_digest, _ = p.build_tick_contexts(source, p._tick_context_profile(profile))
        if context_id != episode_id:
            raise ValueError("Semantic context episode identity differs")
        compute, communication, compute_predicates = p.build_objective_world_inputs(source, p.DEFAULT_COMPUTE_PROFILE_PATH)
        domain = p.build_domain_state_artifacts(source, workdir / "_domain_state_in_memory", p.DEFAULT_DOMAIN_PROFILE_PATH, communication_rows=communication)
        domain_rows = [dict(r) for r in domain.observations]
        controls = p.build_control_response_state_rows(source, communication_rows=communication, domain_rows=domain_rows)
        domain_rows.extend(dict(r) for r in controls)
        observable = p.build_observable_state_rows(source)
        domain_rows.extend(dict(r) for r in observable)
        geometry = p.GeometricStateComputer(source).compute()
        windows = p.PlanWindowComputer(source, geometry).compute()
        domain_rows.extend(dict(r) for r in geometry.rows)
        domain_rows.extend(dict(r) for r in windows)
        predicate_state = sorted(
            [dict(r) for r in domain_rows if str(r.get("observation_family", "")).startswith("predicate_contract_")],
            key=lambda r: (int(r["tick"]), str(r["observation_family"]), str(r["subject_id"])))
        utm = p.build_utm_episode_artifacts(source, workdir / "_utm_in_memory")
        utm_records = {name: p._jsonl_rows_from_text(content, name) for name, content in utm.files.items() if name.endswith(".jsonl")}
        world = p.evaluate_world_truth(source, source_availability=p.build_l0_source_availability(source),
            domain_rows=domain_rows, compute_rows=compute,
            communication_rows=communication, compute_predicate_rows=compute_predicates, utm_records=utm_records)
        projection_ids = sorted(set(p._stage_projection_predicate_ids(stage_contract, epi_id)) | set(p._roster_contract_predicate_ids(source)))
        result = p.run_minimal_semantic_engine(contexts, domain_rows, communication, world.base_graph, world.deltas,
            input_digest=p.digest_object({"context_input_digest": context_digest,
                "domain_input_digest": domain.input_digest,
                "communication_row_count": len(communication), "control_response_row_count": len(controls),
                "observable_state_row_count": len(observable),
                "predicate_state_input_digest": p.digest_object(predicate_state), "predicate_state_row_count": len(predicate_state)}),
            stage_projection_predicate_ids=projection_ids, roster_by_id=p._entity_roster_by_id(source))
        selected = p._select_durable_output_rows(list(result.semantic_state), list(result.predicate_truth),
            list(result.transitions), list(result.continuity_breaks), list(result.occurrences), list(result.outcomes))
        return {"predicate_truth": list(selected["predicate_truth"]),
                "predicate_transitions": list(selected["transitions"]),
                "event_occurrences": list(selected["event_occurrences"]),
                "event_outcomes": list(selected["event_outcomes"])}


def evaluate(episode_id: str, engine: EpisodeStateEngine, trajectories: list[dict],
             roster: Mapping[str, Mapping[str, Any]], *, workdir: Path,
             sumo_frames: Path | None = None) -> dict:
    """Compute the same measured predicates/events without unused release artifacts."""
    root = stage_arm_root(episode_id, engine, trajectories, roster, workdir / episode_id,
                          sumo_frames=sumo_frames)
    result = measured_semantic_rows(root, workdir / "objective")
    for name, values in result.items():
        write_rows(workdir / "objective" / (name + ".jsonl"), values)
    result["predicate_truth_ticks"] = [
        {"tick": r["tick"], "predicate_id": r["predicate_id"], "bindings": r["bindings"],
         "value": r["value"], "truth_id": r["truth_id"], "sampling": "authoritative_capture_grid"}
        for r in result["predicate_truth"]]
    result["materialized_source"] = str(root.relative_to(ROOT))
    write(workdir / "computed.json", result)
    return result
