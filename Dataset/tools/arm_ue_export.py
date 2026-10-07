"""Export measured ARM windows to the existing UE render-host input contract.

Call write_engine_inputs before discarding the real engine/interpreter, then
export_arm after writing manifest.json. No expected predicates or event labels
are inputs. All paths in the resulting package are repository relative.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import batch_generate as bg
import convert_to_render_ready as cv


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def bounds(window: dict) -> tuple[int, int]:
    start, end = window["start_tick"], window["end_tick"]
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError(f"Invalid closed ARM window: {window}")
    return start, end


def event_tick(row: dict) -> int:
    return bg.event_trace_tick(row)


def project_event_display(event_log: list[dict], realizations: list[dict],
                          script: dict, action_audit: list[dict]) -> tuple[list, list, list]:
    """Project explicit action omissions into event titles and visible actors.

    Both omission statuses are current executor contracts. The complete source
    action audit and per-action realization statuses remain execution evidence.
    """
    omitted = {r["action"]["action_id"] for r in action_audit
               if r["result"]["status"] in
               ("omitted_by_explicit_intervention", "omitted_by_configuration")}
    if not omitted:
        return event_log, realizations, []
    dispatched = {r["action"]["action_id"] for r in action_audit
                  if r["result"]["status"] == "ok"}
    definitions = {e["log_event"]["topic"]: e for e in script["events"] if "log_event" in e}
    realized = {r["topic"]: copy.deepcopy(r) for r in realizations}
    labels = {"move_entity": "移动指令", "set_runtime_state": "运行状态更新",
              "set_visual_state": "可视状态设置", "set_weather": "天气更新",
              "capture_screenshot": "截图请求", "spawn_entity": "实体生成",
              "remove_entity": "实体移除", "set_pedestrian_activity": "行人活动更新"}
    kept, results, dropped = [], [], []
    for original in event_log:
        row = copy.deepcopy(original)
        topic = row["source_topic"]
        event = realized[topic]
        definition = definitions[topic]
        authored = [a["action_id"] for a in definition["actions"]]
        excluded = [a for a in authored if a in omitted]
        if authored and len(excluded) == len(authored):
            dropped.append({"dropped_event_id": topic, "activated_tick": event_tick(row),
                            "authored_action_ids": authored,
                            "reason": "all authored actions explicitly omitted"})
            continue
        if excluded:
            executed = [a for a in event["action_realizations"] if a["action_id"] in dispatched]
            targets = list(dict.fromkeys(a["entity_id"] for a in executed if a["entity_id"]))
            descriptions = list(dict.fromkeys(
                (a["entity_id"] + "：" if a["entity_id"] else "") + labels[a["action_type"]]
                for a in executed))
            title = "部分计划动作已派发：" + "；".join(descriptions)
            # The unmodified event_script retains the complete source plan.
            row["payload"]["title"] = title
            if "title" in row:
                row["title"] = title
            row["target_ids"] = targets
            for container in (row["payload"], row["metadata"]):
                if "target_ids" in container:
                    container["target_ids"] = targets
            row["scope"]["entities"] = targets
            row["scope"]["target_id"] = targets[0] if targets else ""
            if row["agent_id"] and row["agent_id"] not in targets:
                row["agent_id"] = ""
            event["title"] = title
            event["target_ids"] = targets
        event["sequence_no"] = len(results) + 1
        kept.append(row)
        results.append(event)
    return kept, results, dropped


def write_engine_inputs(arm_dir: Path, engine, interpreter, window: dict, *, script: dict) -> dict:
    """Persist all authored entities and measured weather, dispatch and events.

    window requires integer start_tick/end_tick (inclusive); extra fork fields
    are retained. script must be the actual interpreter script. Injected actions
    are retained in ue_executed_actions, even when absent from that script.
    This function does not invoke or fabricate engine action dispatch.
    """
    arm_dir = Path(arm_dir)
    start, end = bounds(window)
    raw = arm_dir / "raw"
    trajectories = [copy.deepcopy(r) for r in engine.trajectory_rows if start <= int(r["tick"]) <= end]
    weather = [copy.deepcopy(r) for r in engine.weather_rows if start <= int(r["tick"]) <= end]
    if [int(r["tick"]) for r in weather] != list(range(start, end + 1)):
        raise ValueError("Engine weather does not cover every window tick exactly once")
    roster = {}
    for eid, entity in sorted(engine.entities.items()):
        entry = {"entity_id": eid, "label_class": entity["label_class"],
                 "asset_id": entity["asset_id"], "initial_yaw_deg": entity["initial_yaw_deg"],
                 **bg.preserved_fields_from(entity)}
        # Runtime families must come from each tick, never the engine's final state.
        for field in cv.RUNTIME_STATE_FIELDS:
            entry.pop(field, None)
        entry["activation_tick"] = int(entity["active_from"])
        entry["deactivation_tick"] = entity.get("inactive_from")
        entry["spawned"] = bool(entity["spawned"])
        roster[eid] = entry
    unknown = sorted({r["entity_id"] for r in trajectories} - roster.keys())
    if unknown:
        raise ValueError(f"Trajectory entities missing from engine roster: {unknown}")
    log = copy.deepcopy(interpreter.get_event_log())
    bg.enrich_event_log_validation_fields(log, script, engine.scenario_id)
    actions_log = arm_dir / "raw" / "actions.jsonl"
    action_audit = list(rows(actions_log)) if actions_log.is_file() else []
    log = [r for r in log if start <= event_tick(r) <= end]
    # Restrict evidence to observations available by the window end. Earlier
    # samples support actions dispatched within the window, but future samples
    # cannot make a truncated action appear complete.
    measured = [r for r in engine.trajectory_rows if int(r["tick"]) <= end]
    dispatch = [copy.deepcopy(r) for r in engine.executed_actions if int(r["tick"]) <= end]
    realization = bg.build_event_realizations(
        scenario_id=engine.scenario_id, script=script, event_log=log,
        executed_actions=dispatch, trajectory_rows=measured, params=engine.params)
    by_action = {r["action_id"]: r for r in dispatch if r["action_id"]}
    for event in realization:
        for action in event["action_realizations"]:
            executed = by_action.get(action["action_id"]) if action["action_id"] else next(
                (r for r in dispatch if r["type"] == action["action_type"]
                 and r["entity_id"] == action["entity_id"] and r["tick"] == event["dispatch_tick"]), None)
            if executed is None:
                for key in ("terminal_tick", "first_motion_tick", "first_capture_motion_tick"):
                    action.pop(key, None)
                action.update({"status": "not_dispatched", "result_tick": None, "evidence_tick": None})
            elif action.get("result_tick") is not None and int(action["result_tick"]) > end:
                action["status"] = "pending_at_window_end"
                action["result_tick"] = None
                action["evidence_tick"] = None
            elif action.get("terminal_required") and action.get("terminal_tick") is None:
                action["status"] = "terminal_not_observed_in_window"
                action["result_tick"] = None
                action["evidence_tick"] = None
        for action in event["action_realizations"]:
            if action.get("evidence_tick") is not None and int(action["evidence_tick"]) > end:
                action["evidence_tick"] = None
        event["result_observed"] = bool(event["action_realizations"]) and all(
            a.get("result_tick") is not None and a["status"] == "ok" for a in event["action_realizations"])
        event["window_observation"] = {
            "start_tick": start, "end_tick": end,
            "not_dispatched_count": sum(a["status"] == "not_dispatched" for a in event["action_realizations"])}
        for key in ("result_tick", "evidence_tick"):
            ticks = [int(a[key]) for a in event["action_realizations"] if a.get(key) is not None and start <= int(a[key]) <= end]
            event[key] = max(ticks) if ticks else None
        event["source_truth_snapshots_by_tick"] = {
            k: v for k, v in event["source_truth_snapshots_by_tick"].items() if start <= int(k) <= end}
        event["before_tick"] = max(start, event["before_tick"])
        event["after_tick"] = min(end, event["after_tick"])
    log, realization, dropped = project_event_display(log, realization, script, action_audit)
    if dropped:
        write_rows(raw / "ue_dropped_event_trace.jsonl", dropped)
    write_rows(raw / "ue_trajectories.jsonl", trajectories)
    write_rows(raw / "ue_weather.jsonl", weather)
    write_json(raw / "ue_roster.json", {"entities": list(roster.values())})
    write_rows(raw / "ue_event_trace.jsonl", log)
    write_rows(raw / "ue_event_realization.jsonl", realization)
    write_rows(raw / "ue_executed_actions.jsonl", (r for r in dispatch if start <= int(r["tick"]) <= end))
    report = {"window": window, "scenario_id": engine.scenario_id, "entity_count": len(roster),
              "trajectory_rows": len(trajectories), "weather_rows": len(weather), "event_rows": len(log)}
    write_json(raw / "ue_input_manifest.json", report)
    return report


def _entity(row: dict, entry: dict, frame: dict, contract: dict) -> dict:
    entry = {**entry, "asset_id": row["asset_id"], "label_class": row["label_class"]}
    profile = cv.profile_for_entity(entry, row)
    category = profile["entity_category"]
    position, velocity = row["pos_enu"], row["vel_mps"]
    if len(position) != 3 or len(velocity) != 3 or not all(math.isfinite(float(x)) for x in position + velocity):
        raise ValueError(f"Invalid measured pose: {row}")
    activity = cv.activity_for_sample(entity_id=entry["entity_id"], category=category,
        state=str(row["state"]), row_activity_type=str(row.get("activity_type") or ""),
        velocity_enu_mps=velocity, semantic_idle_when_stationary=cv.is_background_ground_flow_actor(entry))
    entity = {"entity_id": entry["entity_id"], "entity_category": category,
              "entity_kind": profile["entity_kind"], "entity_type": profile["entity_kind"],
              "label_class": entry["label_class"], "site_id": frame["active_site_id"],
              "roi_id": frame["active_roi_id"], "proxy_template_id": profile["proxy_template_id"],
              "logical_asset_id": cv.logical_asset_for(entry, profile),
              "tags": entry.get("tags", []), "truth_pose": cv.truth_pose(position, row["yaw_deg"], velocity),
              "render_presence": cv.render_presence(frame["active_roi_id"]),
              "annotations": cv.build_annotations(activity, row, category),
              "state_revision": int(row["tick"]) + 1, "visual_revision": 1, "state": row["state"],
              "runtime_visibility": cv.runtime_visibility_payload(position, contract)}
    entity.update(cv.preserved_fields_from(row, entry))
    if category == "uav":
        polygon = contract["capture_boundary_polygon_enu_m"]
        eligible = cv.point_in_capture_roi_xy(position, polygon)
        entity["uav_visibility"] = {
            "roi_capture_distance_m": round(cv.distance_to_polygon_xy(position, polygon), 6),
            "roi_capture_eligible": eligible, "selected_for_capture_truth": eligible,
            "capture_roi_policy": cv.UAV_CAMERA_CAPTURE_ROI_POLICY,
            "camera_call_policy": "do_not_call_camera_when_roi_capture_eligible_is_false"}
    return entity



def bind_label_frames(labels: list[dict], frames_by_tick: dict[int, dict]) -> list[dict]:
    """Bind result-frame and dispatch-frame references to this branch's frames."""
    bound = copy.deepcopy(labels)
    for label in bound:
        result_frame = frames_by_tick[int(label["tick"])]
        dispatch_frame = frames_by_tick[int(label["dispatch_tick"])]
        references = dict(label.get("source_frame_references", {}))
        for field, frame in (("frame_id", result_frame), ("source_frame_id", dispatch_frame)):
            if field in label and label[field] != frame["frame_id"]:
                if field not in references:
                    references[field] = label[field]
            label[field] = frame["frame_id"]
        if references:
            label["source_frame_references"] = references
        label["episode_id"] = result_frame["episode_id"]
    return bound


def refresh_arm_label_references(arm_dir: Path) -> dict:
    """Upgrade only dynamic_labels.jsonl and validation.json in an existing package.

    Stream existing truth frames to obtain their actual IDs, preserving all large
    files and all raw engine evidence. The prior successful validation must match
    the window; only the new label-reference check is rerun here. A later full
    validate_arm call returns the same report structure.
    """
    from Dataset.tools.validate_arm_ue import validate_label_frame_references
    arm_dir = Path(arm_dir).resolve()
    out = arm_dir / "ue"
    start, end = bounds(read_json(arm_dir / "manifest.json")["window"])
    validation = read_json(out / "validation.json")
    if validation["ok"] is not True or validation["window"] != {"start_tick": start, "end_tick": end}:
        raise ValueError("Label upgrade requires a successful matching window validation")
    frames = {}
    for frame in rows(out / "truth_frames.jsonl"):
        tick = int(frame["tick"])
        if tick in frames:
            raise ValueError(f"Duplicate truth frame tick: {tick}")
        frames[tick] = {"tick": tick, "frame_id": frame["frame_id"], "episode_id": frame["episode_id"]}
    if list(frames) != list(range(start, end + 1)) or len(frames) != validation["frame_count"]:
        raise ValueError("Label upgrade truth frame coverage differs from validated window")
    labels = bind_label_frames(list(rows(out / "dynamic_labels.jsonl")), frames)
    validate_label_frame_references(labels, frames)
    write_rows(out / "dynamic_labels.jsonl", labels)
    if "dynamic_label_frame_references" not in validation["checks"]:
        validation["checks"].append("dynamic_label_frame_references")
    write_json(out / "validation.json", validation)
    return validation


def event_display_labels(event_log: list[dict], realizations: list[dict],
                         episode_id: str, scenario_id: str, start: int, end: int,
                         frames_by_tick: dict[int, dict]) -> list[dict]:
    labels = cv.build_dynamic_labels(event_log, episode_id, scenario_id=scenario_id,
                                     event_realization_rows=realizations)
    labels = [r for r in labels if start <= int(r["tick"]) <= end]
    realization_by_id = cv.indexed_realizations(realizations, scenario_id)
    for label in labels:
        event = realization_by_id[cv._event_id_from_trace(label, scenario_id)]
        if event is not None:
            label["action_statuses"] = [{"action_id": a["action_id"], "status": a["status"]}
                                        for a in event["action_realizations"]]
            label["observation_basis"] = "interpreter_dispatch_trace; action result is separate"
            label["result_observed"] = event["result_observed"]
    return bind_label_frames(labels, frames_by_tick)


def refresh_arm_event_display(arm_dir: Path) -> dict:
    """Rebuild derived event display without rerunning physics or UE validation.

    Existing raw action/dispatch evidence, objective truth and all trajectory
    files are untouched. The source script remains the complete planned context.
    """
    arm_dir = Path(arm_dir)
    raw, out = arm_dir / "raw", arm_dir / "ue"
    log = list(rows(raw / "ue_event_trace.jsonl"))
    realization = list(rows(raw / "ue_event_realization.jsonl"))
    projected, results, dropped = project_event_display(
        log, realization, read_json(out / "event_script.json"), list(rows(raw / "actions.jsonl")))
    if projected == log and results == realization:
        return {"changed": False, "dropped": 0, "partial": 0}
    start, end = bounds(read_json(arm_dir / "manifest.json")["window"])
    input_meta = read_json(raw / "ue_input_manifest.json")
    episode = read_json(out / "episode_manifest.json")
    frames = {int(f["tick"]): {"tick": f["tick"], "frame_id": f["frame_id"], "episode_id": f["episode_id"]}
              for f in rows(out / "truth_frames.jsonl")}
    labels = event_display_labels(projected, results, episode["episode_id"], input_meta["scenario_id"],
                                  start, end, frames)
    for directory, prefix in ((raw, "ue_"), (out, "")):
        write_rows(directory / (prefix + "event_trace.jsonl"), projected)
        write_rows(directory / (prefix + "event_realization.jsonl"), results)
    write_rows(out / "dynamic_labels.jsonl", labels)
    if dropped:
        dropped_path = raw / "ue_dropped_event_trace.jsonl"
        previous = list(rows(dropped_path)) if dropped_path.exists() else []
        write_rows(dropped_path, previous + dropped)
    input_meta["event_rows"] = len(projected)
    write_json(raw / "ue_input_manifest.json", input_meta)
    episode.update({"n_events": len(projected), "n_event_realizations": len(results)})
    write_json(out / "episode_manifest.json", episode)
    plan = read_json(out / "scenario_plan.json")
    for summary in (plan["compiled_plan_summary"], plan["scenario_plan"]["summary"]):
        summary.update({"event_count": len(projected), "event_realization_count": len(results)})
    write_json(out / "scenario_plan.json", plan)
    return {"changed": True, "dropped": len(dropped),
            "partial": sum(r["payload"]["title"].startswith("部分计划动作已派发：") for r in projected)}


def export_arm(arm_dir: Path, canonical_dir: Path, scene_path: Path, script_path: Path,
               *, frame_source: Path | None = None) -> dict:
    """Write arm_dir/ue from measured raw/ue_* and manifest.window.

    canonical_dir supplies static scene geometry and real background frames.
    frame_source explicitly replaces the background JSONL (e.g. rerun SUMO);
    its frame metadata is preserved. Every engine roster entity is replaced at
    every tick, including removal when absent. No interpolation is performed.
    """
    source_reference = (Path(frame_source).absolute() if frame_source is not None
                        else Path(canonical_dir).absolute() / "truth_frames.jsonl")
    arm_dir, canonical_dir = Path(arm_dir).resolve(), Path(canonical_dir).resolve()
    scene_path, script_path = Path(scene_path).resolve(), Path(script_path).resolve()
    manifest = read_json(arm_dir / "manifest.json")
    start, end = bounds(manifest["window"])
    raw, out = arm_dir / "raw", arm_dir / "ue"
    input_meta = read_json(raw / "ue_input_manifest.json")
    if bounds(input_meta["window"]) != (start, end):
        raise ValueError("Manifest and engine input window disagree")
    roster = {r["entity_id"]: r for r in read_json(raw / "ue_roster.json")["entities"]}
    samples = defaultdict(dict)
    for row in rows(raw / "ue_trajectories.jsonl"):
        tick, eid = int(row["tick"]), row["entity_id"]
        if not start <= tick <= end or eid not in roster or eid in samples[tick]:
            raise ValueError(f"Invalid or duplicate engine row {tick}/{eid}")
        samples[tick][eid] = row
    weather = list(rows(raw / "ue_weather.jsonl"))
    if [int(r["tick"]) for r in weather] != list(range(start, end + 1)):
        raise ValueError("Weather coverage differs from ARM window")
    event_log = list(rows(raw / "ue_event_trace.jsonl"))
    realizations = list(rows(raw / "ue_event_realization.jsonl"))
    scene, script = read_json(scene_path), read_json(script_path)
    contract = {"capture_boundary_polygon_enu_m": cv.source_boundary_polygon(script)}
    episode_id = "__".join(arm_dir.relative_to(ROOT).parts)
    source = Path(frame_source).resolve() if frame_source is not None else canonical_dir / "truth_frames.jsonl"
    frames = []
    canonical_roster = {r["entity_id"]: r for r in read_json(canonical_dir / "global_entity_roster.json")["entities"]}
    output_roster = {}
    for frame in rows(source):
        tick = int(frame["tick"])
        if not start <= tick <= end:
            continue
        frame["episode_id"] = episode_id
        frame["frame_id"] = f"{episode_id}_tick_{tick}"
        entities = [e for e in frame["entities"] if e["entity_id"] not in roster]
        entities.extend(_entity(row, roster[eid], frame, contract) for eid, row in sorted(samples[tick].items()))
        frame["entities"] = entities
        frame["entity_motion_state"] = {e["entity_id"]: e["state"] for e in entities if e["entity_category"] == "uav" and "state" in e}
        frame["roster_summary"] = {"total": len(entities), "by_category": dict(sorted(Counter(e["entity_category"] for e in entities).items()))}
        for entity in entities:
            eid = entity["entity_id"]
            if eid in output_roster:
                continue
            if eid in roster:
                profile = cv.profile_for_entity(roster[eid], samples[tick][eid])
                entry = {**roster[eid], **{k: entity[k] for k in ("site_id", "roi_id", "entity_category", "entity_kind", "entity_type", "proxy_template_id", "logical_asset_id")},
                         "mode": profile["mode"], "initial_position_enu_m": entity["truth_pose"]["position_enu_m"],
                         "initial_yaw_deg": entity["truth_pose"]["rotation_deg"]["yaw_deg"]}
            else:
                if eid not in canonical_roster:
                    raise ValueError(f"Background frame actor missing roster: {eid}")
                entry = copy.deepcopy(canonical_roster[eid])
            output_roster[eid] = entry
        frames.append(frame)
    if [int(f["tick"]) for f in frames] != list(range(start, end + 1)):
        raise ValueError("Background frame coverage differs from ARM window")
    out.mkdir(parents=True, exist_ok=True)
    files = {k: f"{k}.jsonl" for k in ("truth_frames", "trajectories", "weather_meta", "dynamic_labels", "event_trace", "event_realization")}
    files.update({k: f"{k}.json" for k in ("scenario_plan", "episode_manifest", "global_entity_roster", "scene_occupancy_manifest", "capture_plan", "scene_setup", "event_script", "render_host_config")})
    paths = {key: (out / name).relative_to(ROOT).as_posix() for key, name in files.items()}
    write_rows(out / files["truth_frames"], frames)
    write_rows(out / files["trajectories"], (cv.truth_entity_to_trajectory_row(f, e) for f in frames for e in f["entities"]))
    write_rows(out / files["weather_meta"], (cv.normalize_weather_row(r, int(r["tick"])) for r in weather))
    write_rows(out / files["event_trace"], event_log)
    write_rows(out / files["event_realization"], realizations)
    # The label helper expects integer realization times. Dispatch remains the
    # observation time for omitted/pending actions, with explicit status retained.
    labels = event_display_labels(event_log, realizations, episode_id, input_meta["scenario_id"],
                                  start, end, {int(f["tick"]): f for f in frames})
    write_rows(out / files["dynamic_labels"], labels)
    write_json(out / files["global_entity_roster"], {"entities": list(output_roster.values())})
    write_json(out / files["scene_setup"], scene)
    write_json(out / files["event_script"], script)
    occupancy = read_json(canonical_dir / "scene_occupancy_manifest.json")
    occupancy = {k: occupancy[k] for k in ("schema_name", "schema_version", "authority", "geometry_authority", "lane_geometry_source")}
    occupancy.update({"episode_id": episode_id, "inputs": {"episode_dir": out.relative_to(ROOT).as_posix()},
        "entity_counts_by_category": dict(Counter(e["entity_category"] for e in output_roster.values())),
        "validation_summary": {"status": "not_run_for_arm_window", "ok": None},
        "scope": "Geometry authority only; canonical dynamic occupancy conclusions are not inherited."})
    write_json(out / files["scene_occupancy_manifest"], occupancy)
    plan = read_json(canonical_dir / "scenario_plan.json")
    # Retain scene/ROI geometry and policies; all export references are rebound.
    def rebind(value):
        if isinstance(value, dict):
            return {k: (paths[k] if k in paths and isinstance(v, str) else rebind(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [rebind(v) for v in value]
        if isinstance(value, str) and Path(value).name in files.values():
            return (out / Path(value).name).relative_to(ROOT).as_posix()
        return value
    plan = rebind(plan)
    plan["episode_id"] = episode_id
    plan["global_entity_roster"] = list(output_roster.values())
    plan["runtime_contract"].update({"tick_start": start, "tick_end": end})
    plan["export_contract"]["artifacts"] = paths
    for container in (plan["scenario_plan"], plan["compiled_plan_summary"], plan["scenario_plan"]["summary"]):
        for site in container["site_contracts"].values():
            site.update({"tick_start": start, "tick_end": end, "entity_count": len(output_roster)})
        for roi in container.get("roi_windows", {}).values():
            roi.update({"tick_start": start, "tick_end": end})
    for summary in (plan["compiled_plan_summary"], plan["scenario_plan"]["summary"]):
        summary.update({"event_count": len(event_log), "event_realization_count": len(realizations),
                        "entity_counts_by_category": dict(Counter(e["entity_category"] for e in output_roster.values()))})
    plan["scene_occupancy"] = {"artifact": paths["scene_occupancy_manifest"], "validation_summary": occupancy["validation_summary"]}
    write_json(out / files["scenario_plan"], plan)
    capture_ticks = [t for t in range(start, end + 1) if t % 5 == 0]
    write_json(out / files["capture_plan"], {"episode_id": episode_id, "tick_start": start, "tick_end": end,
        "capture_tick_step": 5, "capture_ticks": capture_ticks, "modalities": ["rgb", "depth", "segmentation"]})
    config = read_json(canonical_dir / "render_host_config.json")
    config.update({"episode_dir": out.relative_to(ROOT).as_posix(),
                   "output_dir": (out / "capture").relative_to(ROOT).as_posix(),
                   "event_script_path": paths["event_script"]})
    write_json(out / files["render_host_config"], config)
    write_json(out / files["episode_manifest"], {"episode_id": episode_id, "scenario_id": input_meta["scenario_id"],
        "window": manifest["window"], "duration_ticks": end - start, "tick_start": start, "tick_end": end,
        "n_entities": len(output_roster), "n_events": len(event_log), "n_event_realizations": len(realizations),
        "source_scene_setup_path": paths["scene_setup"], "source_event_script_path": paths["event_script"],
        "artifacts": paths, "generator": "Dataset/tools/arm_ue_export.py",
        "background_source": source_reference.relative_to(ROOT).as_posix() if source_reference.is_relative_to(ROOT) else str(source_reference),
        "capture_executed": False})
    write_json(out / "scenario_package.json", {"scenario_id": input_meta["scenario_id"], "episode_id": episode_id,
        "root_dir": out.relative_to(ROOT).as_posix(), **paths})
    from Dataset.tools.validate_arm_ue import validate_arm
    result = validate_arm(arm_dir)
    write_json(out / "validation.json", result)
    return result
