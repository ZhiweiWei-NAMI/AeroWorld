"""Repair L2-5's retained formal states from its checked-in event chain.

The original episode directories are unavailable. This keeps every retained
spatial sample and replaces only the signal/police state and event projections.
"""

from __future__ import annotations

import copy
import ctypes
import json
import os
import signal
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import orjson

from batch_generate import EpisodeStateEngine
from convert_to_render_ready import build_dynamic_labels, render_truth_sample_summary


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Dataset/scenarios/L2_infrastructure/interaction/L2-5_v1/event_script.json"
SIGNAL = "traffic_signal_l2_5_v1"
POLICE = "police_unit_l2_5_v1"
SOURCE_ROOTS = (
    ROOT / "aw_data/render_ready_episodes",
    ROOT / "aw_data/render_ready_episodes_capture_filtered",
)
INVALIDATED_ROOTS = (
    ROOT / "aw_data/objective_semantic_truth",
    ROOT / "aw_data/event_chains",
)
OBJECTIVE_FILES = {
    "epi_closure.json", "event_occurrences.jsonl", "event_outcomes.jsonl",
    "evidence_observations.jsonl", "l0_predicate_source_availability.json",
    "l0_predicate_state.jsonl", "manifest.json", "predicate_continuity_breaks.jsonl",
    "predicate_transitions.jsonl", "predicate_truth.jsonl", "semantic_graph_base.json",
    "semantic_graph_deltas.jsonl", "semantic_state_observations.jsonl",
    "semantic_truth_graphs.jsonl", "world_truth_graph_base.json",
    "world_truth_graph_deltas.jsonl", "world_truth_summary.json", "simulation_logs",
}
SIMULATION_LOG_FILES = {
    "battery_log.jsonl", "charging_log.jsonl", "communication_log.jsonl",
    "compute_log.jsonl", "facility_log.jsonl", "gnss_log.jsonl",
    "reconciliation.json", "utm_log.jsonl", "weather_log.jsonl",
}
EVENT_CHAIN_FILES = {"event_chain_manifest.json", "event_chain_relations.jsonl"}
EVENT_IDS = (
    "civilian_car_a_fault_entry",
    "civilian_car_b_fault_entry",
    "signal_all_red_fault",
    "traffic_confusion",
    "police_arrival",
    "manual_flow_restore",
    "signal_infrastructure_repair",
    "signal_response_actors_leave",
)
OLD_EVENT_IDS = EVENT_IDS[:6] + EVENT_IDS[7:]
REPLACED_FILES = (
    "truth_frames.jsonl",
    "trajectories.jsonl",
    "event_trace.jsonl",
    "event_realization.jsonl",
    "dynamic_labels.jsonl",
    "episode_manifest.json",
    "scenario_plan.json",
)


@contextmanager
def _defer_termination():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _terminate(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return [orjson.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _projection() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    script = json.loads(SCRIPT.read_text(encoding="utf-8"))
    scene = json.loads(SCRIPT.with_name("scene_setup.json").read_text(encoding="utf-8"))
    trajectories, _, trace, realizations, _ = EpisodeStateEngine(scene, script, SCRIPT, 900).run()
    if tuple(row["event_id"] for row in realizations) != EVENT_IDS:
        raise RuntimeError("checked-in L2-5 event chain has changed")
    if [(row["event_id"], row["dispatch_tick"]) for row in realizations[2:7]] != [
        ("signal_all_red_fault", 210), ("traffic_confusion", 240),
        ("police_arrival", 340), ("manual_flow_restore", 370),
        ("signal_infrastructure_repair", 430),
    ]:
        raise RuntimeError("checked-in L2-5 event times differ from the approved chain")
    expected: dict[str, dict[int, dict[str, Any]]] = {SIGNAL: {}, POLICE: {}}
    for row in trajectories:
        entity_id = row["entity_id"]
        if entity_id in expected:
            expected[entity_id][int(row["tick"])] = row
    if (len(expected[SIGNAL]) != 901 or expected[SIGNAL][430]["facility_state"]["fault"] is not True
            or expected[SIGNAL][431]["facility_state"]["fault"] is not False
            or expected[POLICE][370]["incident_state"]["manual_control_active"] is not False
            or expected[POLICE][371]["incident_state"]["manual_control_active"] is not True):
        raise RuntimeError("L2-5 expected state transition is inconsistent")
    return trace, realizations, expected


def _correct_truth_entity(entity: dict[str, Any], row: dict[str, Any]) -> None:
    entity_id = entity["entity_id"]
    pose = copy.deepcopy(entity["truth_pose"])
    entity["state"] = row["state"]
    if entity_id == SIGNAL:
        entity["facility_state"] = copy.deepcopy(row["facility_state"])
        annotations = entity["annotations"]
        if annotations.pop("signal", None) != {"phase": "green"}:
            raise RuntimeError("L2-5 signal phase differs from the obsolete placeholder")
        annotations["activity_type"] = row["activity_type"]
        activity = annotations["state_facets"]["activity"]
        activity["activity_type"] = row["activity_type"]
        activity["animation_hint"] = row["animation_hint"]
    else:
        entity["incident_state"] = copy.deepcopy(row["incident_state"])
    if entity["truth_pose"] != pose:
        raise RuntimeError(f"spatial truth changed for {entity_id}")


def _correct_trajectory_row(row: dict[str, Any], expected: dict[str, Any]) -> None:
    spatial = (copy.deepcopy(row["pos_enu"]), copy.deepcopy(row["vel_mps"]), row["yaw_deg"])
    row["state"] = expected["state"]
    if row["entity_id"] == SIGNAL:
        row["activity_type"] = expected["activity_type"]
        row["animation_hint"] = expected["animation_hint"]
        row["facility_state"] = copy.deepcopy(expected["facility_state"])
    else:
        row["incident_state"] = copy.deepcopy(expected["incident_state"])
    if (row["pos_enu"], row["vel_mps"], row["yaw_deg"]) != spatial:
        raise RuntimeError(f"trajectory position or motion changed for {row['entity_id']}")


def _stage_frames(
    source: Path, staged: Path, expected: dict[str, dict[int, dict[str, Any]]],
    snapshot_ticks: set[int],
) -> tuple[int, dict[int, dict[str, dict[str, Any]]]]:
    snapshots: dict[int, dict[str, dict[str, Any]]] = {}
    count = 0
    with source.open("rb") as reader, staged.open("wb") as writer:
        for raw_line in reader:
            frame = orjson.loads(raw_line)
            tick = int(frame["tick"])
            if tick != count:
                raise RuntimeError(f"{source}: unexpected frame tick {tick} after {count} rows")
            by_id = {entity["entity_id"]: entity for entity in frame["entities"]}
            if len(by_id) != len(frame["entities"]):
                raise RuntimeError(f"{source}: duplicate entity ID at tick {tick}")
            if SIGNAL not in by_id:
                raise RuntimeError(f"{source}: signal missing at tick {tick}")
            for entity_id in (SIGNAL, POLICE):
                entity = by_id.get(entity_id)
                if (entity is not None) != (tick in expected[entity_id]):
                    raise RuntimeError(f"{source}: {entity_id} presence differs at tick {tick}")
                if entity is not None:
                    _correct_truth_entity(entity, expected[entity_id][tick])
            encoded = orjson.dumps(frame)
            retained = orjson.loads(encoded)
            original = orjson.loads(raw_line)
            if {key: value for key, value in retained.items() if key != "entities"} != {
                key: value for key, value in original.items() if key != "entities"
            }:
                raise RuntimeError(f"{source}: frame context changed at tick {tick}")
            if len(retained["entities"]) != len(original["entities"]):
                raise RuntimeError(f"{source}: entity count changed at tick {tick}")
            for before, after in zip(original["entities"], retained["entities"]):
                if before["entity_id"] != after["entity_id"] or before["truth_pose"] != after["truth_pose"]:
                    raise RuntimeError(f"{source}: entity or pose changed at tick {tick}")
                if before["entity_id"] not in (SIGNAL, POLICE) and before != after:
                    raise RuntimeError(f"{source}: unrelated entity changed at tick {tick}")
            if tick in snapshot_ticks:
                snapshots[tick] = {
                    entity_id: render_truth_sample_summary(entity, tick)
                    for entity_id, entity in by_id.items()
                }
            writer.write(encoded + b"\n")
            count += 1
    if count != 901 or set(snapshots) != snapshot_ticks:
        raise RuntimeError(f"{source}: incomplete truth frames or event snapshots")
    return count, snapshots


def _stage_trajectories(
    source: Path, staged: Path, expected: dict[str, dict[int, dict[str, Any]]],
) -> int:
    count = 0
    seen: dict[str, set[int]] = {SIGNAL: set(), POLICE: set()}
    with source.open("rb") as reader, staged.open("wb") as writer:
        for raw_line in reader:
            row = orjson.loads(raw_line)
            entity_id = row["entity_id"]
            if entity_id in expected:
                tick = int(row["tick"])
                if tick not in expected[entity_id]:
                    raise RuntimeError(f"{source}: {entity_id} has no source row at tick {tick}")
                if tick in seen[entity_id]:
                    raise RuntimeError(f"{source}: duplicate {entity_id} row at tick {tick}")
                seen[entity_id].add(tick)
                spatial = (copy.deepcopy(row["pos_enu"]), copy.deepcopy(row["vel_mps"]), row["yaw_deg"])
                _correct_trajectory_row(row, expected[entity_id][tick])
                encoded = orjson.dumps(row)
                retained = orjson.loads(encoded)
                if (retained["pos_enu"], retained["vel_mps"], retained["yaw_deg"]) != spatial:
                    raise RuntimeError(f"{source}: trajectory pose changed for {entity_id} at tick {tick}")
                writer.write(encoded + b"\n")
            else:
                writer.write(raw_line)
            count += 1
    for entity_id in (SIGNAL, POLICE):
        if seen[entity_id] != set(expected[entity_id]):
            raise RuntimeError(f"{source}: incomplete {entity_id} trajectory rows")
    return count


def _render_snapshots(
    row: dict[str, Any], snapshots: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        str(tick): {
            entity_id: snapshots[tick].get(entity_id, {"present": False})
            for entity_id in row["target_ids"]
        }
        for tick in (row["before_tick"], row["evidence_tick"], row["after_tick"])
    }


def _event_realizations(
    old: list[dict[str, Any]], new: list[dict[str, Any]],
    expected: dict[str, dict[int, dict[str, Any]]],
    snapshots: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    if tuple(row["event_id"] for row in old) != OLD_EVENT_IDS:
        raise RuntimeError("retained L2-5 realizations are not the expected seven-event version")
    old_by_id = {row["event_id"]: row for row in old}
    result = copy.deepcopy(old[:4])
    retained_basis = (
        "Retained formal event_realization source snapshots and formal truth_frames "
        "render samples; checked-in L2-5 event_script.json for the event chain"
    )
    for row in result:
        row["basis"] = retained_basis
    police = copy.deepcopy(new[4])
    police["action_realizations"] = copy.deepcopy(old_by_id["police_arrival"]["action_realizations"][:2])
    police["source_truth_snapshots_by_tick"] = copy.deepcopy(
        old_by_id["police_arrival"]["source_truth_snapshots_by_tick"]
    )
    police["render_truth_snapshots_by_tick"] = _render_snapshots(police, snapshots)
    police["basis"] = retained_basis
    result.append(police)

    manual = copy.deepcopy(new[5])
    old_manual = old_by_id["manual_flow_restore"]
    manual["action_realizations"] = copy.deepcopy(new[5]["action_realizations"][:2]) + [
        copy.deepcopy(action) for action in old_manual["action_realizations"]
        if action["action_type"] == "move_entity"
    ]
    source_snapshots = copy.deepcopy(old_manual["source_truth_snapshots_by_tick"])
    for tick_str, tick_snapshot in source_snapshots.items():
        tick_snapshot.pop(SIGNAL, None)
        police_source = tick_snapshot.get(POLICE)
        if police_source and police_source.get("present"):
            row = expected[POLICE][int(tick_str)]
            police_source["state"] = row["state"]
            police_source["activity_type"] = row["activity_type"]
    manual["source_truth_snapshots_by_tick"] = source_snapshots
    manual["render_truth_snapshots_by_tick"] = _render_snapshots(manual, snapshots)
    manual["basis"] = (
        "Retained source snapshots with police state corrected by the checked-in "
        "event script; render snapshots from retained formal truth_frames"
    )
    result.append(manual)

    repair = copy.deepcopy(new[6])
    repair["render_truth_snapshots_by_tick"] = _render_snapshots(repair, snapshots)
    repair["basis"] = (
        "Checked-in L2-5 event_script.json and scene_setup.json state replay; "
        "render snapshots from retained formal truth_frames"
    )
    result.append(repair)

    leave = copy.deepcopy(new[7])
    old_leave = old_by_id["signal_response_actors_leave"]
    leave["action_realizations"] = copy.deepcopy(old_leave["action_realizations"])
    leave["source_truth_snapshots_by_tick"] = copy.deepcopy(old_leave["source_truth_snapshots_by_tick"])
    leave["render_truth_snapshots_by_tick"] = _render_snapshots(leave, snapshots)
    leave["basis"] = retained_basis
    result.append(leave)
    if (tuple(row["event_id"] for row in result) != EVENT_IDS
            or [row["sequence_no"] for row in result] != list(range(1, 9))):
        raise RuntimeError("L2-5 repaired event sequence is inconsistent")
    return result


def _stage_episode(
    episode_dir: Path, trace: list[dict[str, Any]], realizations: list[dict[str, Any]],
    expected: dict[str, dict[int, dict[str, Any]]],
    owned_candidates: list[tuple[Path, Path]],
) -> Path:
    old_trace = _rows(episode_dir / "event_trace.jsonl")
    old_realizations = _rows(episode_dir / "event_realization.jsonl")
    if (len(old_trace) != 7 or len(old_realizations) != 7
            or tuple(row["event_id"] for row in old_realizations) != OLD_EVENT_IDS):
        raise RuntimeError(f"{episode_dir}: expected old seven-event formal output")
    manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((episode_dir / "scenario_plan.json").read_text(encoding="utf-8"))
    if manifest["record_counts"]["event_trace"] != 7 or plan["compiled_plan_summary"]["event_count"] != 7:
        raise RuntimeError(f"{episode_dir}: old event counts do not agree")
    candidate_dir = episode_dir.with_name(f".{episode_dir.name}.l25_new")
    with _defer_termination():
        candidate_dir.mkdir(exist_ok=False)
        owned_candidates.append((episode_dir, candidate_dir))
    for path in episode_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{episode_dir}: unexpected episode entry {path}")
        if path.name not in REPLACED_FILES:
            os.link(path, candidate_dir / path.name)
    staged = {name: candidate_dir / name for name in REPLACED_FILES}
    snapshot_ticks = {
        int(row[key]) for row in realizations for key in ("before_tick", "evidence_tick", "after_tick")
    }
    frame_count, snapshots = _stage_frames(
        episode_dir / "truth_frames.jsonl", staged["truth_frames.jsonl"], expected, snapshot_ticks
    )
    trajectory_count = _stage_trajectories(
        episode_dir / "trajectories.jsonl", staged["trajectories.jsonl"], expected
    )
    if (frame_count != manifest["record_counts"]["truth_frames"]
            or trajectory_count != manifest["record_counts"]["trajectories"]):
        raise RuntimeError(f"{episode_dir}: spatial record counts changed")
    repaired_realizations = _event_realizations(old_realizations, realizations, expected, snapshots)
    labels = build_dynamic_labels(
        trace, episode_dir.name, scenario_id="L2-5_v1", event_realization_rows=repaired_realizations
    )
    _write_jsonl(staged["event_trace.jsonl"], trace)
    _write_jsonl(staged["event_realization.jsonl"], repaired_realizations)
    _write_jsonl(staged["dynamic_labels.jsonl"], labels)
    manifest["n_events"] = 8
    manifest["n_event_realizations"] = 8
    for counts_key in ("record_counts", "canonical_record_counts"):
        for name in ("event_trace", "event_realization", "dynamic_labels"):
            manifest[counts_key][name] = 8
    manifest["generator"] = "Dataset/tools/repair_l25_signal_truth.py"
    manifest["generation"] = {
        "generator": "Dataset/tools/repair_l25_signal_truth.py",
        "spatial_basis": "Retained formal truth_frames and trajectories for this episode",
        "state_basis": "Dataset/scenarios/L2_infrastructure/interaction/L2-5_v1/event_script.json",
        "scene_basis": "Dataset/scenarios/L2_infrastructure/interaction/L2-5_v1/scene_setup.json",
    }
    manifest["validation_summary"] = {
        "ok": None,
        "errors": [],
        "warnings": ["Full formal release validation pending after targeted L2-5 repair"],
    }
    _write_json(staged["episode_manifest.json"], manifest)
    for summary in (plan["compiled_plan_summary"], plan["scenario_plan"]["summary"]):
        summary["event_count"] = 8
        summary["event_realization_count"] = 8
    _write_json(staged["scenario_plan.json"], plan)
    return candidate_dir


def _exchange_directories(left: Path, right: Path) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(left), str(right))


def _validate_invalidated_dirs(*, allow_missing: bool = False, allow_partial: bool = False) -> None:
    if (ROOT.joinpath("aw_data").is_symlink()
            or not ROOT.joinpath("aw_data").is_dir()):
        raise RuntimeError("unexpected aw_data root")
    for dependent_root in INVALIDATED_ROOTS:
        if (dependent_root.is_symlink() or not dependent_root.is_dir()
                or not dependent_root.resolve().is_relative_to(ROOT)):
            raise RuntimeError(f"unexpected derived root {dependent_root}")
        for seed in range(3):
            episode_id = f"L2-5_v1__seed{seed:02d}"
            stale = dependent_root / episode_id
            if allow_missing and not stale.exists() and not stale.is_symlink():
                continue
            if not stale.is_dir() or stale.is_symlink():
                raise RuntimeError(f"expected derived episode directory {stale}")
            expected_names = OBJECTIVE_FILES if dependent_root.name == "objective_semantic_truth" else EVENT_CHAIN_FILES
            actual_names = {path.name for path in stale.iterdir()}
            if (not actual_names <= expected_names
                    or (not allow_partial and actual_names != expected_names)):
                raise RuntimeError(f"{stale}: unexpected derived file set")
            if any(path.is_symlink() or (path.name != "simulation_logs" and not path.is_file())
                   for path in stale.iterdir()):
                raise RuntimeError(f"{stale}: unexpected derived entry type")
            manifest_name = "manifest.json" if dependent_root.name == "objective_semantic_truth" else "event_chain_manifest.json"
            if manifest_name in actual_names:
                manifest = json.loads((stale / manifest_name).read_text(encoding="utf-8"))
                if manifest.get("episode_id") != episode_id:
                    raise RuntimeError(f"{stale}: derivative episode identity differs")
            elif not allow_partial or actual_names:
                raise RuntimeError(f"{stale}: missing manifest in nonempty derived directory")
            if dependent_root.name == "objective_semantic_truth" and "simulation_logs" in actual_names:
                logs = stale / "simulation_logs"
                if not logs.is_dir() or logs.is_symlink():
                    raise RuntimeError(f"{logs}: unexpected derived directory type")
                actual_log_names = {path.name for path in logs.iterdir()}
                if (not actual_log_names <= SIMULATION_LOG_FILES
                        or (not allow_partial and actual_log_names != SIMULATION_LOG_FILES)):
                    raise RuntimeError(f"{logs}: unexpected derived file set")
                if any(path.is_symlink() or not path.is_file() for path in logs.iterdir()):
                    raise RuntimeError(f"{logs}: unexpected simulation log entry type")


def _remove_invalidated_dirs() -> None:
    for dependent_root in INVALIDATED_ROOTS:
        for seed in range(3):
            stale = dependent_root / f"L2-5_v1__seed{seed:02d}"
            if stale.exists():
                logs = stale / "simulation_logs"
                if logs.exists():
                    for path in logs.iterdir():
                        path.unlink()
                    logs.rmdir()
                manifest_name = (
                    "manifest.json" if dependent_root.name == "objective_semantic_truth"
                    else "event_chain_manifest.json"
                )
                for path in stale.iterdir():
                    if path.name != manifest_name:
                        path.unlink()
                manifest_path = stale / manifest_name
                if manifest_path.exists():
                    manifest_path.unlink()
                stale.rmdir()
                print(f"removed stale derivative {stale}", flush=True)


def _formal_repair_status(episode_dir: Path) -> bool:
    manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("generation", {}).get("generator") != "Dataset/tools/repair_l25_signal_truth.py":
        return False
    realizations = _rows(episode_dir / "event_realization.jsonl")
    if (tuple(row["event_id"] for row in realizations) != EVENT_IDS
            or manifest["record_counts"]["event_realization"] != 8):
        raise RuntimeError(f"{episode_dir}: incomplete prior L2-5 repair")
    return True


def _cleanup_superseded_candidates(episode_dirs: list[Path]) -> None:
    for episode_dir in episode_dirs:
        candidate = episode_dir.with_name(f".{episode_dir.name}.l25_new")
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if not _formal_repair_status(episode_dir) or not candidate.is_dir() or candidate.is_symlink():
            raise RuntimeError(f"unexpected superseded L2-5 candidate {candidate}")
        expected_names = {path.name for path in episode_dir.iterdir()}
        actual_names = {path.name for path in candidate.iterdir()}
        if not actual_names <= expected_names:
            raise RuntimeError(f"{candidate}: unexpected superseded file set")
        if any(path.is_symlink() or not path.is_file() for path in candidate.iterdir()):
            raise RuntimeError(f"{candidate}: unexpected superseded entry type")
        manifest_name = "episode_manifest.json"
        if manifest_name in actual_names:
            manifest = json.loads((candidate / manifest_name).read_text(encoding="utf-8"))
            if (manifest.get("episode_id") != episode_dir.name
                    or manifest.get("n_events") != 7
                    or manifest.get("n_event_realizations") != 7
                    or manifest.get("generator") != "Dataset/tools/batch_generate.py"
                    or manifest.get("generation", {}).get("generator")
                    != "Dataset/tools/convert_to_render_ready.py"
                    or manifest.get("record_counts", {}).get("event_realization") != 7):
                raise RuntimeError(f"{candidate}: superseded manifest differs")
        elif actual_names:
            raise RuntimeError(f"{candidate}: nonempty superseded directory lacks its manifest")
        for path in candidate.iterdir():
            if path.name != manifest_name:
                path.unlink()
        manifest_path = candidate / manifest_name
        if manifest_path.exists():
            manifest_path.unlink()
        candidate.rmdir()
        print(f"removed superseded formal candidate {candidate}", flush=True)


def main() -> None:
    previous_term_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        _run_repair()
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)


def _run_repair() -> None:
    trace, realizations, expected = _projection()
    episode_dirs = [
        source_root / f"L2-5_v1__seed{seed:02d}"
        for source_root in SOURCE_ROOTS for seed in range(3)
    ]
    statuses = [_formal_repair_status(episode_dir) for episode_dir in episode_dirs]
    if any(statuses):
        if not all(statuses):
            raise RuntimeError("L2-5 formal repair is incomplete across the six episodes")
        _validate_invalidated_dirs(allow_missing=True, allow_partial=True)
        _remove_invalidated_dirs()
        _cleanup_superseded_candidates(episode_dirs)
        return
    _validate_invalidated_dirs()
    candidates: list[tuple[Path, Path]] = []
    swapped_count = 0
    all_swapped = False
    safe_cleanup = True
    try:
        for source_root in SOURCE_ROOTS:
            for seed in range(3):
                episode_dir = source_root / f"L2-5_v1__seed{seed:02d}"
                _stage_episode(episode_dir, trace, realizations, expected, candidates)
                print(f"validated candidate {episode_dir}", flush=True)
        for index, (episode_dir, candidate_dir) in enumerate(candidates):
            with _defer_termination():
                _exchange_directories(episode_dir, candidate_dir)
                swapped_count = index + 1
            print(f"replaced {episode_dir}", flush=True)
        all_swapped = True
        _validate_invalidated_dirs()
        _remove_invalidated_dirs()
        _cleanup_superseded_candidates(episode_dirs)
    except BaseException:
        if not all_swapped:
            try:
                with _defer_termination():
                    for index in range(swapped_count - 1, -1, -1):
                        episode_dir, candidate_dir = candidates[index]
                        _exchange_directories(episode_dir, candidate_dir)
            except BaseException:
                safe_cleanup = False
                raise
        raise
    finally:
        if safe_cleanup and not all_swapped:
            for _, candidate_dir in candidates:
                if candidate_dir.exists():
                    shutil.rmtree(candidate_dir)


if __name__ == "__main__":
    main()
