"""Execute explicit L1 action plans and export measured local-window assets.

The registered geometry and minimal semantic engine compute the declared target
family on the actual full simulation prefix. Each dedicated entry provides its
own action plan. Measured states, predicates and outcomes are saved with the
window; no source scheduling adjacency is treated as causality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "Dataset/tools", ROOT / "Plugins/SumoImporter/Scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml
from jsonschema import Draft202012Validator

from batch_generate import EpisodeStateEngine, TICK_HZ, write_json, write_jsonl

# --- authoritative semantic authority -------------------------------------------------
from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
    get_core_predicate_templates,
    get_governed_parameter_defaults,
)
from Dataset.semantic_truth.objective_pipeline import (
    build_episode_objective_artifacts,
)
from Dataset.semantic_truth.outcome_projection import recovery_tick_for_chain

REGISTRY = ROOT / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"
SCHEMA = ROOT / "rules/arm_config.schema.json"
L1_PROFILES = (
    "L1-1_v1", "L1-1_v2", "L1-2_v1", "L1-3_v1", "L1-3_v2", "L1-4_v1", "L1-4_v2",
)
FACTUAL_ROOTS = {
    "L1-1_v1": "aw_data",
    "L1-1_v2": "aw_data",
    "L1-2_v1": "aw_data",
    "L1-3_v1": "aw_data",
    "L1-3_v2": "aw_data",
    "L1-4_v1": "aw_data",
    "L1-4_v2": "aw_data",
}
CAPTURE_DIR = "render_ready_episodes_capture_filtered"
# The engine handlers an authored action type may bind to.  Explicit, closed set.
DISPATCH = {
    "move_entity": "_handle_move_entity",
    "set_visual_state": "_handle_set_visual_state",
    "set_runtime_state": "_handle_set_runtime_state",
    "spawn_entity": "_handle_spawn_entity",
    "remove_entity": "_handle_remove_entity",
    "set_weather": "_handle_set_weather",
    "set_pedestrian_activity": "_handle_set_pedestrian_activity",
    "capture_screenshot": "_handle_capture_screenshot",
}
RAW_FILES = (
    "episode_manifest.json", "global_entity_roster.json", "truth_frames.jsonl",
    "weather_meta.jsonl", "scene_occupancy_manifest.json", "event_realization.jsonl",
    "event_trace.jsonl", "dynamic_labels.jsonl", "scenario_plan.json", "scenario_package.json",
)


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                   separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def project_path(ref: str) -> Path:
    path = ROOT / ref
    if Path(ref).is_absolute() or ".." in Path(ref).parts:
        raise ValueError(f"source must be a repository-relative path: {ref}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_config(config_path: Path) -> dict:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema = read_json(SCHEMA)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config)
    expected = ROOT / "interventions" / config["group_key"] / (config["arm_id"] + ".yaml")
    if config_path != expected:
        raise ValueError(f"configuration identity/path mismatch: {config_path} != {expected}")
    profile, seed, _ = config["group_key"].split("/")
    if profile not in L1_PROFILES:
        raise ValueError(f"unsupported L1 profile: {profile}")
    if config["episode_id"] != f"{profile}__{seed}":
        raise ValueError("configuration seed/profile identity mismatch")
    if config["event_family"] not in FAMILY_PREDICATE:
        raise ValueError(f"unsupported event family: {config['family']}")
    declared_root = (config.get("provenance") or {}).get("parameter_sources", {}).get("factual_root")
    if declared_root != FACTUAL_ROOTS[profile]:
        raise ValueError(
            f"{profile}: factual_root must be {FACTUAL_ROOTS[profile]!r}, got {declared_root!r}")
    for key, ref in config["sources"].items():
        project_path(ref)
    return config


FAMILY_PREDICATE = {
    fam: fam for fam in (
        "uav_takeoff", "uav_landing", "restricted_region_intrusion",
        "altitude_excursion", "uav_separation_conflict", "corridor_congestion",
    )
}


def _authoritative_family(family: str) -> dict:
    for rule in get_core_event_occurrence_types():
        if rule.get("event_family_id") == family:
            return rule
    raise ValueError(f"event family is not in the core occurrence registry: {family}")


def _stage_arm_root(config: dict, trajectories: list[dict], out_dir: Path) -> Path:
    """Materialize a working episode root whose trajectories are the ARM's own."""
    source_dir = project_path(config["sources"]["trajectories"]).parent
    stem = Path(config["sources"]["trajectories"]).name
    root = out_dir / config["episode_id"]
    root.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        src = source_dir / name
        if src.is_file():
            shutil.copy2(src, root / name)
    # The intervention's own trajectory product is the world authority.
    with (root / stem).open("w", encoding="utf-8") as handle:
        for row in trajectories:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    # l0_predicate_source_availability describes which L0 sources exist for this
    # scene/network; it is independent of aircraft pose (verified: the seven L1
    # profiles differ only in episode_id).  Reuse it, do not fabricate it.
    factual_root = ROOT / FACTUAL_ROOTS[config["episode_id"].split("__")[0]]
    availability = factual_root / "objective_semantic_truth" / config["episode_id"] / \
        "l0_predicate_source_availability.json"
    if not availability.is_file():
        raise FileNotFoundError(f"L0 source availability contract is missing: {availability}")
    shutil.copy2(availability, root / "l0_predicate_source_availability.json")
    return root


def engine_duration(config: dict, script: dict) -> int:
    """The authored episode length, so a scheduled write is never mistaken for an overflow."""
    declared = script.get("parameters", {}).get("duration_ticks")
    if isinstance(declared, int) and declared > 0:
        return declared
    return 900


def run_engine(config: dict, plan: dict) -> tuple[EpisodeStateEngine, list[dict], list[dict], dict]:
    """Execute from tick 0 and apply the action plan supplied by the arm's own entry module.

    `plan` is the authoritative action definition.  It is written by hand in
    `scripts/arms/<group>/<arm>.py`; the YAML records the same values and
    :func:`require_plan_matches_config` proves the two agree before execution.
    """
    script_path = project_path(config["sources"]["script"])
    script = read_json(script_path)
    scene = read_json(project_path(config["sources"]["scene"]))
    end = int(config["window"]["end_tick"])
    # The engine must be allowed to run its full authored episode, not just this window: a
    # source action dispatched at or before end_tick can queue state that lands on the next
    # tick (set_runtime_state uses tick + max(1, delay_ticks)), and truncating the engine to
    # end_tick would turn that legitimate scheduled write into a duration overflow.  Rows are
    # still only persisted for [start_tick, end_tick]; the duration only bounds the execution.
    duration = int(engine_duration(config, script))
    if plan["trigger_rewrites"]:
        raise ValueError("L1 direct-action experiments do not rewrite source triggers")
    engine = EpisodeStateEngine(scene, script, script_path, duration)
    from donghu_core.event_script_interpreter import EventScriptInterpreter  # noqa: E402

    interpreter = EventScriptInterpreter(script_path, episode_id=config["episode_id"])
    tick_now = [0]
    audit: list[dict] = []
    snapshots: dict[int, dict] = {}
    window = config["window"]

    def dispatch(action: dict, origin: str):
        atype = str(action["type"])
        if atype not in DISPATCH:
            raise ValueError(f"unsupported action type: {atype}")
        if origin == "source_script" and tick_now[0] >= int(window["fork_tick"]) \
                and action.get("action_id") in set(plan["omit_action_ids"]):
            result = {"status": "omitted_by_explicit_intervention"}
        else:
            result = getattr(engine, DISPATCH[atype])(action, tick_now[0])
            if result.get("status") != "ok":
                raise RuntimeError(f"action failed: {json.dumps(result, ensure_ascii=False)}")
        entry = {"tick": tick_now[0], "origin": origin,
                 "action": copy.deepcopy(action), "result": copy.deepcopy(result)}
        if atype == "move_entity" and result["status"] == "ok":
            entry["created_schedule"] = copy.deepcopy(engine.entities[action["entity_id"]]["schedules"][-1])
        audit.append(entry)
        return result

    for atype in DISPATCH:
        interpreter.register_handler(atype, lambda action: dispatch(action, "source_script"))
    rows: list[dict] = []
    for tick in range(end + 1):
        tick_now[0] = tick
        engine._apply_pending_runtime_state_patches(tick)
        for weather in engine.weather_transitions.get(tick, []):
            engine.weather = weather
        current = engine._record_tick_rows(tick)
        rows.extend(current)
        for row in current:
            interpreter.update_entity_state(row["entity_id"], row["pos_enu"], {}, row["vel_mps"])
            if row["label_class"] == "pedestrian":
                interpreter.update_entity_activity(row["entity_id"], row["activity_type"])
        interpreter.update_weather_state(engine.weather_rows[-1])
        if tick in (int(window["start_tick"]), int(window["fork_tick"])):
            snapshots[tick] = {
                "tick": tick,
                "sampled_states": copy.deepcopy(current),
                "in_flight_actions": copy.deepcopy([
                    a for a in audit
                    if a["origin"] == "source_script"
                    and a["result"].get("scheduled_tick", -1) > tick
                ]),
            }
        for executed in interpreter.tick(tick):
            if executed["result"]["status"] not in ("ok", "omitted_by_explicit_intervention"):
                raise RuntimeError(json.dumps(executed, ensure_ascii=False))
        for step in plan["actions"]:
            if int(step["tick"]) == tick:
                dispatch(step["action"], "intervention")
    omitted = {r["action"]["action_id"] for r in audit
               if r["result"]["status"] == "omitted_by_explicit_intervention"}
    if omitted != set(plan["omit_action_ids"]):
        raise ValueError(f"configured omissions were not all dispatched: {sorted(omitted)}")
    engine.arm_interpreter = interpreter
    return engine, rows, audit, snapshots


def semantic_window(config: dict, trajectories: list[dict], out_dir: Path) -> dict:
    """Authoritative predicate/event computation for this ARM's own trajectories."""
    arm_root = _stage_arm_root(config, trajectories, out_dir)
    produced = build_episode_objective_artifacts(arm_root, out_dir / "_objective")
    rows = {}
    for name in ("predicate_truth", "predicate_transitions", "event_occurrences",
                 "event_outcomes", "predicate_continuity_breaks"):
        text = produced.files.get(f"{name}.jsonl", "")
        rows[name] = [json.loads(l) for l in text.splitlines() if l.strip()]
    l0_text = produced.files.get("l0_predicate_state.jsonl", "")
    rows["l0_predicate_state"] = [json.loads(l) for l in l0_text.splitlines() if l.strip()]
    return rows


def require_plan_matches_config(plan: dict, config: dict) -> None:
    """The entry module's plan is authoritative; the YAML must record exactly the same thing.

    This is the check that makes "one dedicated implementation per arm" verifiable rather than
    nominal: a plan that drifts from its recorded configuration is an error, not a default.
    """
    recorded = config["execution"]
    for key in ("omit_action_ids", "actions", "trigger_rewrites"):
        want = list(recorded.get(key) or [])
        got = list(plan.get(key) or [])
        if key == "actions":
            want = [{"tick": int(s["tick"]), **s["action"]} for s in want]
            got = [{"tick": int(s["tick"]), **s["action"]} for s in got]
        if json.dumps(want, sort_keys=True) != json.dumps(got, sort_keys=True):
            raise ValueError(
                f"entry plan disagrees with the recorded configuration for {key}:\n"
                f"  plan    = {json.dumps(got, sort_keys=True, ensure_ascii=False)}\n"
                f"  config  = {json.dumps(want, sort_keys=True, ensure_ascii=False)}")



def crop_event_evidence(occurrences, outcomes, truth, transitions, start, end, full_truth):
    """Keep local evidence; retain complete excluded rows as boundary diagnostics."""
    occ = copy.deepcopy(occurrences)
    outs = copy.deepcopy(outcomes)
    truth_ids = {r['truth_id'] for r in truth}
    transition_ids = {r['transition_id'] for r in transitions}
    full_byid = {r['truth_id']: r for r in full_truth}
    external_truth = {k for k, r in full_byid.items() if not start <= r['tick'] <= end}
    external_transitions = set()
    diagnostics = []
    for row in outs:
        phases = row['lifecycle_phase_evidence']
        excluded = [p for p in phases if p['start_tick'] < start or p['end_tick'] > end]
        terminal = row['terminal_tick']
        if terminal is not None and terminal < start:
            raise ValueError('In-window event has a terminal before the window')
        if terminal is not None and terminal > end:
            diagnostics.append({'kind': 'event_outcome', 'scope': 'full_run_outside_window',
                                'row': copy.deepcopy(row)})
            row['terminal_tick'] = None
            row['status'] = 'right_censored'
            row['lifecycle_status'] = 'observed_until_window_end'
            row['window_censor_tick'] = end
        elif excluded:
            raise ValueError('Observed terminal hold lacks complete in-window evidence')
        row['lifecycle_phase_evidence'] = [p for p in phases if p not in excluded]
    for row in occ:
        excluded = [p for p in row['event_phases'] if not start <= p['tick'] <= end]
        if excluded:
            diagnostics.append({'kind': 'event_occurrence', 'scope': 'full_run_outside_window',
                                'row': copy.deepcopy(row)})
        external_transitions.update(p['transition_id'] for p in excluded)
        row['event_phases'] = [p for p in row['event_phases'] if p not in excluded]
    for row in occ + outs:
        missing = set(row['supporting_truth_ids']) - truth_ids
        if missing - external_truth:
            raise ValueError('Event has unexplained missing truth evidence: ' + str(missing - external_truth))
        row['supporting_truth_ids'] = [x for x in row['supporting_truth_ids'] if x in truth_ids]
    for row in occ:
        missing = set(row['supporting_transition_ids']) - transition_ids
        if missing - external_transitions:
            raise ValueError('Event has unexplained missing transition evidence: ' + str(missing - external_transitions))
        row['supporting_transition_ids'] = [x for x in row['supporting_transition_ids'] if x in transition_ids]
        local_refs = []
        for ref in row['source_refs']:
            if ref.startswith('raw/') and '#tick=' in ref:
                tick = int(ref.split('#tick=', 1)[1].split('&', 1)[0])
                if not start <= tick <= end:
                    continue
            local_refs.append(ref)
        row['source_refs'] = local_refs
    return occ, outs, diagnostics


def execute_config(config_path: Path, plan: dict | None = None, *, prepared=None) -> dict:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if plan is None:
        # Explicit refusal: this executor no longer infers an action plan from the YAML.  Each
        # arm must pass the definition written in its own entry module.
        raise ValueError(
            f"{config_path.name}: no action plan supplied. Every arm must declare its actions "
            "in its own entry module (scripts/arms/<group>/<arm>.py) and pass them here.")
    require_plan_matches_config(plan, config)
    out = ROOT / config["output"]
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing ARM: {out}")
    start, fork, end = (int(config["window"][k]) for k in ("start_tick", "fork_tick", "end_tick"))
    for name, value in (("start", start), ("fork", fork), ("end", end)):
        if value % 5:
            raise ValueError(f"{name}_tick must lie on the formal 5-tick grid: {value}")
    if not start <= fork <= end:
        raise ValueError("window must satisfy start <= fork <= end")
    source_files = {k: project_path(v) for k, v in config["sources"].items()}
    if prepared is None:
        engine, rows, audit, snapshots = run_engine(config, plan)
        from Dataset.tools.l1_semantics import project
        semantic = project(config, rows, engine.entities)
    else:
        engine, rows, audit, snapshots, semantic = prepared

    scoped = [r for r in rows if start <= r["tick"] <= end and r["entity_id"] in config["entity_ids"]]
    expected = {(t, e) for t in range(start, end + 1) for e in config["entity_ids"]}
    got = {(r["tick"], r["entity_id"]) for r in scoped}
    if len(scoped) != len(got) or got != expected:
        raise ValueError("window state is missing or duplicated ticks")

    # Honest baseline record: the current engine executed from tick 0 must equal the frozen
    # factual over the shared prefix; differences are reported per field, not hidden.
    parent_all = [r for r in read_rows(source_files["trajectories"])
                  if start <= r["tick"] <= end and r["entity_id"] in config["entity_ids"]]
    parent_map = {(r["tick"], r["entity_id"]): r for r in parent_all}
    executed_map = {(r["tick"], r["entity_id"]): r for r in scoped}
    prefix_mismatch = []
    position_mismatch = []
    for key, observed in sorted(parent_map.items()):
        row = executed_map.get(key)
        if row is None:
            continue
        if row["tick"] > fork:
            continue
        if row["pos_enu"] != observed["pos_enu"]:
            position_mismatch.append(row["tick"])
        if max(abs(a - b) for a, b in zip(row["vel_mps"], observed["vel_mps"])) > 1e-4                 or any(row[k] != observed[k] for k in ("state", "activity_type")):
            prefix_mismatch.append(row["tick"])

    is_observation = config["mode"] == "factual_observation"
    pre_spawn_count = 0
    if is_observation:
        # The frozen factual is the observation.  An entity that spawns ON its anchor
        # (every lifecycle_takeoff_* anchor, and L1-4's staggered launches) has no capture
        # row before its own activation tick.  That absence is a real lifecycle boundary:
        # it is recorded as coverage, not treated as a missing measurement, and the pre-spawn
        # samples are taken from the engine's own pre-action sampling.
        parent = [r for r in read_rows(source_files["trajectories"])
                  if start <= r["tick"] <= end and r["entity_id"] in config["entity_ids"]]
        parent_map0 = {(r["tick"], r["entity_id"]): r for r in parent}
        if len(parent_map0) != len(parent):
            raise ValueError("factual has duplicate window state")
        pre_spawn = sorted(expected - set(parent_map0))
        pre_spawn_count = len(pre_spawn)
        engine_map = {(r["tick"], r["entity_id"]): r for r in scoped}
        if not pre_spawn:
            trajectories = parent
        else:
            covered = [engine_map.get(key) for key in pre_spawn]
            if any(row is None for row in covered):
                raise ValueError("pre-spawn samples are unavailable from the engine")
            trajectories = sorted(parent + covered,
                                  key=lambda r: (r["tick"], r["entity_id"]))
            if {(r["tick"], r["entity_id"]) for r in trajectories} != expected:
                raise ValueError("factual window state is incomplete after lifecycle fill")
    else:
        trajectories = scoped

    snapshots = {t: {"tick": t,
                    "sampled_states": [r for r in rows if r["tick"] == t],
                    "source_reconstruction": "current engine execution from tick 0; not standalone restore"}
                 for t in (start, fork)}
    # Occurrences carry trigger_tick, outcomes terminal_tick, transitions to_tick;
    # predicate_truth and the dense L0 projection carry tick.  Filter each by its own field.
    occ = [r for r in semantic["event_occurrences"]
           if start <= int(r["trigger_tick"]) <= end]
    keep = {o["event_id"] for o in occ}
    # An event detected inside the window stays in the product even when its recovery lands
    # beyond the declared boundary; that is right-censoring, and it must not be reported as
    # absence.  Only events with no in-window detection are excluded.
    out_rows = [dict(r) for r in semantic["event_outcomes"] if r.get("event_id") in keep]
    tran = [r for r in semantic["predicate_transitions"]
            if start <= int(r["to_tick"]) <= end]
    truth = [r for r in semantic["predicate_truth"] if start <= int(r["tick"]) <= end]
    dense = [r for r in semantic["l0_predicate_state"] if start <= int(r["tick"]) <= end]
    occ, out_rows, boundary = crop_event_evidence(occ, out_rows, truth, tran, start, end, semantic["predicate_truth"])
    from Dataset.semantic_truth.window_outcome import arm_window_outcome
    out_rows = [arm_window_outcome(row, start_tick=start, end_tick=end) for row in out_rows]

    out.mkdir(parents=True)
    write_json(out / "raw/initial_state.json", snapshots[start])
    write_json(out / "raw/fork_state.json", snapshots[fork])
    write_jsonl(out / "raw/states.jsonl", trajectories)
    write_jsonl(out / "raw/trajectories.jsonl", trajectories)
    write_jsonl(out / "raw/actions.jsonl",
                [a for a in audit if start <= a["tick"] <= end])
    write_jsonl(out / "raw/weather.jsonl",
                [r for r in engine.weather_rows if start <= r["tick"] <= end])
    from Dataset.tools.arm_ue_export import write_engine_inputs
    write_engine_inputs(out, engine, engine.arm_interpreter, config["window"], script=engine.script)
    write_jsonl(out / "objective/predicate_truth.jsonl", truth)
    write_jsonl(out / "objective/predicate_truth_ticks.jsonl", dense)
    write_jsonl(out / "objective/predicate_transitions.jsonl", tran)
    write_jsonl(out / "objective/event_occurrences.jsonl", occ)
    write_jsonl(out / "objective/event_outcomes.jsonl", out_rows)
    nodes, relations = [], []
    for row in tran:
        nodes.append({"node_id": row["transition_id"], "kind": "predicate_transition",
                      "tick": row["to_tick"],
                      "evidence_ref": "objective/predicate_transitions.jsonl#" + row["transition_id"]})
    node_ids = {n["node_id"] for n in nodes}
    recovery_rows = []
    for row in out_rows:
        recovery_tick = recovery_tick_for_chain(
            row, window_start_tick=start, window_end_tick=end)
        if recovery_tick is not None:
            recovery_rows.append(row)
            nodes.append({"node_id": row["outcome_id"], "kind": "event_recovery_confirmed",
                          "within_window": True,
                          "tick": recovery_tick,
                          "evidence_ref": "objective/event_outcomes.jsonl#" + row["outcome_id"]})
    node_ids = {n["node_id"] for n in nodes}
    for row in occ:
        nodes.append({"node_id": row["event_id"], "kind": "objective_event",
                      "tick": row["trigger_tick"],
                      "evidence_ref": "objective/event_occurrences.jsonl#" + row["event_id"]})
        node_ids.add(row["event_id"])
    # Every relation must reference two nodes the window actually contains.  A phase whose
    # transition lies outside the window is recorded as an explicit boundary note instead of
    # being emitted as a dangling edge.
    outside_evidence = []
    for row in occ:
        for phase in row.get("event_phases", ()):
            if phase["transition_id"] not in node_ids:
                outside_evidence.append({"event_id": row["event_id"],
                                         "phase_kind": phase["phase_kind"],
                                         "tick": phase["tick"],
                                         "transition_id": phase["transition_id"],
                                         "note": "evidence transition lies outside the "
                                                 "declared window; not emitted as an edge"})
                continue
            relations.append({"source": phase["transition_id"], "target": row["event_id"],
                              "relation_type": "predicate_" + phase["phase_kind"] + "_evidence",
                              "tick": phase["tick"], "causal_claim": False,
                              "rule_id": row["rule_id"]})
    for row in recovery_rows:
        if row["outcome_id"] not in node_ids:
            continue
        relations.append({"source": row["event_id"], "target": row["outcome_id"],
                          "relation_type": "has_observed_outcome", "causal_claim": False,
                          "tick": row["terminal_tick"]})
    if boundary or outside_evidence:
        write_json(out / "objective/window_boundary_evidence.json", {
            'scope': 'diagnostics_only_excluded_from_local_graph', 'window': config['window'],
            'excluded_full_run_records': boundary,
            'unmaterialized_phase_relations': outside_evidence})
    write_jsonl(out / "chain/events.jsonl", sorted(nodes, key=lambda r: (r["tick"], r["node_id"])))
    write_jsonl(out / "chain/relations.jsonl", relations)

    # The window status describes the TARGET event family declared by this arm.  Other
    # families caught by the shared window are recorded with their own censoring flags but
    # do not decide the target's status.
    target_family = config["event_family"]
    target_occ = [r for r in occ if r["event_family_id"] == target_family]
    target_ids = {r["event_id"] for r in target_occ}
    target_out = [r for r in out_rows if r.get("event_id") in target_ids]
    censored = any(r["status"] == "right_censored" for r in target_out)
    pending = any(r["status"] == "pending" for r in target_out)
    status = ("target_event_absent_in_window" if not target_occ
              else "right_censored" if censored else "pending" if pending else "events_recovered")
    write_json(out / "objective/window_status.json", {
        "window": config["window"], "status": status,
        "absence_is_only_within_window": True,
        "scope": {"entity_ids": config["entity_ids"],
                  "event_family": config["event_family"],
                  "predicate_ids": config["predicate_ids"]},
        "lifecycle_coverage": {
            "pre_activation_samples_filled_from_engine": pre_spawn_count,
            "policy": "spawn-boundary absence is recorded, not treated as a measurement gap",
        },
        "observed": {
            "target_family": target_family,
            "event_occurred": bool(target_occ),
            "onset_ticks": sorted(r["trigger_tick"] for r in target_occ),
            "terminal_ticks": sorted(r["terminal_tick"] for r in target_out
                                     if r.get("terminal_tick") is not None),
            "other_family_events_in_window": sorted(
                {(r["event_family_id"], r["trigger_tick"]) for r in occ
                 if r["event_family_id"] != target_family}),
            "target_censored_by_window": [r["event_id"] for r in target_out
                                          if r["status"] == "right_censored"],
        },
        "expected": config["expected"],
    })
    manifest = {
        "schema": "aeroworld_window_arm", "version": 1,
        "arm_id": config["arm_id"], "group_key": config["group_key"],
        "episode_id": config["episode_id"], "mode": config["mode"],
        "event_family": config["event_family"], "window": config["window"],
        "scope": {"entity_ids": config["entity_ids"], "predicate_ids": config["predicate_ids"],
                  "background_policy": "frozen_parent_reference_no_feedback_claim",
                  "not_claimed": ["all_ontology_predicates", "whole_episode",
                                  "acceleration_feasible_flight", "UE_capture"]},
        "sources": {k: config["sources"][k] for k in sorted(config["sources"])},
        "config": {"path": str(config_path.relative_to(ROOT))},
        "window_status": status,
        "provenance": {"parameter_sources": config["provenance"]["parameter_sources"]},
        "factual_source_audit": {
            "compared_fields": ["pos_enu", "vel_mps", "state", "activity_type"],
            "prefix_position_mismatch_ticks": sorted(set(position_mismatch)),
            "prefix_state_mismatch_ticks": sorted(set(prefix_mismatch)),
            "role": "baseline_drift" if config["mode"] != "intervention"
                    else "prefix_shared_with_factual",
        },
        "snapshot_contract": {
            "sampled_states": "actual_current_engine_rows_before_actions",
            "entity_definitions": "static_engine_definitions_not_current_positions",
            "standalone_restore_supported": False,
            "reconstruction": "execution from tick 0 with the referenced sources and config"},
        "release_status": "research_candidate_not_release_certified",
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print("this module is a library: each arm has its own entry module under scripts/arms/",
          file=sys.stderr)
    return 2



if __name__ == "__main__":
    raise SystemExit(main())
