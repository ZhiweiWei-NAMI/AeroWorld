"""Execute explicitly configured L4 ARM windows for the L4_agents profiles.

Experiment definitions: `design/<profile>/<seed>/<group>/experiment.json`.
Standard: `rules/ARM_WINDOW_STANDARD.md`.
Anchor and incident fragments: `Dataset/tools/l4_chain_truth.py`.

The unit here is the incident fragment, not a per-anchor sweep: one group = the script node
that drives an incident family plus the semantic event that family produces.  For a family the
event authority declares non-executable there is no semantic event, so the group covers the
declaring script stages and the target is reported ``blocked_non_executable_family`` with the
authority's own reason - the handling the L4 handoff prompts prescribe.

Formal predicates use the objective pipeline. L4-9/10/11 use the separate,
explicit L4 business contract for observed ground-vehicle responses. Those
results do not claim to implement the frozen ontology's unavailable families.

Actions come only from the configuration.  `DISPATCH` maps an explicit action
`type` to an engine handler; nothing infers an action from an intent name.
"""
from __future__ import annotations

import copy
import json
import math
import os
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

from batch_generate import (  # noqa: F401
    EpisodeStateEngine,
    RUNTIME_STATE_FIELDS,
    TICK_HZ,
    write_json,
    write_jsonl,
)

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
from Dataset.tools import l4_business_semantics as business
from Dataset.tools import l4_incident_observations as incident_observations

REGISTRY = ROOT / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"
SCHEMA = ROOT / "rules/l4_arm_config.schema.json"
from Dataset.tools.l4_chain_truth import (  # noqa: E402
    L4_INCIDENT_FAMILY,
    authority_chain,
    fragment,
    incident_is_executable,
    profiles as l4_profiles,
)

L4_PROFILES = tuple(l4_profiles())
# The canonical corpus supplies immutable sources and comparison evidence.
# Factual products sample the original configured process.
FACTUAL_ROOTS = {name: "aw_data" for name in L4_PROFILES}
CAPTURE_DIR = "render_ready_episodes_capture_filtered"
# The formal episode duration.  Simulation runs to here; the window bounds only the products.
EPISODE_TICKS = 900
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




def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def project_path(ref: str) -> Path:
    relative = Path(ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"source must be a repository-relative path: {ref}")
    path = ROOT / relative
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
    if profile not in L4_PROFILES:
        raise ValueError(f"unsupported L4 profile: {profile}")
    if config["episode_id"] != f"{profile}__{seed}":
        raise ValueError("configuration seed/profile identity mismatch")
    local_rule = business.profile_rule(profile)
    if local_rule is not None:
        if config["event_family"] != local_rule["event_family"] or config["predicate_ids"] != local_rule["predicate_ids"]:
            raise ValueError("ground-vehicle ARM config differs from its business semantic authority")
    elif config["event_family"] not in L4_INCIDENT_FAMILY.values() and \
            config["event_family"] not in {r["event_family_id"] for r in _authority_rows(profile)}:
        raise ValueError(f"unsupported L4 event family: {config['event_family']}")
    declared_root = (config.get("provenance") or {}).get("parameter_sources", {}).get("factual_root")
    if declared_root != FACTUAL_ROOTS[profile]:
        raise ValueError(
            f"{profile}: factual_root must be {FACTUAL_ROOTS[profile]!r}, got {declared_root!r}")
    for key, ref in config["sources"].items():
        project_path(ref)
    return config


def _authority_rows(profile: str) -> list[dict]:
    return authority_chain(profile)


def incident_fragment(profile: str, seed: str) -> dict:
    return fragment(profile, seed)


def _declared_family_reason(family: str) -> str | None:
    mapping = read_json(ROOT / "Dataset/semantic_rules/profiles/event_family_authority_mapping.json")
    entry = (mapping.get("families") or {}).get(family) or {}
    if entry.get("executable"):
        return None
    return entry.get("non_executable_reason")


def _authoritative_family(family: str) -> dict:
    for rule in get_core_event_occurrence_types():
        if rule.get("event_family_id") == family:
            return rule
    raise ValueError(f"event family is not in the core occurrence registry: {family}")


# Truth-frame entity fields driven by this arm's own engine rows.  Everything else on the
# entity (identity, roster, contract metadata) is immutable scene provenance and stays as
# captured.  `truth_pose` is rebuilt from pos/vel/yaw; runtime state families are taken
# verbatim from the engine rows, so an arm that adds or omits a state patch changes what the
# semantic pipeline reads -- that is the entire point of a state rewrite arm.
_TRUTH_DYNAMIC_FIELDS = ("truth_pose", *RUNTIME_STATE_FIELDS, "state", "state_sequence",
                         "state_revision", "activity_type")


def _rebuild_truth_frames(source_dir: Path, root: Path, trajectories: list[dict]) -> None:
    """Write this arm's own truth_frames.jsonl: frozen frames as the skeleton, the arm's
    engine rows as the authority for pose and runtime state at every sampled tick.

    The objective pipeline reads entity state from truth_frames, not from trajectories, so
    without this rebuild every arm would share the frozen factual state history and state
    rewrites could never change a predicate.  Positions are cross-checked: an engine row
    whose position disagrees with the frozen frame at a tick the arm did not touch signals
    an engine/capture divergence that must be recorded, not silently overwritten.
    """
    frame_path = source_dir / "truth_frames.jsonl"
    if not frame_path.is_file():
        return
    by_tick_entity: dict[tuple[int, str], dict] = {}
    for row in trajectories:
        by_tick_entity[(int(row["tick"]), str(row["entity_id"]))] = row
    overrides: dict[int, dict[str, dict]] = {}
    for (tick, entity_id), row in by_tick_entity.items():
        overrides.setdefault(tick, {})[entity_id] = row
    source_frames = [json.loads(line) for line in frame_path.read_text().splitlines() if line.strip()]
    out_frames = []
    position_divergence = []
    for frame in source_frames:
        tick = int(frame["tick"])
        frame_overrides = overrides.get(tick)
        if frame_overrides:
            for entity in frame.get("entities", []):
                row = frame_overrides.get(str(entity["entity_id"]))
                if row is None:
                    continue
                pose = dict(entity.get("truth_pose") or {})
                pose["position_enu_m"] = list(row["pos_enu"])
                pose["velocity_enu_mps"] = list(row["vel_mps"])
                if "rotation_deg" in pose and isinstance(pose["rotation_deg"], dict):
                    pose["rotation_deg"] = {**pose["rotation_deg"], "yaw_deg": row["yaw_deg"]}
                frozen_pos = list((entity.get("truth_pose") or {}).get("position_enu_m") or [])
                if frozen_pos and frozen_pos != list(row["pos_enu"]):
                    position_divergence.append({"tick": tick, "entity_id": row["entity_id"]})
                entity["truth_pose"] = pose
                for field in _TRUTH_DYNAMIC_FIELDS:
                    if field in ("truth_pose",):
                        continue
                    if field in row:
                        entity[field] = copy.deepcopy(row[field])
                if "activity_type" in row and "state" in entity:
                    entity["state"] = row["activity_type"] if row.get("activity_type") else entity["state"]
            frame["arm_state_authority"] = "this_arm_engine_rows"
        out_frames.append(frame)
    with (root / "truth_frames.jsonl").open("w", encoding="utf-8") as handle:
        for frame in out_frames:
            handle.write(json.dumps(frame, ensure_ascii=False, sort_keys=True) + "\n")
    (root / "arm_truth_frame_provenance.json").write_text(json.dumps({
        "schema": "l4_arm_truth_frame_provenance", "version": 1,
        "authority": "this arm's engine rows for truth_pose and runtime state families; "
                     "identity/roster fields from the frozen episode frames",
        "engine_rows_used": len(by_tick_entity),
        "sampled_ticks_overridden": sorted(overrides),
        "frozen_position_divergence": position_divergence[:50],
        "frozen_position_divergence_count": len(position_divergence),
    }, ensure_ascii=False, indent=1))


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
    _rebuild_truth_frames(source_dir, root, trajectories)
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
    rewrites = list(plan["trigger_rewrites"])
    if rewrites:
        # Delay intent is expressed at the engine layer: the declared trigger threshold is
        # rewritten before the interpreter is constructed, so dispatch, the engine schedule
        # and the logged event trace all move together.  Both mechanisms are the ones the
        # interpreter actually evaluates (`tick` and `min_true_ticks`).
        known = {str(t["trigger_id"]) for t in script.get("triggers") or []}
        for item in rewrites:
            trigger_id = str(item["trigger_id"])
            if trigger_id not in known:
                raise ValueError(f"trigger_rewrites references an unknown trigger: {trigger_id}")
            for trigger in script["triggers"]:
                if str(trigger["trigger_id"]) != trigger_id:
                    continue
                # Each mechanism rewrites THIS anchor's own threshold, the quantity the
                # interpreter actually evaluates for that trigger kind.
                if item["mechanism"] == "tick_shift":
                    if trigger.get("type") != "tick":
                        raise ValueError(f"{trigger_id} is not a tick trigger")
                    trigger["tick"] = int(trigger["tick"]) + int(item["delay_ticks"])
                elif item["mechanism"] == "min_true_ticks_shift":
                    if trigger.get("type") != "entity_proximity":
                        raise ValueError(f"{trigger_id} is not an entity_proximity trigger")
                    base = int(trigger.get("min_true_ticks") or 0)
                    trigger["min_true_ticks"] = base + int(item["delay_ticks"])
                elif item["mechanism"] == "fired_after_delay_shift":
                    if trigger.get("type") != "event_fired_after":
                        raise ValueError(f"{trigger_id} is not an event_fired_after trigger")
                    trigger["delay_ticks"] = int(trigger["delay_ticks"]) + int(item["delay_ticks"])
                else:
                    raise ValueError(f"unsupported delay mechanism: {item['mechanism']}")
    # The engine always simulates the full episode.  The declared window bounds what is
    # WRITTEN, never what is simulated: an authored action scheduled inside the window can
    # dispatch a runtime patch whose effective tick lands after end_tick, and truncating the
    # simulation there would fail the run rather than record the window honestly.
    engine = EpisodeStateEngine(scene, script, script_path, EPISODE_TICKS)
    from donghu_core.event_script_interpreter import EventScriptInterpreter  # noqa: E402

    interpreter = EventScriptInterpreter(script_path, episode_id=config["episode_id"])
    if rewrites:
        # The interpreter is constructed from the file; hand it the same rewritten script the
        # engine simulates, so both advance on identical trigger thresholds.
        interpreter.script = copy.deepcopy(script)
    tick_now = [0]
    audit: list[dict] = []
    snapshots: dict[int, dict] = {}
    window = config["window"]
    local_rule = business.profile_rule(config["episode_id"].split("__")[0])
    mrm_source = None
    mrm_response = None
    if local_rule is not None and local_rule["mechanism"] == "av":
        response_event = next(e for e in script["events"] if e["event_id"] == "av_safe_stop")
        mrm_source = next(a for a in response_event["actions"] if a["type"] == "move_entity")
        stationary_tolerance = business.contract()["stationary_speed_tolerance_mps"]

    def dispatch(action: dict, origin: str):
        nonlocal mrm_response
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
            if (mrm_source is not None and atype == "move_entity"
                    and action["entity_id"] == mrm_source["entity_id"]
                    and ((origin == "source_script" and action["action_id"] == mrm_source["action_id"])
                         or (origin == "intervention" and mrm_response is not None
                             and mrm_response["dispatch_tick"] == tick_now[0]))):
                # A same-dispatch explicit replacement is this arm's actual response.
                # Later movement or a stopping activity interrupts this response.
                mrm_response = {
                    "dispatch_tick": tick_now[0],
                    "schedule_index": len(engine.entities[action["entity_id"]]["schedules"]) - 1,
                }
        audit.append({"tick": tick_now[0], "origin": origin,
                      "action": copy.deepcopy(action), "result": copy.deepcopy(result)})
        return result

    for atype in DISPATCH:
        interpreter.register_handler(atype, lambda action: dispatch(action, "source_script"))
    rows: list[dict] = []
    for tick in range(EPISODE_TICKS + 1):
        tick_now[0] = tick
        engine._apply_pending_runtime_state_patches(tick)
        if mrm_source is not None:
            entity_id = mrm_source["entity_id"]
            entity = engine.entities[entity_id]
            state = entity["vehicle_state"]
            state["stopped_safe"] = False
            if (mrm_response is not None and engine._entity_active_at(entity_id, tick)
                    and state["sensor_fault"] and state["minimal_risk_maneuver_active"]):
                index = mrm_response["schedule_index"]
                response = entity["schedules"][index]
                later = entity["schedules"][index + 1:]
                position, velocity, _ = engine._entity_sample(entity_id, tick)
                # Materialize the actual move's completion before raw/snapshot/UE sampling.
                # No authored timeout or declaration can complete a still-moving response.
                state["stopped_safe"] = (
                    tick > response["tick"] and tick >= engine.keyframes[entity_id][-1][0]
                    and position == response["waypoints_enu_m"][-1]
                    and math.sqrt(sum(round(v, 6) ** 2 for v in velocity)) <= stationary_tolerance
                    and not any(s["type"] == "move" or s.get("stop_motion") is True for s in later)
                )
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
    configured = set(plan["omit_action_ids"])
    if omitted != configured:
        # Report both directions: a configured omission that never reached its dispatch means the
        # source event no longer fires, and an unconfigured omission means the plan is incomplete.
        raise ValueError(
            "configured omissions were not all dispatched: "
            f"never_dispatched={sorted(configured - omitted)} "
            f"unexpectedly_omitted={sorted(omitted - configured)}")
    # Keep the actual interpreter and rewritten script for the common UE producer.
    engine.arm_interpreter = interpreter
    engine.arm_script = script
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


def execute_config(config_path: Path, plan: dict | None = None) -> dict:
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
        # Refuse by default.  A run that supersedes a previous product must say so and name the
        # archive root, so nothing is silently replaced and the old set stays inspectable.
        archive_ref = os.environ.get("L4_SUPERSEDE_TO")
        if not archive_ref:
            raise FileExistsError(f"refusing to overwrite an existing ARM: {out}")
        archive = (ROOT / archive_ref).resolve()
        archive.relative_to(ROOT)
        target = archive / out.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"the supersede archive already holds {target}")
        shutil.move(str(out), str(target))
        print(f"superseded {out.relative_to(ROOT)} -> {target.relative_to(ROOT)}", file=sys.stderr)
    start, fork, end = (int(config["window"][k]) for k in ("start_tick", "fork_tick", "end_tick"))
    for name, value in (("start", start), ("fork", fork), ("end", end)):
        if value % 5:
            raise ValueError(f"{name}_tick must lie on the formal 5-tick grid: {value}")
    if not start <= fork <= end:
        raise ValueError("window must satisfy start <= fork <= end")
    source_files = {k: project_path(v) for k, v in config["sources"].items()}
    engine, rows, audit, snapshots = run_engine(config, plan)

    scoped = [r for r in rows if start <= r["tick"] <= end and r["entity_id"] in config["entity_ids"]]
    expected = {(t, e) for t in range(start, end + 1) for e in config["entity_ids"]}
    got = {(r["tick"], r["entity_id"]) for r in scoped}
    if len(scoped) != len(got):
        raise ValueError("window state has duplicate (tick, entity) rows")
    # A declared subject that spawns ON the anchor has no engine row before its activation tick.
    # That boundary is resolved below (factual: filled from the engine's own sampling and marked;
    # intervention: marked as inactive).  Only an unexpected gap is a failure here.
    # The check runs against the FINAL entity definitions: a subject spawned by an action at the
    # fork is inactive in the tick-240 sample (the sample precedes the action) and active from
    # 241.  Any pair that is still inactive after the whole run, and has a row, is a duplicate;
    # any pair with no row is the lifecycle boundary handled below.
    unexpected = {key for key in expected - got if key[0] > end}
    if unexpected:
        raise ValueError(f"rows outside the declared window: {sorted(unexpected)[:3]}")

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

    pre_spawn_count = 0
    lifecycle_rows: list[dict] = []

    # Factual records the source-script execution used by this group. Several captures
    # predates the current engine (measured on L4-1_v2, where the capture and this engine disagree
    # on the anchor's position from tick 241, and the capture stops the recorded motion at a
    # different tick).  The capture is kept as a digest-referenced comparison, and any difference
    # is recorded rather than hidden.
    lifecycle_actions = [a for a in audit if a["result"]["status"] == "ok"
                         and a["action"]["type"] in ("spawn_entity", "remove_entity")]
    for tick, entity_id in sorted(expected - got):
        previous = [a for a in lifecycle_actions if a["action"]["entity_id"] == entity_id and a["tick"] < tick]
        later_spawn = [a for a in lifecycle_actions if a["action"]["entity_id"] == entity_id
                       and a["action"]["type"] == "spawn_entity" and a["tick"] >= tick]
        if previous and previous[-1]["action"]["type"] == "remove_entity":
            reason, action = "removed_by_engine", previous[-1]
        elif not previous and later_spawn:
            reason, action = "not_yet_spawned", later_spawn[0]
        else:
            raise ValueError(f"unexplained missing engine sample: {entity_id}@{tick}")
        lifecycle_rows.append({"tick": tick, "entity_id": entity_id,
                               "observation_status": reason, "action_tick": action["tick"],
                               "action_id": action["action"]["action_id"],
                               "policy": "actual engine lifecycle; no fabricated state"})
    trajectories = scoped

    # The frozen capture, compared over the same window for the same entities.
    captured = [r for r in read_rows(source_files["trajectories"])
                if start <= r["tick"] <= end and r["entity_id"] in config["entity_ids"]]
    capture_map = {(r["tick"], r["entity_id"]): r for r in captured}
    engine_map = {(r["tick"], r["entity_id"]): r for r in scoped}
    capture_disagreement = []
    for key, observed in sorted(capture_map.items()):
        row = engine_map.get(key)
        if row is None:
            capture_disagreement.append({"tick": key[0], "entity_id": key[1],
                                         "field": "sample_presence",
                                         "note": "the engine has no row here; the capture does"})
            continue
        if row["pos_enu"] != observed["pos_enu"]:
            capture_disagreement.append({"tick": key[0], "entity_id": key[1], "field": "pos_enu",
                                         "capture": observed["pos_enu"], "engine": row["pos_enu"]})
    capture_disagreement.extend(
        {"tick": k[0], "entity_id": k[1], "field": "sample_presence",
         "note": "the engine has a row here; the capture does not"}
        for k in sorted(set(engine_map) - set(capture_map)))
    capture_disagreement = capture_disagreement[:200]

    local_rule = business.profile_rule(config["episode_id"].split("__")[0])
    if local_rule is not None:
        semantic = business.evaluate(config, rows, engine.arm_script, audit)
    else:
        # The staged truth frames must be arm-authoritative over the WHOLE simulated range:
        # passing only the window-scoped rows would leave frozen factual state outside the
        # window, and the pipeline would fabricate a terminal transition from frozen data
        # right after the boundary (measured: a fabricated fall at 645/655 on L4-8).  The
        # full engine rows drive the truth-frame rebuild; the trajectories.jsonl product
        # itself stays the declared-window scope (semantic_window filters its own output).
        full_rows = [r for r in rows if r["entity_id"] in config["entity_ids"]]
        with tempfile.TemporaryDirectory(prefix="l4_arm_semantics_") as tmp:
            semantic = semantic_window(config, full_rows, Path(tmp))
    local_observation = None
    if incident_observations.supported(config):
        local_ticks, local_observation = incident_observations.evaluate(
            config, rows, engine.arm_script, audit, semantic["predicate_truth"])
    # Occurrences carry trigger_tick, outcomes terminal_tick, transitions to_tick;
    # predicate_truth and the dense L0 projection carry tick.  Filter each by its own field.
    truth_ticks = {r["truth_id"]: int(r["tick"]) for r in semantic["predicate_truth"]}
    transition_ticks = {r["transition_id"]: int(r["to_tick"])
                        for r in semantic["predicate_transitions"]}
    boundary_occurrences = []
    occ = [copy.deepcopy(r) for r in semantic["event_occurrences"]
           if start <= int(r["trigger_tick"]) <= end]
    for row in occ:
        if any(int(phase["tick"]) > end for phase in row.get("event_phases", [])):
            boundary_occurrences.append({'kind': 'event_occurrence',
                                         'scope': 'full_run_outside_window',
                                         'row': copy.deepcopy(row)})
        row["event_phases"] = [phase for phase in row.get("event_phases", [])
                               if int(phase["tick"]) <= end]
        row["supporting_transition_ids"] = [ref for ref in row.get("supporting_transition_ids", [])
                                             if transition_ticks[ref] <= end]
        row["supporting_truth_ids"] = [ref for ref in row.get("supporting_truth_ids", [])
                                       if truth_ticks[ref] <= end]
        row["source_refs"] = [ref for ref in row.get("source_refs", [])
                              if '#tick=' not in ref or
                              int(ref.split('#tick=', 1)[1].split('&', 1)[0]) <= end]
    keep = {o["event_id"] for o in occ}
    # An event detected inside the window stays in the product even when its recovery lands
    # beyond the declared boundary; that is right-censoring, and it must not be reported as
    # absence.  Only events with no in-window detection are excluded.
    out_rows = [copy.deepcopy(r) for r in semantic["event_outcomes"]
                if r.get("event_id") in keep]
    boundary_outcomes = []
    for row in out_rows:
        terminal = row.get("terminal_tick")
        if terminal is not None and int(terminal) < start:
            raise ValueError('in-window event has a terminal before the window')
        if terminal is not None and int(terminal) > end:
            boundary_outcomes.append({'kind': 'event_outcome',
                                      'scope': 'full_run_outside_window',
                                      'row': copy.deepcopy(row)})
            row["terminal_tick"] = None
            row["status"] = "right_censored"
            row["lifecycle_status"] = "observed_until_window_end"
            row["window_censor_tick"] = end
            row["supporting_truth_ids"] = [ref for ref in row.get("supporting_truth_ids", [])
                                           if truth_ticks[ref] <= end]
            if "lifecycle_phase_evidence" in row:
                row["lifecycle_phase_evidence"] = [
                    phase for phase in row["lifecycle_phase_evidence"]
                    if int(phase["end_tick"]) <= end]
    from Dataset.semantic_truth.window_outcome import arm_window_outcome, validate_l4_business_outcome
    for index, row in enumerate(out_rows):
        if row.get("schema_name") == "l4_business_event_outcome":
            validate_l4_business_outcome(row, start_tick=start, end_tick=end)
        else:
            out_rows[index] = arm_window_outcome(row, start_tick=start, end_tick=end)
    tran = [r for r in semantic["predicate_transitions"]
            if start <= int(r["to_tick"]) <= end]
    truth = [r for r in semantic["predicate_truth"] if start <= int(r["tick"]) <= end]
    dense = [r for r in semantic["l0_predicate_state"] if start <= int(r["tick"]) <= end]

    out.mkdir(parents=True)
    write_json(out / "raw/initial_state.json", snapshots[start])
    write_json(out / "raw/fork_state.json", snapshots[fork])
    write_jsonl(out / "raw/states.jsonl", trajectories)
    write_jsonl(out / "raw/trajectories.jsonl", trajectories)
    write_jsonl(out / "raw/actions.jsonl",
                [a for a in audit if start <= a["tick"] <= end])
    write_jsonl(out / "raw/lifecycle_absent.jsonl", lifecycle_rows)
    write_jsonl(out / "raw/capture_comparison.jsonl", capture_disagreement)
    write_jsonl(out / "raw/weather.jsonl",
                [r for r in engine.weather_rows if start <= r["tick"] <= end])
    write_jsonl(out / "objective/predicate_truth.jsonl", truth)
    write_jsonl(out / "objective/predicate_truth_ticks.jsonl", dense)
    write_jsonl(out / "objective/predicate_transitions.jsonl", tran)
    write_jsonl(out / "objective/event_occurrences.jsonl", occ)
    write_jsonl(out / "objective/event_outcomes.jsonl", out_rows)
    if local_observation is not None:
        write_jsonl(out / "objective/local_business_truth.jsonl", local_ticks)
        write_json(out / "objective/local_business_result.json", local_observation)

    nodes, relations = [], []
    for row in tran:
        nodes.append({"node_id": row["transition_id"], "kind": "predicate_transition",
                      "tick": row["to_tick"],
                      "evidence_ref": "objective/predicate_transitions.jsonl#" + row["transition_id"]})
    node_ids = {n["node_id"] for n in nodes}
    terminal_rows = []
    for row in out_rows:
        if local_rule:
            terminal_tick = row.get("terminal_tick")
        else:
            terminal_tick = recovery_tick_for_chain(
                row, window_start_tick=start, window_end_tick=end)
        if terminal_tick is not None:
            terminal_rows.append(row)
            nodes.append({"node_id": row["outcome_id"], "kind": "event_response_confirmed" if local_rule else "event_recovery_confirmed",
                          "within_window": True,
                          "tick": terminal_tick,
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
    for row in terminal_rows:
        if row["outcome_id"] not in node_ids:
            continue
        relations.append({"source": row["event_id"], "target": row["outcome_id"],
                          "relation_type": "has_observed_outcome", "causal_claim": False,
                          "tick": row["terminal_tick"]})
    if boundary_occurrences or boundary_outcomes or outside_evidence:
        write_json(out / "objective/window_boundary_evidence.json", {
            'scope': 'diagnostics_only_excluded_from_local_graph',
            'window': config['window'],
            'excluded_full_run_records': boundary_occurrences + boundary_outcomes,
            'unmaterialized_phase_relations': outside_evidence})
    if local_observation is not None and local_observation["onset_tick"] is not None:
        nodes.append({"node_id": local_observation["observation_id"], "kind": "local_business_observation",
                      "tick": local_observation["onset_tick"],
                      "evidence_ref": "objective/local_business_result.json"})
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
    # A family the event authority declares non-executable cannot produce a semantic event.
    # That is a terminal state with a reason, not an absence and not a missing measurement.
    declared_non_executable = local_rule is None and not incident_is_executable(
        config["episode_id"].split("__")[0])
    if declared_non_executable:
        reason = _declared_family_reason(target_family)
        status = "blocked_non_executable_family"
    else:
        reason = None
        status = ("target_event_absent_in_window" if not target_occ
                  else "right_censored" if censored else "pending" if pending
                  else "observed_response_confirmed" if local_rule else "events_recovered")
    # Each delay is the explicit increment in the dedicated plan. The executor
    # neither selects a digest-derived value nor caps it to a later action.
    rewrites = list(plan["trigger_rewrites"])
    write_json(out / "objective/window_status.json", {
        "window": config["window"], "status": status,
        "absence_is_only_within_window": True,
        "target_assessment": status,
        "local_business_observation": local_observation,
        "non_executable_family_reason": reason,
        "declared_family_policy": (
            business.contract()["authority_scope"] if local_rule else
            "Formal ontology executability is read from its authority; unavailable families "
            "are recorded with the stated reason, never inferred from expected outcomes."),
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
        "applied_trigger_rewrites": rewrites,
        "expected": config["expected"],
    })
    from Dataset.tools.arm_ue_export import write_engine_inputs
    write_engine_inputs(out, engine, engine.arm_interpreter, config["window"], script=engine.arm_script)
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
        "business_semantic_authority": ({"path": str(business.RULES.relative_to(ROOT)),
                                         "scope": business.contract()["authority_scope"]} if local_rule else None),
        "window_status": status,
        "provenance": {
            "parameter_sources": config["provenance"]["parameter_sources"],
            "applied_trigger_rewrites": rewrites,
            "trigger_rewrite_semantics": "delay_ticks is added to the source trigger threshold by the explicit dedicated plan",
        },
        "factual_source_audit": {
            "factual_source": "current_engine_factual_source_execution",
            "generation_method": ("execute_source_script_from_tick_zero_then_apply_exact_config_"
                                  "actions_no_snapshot_restore"),
            "historical_capture_role": "comparison_reference_only_never_the_product",
            "capture_vs_engine_disagreement_count": len(capture_disagreement),
            "capture_vs_engine_disagreement_sample": capture_disagreement[:20],
            "prefix_position_mismatch_ticks": sorted(set(position_mismatch)),
            "prefix_state_mismatch_ticks": sorted(set(prefix_mismatch)),
            "role": "factual_source" if config["mode"] == "factual_observation"
                    else "intervention_prefix_must_match_factual",
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

    print(json.dumps({"arm_id": result["arm_id"], "window_status": result["window_status"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
