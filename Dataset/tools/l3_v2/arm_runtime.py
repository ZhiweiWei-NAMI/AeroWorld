"""Single-arm runtime for the L3 v2 workflow.

One arm = one dedicated Python entry under `scripts/arms/`.  The entry declares its
own window, its own mechanism (exact trigger retime, exact runtime-state override,
exact action omissions, exact injected actions) and its own hypothesis.  This module
only carries out what the entry declares; it never derives a mechanism from an arm
id, an intent name or a family name.

Modes
  factual_observation  current engine, no intervention, cropped to the arm window
  intervention         current engine plus the declared mechanism

`factual_observation` here is the CURRENT ENGINE's own zero-intervention run, as the
user requires: every predicate, transition, occurrence and outcome in the window is
recomputed by the repository objective pipeline from that run's self-consistent state.
The historical capture is never used as the factual baseline.
"""
from __future__ import annotations

import copy
import datetime
import json
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from Dataset.semantic_truth.outcome_projection import source_index, validated_recovery_tick
from Dataset.tools.l3_v2.common import (
    CAPTURE, ROOT, SEMANTIC, digest, file_digest, grid_up, load, project_path,
    rows as read_rows, scene_path, script_path, write, write_rows,
)
from Dataset.tools.l3_v2.engine import execute, snapshot
from Dataset.tools.l3_v2.semantics import evaluate


@dataclass
class ArmSpec:
    """Everything one dedicated L3 arm entry declares about itself."""
    profile: str
    seed: str
    arm_id: str
    mode: str
    event_family: str
    target_event_id: str
    fork_event_id: str
    window_start_tick: int
    window_fork_tick: int
    window_end_tick: int
    primary_predicate: str
    entity_ids: tuple[str, ...]
    role: str = ""
    mechanism: Mapping[str, Any] = field(default_factory=dict)
    retime: Mapping[str, Any] | None = None
    state_overrides: tuple[Mapping[str, Any], ...] = ()
    omit_action_ids: tuple[str, ...] = ()
    inject_actions: tuple[Mapping[str, Any], ...] = ()
    hypothesis: str = ""
    falsified_if: str = ""
    window_basis: str = ""
    group_suffix: str = ""
    sumo_intervention: Mapping[str, Any] | None = None

    @property
    def episode_id(self) -> str:
        return f"{self.profile}__{self.seed}"

    @property
    def group_key(self) -> str:
        suffix = f"_{self.group_suffix}" if self.group_suffix else ""
        return f"{self.profile}/{self.seed}/grp_{self.fork_event_id}__{self.event_family}{suffix}"

    @property
    def output(self) -> str:
        return f"arms/{self.group_key}/{self.arm_id}"

    def window(self) -> dict:
        return {"start_tick": self.window_start_tick, "fork_tick": self.window_fork_tick,
                "end_tick": self.window_end_tick,
                "phase": "after_state_sample_before_actions",
                "intervention_order": "source_actions_then_intervention"}


def run_arm(spec: ArmSpec) -> dict:
    out = project_path_allow_new(spec.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing ARM: {out}")
    try:
        return _run_arm(spec)
    except Exception:
        if out.exists():
            shutil.rmtree(out)
        raise


def _run_arm(spec: ArmSpec) -> dict:
    if spec.mode not in ("factual_observation", "intervention"):
        raise ValueError(f"Unsupported ARM mode: {spec.mode}")
    out = project_path_allow_new(spec.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing ARM: {out}")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.perf_counter()
    out.mkdir(parents=True)

    scene, script = scene_path(spec.profile), script_path(spec.profile)
    if spec.mode == "factual_observation":
        run = execute(scene, script, spec.episode_id, end_tick=900,
                      snapshot_ticks=(spec.window_start_tick, spec.window_fork_tick))
        method = "current_engine_no_intervention_factual_baseline"
    else:
        run = execute(scene, script, spec.episode_id, end_tick=900,
                      snapshot_ticks=(spec.window_start_tick, spec.window_fork_tick),
                      retime=spec.retime, state_overrides=spec.state_overrides,
                      omit_action_ids=spec.omit_action_ids,
                      injections=spec.inject_actions)
        method = "current_engine_with_declared_mechanism"

    trajectories = [r for r in run["engine"].trajectory_rows
                    if str(r["entity_id"]) in set(spec.entity_ids)]
    roster = {eid: dict(e) for eid, e in run["engine"].entities.items()}
    window = spec.window()
    observed = [r for r in trajectories if window["start_tick"] <= r["tick"] <= window["end_tick"]]
    sumo_frames = None
    if spec.sumo_intervention is not None:
        from Dataset.tools.l3_v2.sumo_branch import execute_sumo_branch
        sumo_frames = execute_sumo_branch(spec, out)
    signature = digest({"episode": spec.episode_id,
        "trajectories": run["engine"].trajectory_rows, "weather": run["engine"].weather_rows,
        "entities": roster, "sumo": file_digest(sumo_frames) if sumo_frames else None,
        "scene": file_digest(project_path(scene)), "script": file_digest(project_path(script)),
        "semantics": file_digest(ROOT / "Dataset/tools/l3_v2/semantics.py"),
        "pipeline": file_digest(ROOT / "Dataset/semantic_truth/objective_pipeline.py"),
        "sumo_adapter": file_digest(ROOT / "Dataset/tools/l3_v2/sumo_branch.py"),
        "capture": file_digest(CAPTURE / spec.episode_id / "truth_frames.jsonl"),
        "semantic_authorities": {str(p.relative_to(ROOT)): file_digest(p)
            for parent in ("Dataset/semantic_rules", "Dataset/semantic_simulation", "Dataset/semantic_truth", "Dataset/knowledge_graph")
            for p in sorted((ROOT / parent).rglob("*")) if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".ttl"}}})
    semantic_workdir = ROOT / "design/L3_repair/sources" / signature.split(":")[1]
    if (semantic_workdir / "computed.json").is_file():
        computed = load(semantic_workdir / "computed.json")
    else:
        computed = evaluate(spec.episode_id, run["engine"], run["engine"].trajectory_rows,
                            roster, workdir=semantic_workdir, sumo_frames=sumo_frames)
    start, fork, end = window["start_tick"], window["fork_tick"], window["end_tick"]
    occ, outcomes, initial_events = window_events(computed, start, end)
    transitions = [r for r in computed["predicate_transitions"] if start <= int(r["to_tick"]) <= end]
    truth = [r for r in computed["predicate_truth"] if start <= int(r["tick"]) <= end]
    dense = [r for r in computed["predicate_truth_ticks"] if start <= int(r["tick"]) <= end]
    status = window_status(spec, occ, outcomes)
    nodes, relations = chain(
        transitions, occ, outcomes,
        source_truth=computed["predicate_truth"],
        source_transitions=computed["predicate_transitions"],
        window_start_tick=start, window_end_tick=end)

    audit = [a for a in run["audit"] if start <= a["tick"] <= end]
    snapshots = {}
    for tick in (start, fork):
        snapshots[str(tick)] = run["snapshots"][tick]
    write(out / "raw/fork_state.json", snapshots[str(fork)])
    write(out / "raw/initial_state.json", snapshots[str(start)])
    write_rows(out / "raw/trajectories.jsonl", observed)
    write_rows(out / "raw/states.jsonl", observed)
    write(out / "raw/source.json", {"materialized_source": computed["materialized_source"],
          "source_signature": signature, "semantic_sampling": "authoritative_capture_grid",
          "execution_end_tick": 900})
    write_rows(out / "raw/actions.jsonl", audit)
    from Dataset.tools.arm_ue_export import write_engine_inputs
    write_engine_inputs(out, run["engine"], run["interpreter"], window, script=run["interpreter"].script)
    weather = [r for r in run["engine"].weather_rows if start <= r["tick"] <= end]
    write_rows(out / "raw/weather.jsonl", weather)
    write(out / "raw/final_schedules.json",
          {eid: run["engine"].entities[eid].get("schedules", []) for eid in spec.entity_ids})

    write(out / "objective/initial_events.json", initial_events)
    write_rows(out / "objective/event_occurrences.jsonl", occ)
    from Dataset.semantic_truth.window_outcome import arm_window_outcome
    write_rows(out / "objective/event_outcomes.jsonl", [
        arm_window_outcome(row, start_tick=start, end_tick=end) for row in outcomes])
    write_rows(out / "objective/predicate_transitions.jsonl", transitions)
    write_rows(out / "objective/predicate_truth.jsonl", truth)
    write_rows(out / "objective/predicate_truth_ticks.jsonl", dense)

    write(out / "objective/window_status.json", status)
    write_rows(out / "chain/nodes.jsonl", nodes)
    write_rows(out / "chain/relations.jsonl", relations)

    manifest = build_manifest(spec, run, computed, status, out, method, started, clock)
    manifest["materialized_source"] = computed["materialized_source"]
    manifest["semantic_sampling"] = "authoritative_capture_grid"
    manifest["sumo_intervention"] = spec.sumo_intervention
    write(out / "manifest.json", manifest)
    return {"arm_id": spec.arm_id, "group": spec.group_key, "mode": spec.mode,
            "status": status["status"], "events": status["event_family_counts"],
            "elapsed_s": round(time.perf_counter() - clock, 2)}


def window_events(computed: Mapping[str, Any], start: int, end: int) -> tuple[list, list, dict]:
    """Observe full-source lifecycles intersecting the closed window.

    An inherited occurrence keeps its original identity and onset. Its initial
    state references measured pre-window evidence; no new rising edge is made.
    """
    full_outcomes = {row["event_id"]: row for row in computed["event_outcomes"]}
    occ = [row for row in computed["event_occurrences"]
           if int(row["trigger_tick"]) <= end
           and (full_outcomes[row["event_id"]]["terminal_tick"] is None
                or int(full_outcomes[row["event_id"]]["terminal_tick"]) >= start)]
    keep = {o["event_id"] for o in occ}
    truth_ticks = {r["truth_id"]: int(r["tick"]) for r in computed["predicate_truth"]}
    transition_ticks = {r["transition_id"]: int(r["to_tick"]) for r in computed["predicate_transitions"]}
    outcomes = []
    for original in computed["event_outcomes"]:
        if original.get("event_id") not in keep:
            continue
        row = copy.deepcopy(original)
        if row.get("terminal_tick") is not None and int(row["terminal_tick"]) > end:
            row["terminal_tick"] = None
            row["status"] = "right_censored"
            row["window_censor_tick"] = end
            row["lifecycle_status"] = "observed_until_window_end"
        row["supporting_truth_ids"] = [ref for ref in row.get("supporting_truth_ids", []) if truth_ticks[ref] <= end]
        row["lifecycle_phase_evidence"] = [phase for phase in row.get("lifecycle_phase_evidence", [])
                                           if phase.get("end_tick", end) <= end]
        outcomes.append(row)
    occ = [copy.deepcopy(r) for r in occ]
    for row in occ:
        row["event_phases"] = [phase for phase in row.get("event_phases", []) if phase["tick"] <= end]
        row["supporting_truth_ids"] = [ref for ref in row.get("supporting_truth_ids", []) if truth_ticks[ref] <= end]
        row["supporting_transition_ids"] = [ref for ref in row.get("supporting_transition_ids", []) if transition_ticks[ref] <= end]
    source = str(Path(computed["materialized_source"]).parent / "computed.json")
    initial = {
        "tick": start,
        "policy": "inherited_source_lifecycle_not_a_new_window_trigger",
        "source": source,
        "events": [{"event_id": row["event_id"], "event_family_id": row["event_family_id"],
                    "bindings": row["bindings"], "onset_tick": row["trigger_tick"],
                    "status_at_start": "recovery_confirmed_at_start"
                    if full_outcomes[row["event_id"]]["terminal_tick"] == start else "ongoing",
                    "supporting_truth_ids": [ref for ref in row["supporting_truth_ids"]
                                             if truth_ticks[ref] <= start],
                    "supporting_transition_ids": [ref for ref in row["supporting_transition_ids"]
                                                  if transition_ticks[ref] <= start]}
                   for row in occ if row["trigger_tick"] < start],
    }
    return occ, outcomes, initial


def project_path_allow_new(ref: str) -> Path:
    path = (ROOT / ref).resolve()
    path.relative_to(ROOT)
    return path


def window_status(spec: ArmSpec, occurrences: Sequence[Mapping[str, Any]],
                  outcomes: Sequence[Mapping[str, Any]]) -> dict:
    outcome_by_id = {o["event_id"]: o for o in outcomes}
    pending = [o for o in occurrences
               if outcome_by_id[o["event_id"]]["terminal_tick"] is None]
    if pending:
        status = "right_censored"
    elif occurrences:
        status = "events_recovered"
    else:
        status = "no_event_in_window_fully_observed"
    target = [o for o in occurrences if o["event_family_id"] == spec.event_family]
    return {
        "window": spec.window(),
        "status": status,
        "absence_is_only_within_window": True,
        "event_count": len(occurrences),
        "inherited_event_ids": [o["event_id"] for o in occurrences
                                if o["trigger_tick"] < spec.window_start_tick],
        "new_event_ids": [o["event_id"] for o in occurrences
                          if o["trigger_tick"] >= spec.window_start_tick],
        "event_family_counts": dict(sorted(Counter(o["event_family_id"] for o in occurrences).items())),
        "target_family": spec.event_family,
        "target_events": [{"event_id": o["event_id"], "onset_tick": o["trigger_tick"],
                           "terminal_tick": outcome_by_id[o["event_id"]]["terminal_tick"],
                           "status": outcome_by_id[o["event_id"]]["status"]}
                          for o in target],
        "pending_event_ids": [o["event_id"] for o in pending],
    }


def chain(transitions: Sequence[Mapping[str, Any]], occurrences: Sequence[Mapping[str, Any]],
          outcomes: Sequence[Mapping[str, Any]], *, window_start_tick: int,
          window_end_tick: int, source_truth: Sequence[Mapping[str, Any]],
          source_transitions: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    relations: list[dict] = []
    occurrence_by_id = source_index(occurrences, "event_id")
    truth_by_id = source_index(source_truth, "truth_id")
    transition_by_id = source_index(source_transitions, "transition_id")
    for transition in transitions:
        nodes.append({"node_id": transition["transition_id"], "kind": "predicate_transition",
                      "tick": transition["to_tick"],
                      "evidence_ref": "objective/predicate_transitions.jsonl#"
                                      + transition["transition_id"]})
    ids = {n["node_id"] for n in nodes}
    for occurrence in occurrences:
        nodes.append({"node_id": occurrence["event_id"], "kind": "objective_event",
                      "family": occurrence["event_family_id"],
                      "tick": occurrence["trigger_tick"],
                      "evidence_ref": "objective/event_occurrences.jsonl#" + occurrence["event_id"]})
        for phase in occurrence.get("event_phases") or []:
            if phase["transition_id"] not in ids:
                continue
            relations.append({"source": phase["transition_id"], "target": occurrence["event_id"],
                              "relation_type": "predicate_" + phase["phase_kind"] + "_evidence",
                              "tick": phase["tick"], "causal_claim": False,
                              "rule_id": occurrence["rule_id"]})
    for outcome in outcomes:
        recovery_tick = validated_recovery_tick(
            outcome, occurrence=occurrence_by_id.get(outcome.get("event_id")),
            truth_by_id=truth_by_id, transition_by_id=transition_by_id,
            window_start_tick=window_start_tick, window_end_tick=window_end_tick)
        if recovery_tick is None:
            continue
        nodes.append({"node_id": outcome["outcome_id"], "kind": "event_recovery_confirmed",
                      "tick": recovery_tick,
                      "evidence_ref": "objective/event_outcomes.jsonl#" + outcome["outcome_id"]})
        relations.append({"source": outcome["event_id"], "target": outcome["outcome_id"],
                          "relation_type": "has_observed_outcome", "causal_claim": False,
                          "tick": recovery_tick})
    return (sorted(nodes, key=lambda r: (r["tick"], r["node_id"])),
            sorted(relations, key=lambda r: (r["tick"], r["source"], r["target"])))


def build_manifest(spec: ArmSpec, run: Mapping[str, Any], computed: Mapping[str, Any],
                   status: Mapping[str, Any], out: Path, method: str,
                   started: str, clock: float) -> dict:
    return {
        "schema": "aeroworld_l3_v2_arm",
        "version": 1,
        "arm_id": spec.arm_id,
        "group_key": spec.group_key,
        "episode_id": spec.episode_id,
        "profile": spec.profile,
        "seed": spec.seed,
        "mode": spec.mode,
        "event_family": spec.event_family,
        "target_event_id": spec.target_event_id,
        "fork_event_id": spec.fork_event_id,
        "window": spec.window(),
        "window_basis": spec.window_basis,
        "primary_predicate": spec.primary_predicate,
        "entity_ids": sorted(set(spec.entity_ids)),
        "role": spec.role,
        "mechanism": copy.deepcopy(dict(spec.mechanism)),
        "retime": copy.deepcopy(spec.retime),
        "state_overrides": copy.deepcopy(list(spec.state_overrides)),
        "omit_action_ids": sorted(spec.omit_action_ids),
        "injected_action_ids": [a.get("action_id") for a in spec.inject_actions],
        "hypothesis": spec.hypothesis,
        "falsified_if": spec.falsified_if,
        "generation_method": method,
        "factual_baseline": "current_engine_factual_execution",
        "historical_capture_role": "comparison_only_not_the_baseline",
        "semantic_authority": "Dataset.tools.l3_v2.semantics.measured_semantic_rows",
        "window_status": status,
        "event_occurrences": status["event_count"],
        "event_outcomes": status["event_count"],
        "predicate_transitions": sum(spec.window_start_tick <= row["to_tick"] <= spec.window_end_tick
                                     for row in computed["predicate_transitions"]),
        "sources": {
            "scene": {"path": scene_path(spec.profile), "digest": file_digest(project_path(scene_path(spec.profile)))},
            "script": {"path": script_path(spec.profile), "digest": file_digest(project_path(script_path(spec.profile)))},
            "capture": {"path": f"{CAPTURE.relative_to(ROOT)}/{spec.episode_id}", "role": "authority_for_SUMO_background_and_static_identity"},
            "historical_semantic": {"path": f"{SEMANTIC.relative_to(ROOT)}/{spec.episode_id}", "role": "comparison_only"},
        },
        "implementation": {
            "Dataset/tools/l3_v2/engine.py": file_digest(ROOT / "Dataset/tools/l3_v2/engine.py"),
            "Dataset/tools/l3_v2/semantics.py": file_digest(ROOT / "Dataset/tools/l3_v2/semantics.py"),
            "Dataset/tools/l3_v2/arm_runtime.py": file_digest(ROOT / "Dataset/tools/l3_v2/arm_runtime.py"),
            "Dataset/tools/batch_generate.py": file_digest(ROOT / "Dataset/tools/batch_generate.py"),
            "Plugins/SumoImporter/Scripts/donghu_core/event_script_interpreter.py":
                file_digest(ROOT / "Plugins/SumoImporter/Scripts/donghu_core/event_script_interpreter.py"),
            "Dataset/semantic_truth/objective_pipeline.py":
                file_digest(ROOT / "Dataset/semantic_truth/objective_pipeline.py"),
            "Dataset/semantic_simulation/predicate_state_computers.py":
                file_digest(ROOT / "Dataset/semantic_simulation/predicate_state_computers.py"),
            "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json":
                file_digest(ROOT / "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json"),
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "started_utc": started, "elapsed_s": round(time.perf_counter() - clock, 2)},
        "limitations": [
            "local deterministic engine; SUMO/global traffic outside the declared entity scope is referenced, not re-simulated",
            "historical capture is retained as source evidence",
            "no UE RGB/depth/segmentation claim",
        ],
    }
