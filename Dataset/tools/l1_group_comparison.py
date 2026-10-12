"""从 L1 实跑窗口生成目标绑定、动作效果和同组 factual 对照。"""
from __future__ import annotations

import json
import math
from pathlib import Path

from Dataset.tools.l1_window import (
    ROOT, TICK_HZ, load_config, read_json, read_rows, write_json,
)


def action_observations(config: dict, experiment: dict, actions: list, states: dict) -> list:
    """运动只归属到下一条抢占命令；名义终点与实测完成分开保存。"""
    end = config["window"]["end_tick"]
    observations = []
    for action_id in experiment["related_action_ids"]:
        matches = [(i, r) for i, r in enumerate(actions) if r["action"]["action_id"] == action_id]
        if not matches:
            observations.append({"action_id": action_id, "status": "not_dispatched"})
            continue
        for index, row in matches:
            action = row["action"]
            tick, entity = row["tick"], action["entity_id"]
            observed = {"action_id": action_id, "entity_id": entity, "dispatch_tick": tick,
                        "action_type": action["type"], "dispatch_result": row["result"]["status"]}
            if row["result"]["status"] == "omitted_by_explicit_intervention":
                observed["status"] = "omitted"
            elif action["type"] != "move_entity":
                observed["status"] = "applied"
                observed["next_sample"] = states[(tick + 1, entity)] if tick < end else None
            else:
                schedule = row["created_schedule"]
                position = schedule["start_pos_enu"]
                duration = 0
                for waypoint in schedule["waypoints_enu_m"]:
                    distance = math.dist(position, waypoint)
                    if distance > 1e-6:
                        duration += max(1, math.ceil(distance / schedule["velocity_mps"] * TICK_HZ))
                    position = waypoint
                nominal_end = schedule["tick"] + duration
                successors = [r for r in actions[index + 1:]
                              if r["action"].get("entity_id") == entity
                              and r["result"]["status"] == "ok"
                              and (r["action"]["type"] == "move_entity"
                                   or (r["action"]["type"] == "set_visual_state"
                                       and r["action"]["visual_state"]["mode"] in ("hold", "hover")))]
                successor = successors[0] if successors else None
                interrupted = successor is not None and successor["tick"] < nominal_end
                observed_end = min(end, successor["tick"] if interrupted else nominal_end)
                moving = [t for t in range(tick + 1, observed_end + 1)
                          if any(abs(v) > 1e-6 for v in states[(t, entity)]["vel_mps"])]
                observed.update({"uninterrupted_end_tick": nominal_end,
                                 "first_motion_tick_before_takeover": moving[0] if moving else None,
                                 "last_observed_tick": observed_end})
                if interrupted:
                    observed.update({"status": "interrupted", "takeover_tick": successor["tick"],
                                     "takeover_action_id": successor["action"]["action_id"]})
                elif nominal_end <= end:
                    actual = states[(nominal_end, entity)]["pos_enu"]
                    if math.dist(actual, position) > 1e-5:
                        raise ValueError(f"{config['group_key']}/{action_id}: movement endpoint differs")
                    observed.update({"status": "completed", "completion_tick": nominal_end,
                                     "terminal_position_enu_m": actual})
                else:
                    observed["status"] = "ongoing_at_window_end"
            observations.append(observed)
    return observations


def arm_evidence(config: dict, experiment: dict) -> tuple[dict, dict]:
    arm = ROOT / config["output"]
    target = experiment["target"]
    dense = [r for r in read_rows(arm / "objective/predicate_truth_ticks.jsonl")
             if r["predicate_id"] == target["predicate_id"] and r["bindings"] == target["bindings"]]
    start, end = (config["window"][k] for k in ("start_tick", "end_tick"))
    if [r["tick"] for r in dense] != list(range(start, end + 1)):
        raise ValueError(f"{config['output']}: target observations do not cover the window")
    if any(r["value"] not in ("true", "false") for r in dense):
        raise ValueError(f"{config['output']}: target has unobserved samples")
    if dense[0]["value"] != "false":
        raise ValueError(f"{config['output']}: repaired window lacks the selected event's pre-onset state")
    events = [r for r in read_rows(arm / "objective/event_occurrences.jsonl")
              if r["event_family_id"] == config["event_family"] and r["bindings"] == target["bindings"]]
    outcomes = {r["event_id"]: r for r in read_rows(arm / "objective/event_outcomes.jsonl")}
    true_ticks = [r["tick"] for r in dense if r["value"] == "true"]
    mission = set(config["entity_ids"])
    # Mission-attributable overlap: true samples where at least two of the group's
    # mission aircraft occupy the corridor together. Background users are present in
    # both branches; only mission-UAV overlap is the intervention's target cause.
    mission_overlap_ticks = [
        r["tick"] for r in dense if r["value"] == "true"
        and len(mission.intersection(r.get("occupied_aircraft_ids") or [])) >= 2]
    actions = list(read_rows(arm / "raw/actions.jsonl"))
    states = {(r["tick"], r["entity_id"]): r for r in read_rows(arm / "raw/ue_trajectories.jsonl")}
    observations = action_observations(config, experiment, actions, states)
    if any(r["status"] == "ongoing_at_window_end" for r in observations):
        raise ValueError(f"{config['output']}: related action is still running at the window end")
    result = {
        "arm_id": config["arm_id"], "mode": config["mode"],
        "window_status": read_json(arm / "objective/window_status.json")["status"],
        "target_true_tick_count": len(true_ticks),
        "target_first_true_tick": true_ticks[0] if true_ticks else None,
        "target_last_true_tick": true_ticks[-1] if true_ticks else None,
        "target_mission_overlap_tick_count": len(mission_overlap_ticks),
        "target_mission_overlap_first_tick": mission_overlap_ticks[0] if mission_overlap_ticks else None,
        "target_value_at_end": dense[-1]["value"],
        "target_events": [{"event_id": r["event_id"], "trigger_tick": r["trigger_tick"],
                           "outcome_status": outcomes[r["event_id"]]["status"],
                           "terminal_tick": outcomes[r["event_id"]]["terminal_tick"],
                           "window_censor_tick": outcomes[r["event_id"]].get("window_censor_tick")}
                          for r in events],
        "actions": observations,
        "manifest": {"path": str((arm / "manifest.json").relative_to(ROOT))},
        "config": {"path": str((ROOT / "interventions" / config["group_key"] /
                                 (config["arm_id"] + ".yaml")).relative_to(ROOT))},
    }
    return result, states


def build(group_key: str) -> dict:
    configs = [load_config(p) for p in sorted((ROOT / "interventions" / group_key).glob("arm_*.yaml"))]
    factual = next(c for c in configs if c["mode"] == "factual_observation")
    branch = next(c for c in configs if c["mode"] == "intervention")
    experiment_path = ROOT / factual["provenance"]["experiment_ref"]
    experiment = read_json(experiment_path)
    if factual["window"] != branch["window"] or factual["window"] != experiment["window"]:
        raise ValueError(f"{group_key}: group windows differ")
    if factual["window"]["end_tick"] < experiment["window_basis"]["required_end_tick"]:
        raise ValueError(f"{group_key}: selected process ends after the observation window")
    f, fstates = arm_evidence(factual, experiment)
    b, bstates = arm_evidence(branch, experiment)
    fork = factual["window"]["fork_tick"]
    prefix_f = {k: v for k, v in fstates.items() if k[0] <= fork}
    prefix_b = {k: v for k, v in bstates.items() if k[0] <= fork}
    if prefix_f != prefix_b:
        raise ValueError(f"{group_key}: factual and intervention do not share the sampled prefix")
    changed = sorted({k[0] for k in fstates.keys() & bstates.keys() if k[0] > fork
                      and any(fstates[k][name] != bstates[k][name]
                              for name in ("pos_enu", "vel_mps", "state", "activity_type"))})
    delta = b["target_true_tick_count"] - f["target_true_tick_count"]
    metric = experiment["hypothesis"]["metric"]
    if metric == "target_event_absent":
        supported = len(b["target_events"]) == 0 and b["target_true_tick_count"] == 0
    elif metric == "target_true_tick_count_increases":
        supported = delta > 0
    elif metric == "target_true_tick_count_decreases":
        supported = delta < 0
    elif metric == "target_first_true_tick_postpones":
        supported = (b["target_first_true_tick"] is not None
                     and f["target_first_true_tick"] is not None
                     and b["target_first_true_tick"] > f["target_first_true_tick"])
    elif metric == "target_mission_overlap_absent":
        supported = (b["target_mission_overlap_tick_count"] == 0
                     and f["target_mission_overlap_tick_count"] > 0)
    elif metric == "target_mission_overlap_increases":
        supported = b["target_mission_overlap_tick_count"] > f["target_mission_overlap_tick_count"]
    else:
        raise ValueError(f"Unsupported L1 comparison metric: {metric}")
    return {
        "schema": "aeroworld_l1_group_comparison", "version": 1,
        "group_key": group_key, "episode_id": factual["episode_id"],
        "window": factual["window"], "target": experiment["target"],
        "baseline_arm_id": factual["arm_id"], "intervention_arm_id": branch["arm_id"],
        "arms": {f["arm_id"]: f, b["arm_id"]: b},
        "shared_prefix_equal": True,
        "motion_changed_tick_count": len(changed),
        "first_motion_difference_tick": changed[0] if changed else None,
        "target_true_tick_count_delta": delta,
        "target_mission_overlap_tick_count": {
            f["arm_id"]: f["target_mission_overlap_tick_count"],
            b["arm_id"]: b["target_mission_overlap_tick_count"]},
        "hypothesis": experiment["hypothesis"],
        "hypothesis_result": "supported" if supported else "falsified",
        "interpretation_scope": "所选绑定在共同观察窗内的实测比较；负结果保留，不将动作受理或位移变化等同于假设成立。",
        "experiment": {"path": str(experiment_path.relative_to(ROOT))},
        "generator": {"path": "Dataset/tools/l1_group_comparison.py"},
    }


def main() -> int:
    groups = sorted({p.parent.relative_to(ROOT / "interventions").as_posix()
                     for p in (ROOT / "interventions").glob("L1-*/seed*/grp_*/arm_factual.yaml")})
    for group in groups:
        write_json(ROOT / "arms" / group / "comparison.json", build(group))
    print(json.dumps({"groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
