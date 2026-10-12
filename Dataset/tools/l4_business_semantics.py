"""Observed local physical responses for the L4 ARM experiments.

This contract is deliberately separate from the formal ontology families that have no
executable lifecycle authority.  Every value is recomputed from this run's measured
positions and typed runtime state; a declared flag is never accepted as physical evidence.

Two families of value live here:

* ``uav_pedestrian`` and ``uav_vehicle`` read the UAV-pedestrian / UAV-vehicle geometry
  directly from trajectories.  Height, separation, contact, clearance and stop are
  measurements; the only authored inputs are the source trigger thresholds.
* ``ambulance`` reads priority passage as a geometric progression.  Acknowledging the
  request and completing the passage are distinct: completion requires the measured
  crossing, and the yield clearance is carried as passage history so a civilian that has
  already finished yielding cannot erase a passage that actually happened.

``intersection`` and ``av`` are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "Dataset/semantic_rules/profiles/l4_arm_business_semantics.json"


def contract() -> dict:
    return json.loads(RULES.read_text())


def profile_rule(profile: str) -> dict | None:
    return contract()["profiles"].get(profile)


def identifier(kind: str, *parts) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return kind + ":" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def speed(row: dict) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in row["vel_mps"]))


def typed_bool(row: dict, key: str) -> bool:
    value = row["vehicle_state"][key]
    if type(value) is not bool:
        raise ValueError(f"{row['entity_id']}@{row['tick']}: {key} is not boolean")
    return value


def flag(row: dict, family: str, key: str) -> bool:
    """A required typed boolean from a named runtime-state family.

    A missing family or field is an explicit error: a predicate may not silently read
    ``None`` as false when the authored state was never observed.
    """
    payload = row.get(family)
    if not isinstance(payload, dict) or key not in payload:
        raise ValueError(f"{row['entity_id']}@{row['tick']}: {family}.{key} is absent")
    value = payload[key]
    if type(value) is not bool:
        raise ValueError(f"{row['entity_id']}@{row['tick']}: {family}.{key} is not boolean")
    return value


def _roster_ground_reference(config: dict, entity_id: str) -> float:
    """The UAV's declared local ground origin; height is measured against it."""
    ref = config["sources"].get("roster")
    if not ref:
        raise ValueError("a height-based predicate requires the episode roster source")
    payload = json.loads((ROOT / ref).read_text())
    entries = payload["entities"] if isinstance(payload, dict) else payload
    if isinstance(entries, dict):
        entry = entries.get(entity_id)
    else:
        entry = next((e for e in entries if e.get("entity_id") == entity_id), None)
    if entry is None:
        raise ValueError(f"roster has no entry for {entity_id}")
    value = entry.get("ground_reference_z_m")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{entity_id}: roster lacks a numeric ground_reference_z_m")
    return float(value)


def _roles(labels: dict[str, str], mechanism: str) -> dict:
    """Resolve the mechanism's typed roles from measured label classes, never from ids."""
    aircraft = sorted(e for e, c in labels.items() if c == "uav")
    vehicles = sorted(e for e, c in labels.items() if c == "vehicle")
    pedestrians = sorted(e for e, c in labels.items() if c == "pedestrian")
    if mechanism in ("uav_pedestrian", "uav_vehicle"):
        if len(aircraft) != 1:
            raise ValueError(f"{mechanism} requires exactly one aircraft, got {aircraft}")
        others = pedestrians if mechanism == "uav_pedestrian" else vehicles
        if len(others) != 1:
            raise ValueError(f"{mechanism} requires exactly one counterpart, got {others}")
        return {"aircraft": aircraft[0], "counterpart": others[0]}
    if mechanism == "ambulance":
        if len(vehicles) != 2:
            raise ValueError(f"ambulance passage requires two vehicles, got {vehicles}")
        return {"vehicles": vehicles}
    if mechanism == "intersection":
        if len(vehicles) != 2:
            raise ValueError(f"intersection requires two vehicles, got {vehicles}")
        return {"vehicles": vehicles}
    if len(vehicles) != 1:
        raise ValueError(f"AV safe-stop requires exactly one vehicle, got {vehicles}")
    return {"vehicles": vehicles}


def _source_proximity(script: dict, event_id: str, entities: set[str]) -> dict:
    event = next(e for e in script["events"] if e["event_id"] == event_id)
    proximity = next(t for t in script["triggers"] if t["trigger_id"] == event["trigger_ref"])
    if proximity["type"] != "entity_proximity" or proximity["operator"] != "lte":
        raise ValueError(f"{event_id}: unsupported source proximity contract")
    if {proximity["entity_a"], proximity["entity_b"]} != entities:
        raise ValueError(f"{event_id}: source proximity binding differs from the ARM entity scope")
    return proximity


def _approach_geometry(script: dict, config: dict, audit: list[dict]) -> tuple:
    """The authored priority-approach direction, its normal, and the real dispatch tick."""
    event = next(e for e in script["events"] if e["event_id"] == "ambulance_priority_approach")
    if event["max_fire_count"] != 1:
        raise ValueError("Business response contract requires one authored approach instance")
    move = next(a for a in event["actions"] if a["type"] == "move_entity")
    executed = [a for a in audit if a["origin"] == "source_script"
                and a["action"]["action_id"] == move["action_id"] and a["result"]["status"] == "ok"]
    if len(executed) > 1:
        raise ValueError("Approach action executed more than once")
    dispatch = executed[0]["tick"] if executed else None
    points = move["waypoints_enu_m"]
    delta = next([b[i] - a[i] for i in range(2)] for a, b in zip(points, points[1:]) if a[:2] != b[:2])
    length = math.hypot(*delta)
    direction = [v / length for v in delta]
    normal = [-direction[1], direction[0]]
    return move["action_id"], dispatch, direction, normal


def evaluate(config: dict, trajectories: list[dict], script: dict, audit: list[dict]) -> dict:
    authority = contract()
    profile = config["episode_id"].split("__")[0]
    rule = authority["profiles"][profile]
    if config["predicate_ids"] != rule["predicate_ids"] or config["event_family"] != rule["event_family"]:
        raise ValueError("L4 business configuration disagrees with its executable authority")
    entities = set(config["entity_ids"])
    labels: dict[str, str] = {}
    by_tick: dict[int, dict] = {}
    for row in trajectories:
        if row["entity_id"] not in entities:
            continue
        labels.setdefault(row["entity_id"], row["label_class"])
        bucket = by_tick.setdefault(int(row["tick"]), {})
        if row["entity_id"] in bucket:
            raise ValueError("Duplicate business-state sample")
        bucket[row["entity_id"]] = row
    mechanism = rule["mechanism"]
    roles = _roles(labels, mechanism)
    thresholds = authority["geometric_thresholds"]
    tolerance = authority["stationary_speed_tolerance_mps"]
    step = authority["capture_step_ticks"]
    digest = "sha256:" + hashlib.sha256(RULES.read_bytes()).hexdigest()

    proximity = None
    approach_action_id = approach_dispatch = None
    direction = normal = None
    if mechanism in ("intersection", "ambulance"):
        event_id = "vehicle_collision_warning" if mechanism == "intersection" else "civilian_vehicle_yields"
        proximity = _source_proximity(script, event_id, entities)
        if mechanism == "ambulance":
            approach_action_id, approach_dispatch, direction, normal = _approach_geometry(
                script, config, audit)
    elif mechanism == "uav_pedestrian":
        proximity = _source_proximity(script, "pedestrian_near_miss", entities)
        if float(proximity["horizontal_distance_m"]) != thresholds["pedestrian_near_horizontal_m"] \
                or float(proximity["vertical_distance_m"]) != thresholds["pedestrian_near_vertical_m"]:
            raise ValueError("uav_pedestrian thresholds drifted from the recorded geometry contract")
    elif mechanism == "uav_vehicle":
        proximity = _source_proximity(script, "vehicle_roof_contact", entities)

    ground_reference = None
    if mechanism in ("uav_pedestrian", "uav_vehicle"):
        ground_reference = _roster_ground_reference(config, roles["aircraft"])

    dense, truth, transitions = [], [], []
    previous: dict[str, dict] = {}
    near_count = 0
    zone_centre = None
    descent_dispatch = None
    descent_target = None
    if mechanism == "uav_pedestrian":
        descent = next(e for e in script["events"] if e["event_id"] == "uav_low_descent")
        source_move = next(a for a in descent["actions"]
                           if a["type"] == "move_entity" and a["entity_id"] == roles["aircraft"])
        zone_centre = list(source_move["waypoints_enu_m"][-1][:2])
        # Until a descent is actually dispatched, both arms share the source target.
        # Bind the first executed descent, not the birth pad or a later return home.
        for item in audit:
            action = item["action"]
            dispatch = int(item["tick"])
            if (dispatch >= config["window"]["fork_tick"]
                    and item["result"]["status"] == "ok"
                    and action["type"] == "move_entity"
                    and action["entity_id"] == roles["aircraft"]
                    and action["waypoints_enu_m"][-1][2]
                    < by_tick[dispatch][roles["aircraft"]]["pos_enu"][2]):
                descent_dispatch = dispatch
                descent_target = list(action["waypoints_enu_m"][-1][:2])
                break
    clearance_established = False
    contact_held = False
    window = config["window"]
    # Pre-window history seeds dwell and the clearance latch; only
    # the declared window is materialized.  A missing lifecycle sample stays unknown.
    for tick in range(0, int(window["end_tick"]) + 1):
        if descent_dispatch is not None and tick > descent_dispatch:
            zone_centre = descent_target
        states = by_tick.get(tick, {})
        values = {p: None for p in rule["predicate_ids"]}
        metrics = {}
        if set(states) == entities:
            if mechanism == "uav_pedestrian":
                aircraft = states[roles["aircraft"]]
                pedestrian = states[roles["counterpart"]]
                height = aircraft["pos_enu"][2] - ground_reference
                horizontal = math.dist(aircraft["pos_enu"][:2], pedestrian["pos_enu"][:2])
                vertical = abs(aircraft["pos_enu"][2] - pedestrian["pos_enu"][2])
                low = height <= thresholds["low_altitude_max_m"]
                near = (horizontal <= float(proximity["horizontal_distance_m"])
                        and vertical <= float(proximity["vertical_distance_m"]))
                occupied = (None if zone_centre is None
                            else math.dist(pedestrian["pos_enu"][:2], zone_centre)
                            <= thresholds["landing_zone_radius_m"])
                landed = (height <= thresholds["touchdown_max_m"]
                          and speed(aircraft) <= thresholds["stationary_speed_mps"])
                declared_hold = flag(aircraft, "control_state", "hold_active")
                # A declared hold is only a hold if the aircraft actually holds position:
                # measured horizontal speed must be at or below the hold creep threshold.
                holding = declared_hold and math.hypot(*aircraft["vel_mps"][:2]) \
                    <= thresholds["hold_creep_speed_mps"]
                vals = [low, near, occupied, None if occupied is None else not occupied,
                        holding, landed]
                metrics.update(height_agl_m=height, horizontal_distance_m=horizontal,
                               vertical_distance_m=vertical,
                               landing_zone_centre_enu_m=zone_centre,
                               landing_zone_radius_m=thresholds["landing_zone_radius_m"],
                               horizontal_limit_m=proximity["horizontal_distance_m"],
                               vertical_limit_m=proximity["vertical_distance_m"],
                               low_altitude_limit_m=thresholds["low_altitude_max_m"],
                               speed_mps=speed(aircraft))
            elif mechanism == "uav_vehicle":
                aircraft = states[roles["aircraft"]]
                vehicle = states[roles["counterpart"]]
                height = aircraft["pos_enu"][2] - ground_reference
                distance = math.dist(aircraft["pos_enu"], vehicle["pos_enu"])
                airborne = height > thresholds["airborne_min_m"]
                # Contact is the authored trigger geometry itself.  Damage is evidence only after
                # that same measured contact has held (history, so a later separation does not
                # erase it); a fault flag without prior measured contact is a response running
                # ahead of its cause, not damage.
                contact_now = distance <= float(proximity["distance_m"])
                contact_held = contact_held or contact_now
                damaged = (contact_held
                           and flag(aircraft, "incident_state", "physical_fault_active"))
                acknowledgement = flag(vehicle, "vehicle_state", "emergency_stop")
                vals = [contact_now, contact_held, damaged, airborne,
                        acknowledgement,
                        speed(vehicle) <= thresholds["stationary_speed_mps"]]
                metrics.update(distance_3d_m=distance, proximity_limit_m=proximity["distance_m"],
                               contact_held=contact_held,
                               height_agl_m=height, speed_mps=speed(vehicle),
                               stationary_tolerance_mps=thresholds["stationary_speed_mps"])
            elif mechanism in ("intersection", "ambulance"):
                a_id, b_id = proximity["entity_a"], proximity["entity_b"]
                a, b = states[a_id], states[b_id]
                dimensions = 2 if proximity["metric"] == "xy" else 3
                delta = [b["pos_enu"][i] - a["pos_enu"][i] for i in range(dimensions)]
                distance = math.sqrt(sum(v * v for v in delta))
                close = distance <= float(proximity["distance_m"])
                near_count = near_count + 1 if close else 0
                sustained = close and near_count >= max(1, int(proximity["min_true_ticks"]))
                metrics.update(distance_m=distance, threshold_m=proximity["distance_m"],
                               near_dwell_ticks=near_count, required_dwell_ticks=proximity["min_true_ticks"],
                               speeds_mps={e: speed(r) for e, r in states.items()})
                if mechanism == "intersection":
                    braking = all(typed_bool(r, "braking") for r in states.values())
                    vals = [sustained, braking, any(typed_bool(r, "yielding") for r in states.values()),
                            braking and all(speed(r) <= tolerance for r in states.values())]
                else:
                    if a["asset_id"] != "vehicle.emergency.ambulance.v1":
                        raise ValueError("Ambulance binding does not have the required real asset")
                    closing = (sum(delta[i] * (a["vel_mps"][i] - b["vel_mps"][i])
                                   for i in range(dimensions)) / distance) if distance else 0.0
                    downstream = sum((a["pos_enu"][i] - b["pos_enu"][i]) * direction[i] for i in range(2))
                    lateral = sum((b["pos_enu"][i] - a["pos_enu"][i]) * normal[i] for i in range(2))
                    yielding = typed_bool(b, "yielding")
                    approaching = sustained and closing > tolerance
                    dispatch_observed = approach_dispatch is not None and tick > approach_dispatch
                    clearance = yielding and abs(lateral) >= thresholds["min_yield_center_separation_m"]
                    # Passage history: a clearance once physically established during this
                    # approach is not erased by the civilian finishing its yield manoeuvre.
                    clearance_established = clearance_established or clearance
                    passed = downstream >= 0.0
                    # The request is the dispatch plus the observed next sample; it stays active
                    # until the passage completes.  A vehicle waiting for a clearance is still
                    # holding a live request, so "still closing" is deliberately not a condition.
                    request_live = dispatch_observed and not (passed and clearance_established)
                    vals = [approaching, request_live,
                            typed_bool(a, "priority_request_acknowledged"), yielding,
                            clearance, passed, passed and clearance_established]
                    metrics.update(closing_speed_mps=closing, downstream_offset_m=downstream,
                                   lateral_separation_m=lateral,
                                   min_yield_center_separation_m=thresholds["min_yield_center_separation_m"],
                                   approach_direction_xy=direction, approach_normal_xy=normal,
                                   approach_move_action_id=approach_action_id,
                                   clearance_established=clearance_established,
                                   approach_dispatch_tick=approach_dispatch if dispatch_observed else None)
            else:
                if len(states) != 1:
                    raise ValueError("AV safe-stop requires exactly one bound vehicle")
                row = next(iter(states.values()))
                fault = typed_bool(row, "sensor_fault")
                maneuver = fault and typed_bool(row, "minimal_risk_maneuver_active")
                declared = typed_bool(row, "stopped_safe")
                vals = [fault, maneuver, declared, maneuver and declared and speed(row) <= tolerance]
                metrics.update(speed_mps=speed(row), stationary_tolerance_mps=tolerance)
            values = dict(zip(rule["predicate_ids"], vals))
        else:
            near_count = 0
        refs = [f"raw/states.jsonl#tick={tick}&entity={e}" for e in sorted(states)]
        if approach_dispatch is not None and tick > approach_dispatch:
            refs.append(f"raw/actions.jsonl#tick={approach_dispatch}&action_id={approach_action_id}")
        for predicate, value in values.items():
            row = {"schema_name": "l4_business_predicate_truth", "schema_version": "1.0.0",
                   "authority_scope": authority["authority_scope"], "episode_id": config["episode_id"],
                   "predicate_id": predicate, "tick": tick,
                   "value": "unknown" if value is None else str(value).lower(),
                   "bindings": {"entities": sorted(entities)}, "rule_digest": digest,
                   "truth_id": identifier("l4_truth", config["episode_id"], predicate, tick),
                   "source_refs": refs, "metrics": metrics,
                   "observation_status": "inactive_lifecycle" if value is None else "observed"}
            dense.append(row)
            if tick % step:
                continue
            truth.append(row)
            prev = previous.get(predicate)
            if prev and prev["value"] != row["value"] and row["value"] != "unknown":
                transitions.append({"schema_name": "l4_business_predicate_transition", "schema_version": "1.0.0",
                    "transition_id": identifier("l4_transition", config["episode_id"], predicate, tick),
                    "episode_id": config["episode_id"], "predicate_id": predicate,
                    "from_tick": prev["tick"], "to_tick": tick, "from_value": prev["value"],
                    "to_value": row["value"], "direction": "rising" if value else "falling",
                    "observation_boundary": prev["value"] == "unknown", "bindings": row["bindings"],
                    "rule_digest": digest, "source_refs": refs,
                    "evidence_truth_ids": [prev["truth_id"], row["truth_id"]]})
            previous[predicate] = row
    onset = rule["onset_predicate"]
    terminal = rule["terminal_predicate"]
    occurrences, outcomes = [], []
    true_terminal = {r["tick"]: r for r in truth if r["predicate_id"] == terminal and r["value"] == "true"}
    for transition in transitions:
        trigger_tick = transition["to_tick"]
        if transition["predicate_id"] != onset or transition["to_value"] != "true":
            continue
        if not window["start_tick"] <= trigger_tick <= window["end_tick"]:
            continue
        event_id = identifier("l4_event", config["episode_id"], rule["event_family"], trigger_tick)
        phases = [{"phase_kind": "onset", "tick": trigger_tick, "transition_id": transition["transition_id"],
                   "predicate_id": onset}]
        terminal_tick = None
        terminal_truth = []
        for tick in range(trigger_tick, int(window["end_tick"]) + 1, step):
            ticks = [tick + n * step for n in range(authority["terminal_hold_samples"])]
            if all(t in true_terminal for t in ticks):
                terminal_tick = ticks[-1]
                terminal_truth = [true_terminal[t]["truth_id"] for t in ticks]
                terminal_transition = next((r for r in reversed(transitions)
                    if r["predicate_id"] == terminal and r["to_value"] == "true" and r["to_tick"] <= tick), None)
                if terminal_transition:
                    phases.append({"phase_kind": "terminal", "tick": terminal_transition["to_tick"],
                                   "transition_id": terminal_transition["transition_id"], "predicate_id": terminal})
                break
        occurrences.append({"schema_name": "l4_business_event_occurrence", "schema_version": "1.0.0",
            "authority_scope": authority["authority_scope"], "episode_id": config["episode_id"],
            "event_id": event_id, "event_family_id": rule["event_family"], "trigger_tick": trigger_tick,
            "rule_id": "l4_arm_business." + mechanism, "rule_digest": digest,
            "bindings": {"entities": sorted(entities)}, "event_phases": phases,
            "source_refs": transition["source_refs"]})
        outcomes.append({"schema_name": "l4_business_event_outcome", "schema_version": "1.0.0",
            "episode_id": config["episode_id"], "event_id": event_id, "event_family_id": rule["event_family"],
            "outcome_id": identifier("l4_outcome", event_id), "terminal_tick": terminal_tick,
            "status": "succeeded" if terminal_tick is not None else "pending",
            "lifecycle_status": "observed_response_confirmed" if terminal_tick is not None else "response_not_confirmed_in_window",
            "terminal_predicate_id": terminal, "supporting_truth_ids": terminal_truth,
            "terminal_hold_samples": authority["terminal_hold_samples"],
            "interpretation": "Confirms the stated local response only; no claim of complete hazard recovery, legal right of way, or clinical recovery."})
        # Each group binds one authored incident instance (max_fire_count=1).
        # A later geometric re-approach does not create another response instance.
        break
    return {"predicate_truth": truth, "predicate_transitions": transitions,
            "event_occurrences": occurrences, "event_outcomes": outcomes,
            "l0_predicate_state": dense, "predicate_continuity_breaks": []}
