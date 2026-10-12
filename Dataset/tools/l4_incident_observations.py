"""Local physical observations alongside the unchanged formal L4 ontology.

Only the configured subjects and window are assessed. Source triggers supply
distance/dwell limits; landing predicates come from the formal computation.
"""
from __future__ import annotations

import math

from Dataset.tools.l4_business_semantics import identifier, speed
from Dataset.semantic_truth.core_semantic_registry import get_governed_parameter_defaults

PROFILES = {
    "L4-3_v1": "crowd_proximity_response",
    "L4-3_v2": "fault_conditioned_landing", "L4-3_v3": "fault_conditioned_landing",
    "L4-7_v1": "medical_response", "L4-7_v2": "medical_response",
    "L4-8_v1": "crowd_evacuation", "L4-8_v2": "crowd_evacuation",
}
SPEED_TOLERANCE = 1e-6  # Recorded velocities have six decimal places.
POSITION_TOLERANCE = math.sqrt(3) * 1e-6  # Rounding of two serialized positions.


def supported(config):
    return config["episode_id"].split("__")[0] in PROFILES


def evaluate(config, trajectories, script, audit, formal_truth):
    mechanism = PROFILES[config["episode_id"].split("__")[0]]
    subjects = set(config["entity_ids"])
    states = {}
    for row in trajectories:
        if row["entity_id"] in subjects:
            key = (row["tick"], row["entity_id"])
            if key in states:
                raise ValueError("duplicate local business state")
            states[key] = row
    events = {e["event_id"]: e for e in script["events"]}
    triggers = {t["trigger_id"]: t for t in script["triggers"]}
    successful = [a for a in audit if a["result"]["status"] == "ok" and a["origin"] == "source_script"]
    governed = get_governed_parameter_defaults()

    def dispatch(action_id):
        matches = [a["tick"] for a in successful if a["action"]["action_id"] == action_id]
        if len(matches) > 1:
            raise ValueError(f"multiple dispatches for {action_id}")
        return matches[0] if matches else None

    def flag(row, family, key):
        value = row[family][key]
        if type(value) is not bool:
            raise ValueError(f"non-Boolean state {family}.{key}")
        return value

    if mechanism == "crowd_evacuation":
        moves = [a for a in events["evacuation_triggered"]["actions"]
                 if a["type"] == "move_entity" and a["entity_id"] in subjects]
        destinations = {a["entity_id"]: a["waypoints_enu_m"][-1] for a in moves}
        if set(destinations) != subjects:
            raise ValueError("evacuation destination coverage differs from subjects")
        onset_key, terminal_key, hold = "command_active", "arrival_acknowledged", governed["recovery_hold_samples"]
    elif mechanism == "crowd_proximity_response":
        event = "crowd_proximity_response"
        gate = triggers[events[event]["trigger_ref"]]
        if gate["type"] != "entity_proximity" or gate["operator"] != "lte":
            raise ValueError("unsupported local proximity authority")
        a_id, b_id = gate["entity_a"], gate["entity_b"]
        if not {a_id, b_id} <= subjects:
            raise ValueError("proximity subjects outside observation scope")
        # The response volume is the authored geometric gate itself. Its dwell is deliberately
        # NOT read from this trigger: an intervention may rewrite that dwell to postpone the
        # response event, and the exposure measurement must stay the raw geometry so both arms
        # share one onset and differ only in recovery.
        onset_key, terminal_key, hold = "proximity_volume_active", "proximity_recovered", 1
    else:
        event = "ambulance_arrived" if mechanism == "medical_response" else "crowd_proximity_response"
        gate = triggers[events[event]["trigger_ref"]]
        if gate["type"] != "entity_proximity" or gate["operator"] != "lte":
            raise ValueError("unsupported local proximity authority")
        a_id, b_id = gate["entity_a"], gate["entity_b"]
        if not {a_id, b_id} <= subjects:
            raise ValueError("proximity subjects outside observation scope")
        if mechanism == "medical_response":
            move_tick = dispatch("move_ambulance_to_fall")
            spawn_ids = [a["action_id"] for a in events["ambulance_response"]["actions"]
                         if a["type"] == "spawn_entity" and a["entity_id"] == a_id]
            if len(spawn_ids) != 1:
                raise ValueError("medical response requires one ambulance spawn")
            spawn_tick = dispatch(spawn_ids[0])
            if spawn_tick != move_tick:
                raise ValueError("ambulance spawn and initial movement must share the source dispatch")
            arrival_tick = dispatch("capture_fall_arrival")
            handoff_tick = dispatch("capture_fall_handoff")
            handoff_trigger = triggers[events["responder_handoff"]["trigger_ref"]]
            if handoff_trigger["type"] != "event_fired_after" or handoff_trigger["event_id"] != "ambulance_arrived":
                raise ValueError("unsupported handoff source contract")
            onset_key, terminal_key, hold = "physical_dispatch", "near_presence_handoff", 1
        else:
            landing_tick = dispatch("move_uav_touchdown")
            selected_formal = [r for r in formal_truth
                      if r["predicate_id"] in ("airspace.aircraft_touchdown", "airspace.aircraft_in_landing_zone")
                      and r["bindings"].get("aircraft") == a_id]
            formal = {(r["tick"], r["predicate_id"]): r for r in selected_formal}
            if len(formal) != len(selected_formal):
                raise ValueError("ambiguous landing predicate binding")
            onset_key, terminal_key, hold = "fault_descent_motion", "touchdown_in_zone", governed["terminal_hold_samples"]

    dense, seen_active, dispatched, near_count, held = [], False, False, 0, False
    volume_dwell, recovery_dwell, volume_seen = 0, 0, False
    window = config["window"]
    # The pre-window history seeds dwell and lifecycle state; only the window is saved.
    for tick in range(window["end_tick"] + 1):
        present = {e: states[(tick, e)] for e in subjects if (tick, e) in states}
        metrics, refs = {}, [f"raw/states.jsonl#tick={tick}&entity={e}" for e in sorted(present)]
        if mechanism == "crowd_evacuation":
            if set(present) != subjects:
                values = {k: None for k in ("command_active", "actual_motion", "all_at_destination", "arrival_acknowledged")}
            else:
                active = any(flag(r, "pedestrian_state", "evacuation_active") for r in present.values())
                seen_active |= active
                distances = {e: math.dist(r["pos_enu"], destinations[e]) for e, r in present.items()}
                moving = any(speed(r) > SPEED_TOLERANCE for r in present.values())
                arrived = max(distances.values()) <= POSITION_TOLERANCE and not moving
                values = {"command_active": active, "actual_motion": moving,
                          "all_at_destination": arrived, "arrival_acknowledged": seen_active and arrived and not active}
                metrics = {"destination_distance_m": distances, "destinations_enu_m": destinations}
        elif mechanism == "crowd_proximity_response":
            if a_id not in present or b_id not in present:
                raise ValueError("proximity subjects missing within simulation history")
            aircraft, pedestrian = present[a_id], present[b_id]
            horizontal = math.dist(aircraft["pos_enu"][:2], pedestrian["pos_enu"][:2])
            vertical = abs(aircraft["pos_enu"][2] - pedestrian["pos_enu"][2])
            in_volume = (horizontal <= gate["horizontal_distance_m"]
                         and vertical <= gate["vertical_distance_m"])
            if in_volume:
                volume_dwell, recovery_dwell = volume_dwell + 1, 0
            else:
                volume_dwell = 0
                recovery_dwell = recovery_dwell + 1 if volume_seen else 0
            volume_seen |= in_volume
            recovered = volume_seen and recovery_dwell >= governed["recovery_hold_samples"]
            egress = any(speed(r) > SPEED_TOLERANCE for e, r in present.items() if e != a_id)
            values = {"proximity_volume_active": in_volume,
                      "proximity_dwell_ticks": volume_dwell,
                      "proximity_recovered": recovered,
                      "crowd_egress_motion": egress}
            metrics = {"horizontal_distance_m": horizontal, "vertical_distance_m": vertical,
                       "horizontal_limit_m": gate["horizontal_distance_m"],
                       "vertical_limit_m": gate["vertical_distance_m"],
                       "recovery_dwell_ticks": recovery_dwell}
        elif mechanism == "medical_response":
            patient = present.get(b_id)
            declaration = flag(patient, "incident_state", "dispatch_active") if patient else None
            injury = (flag(patient, "pedestrian_state", "injured") and flag(patient, "pedestrian_state", "fallen")) if patient else None
            pair_present = a_id in present and b_id in present
            distance = math.dist(present[a_id]["pos_enu"], present[b_id]["pos_enu"]) if pair_present else None
            near = pair_present and distance <= gate["distance_m"]
            near_count = near_count + 1 if near else 0
            if (pair_present and move_tick is not None and spawn_tick is not None
                    and tick > max(move_tick, spawn_tick) and speed(present[a_id]) > SPEED_TOLERANCE):
                dispatched = True
            if arrival_tick is not None and tick == arrival_tick:
                held = near_count >= gate["min_true_ticks"]
            elif arrival_tick is not None and tick > arrival_tick:
                held = held and near
            completed = (held and handoff_tick is not None and tick >= handoff_tick
                         and handoff_tick - arrival_tick >= handoff_trigger["delay_ticks"])
            values = {"injury_context": injury, "dispatch_declared": declaration,
                      "physical_dispatch": dispatched, "arrival_gate": near_count >= gate["min_true_ticks"] if pair_present else None,
                      "near_presence_held": held if pair_present else None, "near_presence_handoff": completed if pair_present else None}
            metrics = {"distance_m": distance, "distance_limit_m": gate["distance_m"],
                       "near_dwell_ticks": near_count, "required_dwell_ticks": gate["min_true_ticks"],
                       "arrival_dispatch_tick": arrival_tick, "handoff_dispatch_tick": handoff_tick,
                       "dispatch_action_tick": move_tick,
                       "handoff_delay_ticks": handoff_trigger["delay_ticks"]}
            for action_id in ("move_ambulance_to_fall", "capture_fall_arrival", "capture_fall_handoff", spawn_ids[0]):
                at = dispatch(action_id)
                if at is not None and at <= tick:
                    refs.append(f"raw/actions.jsonl#tick={at}&action_id={action_id}")
        else:
            if a_id not in present or b_id not in present:
                raise ValueError("landing subjects missing within simulation history")
            aircraft, pedestrian = present[a_id], present[b_id]
            fault = flag(aircraft, "incident_state", "physical_fault_active")
            horizontal = math.dist(aircraft["pos_enu"][:2], pedestrian["pos_enu"][:2])
            vertical = abs(aircraft["pos_enu"][2] - pedestrian["pos_enu"][2])
            close = horizontal <= gate["horizontal_distance_m"] and vertical <= gate["vertical_distance_m"]
            near_count = near_count + 1 if close else 0
            touchdown = None
            if tick % 5 == 0 and tick >= window["start_tick"]:
                pairs = [formal[(tick, p)] for p in ("airspace.aircraft_touchdown", "airspace.aircraft_in_landing_zone")]
                if all(r["value"] in ("true", "false") for r in pairs):
                    touchdown = all(r["value"] == "true" for r in pairs)
                refs.extend("objective/predicate_truth.jsonl#" + r["truth_id"] for r in pairs)
            values = {"fault_active": fault, "fault_descent_motion": fault and aircraft["vel_mps"][2] < -SPEED_TOLERANCE,
                      "crowd_proximity_sustained": near_count >= gate["min_true_ticks"],
                      "crowd_motion": any(speed(r) > SPEED_TOLERANCE for e, r in present.items() if e != a_id),
                      "landing_descent_motion": landing_tick is not None and tick > landing_tick and aircraft["vel_mps"][2] < -SPEED_TOLERANCE,
                      "touchdown_in_zone": touchdown}
            metrics = {"horizontal_distance_m": horizontal, "vertical_distance_m": vertical,
                       "horizontal_limit_m": gate["horizontal_distance_m"], "vertical_limit_m": gate["vertical_distance_m"],
                       "landing_dispatch_tick": landing_tick}
        if tick >= window["start_tick"]:
            dense.append({"tick": tick, "values": values, "metrics": metrics, "source_refs": refs})
    sampled = [r for r in dense if r["tick"] % 5 == 0]
    onset = next((r["tick"] for r in dense if r["values"][onset_key] is True), None)
    terminal = None
    supporting = []
    terminal_samples = (dense if mechanism in ("medical_response", "crowd_proximity_response")
                        else sampled)
    for i, row in enumerate(terminal_samples):
        span = terminal_samples[i:i + hold]
        if onset is not None and row["tick"] >= onset and len(span) == hold and all(r["values"][terminal_key] is True for r in span):
            terminal, supporting = span[-1]["tick"], [r["tick"] for r in span]
            break
    summary = {"mechanism": mechanism, "authority_scope": "local physical business observation alongside formal ontology",
               "onset_predicate": onset_key, "terminal_predicate": terminal_key, "onset_tick": onset,
               "terminal_tick": terminal, "terminal_hold_samples": hold, "supporting_ticks": supporting,
               "status": "absent_in_window" if onset is None else "confirmed" if terminal is not None else "pending_in_window",
               "not_claimed": ["clinical_recovery", "complete_hazard_recovery", "formal_unavailable_event_implementation"],
               "first_true_ticks": {key: next((r["tick"] for r in dense if r["values"][key] is True), None)
                                    for key in dense[0]["values"]}}
    summary["observation_id"] = identifier("l4_local_response", config["episode_id"], config["arm_id"], mechanism, onset)
    return dense, summary
