"""Gateway-observed degradation -> real downlink receipt -> actor executor.

The authored precheck/planned hold is preserved through its original queue
release. The gateway sees only its declared telemetry schedule and receptions
strictly before observation. Its decisions are not actor-local observations.
Only a received command can reach the actor controller; declared policy and
executor timing then determine action. Motion and ns-3 are recomputed together.
No calibrated city, cellular handover, UE capture or PX4 execution is claimed.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import tempfile

from .command_receipts import CommandReception, CommandTransmission, record_downlink_receipt
from .controller_receipt_consumer import consume_received_commands
from .event_logs import write_jsonl
from .gateway_sequence_metric import (GatewayReception, GatewaySequenceMetric,
                                      ScheduledDatagram)
from .provider_adapter import ProviderAdapter
from .runner import ROOT, run_episode
from .trajectory import STEP_NS, load_episode
from .receipt_scope import DEFAULT_SCOPE, ReceiptScope

OWNER = "uav_digital_l6_2_v1"
GATEWAY = "tower_l6_2_v1"
EPISODE = "L6-2_v1__seed00"
VERSION = "p01.ns3.receipt-replay/v5"
MONITOR_VERSION = GatewaySequenceMetric.namespace
COMMAND_VERSION = "p09.actual-downlink-command-receipt/v1"
TELEMETRY_FLOW = f"{EPISODE}:c2_telemetry"
STEP10_NS = 100_000_000
TTL_NS = 200_000_000
MONITOR_WINDOW_NS = 1_000_000_000
LOSS_THRESHOLD = 0.05
CONSECUTIVE_LIMIT = 3
COMMAND_PAYLOAD_BYTES = 64
COMMAND_PERIOD_NS = 100_000_000
COMMAND_ATTEMPTS = 3
COMMAND_DEADLINE_NS = 2_000_000_000
POLICY_DELAY_NS = 0  # explicit synthetic immediate consumption after RX; scheduling delays remain authored
EVENTS = ("c2_degradation", "uav_delayed_response", "backup_link_lock",
          "nominal_c2_restore", "lifecycle_landing_uav_digital_l6_2_v1")
COMMANDED = (EVENTS[1], EVENTS[2], EVENTS[4])
ACTION_NAMES = {EVENTS[1]: "slow_at_precheck", EVENTS[2]: "lock_backup_route",
                EVENTS[4]: "return_and_land"}


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _authority(script, *, scope=DEFAULT_SCOPE):
    """Authored predecessor chain, delays, and the exact mission moves."""
    if script["scenario_id"] != scope.scenario_id:
        raise ValueError("Authored scenario differs from the explicit receipt binding")
    events = {row["event_id"]: row for row in script["events"]}
    triggers = {row["trigger_id"]: row for row in script["triggers"]}
    for event in (scope.events[0], scope.events[3]):
        gateway_actions = {a["entity_id"] for a in events[event]["actions"]
            if a["type"] in ("set_visual_state", "set_runtime_state")
            and a["entity_id"] != scope.mission_owner}
        if gateway_actions != {scope.gateway_owner}:
            raise ValueError("Gateway binding differs from the actual authored degradation/restore actions")
    delays = {}
    for event, parent in zip(scope.events[1:], scope.events[:-1]):
        trigger = triggers[events[event]["trigger_ref"]]
        if trigger["type"] != "event_fired_after" or trigger["event_id"] != parent:
            raise ValueError(f"authored predecessor differs for {event}")
        delays[event] = trigger["delay_ticks"] * STEP_NS
    moves = {}
    for event in scope.commanded:
        actions = [a for a in events[event]["actions"]
                   if a["type"] == "move_entity" and a["entity_id"] == scope.mission_owner]
        if len(actions) != 1:
            raise ValueError(f"one exact authored mission move required for {event}")
        moves[event] = actions[0]
    return events, delays, moves


def _plan_boundary(script, *, scope=DEFAULT_SCOPE):
    """Original scheduled first response, independent of simulated fault times."""
    events, delays, _ = _authority(script, scope=scope)
    triggers = {row["trigger_id"]: row for row in script["triggers"]}
    root = triggers[events[scope.events[0]]["trigger_ref"]]
    if root["type"] != "tick":
        raise ValueError("pre-agreed anomaly/response schedule requires an explicit tick")
    monitor_enable_ns = root["tick"] * STEP_NS
    return monitor_enable_ns, monitor_enable_ns + delays[scope.events[1]]


def _actor_execution(command_evidence, duration_ns, queue_release_ns, *, scope=DEFAULT_SCOPE):
    """Consume actual accepted RX before a 10Hz executor observation."""
    accepted = [r for r in command_evidence["attempts"] if r["status"] == "accepted"]
    if not accepted:
        command_evidence.update(controller_consumption=None, execution_ns=None, fired=False)
        return None
    first = min(accepted, key=lambda r: (r["accepted_ns"], r["packet_id"]))
    ready = max(first["accepted_ns"] + POLICY_DELAY_NS, queue_release_ns)
    # At a barrier t, same-time RX is unavailable. Consume at the next tick.
    tick = (ready // STEP10_NS + 1) * STEP10_NS
    decisions, _ = consume_received_commands([first], observation_ns=tick,
        receiver_owner=scope.mission_owner, receiver_epoch=0, policy_delay_ns=POLICY_DELAY_NS,
        queue_available_ns=queue_release_ns)
    if len(decisions) != 1:
        raise ValueError("accepted command did not reach the declared executor tick")
    consumption = dict(decisions[0], executor_tick_ns=tick)
    if tick >= first["deadline_ns"]:
        consumption["status"] = "expired_before_execution"
    execute = tick if consumption["status"] == "ready_for_executor" and tick < duration_ns else None
    command_evidence.update(controller_consumption=consumption, execution_ns=execute,
                            fired=execute is not None)
    return execute


def _path(start, waypoints, speed):
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("authored move speed must be positive and finite")
    points = [list(start)] + copy.deepcopy(waypoints)
    distances = [math.dist(a, b) for a, b in zip(points, points[1:])]
    return points, distances, speed


def _on_path(path, elapsed_ns):
    points, distances, speed = path
    remaining = max(0, elapsed_ns) / 1_000_000_000 * speed
    for a, b, distance in zip(points, points[1:], distances):
        if distance and remaining < distance:
            direction = [(bv - av) / distance for av, bv in zip(a, b)]
            return ([av + remaining * d for av, d in zip(a, direction)],
                    [speed * d for d in direction], False)
        remaining -= distance
    return list(points[-1]), [0.0, 0.0, 0.0], True


def derive_monitor(packets, schedule, duration_ns, *, scope=DEFAULT_SCOPE):
    """Gateway-local scheduled-sequence metric; events must be strictly < t.

    Expected slots come from the pre-agreed schedule, not actual sender TX
    acceptance. At expiry+1ns the slot is mature; expiry equality itself is
    still pending. Rejected/missing scheduled TX remains missing at gateway.
    """
    if (schedule["src_owner"], schedule["life_epoch"]) != (scope.mission_owner, scope.mission_epoch):
        raise ValueError("Scheduled telemetry differs from the exact mission owner/epoch")
    period, start, end = schedule["period_ns"], schedule["start_ns"], schedule["end_ns"]
    if type(period) is not int or period <= 0:
        raise ValueError("telemetry period must be positive integer ns")
    tx_times = range(start, min(end, duration_ns - TTL_NS), period)
    expected = [ScheduledDatagram(flow_id=scope.telemetry_flow,
        source_owner=schedule["src_owner"], source_epoch=schedule["life_epoch"],
        receiver_owner=scope.gateway_owner, receiver_epoch=0, sequence=slot,
        generation_ns=tx, ttl_ns=TTL_NS, schedule_available_ns=0)
        for slot, tx in enumerate(tx_times)]
    by_tx = {}
    receptions = []
    for row in packets:
        if row["flow_id"] != scope.telemetry_flow:
            continue
        if (row["owner"],row["life_epoch"],row["receiver_owner"],row["receiver_life_epoch"]) != (
                schedule["src_owner"],schedule["life_epoch"],scope.gateway_owner,0):
            raise ValueError("telemetry exact source/receiver owner epoch differs from schedule")
        sequence, remainder = divmod(row["first_tx_ns"] - start, period)
        if remainder or sequence < 0 or row["first_tx_ns"] >= end:
            raise ValueError("telemetry packet is outside declared scheduled sequence")
        if row["first_tx_ns"] in by_tx:
            raise ValueError("multiple datagrams share one scheduled telemetry slot")
        by_tx[row["first_tx_ns"]] = row
        if row["accepted"] and row["rx_ns"] is not None:
            receptions.append(GatewayReception(flow_id=scope.telemetry_flow,
                source_owner=row["owner"], source_epoch=row["life_epoch"],
                receiver_owner=scope.gateway_owner, receiver_epoch=0, sequence=sequence,
                receive_ns=row["rx_ns"]))
    metric = GatewaySequenceMetric(expected, receptions)
    rows, consecutive = [], 0
    for slot, item in enumerate(expected):
        tx, observation = item.generation_ns, item.generation_ns + TTL_NS + 1
        cohort = metric.observe(observation_ns=observation, window_ns=MONITOR_WINDOW_NS,
                                receiver_owner=scope.gateway_owner, receiver_epoch=0)
        packet = by_tx.get(tx)
        notified = bool(packet is not None and packet["accepted"]
                        and packet["rx_ns"] is not None and packet["rx_ns"] < tx + TTL_NS)
        consecutive = 0 if notified else consecutive + 1
        ratio = cohort["loss_ratio"]
        rows.append({"schema_version": MONITOR_VERSION, "slot": slot, "time_ns": tx,
            "observation_ns": observation, "receiver_owner": scope.gateway_owner, "receiver_epoch": 0,
            "expected_sequence": slot, "notified_sequence": slot if notified else None,
            "slot_missing": not notified, "slot_notified": notified,
            "consecutive_missing_count": consecutive,
            "due_slot_count": cohort["mature_expected"],
            "missing_slot_count": cohort["ttl_not_timely"], "missing_ratio": ratio,
            "metric_status": cohort["availability"],
            "ratio_gt_threshold": ratio is not None and ratio > LOSS_THRESHOLD,
            "bad_communication": None if ratio is None else ratio > LOSS_THRESHOLD,
            "degraded": ratio is not None and ratio > LOSS_THRESHOLD
                        and consecutive >= CONSECUTIVE_LIMIT})
    return rows


def command_flows(send_times, *, scope=DEFAULT_SCOPE):
    """One exact gateway-to-UAV flow per commanded move event (dst_owner declared).

    The single flow window spans the declared attempts: the backend resends at
    period boundaries while now + period < end, so ATTEMPTS consecutive periods
    declare exactly ATTEMPTS bounded tries before the deadline.
    """
    flows = []
    for event in scope.commanded:
        send = send_times.get(event)
        if send is None:
            continue
        flows.append({"flow_id": f"{scope.episode_id}:command:{event}",
                      "link_id": f"{scope.episode_id}:command_link:{event}",
                      "src_owner": scope.gateway_owner, "life_epoch": 0, "dst_owner": scope.mission_owner, "dst_life_epoch": 0,
                      "payload_bytes": COMMAND_PAYLOAD_BYTES,
                      "period_ns": COMMAND_PERIOD_NS,
                      "start_ns": send, "end_ns": send + COMMAND_ATTEMPTS * COMMAND_PERIOD_NS})
    identifiers = [(row["flow_id"], row["link_id"]) for row in flows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate command flow identity")
    return flows


def _command_evidence(packets, send_times, *, scope=DEFAULT_SCOPE):
    """Actual per-attempt receipt rows under the shared receipt contract.

    Sending is not receiving: a command fires only on an accepted UAV-side RX
    strictly before its declared deadline (status "accepted"); late, wrong
    receiver or missing receptions never fire and are reported as-is.
    """
    evidence = {}
    for event in scope.commanded:
        send = send_times.get(event)
        if send is None:
            continue
        flow_id = f"{scope.episode_id}:command:{event}"
        deadline = send + COMMAND_DEADLINE_NS
        attempts = []
        for row in sorted((r for r in packets if r["flow_id"] == flow_id),
                          key=lambda r: r["first_tx_ns"]):
            if (row["owner"], row["life_epoch"]) != (scope.gateway_owner, 0):
                raise ValueError("command TX is not from its exact declared gateway epoch")
            command = CommandTransmission(
                command_id=flow_id, packet_id=row["packet_id"], source_owner=scope.gateway_owner,
                source_epoch=0, receiver_owner=scope.mission_owner, receiver_epoch=0,
                decision_ns=send, first_tx_ns=row["first_tx_ns"], deadline_ns=deadline,
                evidence_available_ns=send, action=scope.action_names[event])
            reception = None
            if row["accepted"] and row["rx_ns"] is not None:
                reception = CommandReception(packet_id=row["packet_id"],
                    receiver_owner=row["receiver_owner"],
                    receiver_epoch=row["receiver_life_epoch"], receive_ns=row["rx_ns"])
            attempts.append(record_downlink_receipt(command, reception))
        accepted = [row["accepted_ns"] for row in attempts if row["status"] == "accepted"]
        evidence[event] = {"command_id": flow_id, "send_ns": send,
                           "deadline_ns": deadline, "attempts": attempts,
                           "receipt_ns": min(accepted) if accepted else None,
                           "fired": False}
    return evidence


def derive_decisions(packets, schedule, script, duration_ns, *, scope=DEFAULT_SCOPE):
    """Gateway decisions schedule commands; actor execution requires RX.

    Authored event delays are gateway scheduling delays. The independent
    planned executor hold and strict-before 10Hz controller are additional
    explicit boundaries. Recovery is a known non-degraded gateway metric.
    """
    _, delays, _ = _authority(script, scope=scope)
    enable_ns, queue_release_ns = _plan_boundary(script, scope=scope)
    monitor = derive_monitor(packets, schedule, duration_ns, scope=scope)
    trigger = next((row for row in monitor if row["degraded"] and row["observation_ns"] >= enable_ns), None)
    times = dict.fromkeys(scope.events)
    sends = dict.fromkeys(scope.commanded)
    evidence = {}
    restore = None
    if trigger is not None:
        times[scope.events[0]] = trigger["observation_ns"]
        sends[scope.events[1]] = times[scope.events[0]] + delays[scope.events[1]]
        evidence.update(_command_evidence(packets, sends, scope=scope))
        fired1 = _actor_execution(evidence[scope.events[1]], duration_ns, queue_release_ns, scope=scope)
        times[scope.events[1]] = fired1
        if fired1 is not None:
            sends[scope.events[2]] = fired1 + delays[scope.events[2]]
            evidence.update(_command_evidence(packets, {scope.events[2]: sends[scope.events[2]]}, scope=scope))
            fired2 = _actor_execution(evidence[scope.events[2]], duration_ns, queue_release_ns, scope=scope)
            times[scope.events[2]] = fired2
            if fired2 is not None:
                earliest = fired2 + delays[scope.events[3]]
                restore = next((row for row in monitor if row["observation_ns"] >= earliest
                                and row["metric_status"] == "KNOWN" and row["bad_communication"] is False), None)
                if restore is not None:
                    times[scope.events[3]] = restore["observation_ns"]
                    sends[scope.events[4]] = times[scope.events[3]] + delays[scope.events[4]]
                    evidence.update(_command_evidence(packets, {scope.events[4]: sends[scope.events[4]]}, scope=scope))
                    times[scope.events[4]] = _actor_execution(evidence[scope.events[4]], duration_ns, queue_release_ns, scope=scope)
    return {"event_times_ns": times, "command_send_times_ns": sends,
            "command_evidence": evidence, "trigger_monitor_row": trigger,
            "restored_monitor_row": restore, "monitor_rows": monitor,
            "authored_delays_ns": delays, "loss_threshold": LOSS_THRESHOLD,
            "consecutive_limit": CONSECUTIVE_LIMIT, "planned_queue_release_ns": queue_release_ns,
            "monitor_enable_ns": enable_ns, "policy_delay_ns": POLICY_DELAY_NS}


def replay_motion(source_episode, decisions, authored_script, *, source_mission_rows=None, scope=DEFAULT_SCOPE):
    """Numerical replay after the authored planned hold queue boundary.

    Commanded phases start after real receipts and executor timing.
    Source position/activity/velocity before the planned release are preserved.
    """
    if source_episode["episode_id"] != scope.episode_id:
        raise ValueError("Source motion differs from the explicit episode binding")
    times = decisions["event_times_ns"]
    _, hold_ns = _plan_boundary(authored_script, scope=scope)
    _, _, moves = _authority(authored_script, scope=scope)
    mission_nodes = [n for n in source_episode["nodes"] if n["owner"] == scope.mission_owner]
    if len(mission_nodes) != 1 or mission_nodes[0]["life_epoch"] != 0:
        raise ValueError("mission requires one exact source radio life epoch")
    node_id = mission_nodes[0]["node_id"]
    source_positions = {f["time_ns"]: p["position_enu_m"]
                        for f in source_episode["mobility"] for p in f["positions"]
                        if p["node_id"] == node_id}
    held = list(source_positions[hold_ns - STEP_NS])
    phase_starts = [(times[event], event) for event in scope.commanded if times[event] is not None]
    paths = []
    current = held
    for start, event in phase_starts:
        if paths:
            previous_start, _, previous_path = paths[-1]
            current = _on_path(previous_path, start - previous_start)[0]
        path = _path(current, moves[event]["waypoints_enu_m"], moves[event]["velocity_mps"])
        paths.append((start, event, path))
    original = {row["time_ns"]: row for row in source_mission_rows} if source_mission_rows is not None else {}
    new_episode = copy.deepcopy(source_episode)
    new_node = next(n for n in new_episode["nodes"] if n["node_id"] == node_id)
    new_node["death_ns"] = None
    trajectory, landing_time = [], None
    for frame in new_episode["mobility"]:
        t = frame["time_ns"]
        if t < hold_ns and t not in source_positions:
            trajectory.append({"episode_id": scope.episode_id, "owner": scope.mission_owner, "life_epoch": 0,
                "node_id": node_id, "time_ns": t, "tick": t // STEP_NS,
                "position_enu_m": None, "velocity_enu_mps": None,
                "activity_type": None, "phase": "source_not_present", "active": False,
                "source_position_enu_m": None, "changed_position": False,
                "source_activity_type": None})
            continue
        if t < hold_ns:
            position = list(source_positions[t])
            if original:
                velocity = original[t]["velocity_enu_mps"]
                activity = original[t]["activity_type"]
            else:
                after = source_positions[min(t + STEP_NS, source_episode["duration_ns"])]
                velocity = [(b - a) * 10 for a, b in zip(position, after)]
                activity = "source_activity_not_loaded"
            phase = "frozen_source"
        else:
            position, velocity, activity, phase = list(held), [0.0] * 3, "airborne", "await_command_receipt"
            active_paths = [p for p in paths if p[0] <= t]
            if active_paths:
                start, event, path = active_paths[-1]
                position, velocity, finished = _on_path(path, t - start)
                phase = event
                if event == scope.events[4]:
                    activity = "landed" if finished else "return_to_pad"
                    if finished and landing_time is None:
                        landing_time = t
                elif event == scope.events[2]:
                    phase = "backup_attempt_hold" if finished else "backup_attempt_motion"
                else:
                    phase = "slowdown_hold" if finished else "slowdown_motion"
            if times[scope.events[3]] is not None and t >= times[scope.events[3]] and (times[scope.events[4]] is None or t < times[scope.events[4]]):
                phase = "nominal_restored_wait_return"
        item = next((p for p in frame["positions"] if p["node_id"] == node_id), None)
        if item is None:
            frame["positions"].append({"node_id": node_id, "position_enu_m": position, "active": True})
        else:
            item.update(position_enu_m=position, active=True)
        trajectory.append({"episode_id": scope.episode_id, "owner": scope.mission_owner, "life_epoch": 0,
            "node_id": node_id, "time_ns": t, "tick": t // STEP_NS,
            "position_enu_m": position, "velocity_enu_mps": velocity,
            "activity_type": activity, "phase": phase, "active": True,
            "source_position_enu_m": source_positions.get(t),
            "changed_position": position != source_positions.get(t),
            "source_activity_type": original[t]["activity_type"] if t in original else None})
    # The provider interpolates sampled positions linearly; report that
    # numerical interval velocity rather than an unsampled corner derivative.
    for row, following in zip(trajectory, trajectory[1:]):
        if row["time_ns"] >= hold_ns:
            row["velocity_enu_mps"] = [(b - a) * 10 for a, b in
                zip(row["position_enu_m"], following["position_enu_m"])]
    for row in trajectory:
        original_row = original.get(row["time_ns"])
        row["source_velocity_enu_mps"] = None if original_row is None else original_row["velocity_enu_mps"]
        row["changed_activity"] = row["activity_type"] != row["source_activity_type"]
    by_time = {r["time_ns"]: r for r in trajectory}
    timeline = []
    for event in scope.events:
        t = times[event]
        observed = by_time.get(t)
        commanded = event in scope.commanded
        sent = decisions["command_send_times_ns"].get(event)
        attempts = decisions["command_evidence"].get(event, {}).get("attempts", [])
        timeline.append({"event_id": event, "time_ns": t,
            "status": "not_triggered" if t is None else "outside_horizon"
                      if t >= source_episode["duration_ns"] else "executed_numerically"
                      if commanded else "gateway_metric_observed",
            "owner": scope.mission_owner if commanded else scope.gateway_owner, "life_epoch": 0,
            "position_enu_m": None if observed is None else observed["position_enu_m"],
            "source_position_enu_m": None if observed is None else observed["source_position_enu_m"],
            "command_id": None if not commanded or sent is None else f"{scope.episode_id}:command:{event}",
            "command_send_ns": sent,
            "command_attempt_count": len(attempts) if commanded else 0,
            "command_receipt_ns": decisions["command_evidence"].get(event, {}).get("receipt_ns"),
            "controller_consumption": decisions["command_evidence"].get(event, {}).get("controller_consumption"),
            "meaning": "executed after actual RX, declared policy/queue, next 10Hz executor tick"
                       if commanded else "gateway-local metric; not available to aircraft until command receipt"})
    dwell = moves[scope.events[4]]["terminal_feasibility"]["touchdown_dwell_ticks"] * STEP_NS
    landing = {"landing_time_ns": landing_time, "dwell_required_ns": dwell,
        "dwell_complete_ns": None if landing_time is None or landing_time + dwell > source_episode["duration_ns"] else landing_time + dwell,
        "landing_reference_enu_m": moves[scope.events[4]]["waypoints_enu_m"][-1],
        "hold_landed_active": True, "physical_death_inferred": False,
        "censored_without_command_receipt": landing_time is None}
    comparison = {
        "source_landed_first_ns": next((r["time_ns"] for r in trajectory
                                       if r["source_activity_type"] == "landed"), None),
        "changed_position_tick_count": sum(r["changed_position"] for r in trajectory),
        "changed_activity_tick_count": sum(r["changed_activity"] for r in trajectory),
        "unreceived_source_script_commands_removed_after_planned_hold": True,
        "other_actor_motion": "unchanged frozen source",
        "velocity_semantics": "post-hold forward difference of 10Hz ENU samples; ns3 linear interpolation"}
    return new_episode, {"decisions": decisions, "timeline": timeline, "trajectory": trajectory,
                         "landing": landing, "held_pose_enu_m": held, "source_comparison": comparison}


class _ReceiptMotionProvider(ProviderAdapter):
    """Use the existing radio backend, describing the receipt replay honestly."""

    def run_episode(self, episode, config, output_dir, *, geometry_export=None):
        result = super().run_episode(episode, config, output_dir, geometry_export=geometry_export)
        result["backend"]["mobility"] = (
            "WaypointMobilityModel; 10Hz ENU linear interpolation; mission is numerical "
            "receipt-causal replay, other actors retain frozen source motion")
        result["backend"]["receipt_motion_source_ref"] = episode["source_ref"]
        result["backend"]["immutable_other_actor_source_ref"] = config["receipt_replay"]["source_truth_ref"]
        result["backend"]["downlink_command_semantics"] = (
            "exact gateway-to-UAV timed flows with declared dst owner/epoch; "
            "actual accepted RX before deadline then policy/queue and next strict-before executor tick")
        return result


def _timed_window_rows(flows):
    """Iteration windows carry dst_owner explicitly; dropping it would silently
    re-point the command flows at the gateway."""
    return [{key: row[key] for key in ("flow_id", "link_id", "src_owner", "life_epoch",
                                       "dst_owner", "dst_life_epoch", "payload_bytes", "period_ns",
                                       "start_ns", "end_ns")}
            for row in flows]


def _monitor_schedule(config, *, scope=DEFAULT_SCOPE):
    telemetry = [row for row in config["network"]["traffic_windows"]
                 if row["flow_id"] == scope.telemetry_flow]
    if len(telemetry) != 1:
        raise ValueError("receipt monitor requires exactly the authored telemetry window")
    return {key: telemetry[0][key] for key in ("src_owner", "life_epoch", "start_ns", "end_ns", "period_ns")}


def solve(source_run, output_root, *, root=ROOT, script_path=None, healthy_control=False, scope=DEFAULT_SCOPE):
    """Execute until the new run's own receipts reproduce the input decisions.

    healthy_control=True keeps the same plan, radio and seed and removes only
    the declared scheduled burst; the published healthy run must show no
    prefault monitor anticipation. Only the stable iteration is published;
    temporary iteration products are removed on exit.
    """
    from .receipt_semantics import write_replay_bindings, VERSION as BINDING_VERSION, RULE_VERSION
    root, source_run, output_root = Path(root), Path(source_run).resolve(), Path(output_root).resolve()
    source_config = json.loads((source_run / "run_config.json").read_text())
    # Config seed is the ns-3 RNG seed, independent of the UE episode seed.
    if (source_config["episode_id"] != scope.episode_id
            or source_config["gateway_selection"]["entity_id"] != scope.gateway_owner
            or source_config["network_profile"] != "timed_flow"):
        raise ValueError("receipt replay requires the exact L6-2 timed source")
    if "geometry" in source_config:
        raise ValueError("this branch preserves the declared R1 source without geometric obstruction")
    script_path = root / scope.script_ref if script_path is None else Path(script_path)
    script = json.loads(script_path.read_text())
    source_episode = load_episode(root, scope.episode_id, gateway_entity_id=source_config["gateway_selection"]["entity_id"])
    for owner, epoch in ((scope.mission_owner, scope.mission_epoch), (scope.gateway_owner, scope.gateway_epoch)):
        owned = [n for n in source_episode["nodes"] if n["owner"] == owner]
        if len(owned) != 1 or owned[0]["life_epoch"] != epoch:
            raise ValueError("Scoped replay requires one exact source lifetime for each bound owner")
    enable_ns, hold_ns = _plan_boundary(script, scope=scope)
    source_mission = []
    for frame in read_rows(source_episode["source_ref"]):
        entities = [e for e in frame["entities"] if e["entity_id"] == scope.mission_owner]
        if not entities:
            continue
        entity, = entities
        source_mission.append({"time_ns": frame["tick"] * STEP_NS,
            "velocity_enu_mps": entity["truth_pose"]["velocity_enu_mps"],
            "activity_type": entity["annotations"]["activity_type"]})
    output = output_root / scope.episode_id / ("healthy_control" if healthy_control else "receipt_replay") / "run1"
    if output.exists():
        raise FileExistsError(f"versioned replay output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(source_config)
    windows = [{key: row[key] for key in ("flow_id", "link_id", "src_owner", "life_epoch",
                                          "payload_bytes", "period_ns", "start_ns", "end_ns")}
               for row in config["network"]["traffic_windows"]]
    if healthy_control:
        windows = [row for row in windows if row["flow_id"] != f"{scope.episode_id}:scheduled_burst"]
        if len(windows) != 1 or windows[0]["flow_id"] != scope.telemetry_flow:
            raise ValueError("healthy control must retain exactly the declared c2 telemetry window")
    schedule = _monitor_schedule({"network": {"traffic_windows": windows}}, scope=scope)
    motion_path = output / "mission_trajectory.jsonl.gz"
    config.update(profile="receipt_replay", output_format_version="ns3_episode_v10_receipt_replay.v5",
                  source_ref=str(motion_path))
    config["receipt_replay"] = {"schema_version": VERSION, "scope": scope.receipt(),
        "episode_seed": scope.seed, "ns3_rng_seed": source_config["seed"], "monitor_schema_version": MONITOR_VERSION,
        "command_schema_version": COMMAND_VERSION, "source_run_ref": str(source_run),
        "predicate_binding_version": BINDING_VERSION, "local_rule_version": RULE_VERSION,
        "source_truth_ref": source_episode["source_ref"], "authored_script_ref": str(script_path),
        "mission_owner": scope.mission_owner, "life_epoch": 0, "planned_queue_release_ns": hold_ns,
        "monitor_enable_ns": enable_ns, "policy_delay_ns": POLICY_DELAY_NS,
        "hold_pose_source_time_ns": hold_ns - STEP_NS,
        "healthy_control": healthy_control,
        "controller_input": "receiver-local scheduled-sequence TTL metric (p09.gateway-scheduled-sequence-ttl/v1) at gateway; actual UAV-side RX before policy/queue/executor only",
        "monitor_semantics": ("slot n expected at telemetry start_ns + n*period_ns; notified = accepted datagram delivered "
                              "on time (rx < tx + 200ms TTL); row n evaluated strictly after expiry at tx + TTL + 1 ns; "
                              "missing_ratio over the trailing 1 s of due slots"),
        "degrade_condition": f"missing_ratio > {LOSS_THRESHOLD} AND >= {CONSECUTIVE_LIMIT} consecutive missing slots",
        "definition_difference": ("old P is the gateway-side accepted-TX mature-datagram loss ratio over 1 s cohorts anchored "
                                  "at packet expiry; the new monitor is the receiver-local expected-vs-notified gap on the "
                                  "pre-agreed 10Hz schedule, gateway-local; command reception is required at the aircraft. Both are kept; the "
                                  "controller reads only the new one."),
        "command_semantics": ("one exact gateway-to-UAV flow per commanded move event, one command identity per event, "
                              f"{COMMAND_ATTEMPTS} declared attempt windows of {COMMAND_PERIOD_NS} ns, "
                              f"{COMMAND_DEADLINE_NS} ns deadline; synthetic protocol parameters, not measured hardware"),
        "backup_semantics": "script action attempt; no alternate radio, socket or handover applied",
        "motion_semantics": "10Hz numerical constant-speed authored waypoints; no UE rendered flight claim",
        "iteration_semantics": "self-consistent numerical motion/network/controller fixed point"}
    seen, history = set(), []
    packets = read_rows(source_run / source_config["artifact_files"]["network_packets.jsonl"])
    with _ReceiptMotionProvider(image=config["image"]) as provider:
        while True:
            decisions = derive_decisions(packets, schedule, script, source_episode["duration_ns"], scope=scope)
            key = (tuple(decisions["event_times_ns"][e] for e in scope.events),
                   tuple(decisions["command_send_times_ns"][e] for e in scope.commanded))
            if key in seen:
                raise RuntimeError(f"causal replay decision cycle: {history}; repeated {key}")
            seen.add(key)
            episode, evidence = replay_motion(source_episode, decisions, script,
                                              source_mission_rows=source_mission, scope=scope)
            episode["source_ref"] = str(motion_path)
            next(n for n in episode["nodes"] if n["owner"] == scope.mission_owner)["source_ref"] = str(motion_path)
            iteration_config = copy.deepcopy(config)
            command_flows_now = command_flows(decisions["command_send_times_ns"], scope=scope)
            iteration_config["network"]["traffic_windows"] = windows + _timed_window_rows(command_flows_now)
            iteration_config["receipt_replay"]["executed_event_times_ns"] = decisions["event_times_ns"]
            iteration_config["receipt_replay"]["executed_command_send_times_ns"] = decisions["command_send_times_ns"]
            with tempfile.TemporaryDirectory(prefix=".receipt-solver-", dir=output.parent) as scratch:
                scratch = Path(scratch)
                summary = run_episode(episode, iteration_config, scratch, provider)
                actual_packets = read_rows(scratch / config["artifact_files"]["network_packets.jsonl"])
                actual = derive_decisions(actual_packets, schedule, script, source_episode["duration_ns"], scope=scope)
                actual_key = (tuple(actual["event_times_ns"][e] for e in scope.events),
                              tuple(actual["command_send_times_ns"][e] for e in scope.commanded))
                history.append({"iteration": len(history) + 1, "input_key": key,
                                "actual_key": actual_key, "self_consistent": key == actual_key})
                if key == actual_key:
                    compression = "gzip"
                    decisions = actual
                    # Stable action times imply the same motion, but not the
                    # same per-attempt RX. Bind all evidence to the final run.
                    _, evidence = replay_motion(source_episode, actual, script,
                                                source_mission_rows=source_mission, scope=scope)
                    evidence["decisions"] = {k: v for k, v in actual.items() if k != "monitor_rows"}
                    monitor_rows = actual.pop("monitor_rows")
                    write_jsonl(scratch / motion_path.name, evidence.pop("trajectory"), compression=compression)
                    write_jsonl(scratch / "receipt_monitor.jsonl.gz", monitor_rows, compression=compression)
                    write_jsonl(scratch / "command_receipts.jsonl.gz",
                                [evidence["decisions"]["command_evidence"][event] for event in scope.commanded
                                 if event in evidence["decisions"]["command_evidence"]], compression=compression)
                    degraded_rows = [row for row in monitor_rows if row["degraded"]]
                    evidence.update(schema_version=VERSION, source_run_ref=str(source_run),
                        source_truth_ref=source_episode["source_ref"], authored_script_ref=str(script_path),
                        solver_history=history, iteration_count=len(history), self_consistent=True,
                        healthy_control=healthy_control,
                        monitor_summary={"row_count": len(monitor_rows),
                            "degraded_row_count": len(degraded_rows),
                            "first_degraded_observation_ns": None if not degraded_rows else degraded_rows[0]["observation_ns"],
                            "max_consecutive_missing": max(row["consecutive_missing_count"] for row in monitor_rows),
                            "prefault_degraded_row_count": sum(row["degraded"] and row["observation_ns"] < enable_ns
                                                               for row in monitor_rows)},
                        trajectory_ref=motion_path.name, monitor_ref="receipt_monitor.jsonl.gz",
                        command_receipts_ref="command_receipts.jsonl.gz",
                        claims="numerical ns-3 coupled motion with real downlink command flows; no UE capture, calibrated realism or radio handover claim")
                    bindings = write_replay_bindings(scratch, evidence, monitor_rows,
                        run_id=f"{scope.episode_id}:{'healthy_control' if healthy_control else 'receipt_replay'}:v5:run1",
                        source_run=str(output), scope=scope)
                    evidence["predicate_bindings"] = {k: v for k, v in bindings.items() if k != "files"}
                    _json(scratch / "receipt_timeline.json", evidence)
                    final_summary = json.loads((scratch / "summary.json").read_text())
                    final_summary["receipt_replay"] = {
                        "schema_version": VERSION, "iteration_count": len(history), "self_consistent": True,
                        "healthy_control": healthy_control,
                        "event_times_ns": actual["event_times_ns"],
                        "command_send_times_ns": actual["command_send_times_ns"],
                        "monitor_summary": evidence["monitor_summary"], "landing": evidence["landing"],
                        "trajectory_ref": motion_path.name, "timeline_ref": "receipt_timeline.json",
                        "monitor_ref": "receipt_monitor.jsonl.gz", "command_receipts_ref": "command_receipts.jsonl.gz"}
                    final_summary["receipt_replay"]["predicate_bindings"] = evidence["predicate_bindings"]
                    final_summary["output_bytes"].update({name: (scratch / name).stat().st_size for name in
                        (motion_path.name, "receipt_monitor.jsonl.gz", "command_receipts.jsonl.gz",
                         "receipt_timeline.json", *bindings["files"])})
                    (scratch / "summary.json").write_text(json.dumps(final_summary, indent=2, allow_nan=False) + "\n")
                    output.mkdir()
                    for product in scratch.iterdir():
                        product.rename(output / product.name)
                    return {"output_dir": str(output), "iteration_count": len(history),
                        "event_times_ns": actual["event_times_ns"],
                        "command_send_times_ns": actual["command_send_times_ns"],
                        "command_evidence": actual["command_evidence"],
                        "trigger_monitor_row": actual["trigger_monitor_row"],
                        "restored_monitor_row": actual["restored_monitor_row"],
                        "monitor_summary": evidence["monitor_summary"],
                        "landing": evidence["landing"], "held_pose_enu_m": evidence["held_pose_enu_m"],
                        "source_comparison": evidence["source_comparison"],
                        "network": summary["network"]}
                packets = actual_packets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--script-json", type=Path)
    parser.add_argument("--binding-json", type=Path, help="Exact versioned five-stage owner/run/script binding")
    parser.add_argument("--healthy-control", action="store_true")
    args = parser.parse_args(argv)
    scope = DEFAULT_SCOPE if args.binding_json is None else ReceiptScope.from_dict(json.loads(args.binding_json.read_text()))
    result = solve(args.source_run, args.output_root, script_path=args.script_json,
                   healthy_control=args.healthy_control, scope=scope)
    network = result.pop("network")
    result["network"] = {key: network[key] for key in
        ("socket_accepted_count", "mature_at_horizon_count", "mature_ttl_loss_count",
         "late_rx_count", "wall_time_s", "peak_rss_kb")}
    result["summary_ref"] = str(Path(result["output_dir"]) / "summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
