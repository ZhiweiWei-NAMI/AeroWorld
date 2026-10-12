"""One approved native coupled trace for L6-5_v1 seed00 (checkpoint 2).

Couples the frozen checkpoint-1 command evidence with the full original chain:
owner-local isolation lockout and its report, the secure recovery notification
on the separately predeclared trusted path, the GCS delay-1 secure runtime patch
applied before any status send, the status record carrying both the original
secure event time and the applied state time, the UAV recovery effect waiting
for the original due time AND actual RX availability, and the landing gate
waiting for the original due time AND RX availability.

The native transport run is NEW here (two extra message flows on the same
frozen episode/mobility/radio/seed: the recovery notification at the original
secure event tick 380 and the original landing command at the original landing
event tick 460). The frozen checkpoint-1 products under ``trace/`` are never
rewritten; this script writes only into ``trace_coupled/``.

Causality notes (numbers from the actual coupled run):
- The receiver controller evaluates on the authored 5-tick action grid (one
  segment is (t-1, t]; a boundary tick satisfies tick % 5 == 0). The movement
  admission uses the same 27.5 s grid barrier as the frozen checkpoint-1
  record; the 497.6737 ms RX->barrier interval is action-grid wait, and the
  2.3263 ms send->RX interval is the native airtime/propagation segment.
- An effect applies at the first gate tick at which BOTH the authored due time
  is reached and the native RX instant is strictly earlier than the gate
  instant. Nothing is backdated to a due time that the RX missed.
- An admission record's ``physical_motion_evidence: null`` is never proof of
  execution. Motion evidence is bound from the actual executed move actions and
  the frozen trajectory rows. At the authored 0.1 s resolution the authored
  motion starts at the send tick, so a post-RX grid admission instant cannot
  precede the motion start; the bind gate records that non-closure honestly
  (``bound: false``) instead of relaxing the inequality.

Run (airfogsim python, from the repository root):
  /home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python -m \
      Dataset.semantic_simulation.p09_l6_5_receipt_v1.coupled_episode
"""
from __future__ import annotations

import json
from pathlib import Path

from Dataset.semantic_simulation.ns3_episode.provider_adapter import ProviderAdapter
from Dataset.semantic_simulation.ns3_episode.linked_engine import execute_linked
from Dataset.semantic_simulation.p09_l6_5_receipt_v1 import adapter as A

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/scene_setup.json"
SCRIPT = ROOT / "Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/event_script.json"
RADIO_REF = ROOT / "design/p01_ns3/radio_reference_R1.json"
TRACE_DIR = Path(__file__).resolve().parent / "trace_coupled"
FROZEN_TRACE_DIR = Path(__file__).resolve().parent / "trace"
NATIVE_SEED = 1  # same p01 convention as the frozen checkpoint-1 trace
RUN = 1

ABNORMAL_MESSAGE_ID = f"{A.EPISODE_ID}:command:{A.MOVEMENT_ACTION}"
NOTIFY_MESSAGE_ID = f"{A.EPISODE_ID}:command:{A.NOTIFICATION_ACTION}"
LANDING_MESSAGE_ID = f"{A.EPISODE_ID}:command:{A.LANDING_ACTION}"

ABNORMAL_SEND_TICK = A.STORY["abnormal_command_event_tick"]     # 270
ABNORMAL_EVIDENCE_TICK = A.STORY["gcs_intrusion_event_tick"]    # 240
NOTIFY_SEND_TICK = A.STORY["secure_recovery_event_tick"]        # 380
NOTIFY_EVIDENCE_TICK = A.STORY["secure_recovery_event_tick"]    # 380
LANDING_SEND_TICK = A.STORY["landing_event_tick"]               # 460
LANDING_EVIDENCE_TICK = A.STORY["landing_event_tick"]           # 460

# Frozen checkpoint-1 receiver action-grid barrier for the movement admission:
# 27.5 s. The 2.3263 ms send->RX interval is native airtime/propagation; the
# 497.6737 ms RX->barrier interval is receiver action-grid wait.
ABNORMAL_ADMISSION_TICK = A.STORY["uav_recovery_due_tick"]      # 275


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_episode_and_engine():
    """Actual frozen trajectory/entities from the immutable original sources."""
    scene = load_json(SCENE)
    script = load_json(SCRIPT)
    run = execute_linked(scene, script, SCRIPT, A.EPISODE_ID, seed=0)
    from Dataset.semantic_simulation.ns3_episode.linked_transport import episode_from_engine
    profile = {
        "episode_id": A.EPISODE_ID, "source_script": str(SCRIPT),
        "actor_ids": [A.UAV], "primary_station": A.GCS,
    }
    episode = episode_from_engine(profile, run["engine"])
    return episode, run["engine"]


def build_binding(message_id, action, role, evidence_tick, send_tick):
    return A.bind_command(
        message_id, action, role,
        {"claimed_roles": [role]},  # recorded, never authoritative
        source_owner=A.GCS, source_epoch=0, receiver_owner=A.UAV, receiver_epoch=0,
    ) | {"evidence_tick": evidence_tick, "send_tick": send_tick}


def native_flow_window(episode, binding):
    """Same existing transport window contract as the frozen checkpoint-1 run."""
    return A.build_message(episode, binding, binding["send_tick"],
                           binding["evidence_tick"] * A.STEP_NS)


def executed_action(engine, suffix):
    """Actual executed action row by original action-id suffix (or None)."""
    for action in engine.executed_actions:
        if str(action.get("action_id")).endswith(suffix):
            return action
    return None


def main():
    if TRACE_DIR.exists() and any(TRACE_DIR.iterdir()):
        raise FileExistsError(f"coupled trace output already exists: {TRACE_DIR}")
    for required in ("command_receipts.json", "controller_admission.json"):
        if not (FROZEN_TRACE_DIR / required).exists():
            raise FileNotFoundError(
                f"frozen checkpoint-1 product missing: {FROZEN_TRACE_DIR / required}")
    TRACE_DIR.mkdir(parents=True)
    episode, engine = build_episode_and_engine()

    # ---- bindings BEFORE transport (predeclared grant table) ----
    abnormal_binding = build_binding(ABNORMAL_MESSAGE_ID, A.MOVEMENT_ACTION,
                                     A.MOVEMENT_ROLE, ABNORMAL_EVIDENCE_TICK,
                                     ABNORMAL_SEND_TICK)
    notify_binding = build_binding(NOTIFY_MESSAGE_ID, A.NOTIFICATION_ACTION,
                                   A.NOTIFICATION_ROLE, NOTIFY_EVIDENCE_TICK,
                                   NOTIFY_SEND_TICK)
    landing_binding = build_binding(LANDING_MESSAGE_ID, A.LANDING_ACTION,
                                    A.MOVEMENT_ROLE, LANDING_EVIDENCE_TICK,
                                    LANDING_SEND_TICK)

    # ---- native transport: three flows, same frozen episode/mobility/radio ----
    messages = [native_flow_window(episode, b) for b in
                (abnormal_binding, notify_binding, landing_binding)]
    config = {
        "network_profile": "timed_flow",
        "network": {"traffic_windows": [
            {key: row[key] for key in ("flow_id", "link_id", "src_owner", "life_epoch",
                                       "dst_owner", "dst_life_epoch", "payload_bytes",
                                       "period_ns", "start_ns", "end_ns")}
            for row in (m["flow"] for m in messages)]},
        "seed": NATIVE_SEED, "run": RUN,
        "radio": load_json(RADIO_REF),
        "radio_actions": [],
        "radio_config_source_ref": str(RADIO_REF),
    }
    (TRACE_DIR / "coupled_input.json").write_text(json.dumps({
        "schema_version": A.SCHEMA, "episode_id": A.EPISODE_ID,
        "scenario_id": A.SCENARIO_ID, "runtime_identity": A.runtime_identity(),
        "source_event_script": str(SCRIPT), "frozen_checkpoint1_trace": str(FROZEN_TRACE_DIR),
        "native_seed": NATIVE_SEED, "native_run": RUN,
        "bindings": [abnormal_binding, notify_binding, landing_binding],
        "command_transport": A.COMMAND,
        "flows": [m["flow"] for m in messages],
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with ProviderAdapter() as provider:
        network = provider.run_episode(episode, config, TRACE_DIR)
    (TRACE_DIR / "network_packets.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                for row in network["network_packets"]), encoding="utf-8")
    (TRACE_DIR / "network_events.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                for row in network["network_events"]), encoding="utf-8")
    (TRACE_DIR / "network_backend.json").write_text(
        json.dumps(network["backend"], indent=2, allow_nan=False) + "\n", encoding="utf-8")

    receipts = {binding["message_id"]: A.extract_receipts(episode, message, network)
                for binding, message in
                ((abnormal_binding, messages[0]), (notify_binding, messages[1]),
                 (landing_binding, messages[2]))}
    (TRACE_DIR / "coupled_receipts.json").write_text(
        json.dumps(receipts, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # ---- phase A: vulnerable-phase movement admission + actual motion ----
    abnormal_receipt = receipts[ABNORMAL_MESSAGE_ID]
    abnormal_rx_ns = abnormal_receipt["accepted"]["accepted_ns"]        # 27_002_326_300
    admission = A.admit_movement_command(
        abnormal_binding, abnormal_receipt, ABNORMAL_ADMISSION_TICK * A.STEP_NS,
        movement_epoch0_active=True)
    # Negative: at the send instant itself no RX can be strictly earlier, so the
    # receiver must reject (no same-instant effect).
    send_instant_attempt = A.admit_movement_command(
        abnormal_binding, abnormal_receipt, ABNORMAL_SEND_TICK * A.STEP_NS,
        movement_epoch0_active=True)
    motion_exec = A.find_motion_execution(
        engine, A.MOVEMENT_ACTION, ABNORMAL_ADMISSION_TICK)
    motion_rows = A.summarize_motion_rows(
        engine.trajectory_rows,
        motion_exec["motion_window_start_ns"], motion_exec["motion_window_end_ns"])
    motion_bound = A.bind_motion_to_admission(admission, motion_exec)
    motion_bound["trajectory_summary"] = motion_rows
    (TRACE_DIR / "movement_admission_motion.json").write_text(json.dumps({
        "abnormal_rx_ns": abnormal_rx_ns,
        "rx_latency_after_send_ns": abnormal_rx_ns - ABNORMAL_SEND_TICK * A.STEP_NS,
        "admission_barrier_tick": ABNORMAL_ADMISSION_TICK,
        "admission_grid_wait_after_rx_ns":
            ABNORMAL_ADMISSION_TICK * A.STEP_NS - abnormal_rx_ns,
        "admission": admission,
        "negative_same_instant_attempt": send_instant_attempt,
        "motion_execution": motion_exec,
        "motion_bound": motion_bound,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # ---- phase B: owner-local isolation lockout + report ----
    lockout_tick = A.STORY["command_lockout_event_tick"]                # 305 event
    gcs_lockout_applied = A.STORY["command_lockout_event_tick"] + 1     # 306 actual
    uav_lockout_applied = A.STORY["command_lockout_event_tick"] + 5     # 310 actual
    gcs_lockout_action = executed_action(engine, "command_lockout_00")
    uav_lockout_action = executed_action(engine, "command_lockout_01")
    gcs_lockout_applied = gcs_lockout_action["result"]["effective_tick"]
    uav_lockout_applied = uav_lockout_action["result"]["effective_tick"]
    authority_before = A.movement_authority_state(engine, lockout_tick)
    authority_after = A.movement_authority_state(engine, uav_lockout_applied)
    report = A.lockout_report(engine, uav_lockout_applied)
    # Negative: the same native command replayed after the isolation revokes
    # epoch0 must be rejected by the receiver.
    revoked_admission = A.admit_movement_command(
        abnormal_binding, abnormal_receipt, (uav_lockout_applied + 1) * A.STEP_NS,
        movement_epoch0_active=authority_after["movement_epoch0_active"])
    (TRACE_DIR / "lockout_report.json").write_text(json.dumps({
        "authority_before_isolation": authority_before,
        "authority_after_isolation": authority_after,
        "lockout_report": report,
        "post_lockout_movement_attempt": revoked_admission,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # ---- phase C: secure recovery notification + GCS patch-before-send ----
    notify_receipt = receipts[NOTIFY_MESSAGE_ID]
    gcs_patch_action = executed_action(engine, "secure_recovery_00")
    uav_recovery_action = executed_action(engine, "secure_recovery_01")
    gcs_patch_applied_tick = gcs_patch_action["result"]["effective_tick"]     # 381
    uav_recovery_applied_tick = uav_recovery_action["result"]["effective_tick"]  # 385
    status_send_tick = A.admission_barrier_tick(gcs_patch_applied_tick)       # 385
    notify_observation_tick = A.admission_barrier_tick(
        A.STORY["secure_recovery_event_tick"] + 1)                            # 385
    notify_admission = A.admit_recovery_notification(
        notify_binding, notify_receipt, notify_observation_tick * A.STEP_NS,
        incident_movement_epoch=A.revoked_movement_epoch())
    order = A.same_tick_effect_order(gcs_patch_applied_tick, status_send_tick)
    secure_engine_record = A.secure_status_record(
        A.STORY["secure_recovery_event_tick"], gcs_patch_applied_tick, status_send_tick)
    secure_engine_record["patch_action_id"] = gcs_patch_action["action_id"]
    secure_engine_record["patch_effective_tick_source"] = \
        "engine.executed_actions result.effective_tick (actual applied state)"
    secure_engine_record["uav_recovery_patch"] = {
        "action_id": uav_recovery_action["action_id"],
        "effective_tick": uav_recovery_applied_tick}
    (TRACE_DIR / "secure_status.json").write_text(json.dumps({
        "notification_receipt_status": [row["status"] for row in notify_receipt["attempts"]],
        "notification_observation_tick": notify_observation_tick,
        "notification_admission": notify_admission,
        "same_tick_order": order,
        "secure_status_record": secure_engine_record,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # ---- phase D: UAV recovery effect (original due AND actual RX, no backdate) ----
    # The original due time IS the actual applied tick of the UAV recovery
    # patch (source event 380, original delay 5, actual effective_tick 385).
    recovery_due_tick = uav_recovery_applied_tick
    recovery_observation_tick = A.admission_barrier_tick(recovery_due_tick)   # 385
    recovery_effect = A.uav_recovery_effect(
        notify_admission, notify_receipt["accepted"] is not None,
        recovery_due_tick, recovery_observation_tick)

    # ---- phase E: landing gate (original due AND RX availability, no backdate) ----
    landing_receipt = receipts[LANDING_MESSAGE_ID]
    landing_rx_ns = landing_receipt["accepted"]["accepted_ns"]          # 46_000_223_477
    landing_due_tick = A.STORY["landing_event_tick"]                    # 460
    landing_observation_tick = A.admission_barrier_tick(landing_due_tick)  # 460
    landing_rx_available_at_due = landing_rx_ns < landing_observation_tick * A.STEP_NS
    landing_effect_at_due = A.landing_effect(
        notify_admission, landing_rx_available_at_due, landing_due_tick,
        landing_observation_tick)
    # The RX instant falls after the due instant (46.000223477 s > 46.0 s), so
    # the due-tick gate must reject; the actual application is the first grid
    # tick at which the RX is strictly available.
    landing_observation_tick = A.next_grid_tick_after_ns(landing_rx_ns)  # 465
    landing_effect = A.landing_effect(
        notify_admission, landing_receipt["accepted"] is not None,
        landing_due_tick, landing_observation_tick)
    landing_motion_exec = A.find_motion_execution(
        engine, A.LANDING_ACTION, landing_observation_tick)
    landing_motion_rows = A.summarize_motion_rows(
        engine.trajectory_rows,
        landing_motion_exec["motion_window_start_ns"],
        landing_motion_exec["motion_window_end_ns"])
    landing_bound = A.bind_motion_to_admission(
        {"decision": "admitted" if landing_effect["applied"] else "rejected",
         "message_id": LANDING_MESSAGE_ID,
         "admission_ns": landing_observation_tick * A.STEP_NS},
        landing_motion_exec)
    landing_bound["trajectory_summary"] = landing_motion_rows
    (TRACE_DIR / "recovery_landing_effects.json").write_text(json.dumps({
        "recovery_effect": recovery_effect,
        "landing_rx_ns": landing_rx_ns,
        "landing_rx_available_at_due": landing_rx_available_at_due,
        "landing_effect_at_due": landing_effect_at_due,
        "landing_effect": landing_effect,
        "landing_motion_execution": landing_motion_exec,
        "landing_motion_bound": landing_bound,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    summary = {
        "schema_version": A.SCHEMA, "episode_id": A.EPISODE_ID,
        "runtime_identity": A.runtime_identity(),
        "backend": {"name": network["backend"]["name"],
                    "ns3_version": network["backend"]["ns3_version"],
                    "image": network["backend"]["image"],
                    "network_profile": network["backend"]["network_profile"]},
        "flows": [{"message_id": m["message_id"],
                   "first_tx_ns": [p["first_tx_ns"] for p in network["network_packets"]
                                   if p["flow_id"] == m["message_id"]],
                   "rx_ns": [p["rx_ns"] for p in network["network_packets"]
                             if p["flow_id"] == m["message_id"]]}
                  for m in messages],
        "receipt_status": {mid: [row["status"] for row in rec["attempts"]]
                           for mid, rec in receipts.items()},
        "movement_admission_decision": admission["decision"],
        "movement_admission_barrier_tick": ABNORMAL_ADMISSION_TICK,
        "negative_same_instant_decision": send_instant_attempt["decision"],
        "motion_bound": motion_bound["bound"],
        "motion_nonclosure_reason": motion_bound["reason"],
        "lockout_reported": report["reported"],
        "post_lockout_movement_decision": revoked_admission["decision"],
        "notification_admission_decision": notify_admission["decision"],
        "notification_revives_movement": notify_admission["revives_revoked_movement_authority"],
        "secure_status": {"applied_tick": secure_engine_record["status_applied_tick"],
                          "sent_tick": secure_engine_record["status_sent_tick"]},
        "recovery_applied_tick": recovery_effect.get("applied_tick"),
        "landing_effect_at_due_applied": landing_effect_at_due["applied"],
        "landing_effect_at_due_reason": landing_effect_at_due["reason"],
        "landing_applied_tick": landing_effect.get("applied_tick"),
        "wall_time_s": network["wall_time_s"],
    }
    (TRACE_DIR / "coupled_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
