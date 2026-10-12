"""One approved native command trace for L6-5_v1 seed00.

Runs the existing shared ns-3.48 Yans/UDP transport
(``Dataset.semantic_simulation.ns3_episode.provider_adapter``) once for the one
approved command representative: the original 9 m/s abnormal command
(``move_uav_intrusion_abnormal``) sent by the original GCS endpoint in its
authorized vulnerable-phase control role, epoch0, to the UAV epoch0.

The run records the raw native trace (packet/tx/rx records), the shared
per-attempt receipt rows bound to the adapter's authority binding, and the
controller admission decision. No analytical delivery, no fabricated TX/RX,
no radio parameter change: the radio configuration is the existing declared R1
reference document read verbatim.

Run (airfogsim python, from the repository root):
  /home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python -m \
      Dataset.semantic_simulation.p09_l6_5_receipt_v1.command_trace
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
TRACE_DIR = Path(__file__).resolve().parent / "trace"
NATIVE_SEED = 1  # existing p01 convention: ns-3 RNG seed independent of episode seed
RUN = 1

# The one approved representative command: original abnormal move, vulnerable
# phase, GCS epoch0 movement control authority, UAV epoch0 receiver.
MESSAGE_ID = f"{A.EPISODE_ID}:command:{A.MOVEMENT_ACTION}"
SEND_TICK = A.STORY["abnormal_command_event_tick"]      # 270
EVIDENCE_TICK = A.STORY["gcs_intrusion_event_tick"]     # 240, sender decision evidence
ADMISSION_TICK = A.STORY["uav_recovery_due_tick"]       # 275 receiver observation barrier


def load_json(path):
    import json as _json
    return _json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_episode():
    """Actual source trajectory for the two bound endpoints, via linked_engine."""
    scene = load_json(SCENE)
    script = load_json(SCRIPT)
    run = execute_linked(scene, script, SCRIPT, A.EPISODE_ID, seed=0)
    from Dataset.semantic_simulation.ns3_episode.linked_transport import episode_from_engine
    profile = {
        "episode_id": A.EPISODE_ID, "source_script": str(SCRIPT),
        "actor_ids": [A.UAV], "primary_station": A.GCS,
    }
    episode = episode_from_engine(profile, run["engine"])
    return episode


def main():
    if TRACE_DIR.exists() and any(TRACE_DIR.iterdir()):
        raise FileExistsError(f"trace output already exists: {TRACE_DIR}")
    TRACE_DIR.mkdir(parents=True)
    episode = build_episode()
    send_ns = SEND_TICK * A.STEP_NS
    evidence_ns = EVIDENCE_TICK * A.STEP_NS

    # Authority binding BEFORE transport: role/epoch from the predeclared
    # grant table; the payload carries no authority.
    binding = A.bind_command(
        MESSAGE_ID, A.MOVEMENT_ACTION, A.MOVEMENT_ROLE,
        {"claimed_roles": [A.MOVEMENT_ROLE]},  # payload claims are recorded, never authoritative
        source_owner=A.GCS, source_epoch=0, receiver_owner=A.UAV, receiver_epoch=0,
        control_epoch=0)

    # Existing transport window contract (attempts x period bounded flow).
    message = A.build_message(episode, binding, SEND_TICK, evidence_ns)
    flows = [message["flow"]]

    config = {
        "network_profile": "timed_flow",
        "network": {"traffic_windows": [
            {key: row[key] for key in ("flow_id", "link_id", "src_owner", "life_epoch",
                                       "dst_owner", "dst_life_epoch", "payload_bytes",
                                       "period_ns", "start_ns", "end_ns")}
            for row in flows]},
        "seed": NATIVE_SEED, "run": RUN,
        "radio": load_json(RADIO_REF),
        "radio_actions": [],
        "radio_config_source_ref": str(RADIO_REF),
    }

    (TRACE_DIR / "command_input.json").write_text(json.dumps({
        "schema_version": A.SCHEMA, "episode_id": A.EPISODE_ID, "scenario_id": A.SCENARIO_ID,
        "runtime_identity": A.runtime_identity(),
        "source_event_script": str(SCRIPT),
        "message_id": MESSAGE_ID, "action": A.MOVEMENT_ACTION,
        "declared_role": A.MOVEMENT_ROLE, "source_owner": A.GCS, "source_epoch": 0,
        "receiver_owner": A.UAV, "receiver_epoch": 0,
        "send_tick": SEND_TICK, "send_ns": send_ns,
        "evidence_tick": EVIDENCE_TICK, "evidence_ns": evidence_ns,
        "admission_barrier_tick": ADMISSION_TICK,
        "command_transport": A.COMMAND,
        "flow": message["flow"],
        "payload_claimed_roles": binding["payload_claimed_roles"],
        "authority_source": binding["authority_source"],
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

    receipts = A.extract_receipts(episode, message, network)
    (TRACE_DIR / "command_receipts.json").write_text(
        json.dumps(receipts, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    barrier_ns = ADMISSION_TICK * A.STEP_NS
    admission = A.admit_movement_command(binding, receipts, barrier_ns,
                                         movement_epoch0_active=True)
    (TRACE_DIR / "controller_admission.json").write_text(
        json.dumps(admission, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    summary = {
        "schema_version": A.SCHEMA,
        "episode_id": A.EPISODE_ID,
        "message_id": MESSAGE_ID,
        "runtime_identity": A.runtime_identity(),
        "native_first_tx_ns": [row["first_tx_ns"] for row in network["network_packets"]],
        "native_rx_ns": [row["rx_ns"] for row in network["network_packets"]],
        "attempt_receipt_status": [row["status"] for row in receipts["attempts"]],
        "accepted_ns": None if receipts["accepted"] is None else receipts["accepted"]["accepted_ns"],
        "admission_barrier_tick": ADMISSION_TICK,
        "admission_decision": admission["decision"],
        "admission_reason": admission["reason"],
        "backend": {"name": network["backend"]["name"], "ns3_version": network["backend"]["ns3_version"],
                    "image": network["backend"]["image"], "network_profile": network["backend"]["network_profile"]},
        "wall_time_s": network["wall_time_s"],
    }
    (TRACE_DIR / "trace_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
