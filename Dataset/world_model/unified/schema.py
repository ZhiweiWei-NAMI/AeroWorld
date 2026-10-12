"""Declared fields, quanta, horizons and answer alphabets (RESEARCH_PLAN §2-§4).

Every numeric field carries a quantum eps_f; values are stored as integers
k = round(x / eps_f) so quantization (the only lossy step, at most eps_f/2) is
explicit and compaction can be verified exactly on integers.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AW = ROOT/"aw_data"
WORKSPACE = ROOT/"Dataset/world_model/runs/unified"
QWEN_2B = "/mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/hf/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
QWEN_08B = "/mnt/data1/weizhiwei/AERO_WORLD_runtime/qwen35/hf/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

TICK_S = 0.1
HISTORY_TICKS = 50
HORIZONS = (5, 10, 20, 30, 50)
FORMAL_STEP = 5                       # world-graph, communication and compute cadence
LAST_TICK = 900
MOVING = ("uav", "vehicle", "pedestrian")

# Motion quanta: position 0.01 m, velocity 0.01 m/s (integers in cm, cm/s).
POS_EPS, VEL_EPS = 0.01, 0.01

# Entity fields read from truth frames besides motion.  Prefix entries select
# every bool/str leaf below them (the per-category semantic state flags).
FRAME_FIELDS = ("state", "annotations.activity_type", "sumo_vehicle.lane_id", "sumo_vehicle.controlling_signal_state",
                "sumo_vehicle.crossed_stop_line", "lifecycle.terminal_flight_state")
FRAME_PREFIXES = ("control_state.", "mission_state.", "navigation_state.", "security_state.", "sensor_state.",
                  "communication_state.", "incident_state.", "pedestrian_state.")
FRAME_NUMERIC = {"truth_pose.rotation_deg.yaw_deg": 1.0}

COMM_NUMERIC = {"link_quality.latency_ms": .1, "link_quality.packet_loss_ratio": 1e-4, "link_quality.quality_score": 1e-3,
                "link_quality.distance_m": .1, "link_quality.availability": 1e-3, "link_quality.capacity_factor": 1e-3,
                "link_quality.weather_attenuation": 1e-3, "bandwidth.allocated_mbps": .01, "bandwidth.dropped_mbps": .01,
                "channel.queued_bandwidth_mbps": .01, "heartbeat_age_ms": .1, "retransmission.count": 1.,
                "handover_policy.dwell_elapsed_ticks": 1.}
COMM_CATEGORICAL = ("link_quality.quality_level", "handover.active", "handover.current_station_id", "handover_policy.decision",
                    "handover_policy.switch_allowed", "station_id", "station_operational_state.state_mode", "scope_active",
                    "retransmission.reason")
COMPUTE_NUMERIC = {"cpu_utilization_pct": .01, "gpu_utilization_pct": .01, "memory_used_mb": .1, "queue_depth": 1.,
                   "allocation.cpu_cores": 1e-3, "offload.accepted_cpu_cores": 1e-3, "migration.transferred_cpu_cores": 1e-3,
                   "resource_balance.cpu_queued": 1e-3, "resource_balance.cpu_requested": 1e-3, "deadline.slack_ticks": 1.}
COMPUTE_CATEGORICAL = ("execution_state", "node_availability.available", "migration.active", "migration.target_node_id",
                       "offload.enabled", "offload.target_node_id", "failure.mode", "deadline.status", "task_kind")
# Communication / compute object identifiers and the owner entity's row field that holds them.
COMM_OBJECTS = {"flow_id": "flow", "handover_id": "handover", "session_id": "session", "channel_allocation_id": "allocation",
                "transmission_attempt_id": "attempt", "retransmission_id": "retransmission", "message_id": "message"}
COMPUTE_OBJECTS = {"node_id": "node", "task_id": "task", "queue_id": "queue", "execution_id": "execution",
                   "allocation_id": "resource_allocation", "migration_id": "migration", "offload_id": "offload",
                   "failure_state_id": "failure_state", "resource_id": "resource"}

# Evidence quanta by path suffix (world-graph assertion observations).
EVIDENCE_EPS = ((("_m", "_meters"), .01), (("_mps",), .01), (("_s", "_seconds"), .1), (("_deg",), .1),
                (("_ratio",), 1e-4), (("_pct",), .01), (("_tick", "_ticks", "_count"), 1.))
EVIDENCE_DEFAULT_EPS = 1e-3
EVIDENCE_SKIP = {"direct_predicate_value"}   # repeats the relation value itself


def evidence_eps(path: str) -> float:
    for suffixes, eps in EVIDENCE_EPS:
        if path.endswith(suffixes):
            return eps
    return EVIDENCE_DEFAULT_EPS


# Numeric query targets: (family, source, field, quantum, categories).  Position is a 3-vector.
NUMERIC_TARGETS = {"position": ("frame", "position", POS_EPS),
                   "latency_ms": ("comm", "link_quality.latency_ms", .1),
                   "cpu_utilization_pct": ("compute", "cpu_utilization_pct", .01)}

FLIP_BINS = ((0, 5), (5, 10), (10, 20), (20, 30), (30, 50))     # (a, b] in ticks after the origin; class 5 is ">5 s"
FLIP_LABELS = ("(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]", "(3,5]", ">5")
EVENT_LABELS = ("none",) + FLIP_LABELS[:5]

# Answer alphabet: single Qwen tokens; boolean domains use the native words.
ALPHABET = tuple(" "+c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
BOOLEAN_TOKENS = (" false", " true")

ALIAS_PREFIX = {"uav": "u", "vehicle": "v", "pedestrian": "p", "airspace_corridor": "c", "facility": "f", "prop": "o",
                "crowd_anchor": "a", "facade_anchor": "b", "vehicle_anchor": "w", "traffic_light": "l"}
OBJECT_PREFIX = {"flow": "fl", "handover": "ho", "session": "se", "allocation": "al", "attempt": "at",
                 "retransmission": "rt", "message": "ms", "node": "nd", "task": "tk", "queue": "qu", "execution": "ex",
                 "resource_allocation": "ra", "migration": "mg", "offload": "of", "failure_state": "fs", "resource": "rs",
                 "station": "st", "other": "x"}

# Short field names used in serialized text (unique within their source).
SHORT_NAMES = {
    "annotations.activity_type": "activity", "truth_pose.rotation_deg.yaw_deg": "yaw_deg",
    "sumo_vehicle.lane_id": "lane", "sumo_vehicle.controlling_signal_state": "signal", "sumo_vehicle.crossed_stop_line": "crossed_stop_line",
    "lifecycle.terminal_flight_state": "terminal_flight_state",
    "link_quality.latency_ms": "latency_ms", "link_quality.packet_loss_ratio": "loss", "link_quality.quality_score": "quality",
    "link_quality.distance_m": "distance_m", "link_quality.availability": "availability", "link_quality.capacity_factor": "capacity_factor",
    "link_quality.weather_attenuation": "weather_att", "bandwidth.allocated_mbps": "bw_alloc_mbps", "bandwidth.dropped_mbps": "bw_drop_mbps",
    "channel.queued_bandwidth_mbps": "bw_queued_mbps", "heartbeat_age_ms": "heartbeat_ms", "retransmission.count": "retx",
    "handover_policy.dwell_elapsed_ticks": "dwell_ticks", "link_quality.quality_level": "level", "handover.active": "handover",
    "handover.current_station_id": "serving", "handover_policy.decision": "ho_decision", "handover_policy.switch_allowed": "switch_allowed",
    "station_id": "station", "station_operational_state.state_mode": "station_mode", "scope_active": "active",
    "retransmission.reason": "retx_reason",
    "cpu_utilization_pct": "cpu_utilization_pct", "gpu_utilization_pct": "gpu_pct", "memory_used_mb": "mem_mb", "queue_depth": "queue",
    "allocation.cpu_cores": "cpu_alloc", "offload.accepted_cpu_cores": "offload_cpu", "migration.transferred_cpu_cores": "migrated_cpu",
    "resource_balance.cpu_queued": "cpu_queued", "resource_balance.cpu_requested": "cpu_requested", "deadline.slack_ticks": "slack_ticks",
    "execution_state": "exec", "node_availability.available": "node_available", "migration.active": "migrating",
    "migration.target_node_id": "migrate_to", "offload.enabled": "offloading", "offload.target_node_id": "offload_to",
    "failure.mode": "failure", "deadline.status": "deadline", "task_kind": "task_kind"}


def short(name: str) -> str:
    return SHORT_NAMES.get(name, name.split(".")[-1])


def code_version():
    """Commit of the working tree and whether the unified code differs from it (provenance of runs and records)."""
    import subprocess
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "unified_dirty": bool(run("status", "--porcelain", "--", "Dataset/world_model/unified"))}
