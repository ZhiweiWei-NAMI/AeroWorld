"""Primary/derived state contract for future state prediction.

Only PRIMARY_FIELDS are predicted as independent distributions.  Every other
modelled field is computed from primary values (and declared history context)
by the rules below; derived fields remain model inputs in the history graph.

Rules follow the producers (Dataset/tools/convert_to_render_ready.py,
Dataset/semantic_simulation/compute_comm.py) with three declared modelling
assumptions: yaw is the horizontal direction of motion, the observer body is
level, and the camera position equals the observer UAV's position.
"""
from __future__ import annotations

import math

# Position and instantaneous velocity are both primary (no kinematic tie).
PRIMARY_FIELDS = frozenset({"pose.position_enu_m", "pose.velocity_enu_mps",
                            "communication.latency_ms", "communication.handover_active"})

# Horizontal speed below which the motion direction is undefined and yaw holds
# (convert_to_render_ready.py heading rule).
YAW_HOLD_SPEED_MPS = 1e-5
# Link model (compute_comm.py; compute_comm_supplement_profile.json):
# latency = 20 + 200(1-q), loss = 0.25(1-q)^2, quality thresholds on q.
LATENCY_FLOOR_MS, LATENCY_SPAN_MS, LOSS_SCALE = 20., 200., .25
QUALITY_THRESHOLDS = ((.75, "excellent"), (.5, "good"), (.25, "fair"))

DERIVED_FIELDS = {
    "motion.speed_mps": {"inputs": ("pose.velocity_enu_mps",), "rule": "|v| (3D)", "kind": "definition"},
    "pose.yaw_deg": {"inputs": ("pose.velocity_enu_mps", "history pose.yaw_deg"),
                     "rule": f"atan2(vy, vx) when hypot(vx, vy) > {YAW_HOLD_SPEED_MPS}, else latest yaw",
                     "kind": "assumption: yaw is the direction of motion"},
    "communication.packet_loss_ratio": {"inputs": ("communication.latency_ms",),
                                        "rule": "0.25 ((latency-20)/200)^2", "kind": "simulator link model"},
    "communication.quality_level": {"inputs": ("communication.latency_ms",),
                                    "rule": "q = 1-(latency-20)/200; excellent>=.75, good>=.5, fair>=.25, poor>0, else down",
                                    "kind": "definition (threshold label)"},
    "observer.capture_position_world_enu_m": {"inputs": ("pose.position_enu_m",),
                                              "rule": "camera position = observer UAV position",
                                              "kind": "assumption: camera and UAV share one position (frame of pose.position_enu_m)"},
    "observer.body_orientation_ned_6d": {"inputs": ("pose.yaw_deg",),
                                         "rule": "[cos y, sin y, 0, -sin y, cos y, 0] (forward, right axes in NED)",
                                         "kind": "assumption: level body"},
}


def speed(velocity):
    return math.sqrt(sum(float(v)*float(v) for v in velocity))


def yaw(velocity, latest_yaw):
    vx, vy = float(velocity[0]), float(velocity[1])
    if math.hypot(vx, vy) > YAW_HOLD_SPEED_MPS:
        return math.degrees(math.atan2(vy, vx))
    if latest_yaw is None:
        raise ValueError("yaw is undefined: no horizontal motion and no yaw history")
    return float(latest_yaw)


def link_score(latency):
    return min(1., max(0., 1.-(float(latency)-LATENCY_FLOOR_MS)/LATENCY_SPAN_MS))


def packet_loss(latency):
    return LOSS_SCALE*(1.-link_score(latency))**2


def quality_level(latency):
    score = link_score(latency)
    for threshold, label in QUALITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "poor" if score > 0 else "down"


def capture_position(position):
    return [float(position[0]), float(position[1]), float(position[2])]


def body_orientation(yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [c, s, 0., -s, c, 0.]


def derive(primary, context):
    """Derived fields of one entity at one tick.

    ``primary`` maps primary families to values (vectors as lists);
    ``context`` carries ``latest_yaw`` and ``observer`` (whether the entity
    carries the camera).  Fields whose inputs are absent are omitted.
    """
    out = {}
    velocity = primary.get("pose.velocity_enu_mps")
    if velocity is not None:
        out["motion.speed_mps"] = speed(velocity)
        out["pose.yaw_deg"] = yaw(velocity, context.get("latest_yaw"))
    latency = primary.get("communication.latency_ms")
    if latency is not None:
        out["communication.packet_loss_ratio"] = packet_loss(latency)
        out["communication.quality_level"] = quality_level(latency)
    if context.get("observer") and primary.get("pose.position_enu_m") is not None:
        out["observer.capture_position_world_enu_m"] = capture_position(primary["pose.position_enu_m"])
        if "pose.yaw_deg" in out:
            out["observer.body_orientation_ned_6d"] = body_orientation(out["pose.yaw_deg"])
    return out
