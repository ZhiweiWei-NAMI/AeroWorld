"""Versioned packet/task finals and strict-before-time 2Hz state projections."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

VERSION = "p01.ns3.v1"
STEP_NS = 500_000_000
JSONL_ARTIFACTS = (
    "node_registry.jsonl", "network_events.jsonl", "network_packets.jsonl",
    "network_queue_samples.jsonl", "network_diagnostics.jsonl", "compute_events.jsonl",
    "compute_tasks.jsonl", "network_states.jsonl", "compute_states.jsonl", "predicates.jsonl",
)


def make_artifact_files(compression):
    """Declare exact storage names once, before simulation or projection."""
    if compression not in ("none", "gzip"):
        raise ValueError("log_compression must be explicitly none or gzip")
    suffix = ".gz" if compression == "gzip" else ""
    return {name: name + suffix for name in JSONL_ARTIFACTS}


def resolve_artifact_ref(reference, artifact_files):
    """Resolve exact local logical references; preserve external input paths."""
    return artifact_files[reference] if reference in artifact_files else reference


def _metadata(episode, source, artifact_files):
    return {"episode_id": episode["episode_id"], "producer_version": VERSION,
            "source_ref": resolve_artifact_ref(source, artifact_files), "config_ref": "run_config.json",
            "units": {"time": "ns"}, "dimensions": {"time": "time"}}


def write_jsonl(path, rows, *, compression):
    """Exclusive JSONL output in one of the two declared storage formats."""
    if compression == "gzip":
        stream = gzip.open(path, "xt", encoding="utf-8", compresslevel=1)
    elif compression == "none":
        stream = path.open("x", encoding="utf-8")
    else:
        raise ValueError("log_compression must be explicitly none or gzip")
    with stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def _stat(value, unit, dimension, reason=None):
    return {"value": value, "status": "KNOWN" if reason is None else "UNKNOWN",
            "reason": reason, "unit": unit, "dimension": dimension}


def _active(node, t):
    # Registry availability scope is [birth, epoch end); event statistics use times < t.
    return node["birth_ns"] <= t and (node["death_ns"] is None or t < node["death_ns"])


def write_episode_logs(output_dir, episode, config, network, compute):
    """Write the owned JSONL products and return state lists plus actual summary.

    Cohorts contain accepted datagrams whose expiry falls in [t-window,t).
    Latency uses only RX strictly before expiry, in that same mature cohort.
    Packet TTL loss, local deadline misses and observed local executor unavailability are distinct.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    compression, artifact_files = config["log_compression"], config["artifact_files"]
    duration = episode["duration_ns"]
    ttl, window = config["network"]["packet_ttl_ns"], config["network"]["loss_window_ns"]
    if type(ttl) is not int or type(window) is not int or ttl <= 0 or window <= 0:
        raise ValueError("TTL and cohort window must be positive integer nanoseconds")
    if duration % STEP_NS:
        raise ValueError("episode horizon must align with the declared 2Hz state grid")
    nodes = {row["node_id"]: row for row in episode["nodes"]}
    links = {row["link_id"]: row for row in network["links"]}
    grouped, packet_rows, ttl_events = defaultdict(list), [], []
    queue_rows = []
    for sample in network["network_queue_samples"]:
        node = nodes[sample["node_id"]]
        queue_rows.append({**sample, **_metadata(episode, network["backend"]["source_ref"], artifact_files),
            "owner": node["owner"], "life_epoch": node["life_epoch"],
            "status": "KNOWN", "reason": None,
            "time_support": {"start_ns": max(0, sample["observed_until_ns"]), "end_ns": sample["time_ns"], "end_exclusive": True},
            "units": {"time": "ns", "queue_packets": "packet", "queue_bytes": "byte"},
            "dimensions": {"time": "time", "queue_packets": "count", "queue_bytes": "information"}})
    queue_index = {(row["time_ns"], row["node_id"]): row for row in queue_rows}
    diagnostic_rows = []
    diagnostic_kinds, diagnostic_scopes, diagnostic_reasons = Counter(), Counter(), Counter()
    for diagnostic in network["network_diagnostics"]:
        node = nodes[diagnostic["node_id"]]
        diagnostic_rows.append({**diagnostic, **_metadata(episode, network["backend"]["source_ref"], artifact_files),
            "owner": node["owner"], "life_epoch": node["life_epoch"], "status": "KNOWN",
            "reason_status": "UNKNOWN" if diagnostic["reason"] is None else "KNOWN",
            "time_support": {"start_ns": diagnostic["first_time_ns"], "end_ns": diagnostic["last_time_ns"], "end_exclusive": False},
            "units": {"time": "ns", "count": "event"}, "dimensions": {"time": "time", "count": "count"}})
        diagnostic_kinds[diagnostic["kind"]] += diagnostic["count"]
        diagnostic_scopes[diagnostic["scope"]] += diagnostic["count"]
        diagnostic_reasons[(diagnostic["kind"], diagnostic["reason"], diagnostic["scope"])] += diagnostic["count"]
    for packet in network["network_packets"]:
        identifier = packet["packet_id"]
        link = links[packet["link_id"]]
        src, dst = nodes[packet["src_node_id"]], nodes[packet["dst_node_id"]]
        expiry = packet["first_tx_ns"] + ttl
        rx = packet["rx_ns"]
        on_time = packet["accepted"] and rx is not None and rx < expiry and rx < duration
        mature = packet["accepted"] and expiry < duration
        if not packet["accepted"]:
            status, reason = "tx_rejected", "socket_not_accepted"
        elif on_time:
            status, reason = "delivered_on_time", None
        elif mature:
            status, reason = "ttl_expired", "no_rx_strictly_before_ttl"
        else:
            status, reason = "right_censored", "episode_horizon"
        row = {**packet, **_metadata(episode, "network_events.jsonl", artifact_files),
            "owner": src["owner"], "life_epoch": src["life_epoch"],
            "receiver_owner": dst["owner"], "receiver_life_epoch": dst["life_epoch"],
            "expiry_ns": expiry, "on_time_delivery": on_time, "mature_at_horizon": mature,
            "status": status, "reason": reason,
            "late_rx_ns": rx if packet["accepted"] and rx is not None and rx >= expiry else None,
            "time_support": {"start_ns": packet["first_tx_ns"], "end_ns": duration, "end_exclusive": True},
            "payload_unit": "byte", "payload_dimension": "information"}
        packet_rows.append(row)
        if packet["accepted"]:
            grouped[packet["link_id"]].append(row)
            if mature:
                ttl_events.append({"event_id": f"{identifier}:ttl_outcome", "event_type": "ttl_delivered" if on_time else "ttl_expired",
                    "time_ns": expiry, "packet_id": identifier, "link_id": packet["link_id"],
                    "src_node_id": src["node_id"], "dst_node_id": dst["node_id"], "flow_id": packet["flow_id"],
                    "owner": src["owner"], "life_epoch": src["life_epoch"], "status": "KNOWN",
                    "reason": None if on_time else "no_rx_strictly_before_ttl",
                    "time_support": {"start_ns": packet["first_tx_ns"], "end_ns": expiry, "end_exclusive": True},
                    **_metadata(episode, "network_packets.jsonl", artifact_files)})
    indexed = {}
    for link_id, rows in grouped.items():
        rows.sort(key=lambda row: (row["expiry_ns"], row["packet_id"]))
        expiries = [row["expiry_ns"] for row in rows]
        delivered, delay = [0], [0]
        for row in rows:
            delivered.append(delivered[-1] + int(row["on_time_delivery"]))
            delay.append(delay[-1] + (row["rx_ns"] - row["first_tx_ns"] if row["on_time_delivery"] else 0))
        indexed[link_id] = expiries, delivered, delay
    network_states = []
    for t in range(0, duration + 1, STEP_NS):
        for link_id, link in links.items():
            src, dst = nodes[link["src_node_id"]], nodes[link["dst_node_id"]]
            expiries, successes, delays = indexed.get(link_id, ([], [0], [0]))
            left, right = bisect_left(expiries, t - window), bisect_left(expiries, t)
            count = right - left
            delivered = successes[right] - successes[left]
            lost = count - delivered
            active = _active(src, t) and _active(dst, t)
            reason = None if active else "owner_life_epoch_inactive"
            loss_reason = reason or ("no_mature_accepted_datagrams" if count == 0 else None)
            latency_reason = reason or ("no_on_time_delivered_mature_datagrams" if delivered == 0 else None)
            queue = queue_index.get((t, src["node_id"]))
            queue_reason = reason or ("no_mac_queue_sample" if queue is None else None)
            network_states.append({**_metadata(episode, "network_packets.jsonl", artifact_files),
                "time_ns": t, "link_id": link_id, "flow_id": link["flow_id"],
                "src_node_id": src["node_id"], "dst_node_id": dst["node_id"],
                "owner": src["owner"], "life_epoch": src["life_epoch"],
                "receiver_owner": dst["owner"], "receiver_life_epoch": dst["life_epoch"],
                "status": "IN_SCOPE" if active else "OUT_OF_SCOPE", "reason": reason,
                "time_support": {"start_ns": max(0, t - window), "end_ns": t, "end_exclusive": True,
                                 "anchor": "packet_expiry_ns", "ttl_ns": ttl},
                "mature_accepted_count": count, "on_time_delivered_count": delivered, "ttl_loss_count": lost,
                "latency_mean_ms": _stat((delays[right] - delays[left]) / delivered / 1_000_000 if latency_reason is None else None, "ms", "time", latency_reason),
                "packet_loss_ratio": _stat(lost / count if loss_reason is None else None, "1", "dimensionless", loss_reason),
                "mac_queue_depth": _stat(queue["queue_packets"] if queue_reason is None else None, "packet", "count", queue_reason),
                "mac_queue_bytes": _stat(queue["queue_bytes"] if queue_reason is None else None, "byte", "information", queue_reason),
                "mac_queue_observed_until_ns": None if queue is None else queue["observed_until_ns"],
                "mac_queue_source_ref": artifact_files["network_queue_samples.jsonl"]})
    task_groups = defaultdict(list)
    for task in compute["tasks"]:
        task_groups[task["executor_node_id"]].append(task)
    compute_states = []
    for t in range(0, duration + 1, STEP_NS):
        for node in episode["nodes"]:
            if node["node_type"] != "uav":
                continue
            rows = task_groups[node["node_id"]]
            arrived = [row for row in rows if row["arrival_ns"] < t]
            completed = [row for row in arrived if row["completion_ns"] is not None and row["completion_ns"] < t]
            aborted = [row for row in arrived if row["status"] == "aborted" and row["terminal_ns"] < t]
            live = [row for row in arrived if (row["completion_ns"] is None or row["completion_ns"] >= t)
                    and (row["terminal_ns"] is None or row["terminal_ns"] >= t)]
            waiting = [row for row in live if row["start_ns"] is None or row["start_ns"] >= t]
            busy = len(live) - len(waiting)
            active = _active(node, t)
            reason = None if active else "owner_life_epoch_inactive"
            waits = [row["wait_ns"] for row in arrived if row["start_ns"] is not None and row["start_ns"] < t]
            known_deadlines = [row for row in arrived if row["deadline_missed_available_ns"] is not None and row["deadline_missed_available_ns"] < t]
            missed = sum(row["deadline_missed"] is True for row in known_deadlines)
            met = sum(row["deadline_missed"] is False for row in known_deadlines)
            out_of_scope = sum(row["deadline_missed_status"] == "OUT_OF_SCOPE" for row in known_deadlines)
            compute_states.append({**_metadata(episode, "compute_events.jsonl", artifact_files),
                "time_ns": t, "executor_node_id": node["node_id"], "owner": node["owner"], "life_epoch": node["life_epoch"],
                "status": "IN_SCOPE" if active else "OUT_OF_SCOPE", "reason": reason,
                "time_support": {"start_ns": min(node["birth_ns"], t), "end_ns": t, "end_exclusive": True},
                "arrived_count": len(arrived), "completed_count": len(completed), "aborted_count": len(aborted),
                "deadline_missed_count": missed, "deadline_met_count": met,
                "deadline_unknown_count": len(arrived) - missed - met - out_of_scope,
                "deadline_out_of_scope_count": out_of_scope,
                "queue_depth": _stat(len(waiting) if active else None, "task", "count", reason),
                "running_count": _stat(busy if active else None, "task", "count", reason),
                "wait_mean_ms": _stat(sum(waits) / len(waits) / 1_000_000 if waits and active else None, "ms", "time", reason or (None if waits else "no_observed_task_starts")),
                "cpu_hz": config["compute"]["cpu_hz"], "memory_bytes": config["compute"]["memory_bytes"],
                "units": {"time": "ns", "cpu": "Hz", "memory": "byte", "count": "task"},
                "dimensions": {"time": "time", "cpu": "frequency", "memory": "information", "count": "count"}})
    events = []
    for raw in network["network_events"]:
        src = nodes[raw["src_node_id"]]
        events.append({**raw, **_metadata(episode, episode["source_ref"], artifact_files),
            "owner": src["owner"], "life_epoch": src["life_epoch"], "status": "KNOWN", "reason": None,
            "time_support": {"start_ns": raw["time_ns"], "end_ns": raw["time_ns"], "end_exclusive": False},
            "units": {"time": "ns", "payload": "byte"}, "dimensions": {"time": "time", "payload": "information"}})
    events.extend(ttl_events)
    events.sort(key=lambda row: (row["time_ns"], row["event_id"]))
    compute_events = [{**row, **_metadata(episode, episode["source_ref"], artifact_files), "status": "KNOWN", "reason": row["reason"],
        "time_support": {"start_ns": row["time_ns"], "end_ns": row["time_ns"], "end_exclusive": False},
        "units": {"time": "ns", "cycles": "cycle", "memory": "byte"},
        "dimensions": {"time": "time", "cycles": "count", "memory": "information"}} for row in compute["events"]]
    products = {"network_events.jsonl": events, "network_packets.jsonl": packet_rows,
        "compute_events.jsonl": compute_events, "compute_tasks.jsonl": compute["tasks"],
        "network_queue_samples.jsonl": queue_rows, "network_diagnostics.jsonl": diagnostic_rows,
        "network_states.jsonl": network_states, "compute_states.jsonl": compute_states}
    if "network_geometry_samples.jsonl" in artifact_files:
        geometry_rows = []
        for sample in network["geometry_samples"]:
            src, dst = nodes[sample["src_node_id"]], nodes[sample["dst_node_id"]]
            active = _active(src, sample["time_ns"]) and _active(dst, sample["time_ns"])
            geometry_rows.append({**sample,
                **_metadata(episode, network["backend"]["source_ref"], artifact_files),
                "status": "KNOWN" if active else "OUT_OF_SCOPE",
                "reason": None if active else "source_or_receiver_epoch_unavailable",
                "coordinate_contract": "coord.external_enu_m.v1",
                "units": {"time": "ns", "position": "m", "distance": "m", "loss": "dB", "exponent": "1"},
                "dimensions": {"time": "time", "position": "length", "distance": "length", "loss": "power_ratio", "exponent": "dimensionless"}})
        products["network_geometry_samples.jsonl"] = geometry_rows
    for name, rows in products.items():
        write_jsonl(output / artifact_files[name], rows, compression=compression)
    task_waits = [row["wait_ns"] for row in compute["tasks"] if row["wait_ns"] is not None]
    mature_packets = [row for row in packet_rows if row["mature_at_horizon"]]
    accepted = sum(row["accepted"] for row in packet_rows)
    mature_delivered = [row for row in mature_packets if row["on_time_delivery"]]
    summary = {"producer_version": VERSION, "episode_id": episode["episode_id"], "duration_ns": duration,
        "config_ref": "run_config.json", "source_ref": episode["source_ref"],
        "state_time_count": duration // STEP_NS + 1, "network_state_rows": len(network_states), "compute_state_rows": len(compute_states),
        "network": {"backend": network["backend"], "queue_sample_count": len(queue_rows),
            "diagnostic_row_count": len(diagnostic_rows), "diagnostic_kind_counts": dict(diagnostic_kinds),
            "diagnostic_scope_counts": dict(diagnostic_scopes),
            "diagnostic_kind_reason_scope_counts": [{"kind": kind, "reason": reason, "scope": scope, "count": count}
                for (kind, reason, scope), count in diagnostic_reasons.items()], "packet_count": len(packet_rows), "socket_accepted_count": accepted,
            "socket_rejected_count": len(packet_rows) - accepted,
            "right_censored_count": sum(row["status"] == "right_censored" for row in packet_rows),
            "mature_on_time_delivered_count": len(mature_delivered),
            "mature_latency_mean_ms": _stat(sum(row["rx_ns"] - row["first_tx_ns"] for row in mature_delivered) / len(mature_delivered) / 1_000_000 if mature_delivered else None, "ms", "time", None if mature_delivered else "no_on_time_delivered_mature_datagrams"),
            "mature_at_horizon_count": len(mature_packets), "mature_ttl_loss_count": sum(not row["on_time_delivery"] for row in mature_packets),
            "packet_status_counts": dict(Counter(row["status"] for row in packet_rows)),
            "late_rx_count": sum(row["late_rx_ns"] is not None for row in packet_rows),
            "ttl_ns": ttl, "cohort_window_ns": window, "cohort_anchor": "expiry_deadline",
            "wall_time_s": network["wall_time_s"], "peak_rss_kb": network["peak_rss_kb"]},
        "compute": {"task_count": len(compute["tasks"]), "task_status_counts": dict(Counter(row["status"] for row in compute["tasks"])),
            "task_deadline_missed_count": sum(row["deadline_missed"] is True for row in compute["tasks"]),
            "task_deadline_met_count": sum(row["deadline_missed"] is False for row in compute["tasks"]),
            "task_deadline_unknown_count": sum(row["deadline_missed_status"] == "UNKNOWN" for row in compute["tasks"]),
            "task_deadline_out_of_scope_count": sum(row["deadline_missed_status"] == "OUT_OF_SCOPE" for row in compute["tasks"]),
            "mean_observed_wait_ns": sum(task_waits) / len(task_waits) if task_waits else None,
            "service_ns": config["compute"]["cycles_per_job"] * 1_000_000_000 // config["compute"]["cpu_hz"],
            "task_deadline_ns": config["compute"]["deadline_ns"],
            "arrival_period_ns": config["compute"]["arrival_period_ns"],
            "service_capacity_jobs_per_s": config["compute"]["cpu_hz"] / config["compute"]["cycles_per_job"],
            "offered_jobs_per_s_per_active_executor": 1_000_000_000 / config["compute"]["arrival_period_ns"],
            "offered_utilization": (config["compute"]["cycles_per_job"] * 1_000_000_000 / config["compute"]["cpu_hz"]) / config["compute"]["arrival_period_ns"],
            "memory_parameter_bytes": config["compute"]["memory_bytes"],
            "memory_model": "declared_parameter_no_memory_admission_or_queue_storage_model",
            "termination_semantics": "source_epoch_departure_or_unavailability_not_proven_physical_death",
            "model": "synthetic_dedicated_core_local_nonpreemptive_fifo"},
        "log_compression": compression,
        "artifact_files": {name: artifact_files[name] for name in products},
        "output_bytes": {artifact_files[name]: (output / artifact_files[name]).stat().st_size for name in products}}
    return {"network_states": network_states, "compute_states": compute_states, "summary": summary}
