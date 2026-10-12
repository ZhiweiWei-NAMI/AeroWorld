"""Declared local dedicated-core FIFO jobs on an integer nanosecond clock."""
from __future__ import annotations

VERSION = "p01.ns3.v1"
NS = 1_000_000_000


def simulate_local_tasks(episode, config):
    """Return events/tasks; observed executor departure/unavailability affects local jobs only.

    No hardware performance is inferred. Service is cycles/cpu_hz and jobs
    arrive at each life epoch's birth plus integer periods, before the horizon.
    memory_bytes is retained as a declared parameter; memory admission and
    storage occupied by queued jobs are not simulated.
    """
    parameters = config["compute"]
    cpu = parameters["cpu_hz"]
    cycles = parameters["cycles_per_job"]
    period = parameters["arrival_period_ns"]
    deadline = parameters["deadline_ns"]
    memory = parameters["memory_bytes"]
    if any(type(x) is not int or x <= 0 for x in (cpu, cycles, period, deadline, memory)):
        raise ValueError("local compute parameters must be positive integers")
    if (cycles * NS) % cpu:
        raise ValueError("declared service must have an exact integer-nanosecond duration")
    service = cycles * NS // cpu
    duration = episode["duration_ns"]
    events, tasks = [], []
    for node in episode["nodes"]:
        if node["node_type"] != "uav":
            continue
        birth, epoch_end = node["birth_ns"], node["death_ns"]
        stop = duration if epoch_end is None else min(duration, epoch_end)
        free = birth
        for number, arrival in enumerate(range(birth, stop, period)):
            task_id = f"{node['node_id']}:local:{number}"
            due = arrival + deadline
            start = max(arrival, free)
            finish = start + service
            base = {"episode_id": episode["episode_id"], "task_id": task_id, "executor_node_id": node["node_id"],
                    "owner": node["owner"], "life_epoch": node["life_epoch"],
                    "arrival_ns": arrival, "deadline_ns": due,
                    "cycles": cycles, "memory_bytes": memory,
                    "source_ref": episode["source_ref"], "config_ref": "run_config.json",
                    "producer_version": VERSION}
            def event(kind, when):
                events.append({"event_id": f"{task_id}:{kind}", "event_type": kind,
                    "time_ns": when, **base,
                    "reason": "observed_source_epoch_unavailability" if kind == "task_aborted_executor_unavailable" else None})
            event("task_arrival", arrival)
            observed_unavailability = epoch_end is not None and epoch_end < duration and finish >= epoch_end
            terminal = epoch_end if observed_unavailability else finish
            actual_start = start if start < min(stop, terminal) else None
            if actual_start is not None:
                event("task_start", actual_start)
            completed = not observed_unavailability and finish < duration
            if completed and finish <= due:
                deadline_missed, deadline_status, deadline_reason, deadline_available = False, "KNOWN", None, finish
            elif due < duration and (epoch_end is None or due < epoch_end):
                deadline_missed, deadline_status, deadline_reason, deadline_available = True, "KNOWN", None, due
                event("task_deadline_missed", due)
            elif observed_unavailability and epoch_end <= due:
                deadline_missed, deadline_status = None, "OUT_OF_SCOPE"
                deadline_reason, deadline_available = "executor_unavailable_before_or_at_deadline", epoch_end
            else:
                deadline_missed, deadline_status = None, "UNKNOWN"
                deadline_reason, deadline_available = "episode_horizon_before_deadline_outcome", None
            if observed_unavailability:
                event("task_aborted_executor_unavailable", epoch_end)
                status, reason, completion = "aborted", "observed_source_epoch_unavailability", None
                terminal_ns = epoch_end
            elif completed:
                event("task_complete", finish)
                status = "completed_late" if finish > due else "completed_on_time"
                reason, completion, terminal_ns = None, finish, finish
            else:
                status, reason, completion, terminal_ns = "right_censored", "episode_horizon", None, None
            tasks.append({**base, "start_ns": actual_start, "completion_ns": completion,
                "terminal_ns": terminal_ns, "service_ns": service,
                "wait_ns": None if actual_start is None else actual_start - arrival,
                "status": status, "reason": reason,
                "deadline_missed": deadline_missed, "deadline_missed_status": deadline_status,
                "deadline_missed_reason": deadline_reason, "deadline_missed_available_ns": deadline_available,
                "time_support": {"start_ns": arrival, "end_ns": duration, "end_exclusive": True},
                "units": {"time": "ns", "cycles": "cycle", "memory": "byte"},
                "dimensions": {"time": "time", "cycles": "count", "memory": "information"}})
            free = finish
        if epoch_end is not None and birth <= epoch_end < duration:
            events.append({"event_id": f"{node['node_id']}:executor_unavailable", "event_type": "executor_unavailable",
                "time_ns": epoch_end, "reason": "observed_source_epoch_unavailability", "executor_node_id": node["node_id"], "owner": node["owner"],
                "life_epoch": node["life_epoch"], "source_ref": episode["source_ref"],
                "config_ref": "run_config.json", "producer_version": VERSION})
    events.sort(key=lambda row: (row["time_ns"], row["event_id"]))
    return {"events": events, "tasks": tasks}
