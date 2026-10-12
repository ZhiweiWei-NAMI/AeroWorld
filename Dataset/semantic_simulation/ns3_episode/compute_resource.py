"""Explicit local CPU/memory FIFO events; no instantaneous offload fiction.

All durations are integer nanoseconds. Workloads and capacities are declared
simulation assumptions, not measured aircraft hardware. The actor lifecycle is
read from frozen truth; these jobs do not command or modify actor motion.
"""
from __future__ import annotations

from collections import deque
import heapq

VERSION = "p09.compute.local-resource-fifo/v3"
NS = 1_000_000_000


def _arrival_specs(node, stop, p):
    """Resolve one explicit arrival policy before scheduling; no label inputs."""
    if p['arrival_process'] == 'declared_periodic_bursts':
        if not isinstance(p['job_class'], str) or not p['job_class']:
            raise ValueError('periodic jobs require an explicit job_class')
        for key in ('cycles_per_job', 'arrival_period_ns', 'burst_jobs', 'task_deadline_duration_ns'):
            if type(p[key]) is not int or p[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        footprints = p['job_memory_cycle_bytes']
        if not isinstance(footprints, list) or not footprints or any(type(x) is not int or x <= 0 for x in footprints):
            raise ValueError('job_memory_cycle_bytes must be explicit positive integer bytes')
        number = 0
        for when in range(node['birth_ns'], stop, p['arrival_period_ns']):
            for _ in range(p['burst_jobs']):
                yield {'job_id':str(number), 'arrival_ns':when, 'cycles':p['cycles_per_job'],
                       'memory_bytes':footprints[number % len(footprints)],
                       'deadline_duration_ns':p['task_deadline_duration_ns'],
                       'job_class':p['job_class'], 'workload_source_ref':'run_config.json#/compute'}
                number += 1
    elif p['arrival_process'] == 'declared_task_manifest':
        rows = p['task_manifest']
        if not isinstance(rows, list):
            raise ValueError('task_manifest must be a list, including explicit empty arrivals')
        ids = set()
        fields = {'job_id','owner_entity_id','actor_epoch','arrival_ns','cycles','memory_bytes',
                  'deadline_duration_ns','job_class','workload_source_ref'}
        for row in rows:
            if set(row) != fields:
                raise ValueError('task manifest row fields do not match the explicit contract')
            for key in ('job_id','owner_entity_id','job_class','workload_source_ref'):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f'{key} must be a nonempty string')
            for key in ('cycles','memory_bytes','deadline_duration_ns'):
                if type(row[key]) is not int or row[key] <= 0:
                    raise ValueError(f'{key} must be a positive integer')
            for key in ('arrival_ns','actor_epoch'):
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError(f'{key} must be a nonnegative integer')
            if row['job_id'] in ids:
                raise ValueError('duplicate task-manifest job ID')
            ids.add(row['job_id'])
            if row['owner_entity_id'] == node['owner'] and row['actor_epoch'] == node['life_epoch']:
                if not node['birth_ns'] <= row['arrival_ns'] < stop:
                    raise ValueError('task arrival is outside its exact owner life epoch')
                yield row
    else:
        raise ValueError('unsupported arrival process')


def simulate_resource_tasks(episode, config):
    """Execute declared burst arrivals on each explicitly selected local node.

    Source departure and authored local unavailability take priority over completion.
    Otherwise completion releases resources before arrival/admission and before
    deadline evaluation. Completion at the deadline is timely. Source departure
    aborts only this local executor. Horizon H is a right censoring boundary:
    neither a completion nor an executor death is invented at H.
    """
    p = config["compute"]
    for key in ("cpu_cores", "cpu_hz", "memory_capacity_bytes"):
        if type(p[key]) is not int or p[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(p["waiting_queue_limit_jobs"]) is not int or p["waiting_queue_limit_jobs"] < 0:
        raise ValueError("waiting_queue_limit_jobs must be a nonnegative integer")
    if p["scheduling"] != "strict_fifo_nonpreemptive":
        raise ValueError("unsupported compute process or scheduling discipline")
    if p["service_randomness"] != "none" or config["seed_use"] != "not_used_deterministic_workload":
        raise ValueError("this version has no stochastic service/arrival model")
    selected = set(config["owner_entity_ids"])
    if not selected or len(selected) != len(config["owner_entity_ids"]):
        raise ValueError("explicit unique owner_entity_ids required")
    nodes = [n for n in episode["nodes"] if n["owner"] in selected and n["node_type"] in ("uav", "vehicle")]
    if {n["owner"] for n in nodes} != selected:
        raise ValueError("selected owners must resolve to actual UAV/vehicle life epochs")
    unavailable = config["local_unavailability_intervals_ns"]
    if config["local_unavailability_policy"] != "cancel_local_jobs_reject_arrivals_then_resume":
        raise ValueError("explicit supported local unavailability policy required")
    for row in unavailable:
        if set(row) != {"owner_entity_id", "start_ns", "end_ns", "source_ref"} or row["owner_entity_id"] not in selected:
            raise ValueError("unavailability must bind one selected exact owner and explicit source")
        if type(row["start_ns"]) is not int or type(row["end_ns"]) is not int or not 0 <= row["start_ns"] < row["end_ns"] <= episode["duration_ns"]:
            raise ValueError("unavailability requires a half-open integer ns interval within the horizon")
        if not isinstance(row["source_ref"], str) or not row["source_ref"]:
            raise ValueError("unavailability requires source_ref")
    for owner_id in selected:
        intervals = sorted((row["start_ns"], row["end_ns"]) for row in unavailable if row["owner_entity_id"] == owner_id)
        if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("unavailability intervals must be disjoint and coalesced per owner")
    h = episode["duration_ns"]
    if p['arrival_process'] == 'declared_task_manifest':
        epochs = {(n['owner'], n['life_epoch']) for n in nodes}
        if any((row['owner_entity_id'], row['actor_epoch']) not in epochs for row in p['task_manifest']):
            raise ValueError('task manifest owner/epoch is not an explicitly selected source executor')
    events, tasks = [], []
    for node in nodes:
        birth, death = node["birth_ns"], node["death_ns"]
        stop = h if death is None else min(death, h)
        queue, running, clock_events = deque(), {}, []
        serial = 0
        available = True

        def schedule(time_ns, priority, kind, task):
            nonlocal serial
            heapq.heappush(clock_events, (time_ns, priority, serial, kind, task))
            serial += 1

        def emit(kind, when, task=None, reason=None):
            event = {"schema_version": "p09.compute.resource-event/v3", "time_ns": when,
                     "sequence": len(events),
                     "event_type": kind, "owner": owner, "executor_node_id": node["node_id"],
                     "task_id": None if task is None else task["task_id"], "reason": reason,
                     "job_class": None if task is None else task['job_class'],
                     "queue_depth_jobs": len(queue), "executing_jobs": len(running),
                     "cpu_busy_cores": len(running),
                     "execution_memory_bytes": sum(j["memory_bytes"] for j in running.values()),
                     "executor_available": available,
                     "source_ref": episode["source_ref"], "config_ref": "run_config.json",
                     "producer_version": VERSION}
            event["event_id"] = f"{node['node_id']}:{len(events)}"
            events.append(event)

        def dispatch(now):
            while available and queue and len(running) < p["cpu_cores"]:
                task = queue[0]
                free = p["memory_capacity_bytes"] - sum(j["memory_bytes"] for j in running.values())
                if task["memory_bytes"] > free:
                    break  # Strict head-of-line FIFO, no implicit backfill.
                queue.popleft()
                task["start_ns"] = now
                task["wait_ns"] = now - task["arrival_ns"]
                task["status"] = "executing"
                running[task["task_id"]] = task
                emit("task_start", now, task)
                schedule(now + task['service_ns'], 0, "complete", task)

        owner = {"episode_id": episode["episode_id"], "entity_id": node["owner"], "actor_epoch": node["life_epoch"]}
        for spec in _arrival_specs(node, stop, p):
                if spec['cycles'] * NS % p['cpu_hz']:
                    raise ValueError('service duration must resolve to exact integer ns')
                service_ns = spec['cycles'] * NS // p['cpu_hz']
                task = {"schema_version": "p09.compute.resource-task/v3", "producer_version": VERSION,
                        "task_id": f"{node['node_id']}:resource-job:{spec['job_id']}", "owner": owner,
                        "executor_node_id": node["node_id"], "arrival_ns": spec['arrival_ns'],
                        "deadline_ns": spec['arrival_ns'] + spec['deadline_duration_ns'],
                        "cycles": spec['cycles'], "required_cores": 1,
                        "memory_bytes": spec['memory_bytes'], "service_ns": service_ns,
                        "job_class":spec['job_class'], "workload_source_ref":spec['workload_source_ref'],
                        "start_ns": None, "completion_ns": None, "terminal_ns": None, "wait_ns": None,
                        "status": "not_arrived", "terminal_reason": None,
                        "deadline_missed": None, "deadline_status": "UNKNOWN",
                        "deadline_reason": "horizon_before_deadline_outcome", "deadline_available_ns": None,
                        "source_ref": episode["source_ref"], "config_ref": "run_config.json"}
                tasks.append(task)
                schedule(spec['arrival_ns'], 2, "arrival", task)
                schedule(task["deadline_ns"], 3, "deadline", task)
        if death is not None and death < h:
            schedule(death, -3, "executor_exit", None)
        for interval in unavailable:
            if interval["owner_entity_id"] != node["owner"]:
                continue
            start, end = max(birth, interval["start_ns"]), min(stop, interval["end_ns"])
            if start < end:
                schedule(start, -2, "executor_unavailable", None)
                if end < stop:
                    schedule(end, -2, "executor_recovered", None)
        while clock_events:
            now, _, _, kind, task = heapq.heappop(clock_events)
            if now >= h:
                break
            if kind == "complete":
                if task["task_id"] not in running:
                    continue  # A recorded executor exit canceled this local job.
                del running[task["task_id"]]
                task.update(completion_ns=now, terminal_ns=now,
                            status="completed_on_time" if now <= task["deadline_ns"] else "completed_late")
                if now <= task["deadline_ns"]:
                    task.update(deadline_missed=False, deadline_status="KNOWN", deadline_reason=None,
                                deadline_available_ns=now)
                emit("task_complete", now, task)
                dispatch(now)
            elif kind == "executor_exit":
                available = False
                local = [*running.values(), *queue]
                running.clear()
                queue.clear()
                for job in local:
                    job.update(status="aborted_executor_exit", terminal_ns=now,
                               terminal_reason="observed_local_executor_exit")
                    if job["deadline_status"] == "UNKNOWN":
                        job.update(deadline_status="OUT_OF_SCOPE", deadline_reason="executor_exit_before_deadline",
                                   deadline_available_ns=now)
                    emit("task_aborted_executor_exit", now, job, job["terminal_reason"])
                emit("executor_exit", now, reason="observed_source_lifecycle")
            elif kind == "executor_unavailable":
                available = False
                local = [*running.values(), *queue]
                running.clear()
                queue.clear()
                for job in local:
                    job.update(status="canceled_local_unavailability", terminal_ns=now,
                               terminal_reason="authored_local_executor_unavailability")
                    emit("task_canceled", now, job, job["terminal_reason"])
                emit("executor_unavailable", now, reason="authored_local_executor_unavailability")
            elif kind == "executor_recovered":
                available = True
                emit("executor_recovered", now, reason="authored_local_executor_recovery")
                dispatch(now)
            elif kind == "arrival":
                emit("task_arrival", now, task)
                if not available:
                    task.update(status="rejected_executor_unavailable", terminal_ns=now,
                                terminal_reason="authored_local_executor_unavailability")
                    emit("task_rejected", now, task, task["terminal_reason"])
                    continue
                if task["memory_bytes"] > p["memory_capacity_bytes"]:
                    task.update(status="rejected_resource_request", terminal_ns=now,
                                terminal_reason="job_memory_exceeds_executor_capacity")
                    emit("task_rejected", now, task, task["terminal_reason"])
                    continue
                # An immediately dispatchable job does not occupy the wait queue.
                free = p["memory_capacity_bytes"] - sum(j["memory_bytes"] for j in running.values())
                immediate = not queue and len(running) < p["cpu_cores"] and task["memory_bytes"] <= free
                if not immediate and len(queue) >= p["waiting_queue_limit_jobs"]:
                    task.update(status="rejected_queue_full", terminal_ns=now, terminal_reason="waiting_queue_full")
                    emit("task_rejected", now, task, task["terminal_reason"])
                    continue
                task["status"] = "queued"
                queue.append(task)
                emit("task_admitted", now, task)
                dispatch(now)
            elif kind == "deadline":
                if task["deadline_status"] == "UNKNOWN":
                    task.update(deadline_missed=True, deadline_status="KNOWN", deadline_reason=None,
                                deadline_available_ns=now)
                    emit("task_deadline_missed", now, task)
            else:
                raise ValueError(f"unsupported event {kind}")
        for job in [*running.values(), *queue]:
            job.update(status="right_censored", terminal_reason="episode_horizon")
    events.sort(key=lambda e: (e["time_ns"], e["sequence"]))
    return {"events": events, "tasks": tasks, "nodes": nodes}


def project_resource_states(result, episode, config):
    """181 half-open observations using events <t and mature deadline cohorts."""
    h = episode["duration_ns"]
    p = config["compute"]
    states = []
    for node in result["nodes"]:
        jobs = [j for j in result["tasks"] if j["executor_node_id"] == node["node_id"]]
        for t in range(0, h + 1, 500_000_000):
            owner = {"episode_id": episode["episode_id"], "entity_id": node["owner"], "actor_epoch": node["life_epoch"]}
            if t < node["birth_ns"] or node["death_ns"] is not None and t >= node["death_ns"]:
                states.append({"schema_version": "p09.compute.resource-state/v3", "time_ns": t,
                    "owner": owner, "executor_node_id": node["node_id"], "availability": "OUT_OF_SCOPE",
                    "availability_reason": "outside_observed_executor_life_epoch", "queue_depth_jobs": None,
                    "executing_jobs": None, "wait_mean_ns": None, "deadline_miss_ratio": None,
                    "time_support": {"start_ns":max(0,t-500_000_000),"end_ns":t,"end_exclusive":True},
                    "source_ref": episode["source_ref"], "config_ref":"run_config.json", "producer_version":VERSION})
                continue
            arrived = [j for j in jobs if j["arrival_ns"] < t]
            executing = [j for j in arrived if j["start_ns"] is not None and j["start_ns"] < t
                         and (j["terminal_ns"] is None or j["terminal_ns"] >= t)]
            waiting = [j for j in arrived if (j["start_ns"] is None or j["start_ns"] >= t)
                       and (j["terminal_ns"] is None or j["terminal_ns"] >= t)]
            completed = [j for j in arrived if j["completion_ns"] is not None and j["completion_ns"] < t]
            wait_cohort = [j["wait_ns"] for j in arrived if j["start_ns"] is not None
                           and max(0, t-500_000_000) <= j["start_ns"] < t]
            deadline_cohort = [j for j in arrived if max(0, t-500_000_000) <= j["deadline_ns"] < t
                               and j["deadline_status"] == "KNOWN"]
            changes = [event for event in result["events"] if event["executor_node_id"] == node["node_id"] and
                       event["time_ns"] < t and event["event_type"] in ("executor_unavailable", "executor_recovered")]
            available = not changes or changes[-1]["event_type"] == "executor_recovered"
            states.append({"schema_version": "p09.compute.resource-state/v3", "time_ns": t, "owner": owner,
                "availability":"KNOWN", "availability_reason":None,
                "executor_available": available,
                "executor_node_id": node["node_id"], "queue_depth_jobs": len(waiting), "executing_jobs": len(executing),
                "cpu_busy_cores": len(executing), "execution_memory_bytes": sum(j["memory_bytes"] for j in executing),
                "arrived_jobs": len(arrived), "completed_jobs": len(completed),
                "wait_mean_ns": sum(wait_cohort)/len(wait_cohort) if wait_cohort else None,
                "wait_status": "KNOWN" if wait_cohort else "UNKNOWN", "wait_reason": None if wait_cohort else "no_started_jobs_in_window",
                "deadline_mature_jobs": len(deadline_cohort),
                "deadline_missed_jobs": sum(j["deadline_missed"] for j in deadline_cohort),
                "deadline_miss_ratio": sum(j["deadline_missed"] for j in deadline_cohort)/len(deadline_cohort) if deadline_cohort else None,
                "deadline_status": "KNOWN" if deadline_cohort else "UNKNOWN", "deadline_reason": None if deadline_cohort else "no_mature_deadlines_in_window",
                "gpu_service": {"value": None, "status": "UNKNOWN", "reason": "no_calibrated_gpu_workload_or_service_profile"},
                "offload": {"value": None, "status": "UNKNOWN", "reason": "no_modeled_request_service_response_transport"},
                "migration": {"value": None, "status": "UNKNOWN", "reason": "no_modeled_checkpoint_transfer_resume"},
                "cpu_capacity_cores": p["cpu_cores"], "memory_capacity_bytes": p["memory_capacity_bytes"],
                "time_support": {"start_ns": max(0,t-500_000_000), "end_ns": t, "end_exclusive": True},
                "source_ref": episode["source_ref"], "config_ref": "run_config.json", "producer_version": VERSION})
    return states
