"""Receiver-visible schedule-cohort TTL metric, separate from accepted-TX loss.

The gateway knows this declared application schedule, not future sender socket
acceptance. Only its own receptions before the observation time are consumed.
This metric must never overwrite p01.ns3.mature-datagram-lp/v1 packet loss.
"""
from dataclasses import dataclass
from collections import defaultdict
from bisect import bisect_left


@dataclass(frozen=True)
class ScheduledDatagram:
    flow_id: str
    source_owner: str
    source_epoch: int
    receiver_owner: str
    receiver_epoch: int
    sequence: int
    generation_ns: int
    ttl_ns: int
    schedule_available_ns: int


@dataclass(frozen=True)
class GatewayReception:
    flow_id: str
    source_owner: str
    source_epoch: int
    receiver_owner: str
    receiver_epoch: int
    sequence: int
    receive_ns: int


def sequence_key(row):
    return (row.flow_id, row.source_owner, row.source_epoch,
            row.receiver_owner, row.receiver_epoch, row.sequence)


class GatewaySequenceMetric:
    """Build an immutable receiver index once; query one bounded time cohort."""

    namespace = "p09.gateway-scheduled-sequence-ttl/v1"

    def __init__(self, schedule, receptions):
        self.schedule = tuple(sorted(schedule, key=lambda r: r.generation_ns))
        keys = tuple(map(sequence_key, self.schedule))
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate exact scheduled datagram identity")
        if any(r.ttl_ns <= 0 or r.generation_ns < r.schedule_available_ns
               for r in self.schedule):
            raise ValueError("Schedule must be known by generation and declare positive TTL")
        self.times = tuple(r.generation_ns for r in self.schedule)
        rx = defaultdict(list)
        for row in receptions:
            rx[sequence_key(row)].append(row.receive_ns)
        self.receptions = {key: tuple(sorted(times)) for key, times in rx.items()}

    def observe(self, *, observation_ns, window_ns, receiver_owner, receiver_epoch):
        """Events at t are unavailable; TTL receipt equality is not timely.

        Cohort generation lies in [t-window,t), and TTL must mature strictly
        before t. Pending and zero-denominator UNKNOWN remain explicit.
        """
        lo = bisect_left(self.times, observation_ns - window_ns)
        hi = bisect_left(self.times, observation_ns)
        mature = timely = late = pending = 0
        for row in self.schedule[lo:hi]:
            if (row.receiver_owner, row.receiver_epoch) != (receiver_owner, receiver_epoch):
                continue
            if row.schedule_available_ns >= observation_ns:
                continue
            expiry = row.generation_ns + row.ttl_ns
            if expiry >= observation_ns:
                pending += 1
                continue
            mature += 1
            available_rx = self.receptions.get(sequence_key(row), ())
            at = bisect_left(available_rx, row.generation_ns)
            first_rx = available_rx[at] if at < len(available_rx) else None
            if first_rx is not None and first_rx < expiry:
                timely += 1
            elif first_rx is not None and first_rx < observation_ns:
                late += 1
        return {"schema_version": self.namespace, "observation_ns": observation_ns,
                "receiver_owner": receiver_owner, "receiver_epoch": receiver_epoch,
                "window_ns": window_ns, "mature_expected": mature,
                "timely_received": timely, "ttl_not_timely": mature - timely,
                "observed_late": late, "pending_expected": pending,
                "loss_ratio": (mature - timely) / mature if mature else None,
                "availability": "KNOWN" if mature else "UNKNOWN",
                "reason": None if mature else "no_mature_expected_sequence"}
