"""Episode-local clock contract shared by formal truth writers and readers."""

from __future__ import annotations

import math
from typing import Any, Mapping


FORMAL_TICK_HZ = 10


def validate_formal_frame_clock(frame: Mapping[str, Any], *, expected_tick: int,
                                source: str) -> float:
    """Check one formal frame without conflating source-segment timestamps."""
    tick, seq, hz, step, sim_time = (
        frame.get(name) for name in ("tick", "frame_seq", "tick_hz", "dt_s", "sim_time_s")
    )
    if (type(tick) is not int or tick != expected_tick or
            type(seq) is not int or seq != tick or
            type(hz) not in (int, float) or hz != FORMAL_TICK_HZ or
            type(step) not in (int, float) or step <= 0 or
            type(sim_time) not in (int, float) or
            not math.isclose(step * hz, 1.0, rel_tol=1e-9, abs_tol=1e-9) or
            not math.isclose(sim_time, tick / hz, rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError(f"formal frame clock differs from source time contract: {source}")
    return float(hz)
