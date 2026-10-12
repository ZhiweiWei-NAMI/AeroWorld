"""Reconstructible compaction of quantized streams (RESEARCH_PLAN §3.2).

Quantization is the only lossy step (values are integer quanta).  Compaction
keeps presence changes, change points and a named reconstruction rule; every
window checks R(compact(x)) == x on every present history tick, and a stream
that fails keeps all of its samples.

hold-last : records (tick, from, to) at every tick whose value differs from the
            previous present sample, plus appear (tick, value) / leave (tick,
            last value).  Read backwards from the origin value (or the leave
            record) this regenerates every sample.
velocity-hold knots (motion): knots (t_k, p_k, v_k); between knots
            p_t = p_k + round(v_k (t - t_k) dt) in quanta and v_t = v_k.  The first
            present tick is a knot and a new knot is placed at the first
            mismatch.  Knots carry their tick, so timing is exact.
"""
from __future__ import annotations

POS_PER_VEL_TICK = 10      # cm per (cm/s * tick) divisor: dt = 0.1 s


def presence_runs(ticks, step):
    """Maximal runs of consecutive grid ticks: [(first, last), ...]."""
    runs = []
    for t in ticks:
        if runs and t - runs[-1][1] == step:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return [tuple(r) for r in runs]


def hold_last(samples: dict, ticks: list, step: int, origin: int):
    """Compact one stream over the window's sample ticks (sorted, all inside the window)."""
    present = [t for t in ticks if samples.get(t) is not None]
    records = []
    runs = presence_runs(present, step)
    for first, last in runs:
        if first != ticks[0]:
            records.append(("appear", first, samples[first]))
        prev = samples[first]
        for t in range(first+step, last+1, step):
            if samples[t] != prev:
                records.append(("change", t, prev, samples[t]))
                prev = samples[t]
        if last != origin:
            records.append(("leave", last+step, prev))
    return records


def reconstruct_hold(records, ticks, step, origin_value):
    """Inverse of hold_last, read backwards from the origin value (None when absent at the origin)."""
    leave, change, appear = {}, {}, set()
    for r in records:
        if r[0] == "leave":
            leave[r[1]] = r[2]
        elif r[0] == "change":
            change[r[1]] = r[2]
        else:
            appear.add(r[1])
    out, value = {}, origin_value
    for t in reversed(ticks):
        if t+step in leave:          # present at t, absent from t+step on
            value = leave[t+step]
        if value is not None:
            out[t] = value
        if t in change:
            value = change[t]
        elif t in appear:
            value = None
    return out


def verify_hold(samples, ticks, step, origin, records):
    expected = {t: samples[t] for t in ticks if samples.get(t) is not None}
    return reconstruct_hold(records, ticks, step, samples.get(origin)) == expected


def motion_knots(pos: dict, vel: dict, ticks: list):
    """Velocity-hold knots over present ticks (step 1); returns [(tick, p, v)] and the leave tick if any."""
    present = [t for t in ticks if t in pos and t in vel]
    knots = []
    for first, last in presence_runs(present, 1):
        k = None
        for t in range(first, last+1):
            p, v = pos[t], vel[t]
            if k is not None:
                tk, pk, vk = k
                predicted = tuple(pk[i] + round(vk[i]*(t-tk)/POS_PER_VEL_TICK) for i in range(3))
                if predicted == p and vk == v:
                    continue
            k = (t, p, v)
            knots.append(k)
    return knots, presence_runs(present, 1)


def reconstruct_motion(knots, runs):
    out = {}
    starts = {first for first, _ in runs}
    ordered = sorted(knots)
    for first, last in runs:
        ks = [k for k in ordered if first <= k[0] <= last]
        for i, (tk, pk, vk) in enumerate(ks):
            end = ks[i+1][0] if i+1 < len(ks) else last+1
            for t in range(tk, end):
                out[t] = (tuple(pk[j] + round(vk[j]*(t-tk)/POS_PER_VEL_TICK) for j in range(3)), vk)
        if not ks or ks[0][0] != first or first not in starts:
            return None
    return out


def verify_motion(pos, vel, ticks, knots, runs):
    expected = {t: (pos[t], vel[t]) for t in ticks if t in pos and t in vel}
    return reconstruct_motion(knots, runs) == expected
