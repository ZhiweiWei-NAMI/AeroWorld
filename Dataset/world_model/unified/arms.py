"""Paired ARM evaluation on shared parent histories; ARM data never fit parameters."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import functools
import itertools
import json
import random
import sys

import torch
import yaml

from Dataset.world_model.unified import assemble as A, codes as K, data as D, schema as S
from Dataset.world_model.unified import sources as R, visual as V, windows as W

INDEX = S.ROOT/"arms/index.jsonl"
MAPPING = S.ROOT/"Dataset/world_model/configs/capture_sources.json"
OUT = S.WORKSPACE/"arms"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _path(value):
    if isinstance(value, dict):
        value = value["path"]
    path = S.ROOT/value
    if not path.resolve().is_relative_to(S.ROOT.resolve()):
        raise ValueError(f"ARM path outside repository: {value}")
    return path


@functools.lru_cache(maxsize=1)
def _catalog():
    return {r["key"]: r for r in R.rows(INDEX)}


@functools.lru_cache(maxsize=1)
def _mapping():
    return json.loads(MAPPING.read_text())["arms"]


def _factual_key(key):
    group = key.rsplit("/", 1)[0]
    matches = [k for k in _catalog() if k.rsplit("/", 1)[0] == group and _mapping()[k]["factual"]]
    if len(matches) != 1:
        raise ValueError(f"expected one factual arm in {group}, found {len(matches)}")
    return matches[0]


@functools.lru_cache(maxsize=None)
def _meta(key):
    row = _catalog()[key]
    manifest = json.loads((_path(row["path"])/"manifest.json").read_text())
    config = yaml.safe_load(_path(row["configuration"]).read_text())
    config = config.get("spec", config)
    scope = manifest.get("scope", {})
    entities = scope.get("entity_ids", manifest.get("entity_ids", scope.get("simulated_entity_ids")))
    if not entities:
        raise ValueError("no declared entity scope")
    predicates = scope.get("predicate_ids", manifest.get("predicate_ids", config.get("predicate_ids")))
    if predicates is None and manifest.get("primary_predicate"):
        predicates = [manifest["primary_predicate"]]
    families = manifest.get("event_families", manifest.get("target_families", config.get("event_families")))
    if families is None and manifest.get("event_family"):
        families = [manifest["event_family"]]
    window = manifest["window"]
    return {"manifest": manifest, "config": config, "entities": sorted(set(entities)),
            "predicates": None if predicates is None else sorted(set(predicates)),
            "event_families": sorted(set(families or ())), "window": window,
            "parent_episode": _mapping()[key]["parent_episode"]}


@functools.lru_cache(maxsize=16)
def _actions(key):
    rows = list(R.rows(_path(_catalog()[key]["path"])/"raw/actions.jsonl"))
    for row in rows:
        if not isinstance(row.get("tick"), int) or not isinstance(row.get("action"), dict):
            raise ValueError("action missing tick or action body")
        if not row.get("result", {}).get("status"):
            raise ValueError("action missing result.status")
    return rows


def _fork(key, factual):
    meta = _meta(key)
    tick = meta["window"].get("fork_tick", meta["manifest"].get("fork_tick"))
    if tick is not None:
        return tick, "manifest fork_tick"
    by_tick = []
    for arm in (key, factual):
        table = defaultdict(list)
        for row in _actions(arm):
            table[row["tick"]].append(_json({"action": row["action"], "status": row["result"]["status"]}))
        by_tick.append(table)
    differing = [t for t in set(by_tick[0]) | set(by_tick[1])
                 if by_tick[0][t] != by_tick[1][t]]
    if not differing:
        raise ValueError("no explicit fork tick and no differing actions")
    return min(differing), "first differing action tick"


def _state(row):
    if row.get("present") is False or row.get("observation_status") == "unavailable":
        return None
    body = row.get("values", row)
    if not isinstance(body, dict):
        return None
    return {**body, **body.get("runtime_state", {})}


def _vector(body, name, eps):
    value = body.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    result = tuple(R.quantize(v, eps) for v in value)
    return result if all(v is not None for v in result) else None


def _tuple(predicate, bindings):
    return predicate, _json(bindings)


def _scoped(bindings, entities, src):
    def refs(value):
        if isinstance(value, dict):
            return [r for v in value.values() for r in refs(v)]
        if isinstance(value, list):
            return [r for v in value for r in refs(v)]
        owner = src.owner(value) if isinstance(value, str) else None
        return [owner] if owner is not None else []
    owners = refs(bindings)
    return not owners or bool(set(owners) & set(entities))


@functools.lru_cache(maxsize=16)
def _labels(key, origin):
    meta, directory = _meta(key), _path(_catalog()[key]["path"])
    entities = set(meta["entities"])
    path = directory/"raw/states.jsonl"
    if not path.exists():
        path = directory/"raw/trajectories.jsonl"
    positions, states, observed = defaultdict(dict), {}, defaultdict(set)
    last_tick = None
    end = min(meta["window"]["end_tick"], origin+50)
    for row in R.rows(path):
        t, e = row["tick"], row["entity_id"]
        last_tick = t if last_tick is None else max(last_tick, t)
        if e not in entities or not origin <= t <= end:
            continue
        observed[e].add(t)
        body = _state(row)
        if body is None:
            continue
        if t == origin:
            states[e] = body
        p = body.get("pos_enu")
        if isinstance(p, (list, tuple)) and len(p) == 3:
            p = tuple(float(c) for c in p)
            if t in positions[e] and positions[e][t] != p:
                raise ValueError(f"conflicting positions for {e}@{t}")
            positions[e][t] = p          # storage precision (m); residual labels are quantized against the context
    relations = {}
    predicate_path = directory/"objective/predicate_truth_ticks.jsonl"
    predicate_rows = R.rows(predicate_path)
    first = next(predicate_rows, None)
    if first is not None and first.get("schema_name") == "domain_state_observation":
        # Some L4 releases put L0 operands here; their compiled ARM truths live beside them.
        predicate_path = directory/"objective/predicate_truth.jsonl"
        predicate_rows = R.rows(predicate_path)
    else:
        predicate_rows = itertools.chain(() if first is None else (first,), predicate_rows)
    for row in predicate_rows:
        t = row["tick"]
        if not origin - S.HISTORY_TICKS <= t <= end:
            continue
        pid = row["predicate_id"]
        if meta["predicates"] is not None and pid not in meta["predicates"]:
            continue
        bindings = row["bindings"]
        tid = _tuple(pid, bindings)
        rel = relations.setdefault(tid, R.Relation(_json(tid), pid, bindings))
        value = R.categorical(row["value"])
        if t in rel.samples and rel.samples[t] != value:
            raise ValueError(f"conflicting predicate labels for {tid}@{t}")
        rel.samples[t] = value
    events, unsupported = [], Counter()
    for row in R.rows(directory/"objective/event_occurrences.jsonl"):
        if not row.get("event_type_id"):
            unsupported[row.get("kind", row.get("schema_name", "missing event_type_id"))] += 1
            continue
        tick = row.get("detection_tick", row.get("trigger_tick", row.get("tick")))
        if not isinstance(tick, int):
            raise ValueError("typed event missing detection/trigger tick")
        events.append({"tick": tick, "type": row["event_type_id"], "family": row.get("event_family_id"),
                       "bindings": row["bindings"]})
    if last_tick is None:
        raise ValueError("empty arm state stream")
    return {"positions": dict(positions), "states": states, "observed": dict(observed),
            "relations": relations, "events": sorted(events, key=lambda e: (e["tick"], e["type"])),
            "horizon_end": min(meta["window"]["end_tick"], last_tick),
            "unsupported_events": dict(unsupported), "predicate_source": str(predicate_path.relative_to(S.ROOT))}


def arm_pairs():
    """One pair per intervention, with skips printed and retained in ``diagnostics``.

    Use the manifest's window.fork_tick (or top-level fork_tick). If absent,
    compare ordered action-body/status lists per tick with the same group's factual
    arm and take the first difference, ignoring execution-effect summaries.
    Floor that tick to FORMAL_STEP. Undeclared predicate scope is the set in
    the arm's own origin records; future records never enlarge query scope.
    """
    pairs, skipped = [], []
    total = 0
    for key in _catalog():
        if _mapping().get(key, {}).get("factual") is True:
            continue
        total += 1
        try:
            factual = _factual_key(key)
            meta = _meta(key)
            if meta["parent_episode"] != _meta(factual)["parent_episode"]:
                raise ValueError("factual and intervention map to different parents")
            tick, rule = _fork(key, factual)
            if not isinstance(tick, int):
                raise ValueError("fork tick is not an integer")
            origin = tick // S.FORMAL_STEP * S.FORMAL_STEP
            _actions(key), _actions(factual)
            labels = [_labels(k, origin) for k in (key, factual)]
            if any(origin < _meta(k)["window"]["start_tick"] or origin >= d["horizon_end"]
                   for k, d in zip((key, factual), labels)):
                raise ValueError("origin outside recorded arm window or no future coverage")
            src = D.cached_source(meta["parent_episode"])
            parent_rels = {_tuple(r.predicate, r.bindings): r for r in src.relations.values()
                           if r.samples.get(origin) in W.KNOWN and (r.tuple_id, origin) not in src.masked_labels}
            for k, d in zip((key, factual), labels):
                motion = any(e in src.category and src.category[e] in S.MOVING and origin in src.present[e]
                             and origin in p and any(origin+h in p for h in S.HORIZONS)
                             for e, p in d["positions"].items())
                predicates = any(tid in parent_rels and _scoped(r.bindings, _meta(k)["entities"], src)
                                 and r.samples.get(origin) in W.KNOWN
                                 and any(r.samples.get(origin+h) in W.KNOWN for h in S.HORIZONS)
                                 for tid, r in d["relations"].items())
                if not motion and not predicates:
                    raise ValueError(f"no parent-addressable future position/predicate labels: {k}")
            predicates = meta["predicates"]
            if predicates is None:
                predicates = sorted({r.predicate for r in labels[0]["relations"].values() if origin in r.samples})
            pairs.append({"arm_key": key, "factual_key": factual, "parent_episode": meta["parent_episode"],
                          "origin": origin, "entities": meta["entities"], "predicates": predicates,
                          "event_families": meta["event_families"],
                          "horizon_end": min(d["horizon_end"] for d in labels),
                          "diagnostics": {"fork_tick": tick, "origin_rule": rule,
                                          "unsupported_events": {k: d["unsupported_events"]
                                                                 for k, d in zip((key, factual), labels)
                                                                 if d["unsupported_events"]}}})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rec = {"arm_key": key, "reason": str(exc)}
            skipped.append(rec)
            print(_json({"arm_pair_skipped": rec}), file=sys.stderr, flush=True)
    arm_pairs.diagnostics = {"counterfactual_arms": total, "usable": len(pairs), "skipped": skipped,
                            "skip_reasons": dict(Counter(r["reason"].split(": ")[0] for r in skipped))}
    return pairs


def _alias(value, aliases):
    if isinstance(value, dict):
        return {aliases.get(k, k): _alias(v, aliases) for k, v in value.items()}
    if isinstance(value, list):
        return [_alias(v, aliases) for v in value]
    return aliases.get(value, value) if isinstance(value, str) else value


@functools.lru_cache(maxsize=32)
def _script(key):
    meta = _meta(key)
    config = meta["config"]
    if "intervention_actions" in config and "definition" in config:
        # X producer contract: the configuration's sibling is the executable branch script.
        path = str(_path(_catalog()[key]["configuration"]).with_name(
            _catalog()[key]["arm_id"]+"_event_script.json").relative_to(S.ROOT))
    else:
        sources = config.get("sources", meta["manifest"].get("sources", meta["manifest"].get("source_refs", {})))
        path = sources.get("script")
    if path is None:
        return {}
    raw = json.loads(_path(path).read_text())
    out = {}
    for kind, identity in (("events", "event_id"), ("triggers", "trigger_id")):
        for row in raw.get(kind, ()):
            out[(kind, row[identity])] = {k: v for k, v in row.items() if k not in ("title", "description")}
    return out


def action_lines(arm_key, origin, aliases):
    """Origin actions/statuses and static branch plans, never future realized effects.

    execution.actions/inject_actions and conditional intervention_actions are
    declarations fixed by the arm configuration, as are omissions, retiming
    and trigger rewrites. Differing authored script definitions are included
    symmetrically across a group (not their future dispatch ticks). No result
    fields except origin-time status enter inputs; schedules/effects recorded
    by the execution are excluded.
    """
    lines = []
    for row in _actions(arm_key):
        if row["tick"] == origin:
            body = {"action": row["action"], "status": row["result"]["status"]}
            lines.append("  t=0 " + _json(_alias(body, aliases)))
    config = _meta(arm_key)["config"]
    execution = config.get("execution", {})
    plans = list(execution.get("actions", ())) + list(config.get("inject_actions", ()))
    plans += list(config.get("intervention_actions", ()))
    for plan in plans:
        tick = plan.get("tick")
        if tick == origin:
            continue
        # Only declaration timing is exposed; conditional triggers keep their condition.
        lines.append("  fixed schedule " + _json(_alias(plan, aliases)))
    conditions = {k: execution[k] for k in ("omit_action_ids", "trigger_rewrites") if execution.get(k)}
    conditions.update({k: config[k] for k in ("omit_action_ids", "cancel_event_ids", "removed_event_ids",
                                            "retime", "state_overrides", "sumo_intervention", "gates")
                       if config.get(k) is not None and config.get(k) != [] and config.get(k) != {}})
    removed = config.get("definition", {}).get("branch_changes", {}).get("removed_action_ids")
    if removed:
        conditions["removed_action_ids"] = removed
    if conditions:
        lines.append("  fixed conditions " + _json(_alias(conditions, aliases)))
    factual = _factual_key(arm_key)
    baseline = _script(factual)
    changed = set()
    for key in _catalog():
        if key.rsplit("/", 1)[0] == arm_key.rsplit("/", 1)[0]:
            script = _script(key)
            changed.update(k for k in set(script) | set(baseline) if script.get(k) != baseline.get(k))
    script = _script(arm_key)
    triggers = {identity for kind, identity in changed if kind == "triggers"}
    for definitions in (baseline, script):
        changed.update(k for k, v in definitions.items() if k[0] == "events" and v.get("trigger_ref") in triggers)
    for kind, identity in sorted(changed):
        body = script.get((kind, identity))
        declaration = {"kind": kind, "id": identity, "definition": body,
                       "status": "declared" if body is not None else "omitted_by_configuration"}
        lines.append("  fixed script " + _json(_alias(declaration, aliases)))
    return lines


@functools.lru_cache(maxsize=1)
def _event_types():
    raw = json.loads((S.ROOT/"Dataset/semantic_rules/profiles/event_family_authority_mapping.json").read_text())
    return {family: body["ontology_event_type_ids"] for family, body in raw["families"].items()}


def _relation_aliases(src, window, origin):
    # build_window enumerates these exact history-present tuples, ordered this way.
    rels = sorted((r for r in src.relations.values() if any(t in r.samples for t in W.formal_ticks(origin))),
                  key=lambda r: (r.predicate, r.tuple_id))
    out = {}
    for i, rel in enumerate(rels, 1):
        alias = f"r{i}"
        if f"rel|{alias}" in window["headings"] and rel.samples.get(origin) in W.KNOWN:
            if (rel.tuple_id, origin) not in src.masked_labels:
                out[_tuple(rel.predicate, rel.bindings)] = (alias, rel)
    return out


def _origin_difference(e, body, src, origin):
    difference = {}
    for name, samples, eps in (("position_cm", src.position, S.POS_EPS),
                               ("velocity_cm_s", src.velocity, S.VEL_EPS)):
        arm = _vector(body, "pos_enu" if name == "position_cm" else "vel_mps", eps)
        parent = samples.get(e, {}).get(origin)
        if arm is not None and arm != parent:
            difference[name] = {"arm": arm, "parent": parent}
    fields = R._frame_fields({**body, "annotations": {"activity_type": body.get("activity_type")}})
    for name, value in fields.items():
        stream = src.fields.get((e, "frame", name))
        parent = stream.samples.get(origin) if stream else None
        if parent is not None and value != parent:
            difference[name] = {"arm": value, "parent": parent}
    yaw = R.quantize(body.get("yaw_deg"), 1.)
    stream = src.fields.get((e, "frame", "truth_pose.rotation_deg.yaw_deg"))
    if yaw is not None and stream is not None and stream.samples.get(origin) != yaw:
        difference["yaw_deg"] = {"arm": yaw, "parent": stream.samples.get(origin)}
    return difference


def arm_queries(arm_key, origin, window):
    """Parent query templates with scoped ARM labels, quantized before subtraction.

    Predicate bindings must match a history-present parent tuple exactly.
    Missing/unknown future values remain masked; first change uses the same
    formal-grid censoring as windows.flip_label. Untyped business events do
    not become ontology events or negative event labels. Diagnostics are
    attached to the supplied window and to affected position queries.
    """
    meta, labels = _meta(arm_key), _labels(arm_key, origin)
    src = D.cached_source(meta["parent_episode"])
    aliases, queries, diagnostics = window["aliases"], [], {
        "origin_state_differences": {}, "origin_predicate_differences": [], "skipped": [],
        "predicate_source": labels["predicate_source"]}
    present = []
    for e in meta["entities"]:
        if e not in aliases or origin not in src.present.get(e, ()):
            diagnostics["skipped"].append({"entity": e, "reason": "absent from parent at origin"})
            continue
        present.append(e)
        body = labels["states"].get(e)
        if body is not None:
            diff = _origin_difference(e, body, src, origin)
            if diff:
                diagnostics["origin_state_differences"][e] = diff
        p = labels["positions"].get(e, {})
        if src.category[e] not in S.MOVING:
            continue
        ref = src.position_raw.get(e, {}).get(origin)
        if ref is None:
            diagnostics["skipped"].append({"entity": e, "reason": "missing parent origin position"})
            continue
        a, cat = aliases[e], src.category[e]
        for h in S.HORIZONS:
            # R0: the reference is the origin value the context shows (the shared parent history).
            residual = [int(round((p[origin+h][i] - ref[i]) / S.POS_EPS)) for i in range(3)] if origin+h in p else None
            q = {"group": "position", "entity": a, "category": cat, "horizon": h,
                 "text": f"{a} position@+{h*S.TICK_S:.1f}s:", "residual": residual, "eps": S.POS_EPS}
            if e in diagnostics["origin_state_differences"]:
                q["diagnostics"] = {"origin_state_difference": diagnostics["origin_state_differences"][e]}
            queries.append(q)
    parent = _relation_aliases(src, window, origin)
    for tid, rel in sorted(labels["relations"].items()):
        if origin not in rel.samples or not _scoped(rel.bindings, meta["entities"], src):
            continue
        if tid not in parent or rel.samples[origin] not in W.KNOWN:
            diagnostics["skipped"].append({"predicate": rel.predicate, "bindings": rel.bindings,
                                           "reason": "no identical known parent/ARM tuple at origin"})
            continue
        a, pr = parent[tid]
        if rel.samples[origin] != pr.samples[origin]:
            diagnostics["origin_predicate_differences"].append({
                "tuple": a, "arm": rel.samples[origin], "parent": pr.samples[origin]})
        hist = sum(pr.samples.get(t) != pr.samples.get(t-S.FORMAL_STEP)
                   for t in range(origin-S.HISTORY_TICKS+S.FORMAL_STEP, origin+1, S.FORMAL_STEP))
        base = {"tuple": a, "predicate": rel.predicate, "family": rel.predicate.split(".")[0],
                "state": rel.samples[origin], "history_changes": hist}
        for h in S.HORIZONS:
            v = rel.samples.get(origin+h)
            queries.append({**base, "group": "predicate", "horizon": h,
                            "text": f"{a} value@+{h*S.TICK_S:.1f}s:",
                            "label": W.KNOWN.index(v) if v in W.KNOWN else None})
        label, compatible = W.flip_label(rel.samples, set(), rel.tuple_id, origin, rel.samples[origin])
        queries.append({**base, "group": "first_change", "text": f"{a} first_change:",
                        "label": label, "compatible": compatible})
    types = {t for family in meta["event_families"] for t in _event_types().get(family, ())}
    if labels["unsupported_events"]:
        diagnostics["unsupported_events"] = labels["unsupported_events"]
    for e in sorted(present, key=lambda e: aliases[e]):
        cat, a = src.category[e], aliases[e]
        coverage = (origin+50 <= labels["horizon_end"] and not labels["unsupported_events"]
                    and all(t in labels["observed"].get(e, ()) for t in range(origin, origin+51, S.FORMAL_STEP)))
        for etype in sorted(types & set(W.EVENT_TYPES.get(cat, ()))):
            hit = next((ev["tick"] for ev in labels["events"] if ev["type"] == etype
                        and origin < ev["tick"] <= origin+50
                        and any(src.owner(v) == e for v in ev["bindings"].values() if isinstance(v, str))), None)
            queries.append({"group": "event", "entity": a, "category": cat, "event_type": etype,
                            "text": f"{a} next {etype}:",
                            "label": (0 if hit is None else 1+W.time_bin(hit-origin)) if coverage else None})
    window.setdefault("arm_diagnostics", {})[arm_key] = diagnostics
    return queries


def _query_key(q):
    return tuple(q.get(k) for k in ("group", "entity", "tuple", "horizon", "event_type"))


def _prepared(enc, pair, images):
    key = (pair["arm_key"], pair["origin"], images)
    cached = getattr(enc, "_arm_window", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    src = D.cached_source(pair["parent_episode"])
    window = W.build_window(src, pair["origin"])
    # Candidate captures are the parent's (the history before the origin is shared by both branches); their
    # geometric relevance is ranked on the arm's scoped entities.
    caps = []
    if images:
        for c in V.window_captures(pair["parent_episode"], pair["origin"]):
            caps.append((c, V.describe(c, src, window, W.ts)))
    queries = {k: arm_queries(k, pair["origin"], window) for k in (pair["arm_key"], pair["factual_key"])}
    union = {}
    for qs in queries.values():
        for q in qs:
            union.setdefault(_query_key(q), q)
    window["queries"] = list(union.values())
    pw = A.PreparedWindow(window, enc.tok, caps)
    for u in pw.captures():
        u.relevance = V.relevance(u.capture, src, set(pair["entities"]))
    # Both branches share one kept set, and swapped conditions pair one branch's queries with the other's actions,
    # so the reserve is the larger label-free query block plus the larger action block.
    reserve = max(D.query_reserve(enc, qs, range(len(qs))) for qs in queries.values()) + \
        max(len(_action_block(enc, k, pair["origin"], window)) for k in queries)
    window["arm_reserve"] = reserve
    prepared = window, pw, queries
    enc._arm_window = key, prepared
    return prepared


def _action_block(enc, key, origin, window):
    text = ("actions fixed at t=0\n"
            f"  origin_tick={origin}; scheduled ticks are simulator ticks; action coordinates are absolute ENU metres; "
            f"window reference o={W.vec(window['frame_origin_cm'], S.POS_EPS)}\n")
    text += "\n".join(action_lines(key, origin, window["aliases"])) + "\n"
    return enc.tok(text, add_special_tokens=False)["input_ids"]


def arm_item(enc, pair, which, budget, policy, images=True, selector=None, visual="learned"):
    """Encoder-compatible item; both branches select the same parent evidence.

    budget is the total sequence budget.  All scoped queries form a shared union for fixing origin lines and
    choosing optional evidence (text records and parent captures). Labels, action text and future-derived
    diagnostics cannot change the selected parent records or captures.
    """
    if which not in ("arm", "factual"):
        raise ValueError(which)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    key = pair["arm_key"] if which == "arm" else pair["factual_key"]
    w, pw, by_arm = _prepared(enc, pair, images)
    qids = list(range(len(w["queries"])))
    optional = budget - pw.fixed_cost(qids) - w["arm_reserve"] - A.PAD_RESERVE
    if optional < 0:
        raise ValueError(f"mandatory ARM inputs exceed the total budget {budget}: {pair['arm_key']}")
    if selector is not None:
        kept, _ = selector.select(pw, enc.tok, qids, optional, visual=visual)
    else:
        kept, _ = A.select(pw, qids, optional, policy,
                           rng=random.Random(W.stable_hash(pair["parent_episode"], pair["origin"])))
    segments, counts = pw.assemble(kept, qids)
    actions = _action_block(enc, key, pair["origin"], w)
    segments.insert(1, ("text", actions))
    counts["N_fixed"] += len(actions)
    encs, idx, omitted = [], [], []
    qs = by_arm[key]
    for i, q in enumerate(qs):
        if q["group"] == "event" and q["label"] is None:
            omitted.append({"text": q["text"], "reason": "untyped events or incomplete observation interval"})
            continue
        encoded = K.encode_query(q, enc.answers, enc.vocab, enc.tok, enc.text_cache)
        if encoded is None:
            _, code, _ = enc.answers.query_domain(q)
            omitted.append({"text": q["text"], "reason": "out_of_support" if code == "out_of_support" else "undefined label"})
        else:
            encs.append(encoded)
            idx.append(i)
    if not encs:
        raise ValueError(f"no encodable scoped ARM labels: {key}")
    visual = {V.ref_key(r): V.load_features(r) for kind, r in segments if kind == "image"}
    joint = D.joint_order(qs, idx, encs)
    counts["N_query"] = D.query_positions(encs, joint)
    counts["N_total"] = D.padded_total(counts, encs, joint)
    return {"episode": pair["parent_episode"], "origin": pair["origin"], "budget": budget, "policy": policy,
            "segments": segments, "visual": visual, "encs": encs, "query_index": idx,
            "joint": joint, "counts": counts, "kept": kept,
            "groups": [e.group for e in encs], "arm_key": key, "queries": qs,
            "diagnostics": {**w["arm_diagnostics"][key], "omitted_queries": omitted}}


def _swap_actions(item, other):
    segments = list(item["segments"])
    segments[1] = other["segments"][1]
    counts = dict(item["counts"])
    counts["N_fixed"] += len(segments[1][1]) - len(item["segments"][1][1])
    counts["N_total"] = D.padded_total(counts, item["encs"], item["joint"])
    return {**item, "segments": segments, "counts": counts}


def _report(rows, diagnostics):
    parents = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        table = parents[(row["budget"], row["policy"])][row["parent_episode"]]
        for condition in ("arm|arm", "factual|factual"):
            for group, summary in row["scores"][condition].items():
                table[f"{condition}/{group}"].append(summary["nll"])
        for group, matched in row["scores"]["arm|arm"].items():
            alternate = row["scores"]["arm|factual"][group]
            table[f"sensitivity/{group}"].append(alternate["nll"]-matched["nll"])
    mean = lambda values: sum(values)/len(values)
    result = {}
    for (budget, policy), parent in sorted(parents.items()):
        per_parent = {p: {metric: mean(v) for metric, v in metrics.items()} for p, metrics in parent.items()}
        metrics = sorted({m for p in per_parent.values() for m in p})
        result[f"{budget}/{policy}"] = {
            "parents": len(parent), "pairs": sum(r["budget"] == budget and r["policy"] == policy for r in rows),
            "per_parent": per_parent,
            "mean": {m: mean([p[m] for p in per_parent.values() if m in p]) for m in metrics},
            "parents_per_metric": {m: sum(m in p for p in per_parent.values()) for m in metrics}}
    return {"statement": "Arms never fit any predictor, selector or calibration parameter. Probabilities refer to "
                         "the intervention distribution, not natural frequencies. ARM evaluates branch futures on "
                         "seen parent backgrounds and does not establish unseen-family generalization.",
            "aggregation": "Mean query NLL per group within each pair; mean pairs per parent; equal-weight mean parents.",
            "sensitivity": "NLL(arm labels | factual actions) - NLL(arm labels | arm actions); positive uses intervention.",
            "diagnostics": diagnostics, "results": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[A.TOTAL_BUDGET], help="total budgets")
    parser.add_argument("--policy", choices=A.POLICIES, default="recency")
    parser.add_argument("--selector", default=None, help="selector state: choose the shared evidence with it")
    parser.add_argument("--visual", choices=("learned", "fixed"), default="learned")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 0) or any(b < 0 for b in args.budgets):
        parser.error("limit and budgets must be nonnegative")
    if not args.name or "/" in args.name or args.name in (".", ".."):
        parser.error("name must be one directory name")
    pairs = arm_pairs()
    audit = {**arm_pairs.diagnostics, "item_skipped": []}
    if args.limit is not None:
        pairs = pairs[:args.limit]
    from Dataset.world_model.unified import evaluate as E
    from Dataset.world_model.unified.predictor import Predictor
    from Dataset.world_model.unified.train_predictor import load_trainable

    model = Predictor(device="cuda", query_chunk=8)
    load_trainable(model, args.checkpoint)
    model.eval()
    enc = D.Encoder()
    selector = None
    if args.selector:
        from Dataset.world_model.unified.selector import Selector
        selector = Selector()
        selector.load_state_dict(torch.load(args.selector, map_location="cpu"), strict=False)
        selector.eval()
    directory = OUT/args.name
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with (directory/"rows.jsonl").open("w") as handle:
        for pair in pairs:
            for budget in args.budgets:
                try:
                    arm = arm_item(enc, pair, "arm", budget, args.policy, images=not args.no_images, selector=selector,
                                   visual=args.visual)
                    factual = arm_item(enc, pair, "factual", budget, args.policy, images=not args.no_images,
                                       selector=selector, visual=args.visual)
                except (OSError, ValueError, KeyError) as exc:
                    skipped = {"arm_key": pair["arm_key"], "budget": budget, "reason": str(exc)}
                    audit["item_skipped"].append(skipped)
                    print(_json({"arm_item_skipped": skipped}), file=sys.stderr, flush=True)
                    continue
                inputs = {"arm|arm": arm, "arm|factual": _swap_actions(arm, factual),
                          "factual|factual": factual, "factual|arm": _swap_actions(factual, arm)}
                scores = {name: E.summarize(E.score_item(model, item)["rows"]) for name, item in inputs.items()}
                rec = {**pair, "budget": budget, "policy": "selector" if selector else args.policy, "scores": scores,
                       "counts": {k: item["counts"] for k, item in inputs.items()},
                       "item_diagnostics": {"arm": arm["diagnostics"], "factual": factual["diagnostics"]}}
                handle.write(_json(rec)+"\n")
                handle.flush()
                rows.append(rec)
    report = _report(rows, audit)
    (directory/"report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False)+"\n")
    print(_json({"rows": len(rows), "report": str(directory/"report.json")}))


if __name__ == "__main__":
    main()
