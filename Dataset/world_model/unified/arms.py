"""TRAIN ARM items and paired held-out intervention evaluation (RESEARCH_PLAN §2, §8, §9.1, §10).

Both members use the parent history, union query scope and one evidence capacity. Only origin actions and
authored plans condition their future labels. TRAIN arms fit the predictor and selection supervision; TEST
arms evaluate held-out scenario templates with matched scores, calibration inputs and two-way action swaps.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import functools
import itertools
import json
import random
import re
from statistics import fmean, stdev
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


@functools.lru_cache(maxsize=1)
def arm_pairs():
    """Cached pairs with source-validity exclusions printed and retained (RESEARCH_PLAN §2, §8).

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
                          "group": key.rsplit("/", 1)[0], "profile": D.profile(meta["parent_episode"]),
                          "intervention_type": _catalog()[key]["group"],
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


@functools.lru_cache(maxsize=4)
def pairs(split):
    """Pairs inherit their parent's TRAIN/DEV/TEST split; subsequent calls are cheap (RESEARCH_PLAN §8)."""
    if split == "all":
        return arm_pairs()
    if split not in ("train", "dev", "test"):
        raise ValueError(split)
    parents = set(D.splits()[split])
    return [p for p in arm_pairs() if p["parent_episode"] in parents]


@functools.lru_cache(maxsize=4)
def groups(split):
    """Intervention groups with one shared factual member for training sampling (RESEARCH_PLAN §8, §9.1)."""
    out = defaultdict(list)
    for pair in pairs(split):
        out[pair["group"]].append(pair)
    return dict(out)


ACTION_METADATA = frozenset((
    "arm_id", "arm_key", "branch", "branch_id", "group", "group_key", "group_name", "episode_id",
    "scenario_id", "seed", "name", "label", "title", "description", "comment", "comments", "notes",
    "reason", "rationale", "hypothesis", "expected", "falsified_if", "research_question", "provenance",
    "sources", "source_refs", "source", "output", "script", "entrypoint", "path", "file", "filename",
    "intent", "intent_stage", "causal_chain_id", "causal_predecessor_intent", "target_roles", "log_event",
    "uav_corridor_validation", "audit", "diagnostics", "parameter_sources", "schema", "$schema", "version"))
SCHEMA_IDS = frozenset(("predicate_id", "predicate_ids", "event_type_id", "event_type_ids",
                        "event_family_id", "event_family_ids"))
FILE_NAME = re.compile(r"(?:^/|(?:^|/)(?:arms|interventions|Dataset|aw_data)/|\.(?:jsonl?|ya?ml|py|npz|pt)(?:$|#))")
BRANCH_NAME = re.compile(r"(?:^|[/_.-])(?:arm|grp|factual|counterfactual)(?:$|[/_.-])")


def _action_body(value, aliases):
    """Strip metadata/paths and alias authored identifiers, preserving executable values (RESEARCH_PLAN §8)."""
    is_path = lambda v: isinstance(v, str) and FILE_NAME.search(v) is not None
    if isinstance(value, dict):
        return {aliases.get(k, k): _action_body(v, aliases) for k, v in value.items()
                if k not in ACTION_METADATA and not k.endswith(("_path", "_file", "_filename")) and not is_path(v)}
    if isinstance(value, list):
        return [_action_body(v, aliases) for v in value if not is_path(v)]
    return aliases.get(value, value) if isinstance(value, str) else value


@functools.lru_cache(maxsize=None)
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
            out[(kind, row[identity])] = _action_body(row, {})
    return out


@functools.lru_cache(maxsize=None)
def _members(group):
    return tuple(k for k in _catalog() if k.rsplit("/", 1)[0] == group)


@functools.lru_cache(maxsize=None)
def _plans(key):
    """Actions and conditions declared by configuration, excluding outcome prose (RESEARCH_PLAN §8)."""
    config = _meta(key)["config"]
    execution = config.get("execution", {})
    plans = list(execution.get("actions", ())) + list(config.get("inject_actions", ()))
    plans += list(config.get("intervention_actions", ()))
    conditions = {k: execution[k] for k in ("omit_action_ids", "trigger_rewrites") if execution.get(k)}
    conditions.update({k: config[k] for k in ("omit_action_ids", "cancel_event_ids", "removed_event_ids",
                                            "retime", "state_overrides", "sumo_intervention", "gates")
                       if config.get(k) is not None and config.get(k) != [] and config.get(k) != {}})
    removed = config.get("definition", {}).get("branch_changes", {}).get("removed_action_ids")
    if removed:
        conditions["removed_action_ids"] = removed
    return _action_body(plans, {}), _action_body(conditions, {})


def _strings(value):
    if isinstance(value, dict):
        return {s for k, v in value.items() for s in {k} | _strings(v)}
    if isinstance(value, list):
        return {s for v in value for s in _strings(v)}
    return {value} if isinstance(value, str) else set()


@functools.lru_cache(maxsize=None)
def _script_scope(group):
    """Shared declarations and dependencies for changes and referenced omissions (RESEARCH_PLAN §8)."""
    members = _members(group)
    factual = _factual_key(members[0])
    baseline = _script(factual)
    changed = set()
    references = {s for key in members for body in _plans(key) for s in _strings(body)}
    scripts = [_script(key) for key in members]
    for script in scripts:
        changed.update(k for k in set(script) | set(baseline) if script.get(k) != baseline.get(k))
        for k, body in script.items():
            if k[1] in references or any(a.get("action_id") in references for a in body.get("actions", ())):
                changed.add(k)
    # Resolve anonymous references through authored trigger/event definitions, never execution histories.
    while True:
        before = set(changed)
        triggers = {identity for kind, identity in changed if kind == "triggers"}
        for script in scripts:
            for k, body in script.items():
                if k[0] == "events" and body.get("trigger_ref") in triggers:
                    changed.add(k)
                if k not in changed:
                    continue
                for identity in body.get("require_conditions", ()):
                    changed.add(("triggers", identity))
                if body.get("trigger_ref"):
                    changed.add(("triggers", body["trigger_ref"]))
                if k[0] == "triggers" and body.get("event_id"):
                    changed.add(("events", body["event_id"]))
        if changed == before:
            return tuple(sorted(changed))


def _status(status):
    """Operational status only: 'omitted_by_explicit_intervention' and 'omitted_by_configuration' both read
    'omitted', so the text states what happens, not which branch authored it (RESEARCH_PLAN §8)."""
    return "omitted" if status.startswith("omitted") else status


@functools.lru_cache(maxsize=None)
def _action_records(arm_key, origin):
    records = []
    for row in _actions(arm_key):
        if row["tick"] == origin:
            records.append(("t=0", _action_body({"action": row["action"], "status": _status(row["result"]["status"])}, {})))
    plans, conditions = _plans(arm_key)
    # Include declarations at the origin too: an authored conditional action need not have executed yet.
    records.extend(("fixed schedule", p) for p in plans)
    if conditions:
        records.append(("fixed conditions", conditions))
    script = _script(arm_key)
    for kind, identity in _script_scope(arm_key.rsplit("/", 1)[0]):
        body = script.get((kind, identity))
        declaration = {"kind": kind, "id": identity, "definition": body,
                       "status": "declared" if body is not None else "omitted"}
        records.append(("fixed script", declaration))
    return records


@functools.lru_cache(maxsize=None)
def _action_aliases(group, origin):
    """One anonymous reference map for every member, based only on origin inputs (RESEARCH_PLAN §8)."""
    identities = set()
    def collect(value, field=""):
        if isinstance(value, dict):
            for k, v in value.items():
                collect(v, k)
        elif isinstance(value, list):
            for v in value:
                collect(v, field)
        elif isinstance(value, str) and field not in SCHEMA_IDS:
            if field == "id" or field.endswith(("_id", "_ids", "_ref", "_refs")) or BRANCH_NAME.search(value):
                identities.add(value)
    for key in _members(group):
        for kind, identity in _script(key):
            identities.add(identity)
        for _, body in _action_records(key, origin):
            collect(body)
    return {identity: f"x{i}" for i, identity in enumerate(sorted(identities), 1)}


def action_lines(arm_key, origin, aliases):
    """Origin actions/statuses and static plans with anonymous identifiers (RESEARCH_PLAN §8).

    Remove branch/group/episode names, paths, file names, titles, descriptions, intent/causal annotations,
    provenance, expected outcomes and compiler diagnostics. Alias every authored action/event/trigger ID,
    reference and other non-schema ID (including outcome words embedded in IDs), consistently across the
    group; parent entity aliases take precedence. Retain executable types, conditions, timing, coordinates,
    state patches and origin status. Future dispatch ticks, results and effects never supply input text.
    """
    refs = {**_action_aliases(arm_key.rsplit("/", 1)[0], origin), **aliases}
    return ["  " + prefix + " " + _json(_action_body(body, refs)) for prefix, body in _action_records(arm_key, origin)]


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
                 "text": f"{a} position@+{h*S.TICK_S:.1f}s:", "reference": list(ref),
                 "residual": residual, "eps": S.POS_EPS}
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


def _masked_query(q):
    """A union-scope query absent from one branch has an undefined label (RESEARCH_PLAN §2, §8)."""
    q = {k: v for k, v in q.items() if k != "diagnostics"}
    q["residual" if "residual" in q else "label"] = None
    if "compatible" in q:
        q["compatible"] = []
    return q


def arm_window(enc, pair, images=True):
    """Shared parent evidence, union query scope and branch queries at the fork (RESEARCH_PLAN §8)."""
    key = (pair["arm_key"], pair["origin"], images)
    cached = getattr(enc, "_arm_window", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    src = D.cached_source(pair["parent_episode"])
    window = W.build_window(src, pair["origin"])
    # Captures and their relevance use the parent's history and the shared union query scope.
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
    for k, qs in queries.items():
        lookup = {_query_key(q): q for q in qs}
        queries[k] = [lookup[_query_key(q)] if _query_key(q) in lookup else _masked_query(q)
                      for q in window["queries"]]
    pw = A.PreparedWindow(window, enc.tok, caps)
    enc.set_relevance(window, pw, range(len(window["queries"])))
    window["arm_action_ids"] = {k: _action_block(enc, k, pair["origin"], window) for k in queries}
    window["arm_query_reserve"] = max(D.query_reserve(enc, qs, range(len(qs))) for qs in queries.values())
    # fixed_cost already counts the default block. Reserve only its replacement delta, plus the larger
    # label-free query block, so either branch and either action swap fits with the same optional evidence.
    window["arm_reserve"] = window["arm_query_reserve"] + \
        max(len(ids) for ids in window["arm_action_ids"].values()) - len(pw.action_ids)
    window["arm_features"] = {}
    prepared = window, pw, queries
    enc._arm_window = key, prepared
    return prepared


def _action_block(enc, key, origin, window):
    """Anonymous origin conditions in the same action-block position as parent items (RESEARCH_PLAN §8)."""
    text = ("actions fixed at t=0\n"
            f"  origin_tick={origin}; scheduled ticks are simulator ticks; action coordinates are absolute ENU metres; "
            f"window reference o={W.vec(window['frame_origin_cm'], S.POS_EPS)}\n")
    lines = action_lines(key, origin, window["aliases"])
    text += "\n".join(lines) + "\n" if lines else A.NO_ACTIONS_TEXT.split("\n", 1)[1]
    return enc.tok(text, add_special_tokens=False)["input_ids"]


def optional_budget(enc, pair, budget):
    """Shared optional capacity, using the prepared window's image setting (RESEARCH_PLAN §6, §8)."""
    cached = getattr(enc, "_arm_window", None)
    w, pw, _ = cached[1] if cached is not None and cached[0][:2] == (pair["arm_key"], pair["origin"]) \
        else arm_window(enc, pair)
    qids = list(range(len(w["queries"])))
    optional = budget - pw.fixed_cost(qids) - w["arm_reserve"] - A.PAD_RESERVE
    if optional < 0:
        raise ValueError(f"mandatory ARM inputs exceed the total budget {budget}: {pair['arm_key']}")
    return optional


def arm_item(enc, pair, which, budget, policy, images=True, selector=None, kept=None, rng=None):
    """Encoder-compatible branch item; selection uses shared parent evidence (RESEARCH_PLAN §8, §9.1).

    budget is the total sequence budget.  All scoped queries form a shared union for fixing origin lines and
    choosing optional evidence (text records and parent captures). The selector may condition on the declared
    origin actions; future labels and diagnostics never enter it. kept overrides selection; policy is a
    baseline name or {text, visual, share}, with share=0 keeping no captures. rng drives random reductions.
    """
    if which not in ("arm", "factual"):
        raise ValueError(which)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    key = pair["arm_key"] if which == "arm" else pair["factual_key"]
    w, pw, by_arm = arm_window(enc, pair, images)
    qids = list(range(len(w["queries"])))
    optional = optional_budget(enc, pair, budget)
    if kept is None:
        rng = rng if rng is not None else random.Random(W.stable_hash(pair["parent_episode"], pair["origin"]))
        if selector is not None:
            kept, _ = selector.select(pw, enc.tok, qids, optional, actions_ids=w["arm_action_ids"][key])
        elif isinstance(policy, dict):
            if policy["share"] == 0:
                kept, _ = pw.greedy_fill(A.policy_order(pw, qids, policy["text"], rng, kinds="text"), optional, qids)
            else:
                kept, _ = A.mixed_fill(pw, qids, optional, policy["text"], policy["visual"], policy["share"], rng)
        else:
            kept, _ = A.select(pw, qids, optional, policy, rng)
    kept = list(dict.fromkeys(kept))
    fixed, _ = pw.fixed_keys(qids)
    used = pw.set_cost([k for k in kept if k not in fixed], qids)
    if used > optional:
        raise ValueError(f"kept ARM evidence costs {used} positions but shared capacity is {optional}: {key}")
    segments, counts = pw.assemble(kept, qids, actions_ids=w["arm_action_ids"][key])
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
    features, visual_features = w["arm_features"], {}
    for kind, ref in segments:
        if kind == "image":
            ref_key = V.ref_key(ref)
            if ref_key not in features:
                features[ref_key] = V.load_features(ref)
            visual_features[ref_key] = features[ref_key]
    joint = D.joint_order(qs, idx, encs)
    counts["N_query"] = D.query_positions(encs, joint)
    counts["N_query_reserved"] = w["arm_query_reserve"]
    counts["N_total"] = D.padded_total(counts, encs, joint)
    if counts["N_total"] > budget:
        raise ValueError(f"ARM sequence of {counts['N_total']} positions exceeds the total budget {budget}: {key}")
    counts["optional_budget"], counts["optional_used"] = optional, used
    counts["out_of_support"] = sum(q["reason"] == "out_of_support" for q in omitted)
    counts["N_text_full"] = sum(len(u.ids) for u in pw.candidates(qids, "text")) + \
        sum(len(v) for h, v in pw.headings.items() if h not in pw.base_heads(qids) and h != "images")
    counts["N_captures_full"] = len(pw.captures())
    counts["N_visual_full"] = sum(u.extra - 1 for u in pw.captures())
    meta = [{k: qs[i].get(k) for k in ("group", "tuple", "entity", "horizon", "state", "residual")} for i in idx]
    return {"episode": pair["parent_episode"], "origin": pair["origin"], "budget": budget,
            "policy": policy if not isinstance(policy, dict) else "mix:" + json.dumps(policy, sort_keys=True),
            "segments": segments, "visual": visual_features, "encs": encs, "query_index": idx, "query_meta": meta,
            "visual_refs": [[r["observer"], r["tick"]] for kind, r in segments if kind == "image"],
            "joint": joint, "counts": counts, "kept": kept,
            "groups": [e.group for e in encs], "arm_key": key, "group": pair["group"], "which": which,
            "query_ids": qids, "queries": qs,
            "diagnostics": {**w["arm_diagnostics"][key], "omitted_queries": omitted}}


def _swap_actions(item, other):
    """Replace assemble's action segment immediately after background (RESEARCH_PLAN §8, §10)."""
    segments = list(item["segments"])
    segments[1] = other["segments"][1]
    counts = dict(item["counts"])
    counts["N_fixed"] += len(segments[1][1]) - len(item["segments"][1][1])
    counts["N_total"] = D.padded_total(counts, item["encs"], item["joint"])
    if counts["N_total"] > item["budget"]:
        raise ValueError(f"swapped ARM sequence exceeds the total budget {item['budget']}: {item['arm_key']}")
    return {**item, "segments": segments, "counts": counts}


@torch.no_grad()
def _score_item(model, item):
    """Marginal scores and calibration inputs from the same forward pass (RESEARCH_PLAN §8, §10)."""
    ctx = model.build_context(item["segments"], D.visual_loader(item))
    lps = model.marginal(ctx, [(e.ids, e.steps) for e in item["encs"]])
    rows = []
    for i, e, lp in zip(item["query_index"], item["encs"], lps):
        q = item["queries"][i]
        probs = [p.exp().cpu().tolist() for p in lp]
        row = {"query_index": i, "query_key": _query_key(q), "group": e.group, "text": q["text"],
               "nll": float(K.query_loss(lp, e)), "targets": e.targets, "compatible": e.compatible,
               "target_code": e.path, "step_probabilities": probs, "step_candidates": [s[1] for s in e.steps]}
        if len(probs) == 1 and not isinstance(e.trees[0], K.IntervalCode):
            row["probabilities"] = probs[0]
            row["label"] = e.targets[0] if e.targets is not None else None
            if e.targets is not None:
                row["brier"] = sum((p - int(j == e.targets[0])) ** 2 for j, p in enumerate(probs[0]))
        if e.group in ("position", "latency_ms", "cpu_utilization_pct"):
            row["residual"], row["eps"] = q["residual"], q["eps"]
            kids = e.trees[0].tree.children[()]
            rank = lambda s: 0 if s == "z" else (-1 if s[0] == "-" else 1) * (int(s[1:]) + 1)
            order = sorted(range(len(kids)), key=lambda j: rank(kids[j]))
            row["coarse_rps"] = K.ranked_probability_score([probs[0][j] for j in order], order.index(e.targets[0]))
        rows.append(row)
    return rows


def _summarize(rows):
    table = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in ("nll", "brier", "coarse_rps"):
            if metric in row:
                table[row["group"]][metric].append(row[metric])
    return {g: {k: fmean(v) for k, v in metrics.items()} | {"n": len(metrics["nll"])} |
            {k + "_n": len(v) for k, v in metrics.items() if k != "nll"} for g, metrics in table.items()}


def _common_queries(arm, factual):
    """Common encodable queries; censoring never implies equality or a changed label (RESEARCH_PLAN §2, §8)."""
    def lookup(item):
        return {_query_key(item["queries"][i]): (item["queries"][i], e)
                for i, e in zip(item["query_index"], item["encs"])}
    a, f = lookup(arm), lookup(factual)
    common = {}
    for key in sorted(a.keys() & f.keys(), key=_json):
        aq, ae = a[key]
        fq, fe = f[key]
        if ae.targets is None or fe.targets is None:
            common[key] = "censored"
        else:
            field = "residual" if "residual" in aq else "label"
            common[key] = "changed" if aq[field] != fq[field] else "unchanged"
    return common


def _common_item(item, common):
    """Restrict scored queries while retaining the union context and evidence capacity (RESEARCH_PLAN §8)."""
    take = [j for j, i in enumerate(item["query_index"]) if _query_key(item["queries"][i]) in common]
    encs = [item["encs"][j] for j in take]
    idx = [item["query_index"][j] for j in take]
    joint = D.joint_order(item["queries"], idx, encs)
    counts = dict(item["counts"])
    counts["N_query"] = D.query_positions(encs, joint)
    counts["N_total"] = D.padded_total(counts, encs, joint)
    return {**item, "encs": encs, "query_index": idx, "query_meta": [item["query_meta"][j] for j in take],
            "groups": [e.group for e in encs], "joint": joint, "counts": counts}


def _swap_scores(query_rows, common):
    """Two paired NLL differences, stratified by common-query label changes (RESEARCH_PLAN §8, §10)."""
    out = {}
    for direction, matched in (("arm|factual", "arm|arm"), ("factual|arm", "factual|factual")):
        base = {r["query_key"]: r for r in query_rows[matched]}
        swapped = {r["query_key"]: r for r in query_rows[direction]}
        out[direction] = {}
        for bucket in ("all", "changed", "unchanged", "censored"):
            groups = defaultdict(list)
            for key, change in common.items():
                if bucket == "all" or change == bucket:
                    groups[base[key]["group"]].append((base[key]["nll"], swapped[key]["nll"]))
            out[direction][bucket] = {
                g: {"n": len(vals), "matched_nll": fmean(a for a, _ in vals),
                    "swapped_nll": fmean(b for _, b in vals), "sensitivity": fmean(b-a for a, b in vals)}
                for g, vals in groups.items()}
    return out


def _pair_metrics(row):
    values, counts = {}, {}
    matched = defaultdict(list)
    for condition in ("arm|arm", "factual|factual"):
        summary = row["scores"][condition]
        for group, score in summary.items():
            for metric in ("nll", "brier", "coarse_rps"):
                if metric not in score:
                    continue
                name = f"{condition}/{group}/{metric}"
                values[name] = score[metric]
                counts[name] = score["n"] if metric == "nll" else score[metric + "_n"]
                matched[f"matched/{group}/{metric}"].append((values[name], counts[name]))
        if summary:
            name = f"{condition}/weighted_nll"
            values[name] = fmean(s["nll"] for s in summary.values())
            counts[name] = sum(s["n"] for s in summary.values())
    for name, scores in matched.items():
        values[name], counts[name] = fmean(v for v, _ in scores), sum(n for _, n in scores)
    if matched:
        names = [k for k in matched if k.endswith("/nll")]
        values["matched/weighted_nll"] = fmean(values[k] for k in names)
        counts["matched/weighted_nll"] = sum(counts[k] for k in names)
    for direction, buckets in row["swaps"].items():
        for bucket, summary in buckets.items():
            for group, score in summary.items():
                for metric in ("matched_nll", "swapped_nll", "sensitivity"):
                    name = f"swap/{direction}/{bucket}/{group}/{metric}"
                    values[name], counts[name] = score[metric], score["n"]
    return values, counts


def _profile_se(clusters, draws, seed):
    """Bootstrap profiles, retaining every parent within a sampled profile (RESEARCH_PLAN §8)."""
    profiles = sorted(clusters)
    if len(profiles) < 2:
        return None
    sums = {p: (sum(clusters[p]), len(clusters[p])) for p in profiles}
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        picked = [sums[rng.choice(profiles)] for _ in profiles]
        samples.append(sum(total for total, _ in picked) / sum(n for _, n in picked))
    return stdev(samples)


def _aggregate(rows, draws=2000, seed=0):
    """Group-normalized pair scores, equal-weight parents, profile-clustered SEs (RESEARCH_PLAN §8, §10)."""
    parents = defaultdict(lambda: defaultdict(list))
    query_counts = Counter()
    for row in rows:
        metrics, counts = _pair_metrics(row)
        for metric, value in metrics.items():
            parents[row["parent_episode"]][metric].append(value)
            query_counts[metric] += counts[metric]
    per_parent = {p: {m: fmean(v) for m, v in metrics.items()} for p, metrics in sorted(parents.items())}
    metrics = sorted({m for values in per_parent.values() for m in values})
    ses, cluster_counts = {}, {}
    for metric in metrics:
        clusters = defaultdict(list)
        for parent, scores in per_parent.items():
            if metric in scores:
                clusters[D.profile(parent)].append(scores[metric])
        ses[metric] = _profile_se(clusters, draws, W.stable_hash("arm-bootstrap", seed, metric))
        cluster_counts[metric] = len(clusters)
    return {"parents": len(per_parent), "profiles": len({D.profile(p) for p in per_parent}), "pairs": len(rows),
            "per_parent": per_parent, "profile_per_parent": {p: D.profile(p) for p in per_parent},
            "mean": {m: fmean(v[m] for v in per_parent.values() if m in v) for m in metrics},
            "profile_clustered_se": ses, "profiles_per_metric": cluster_counts,
            "parents_per_metric": {m: sum(m in v for v in per_parent.values()) for m in metrics},
            "queries_per_metric": dict(query_counts)}


def _report(rows, diagnostics, split="test", train_types=None):
    """Matched quality and two-way swaps by intervention exposure in TRAIN (RESEARCH_PLAN §8, §10)."""
    train_types = {p["intervention_type"] for p in pairs("train")} if train_types is None else set(train_types)
    table = defaultdict(list)
    for row in rows:
        table[(row["budget"], row["policy"])].append(row)
    results = {}
    for (budget, policy), selected in sorted(table.items()):
        by_type = defaultdict(list)
        for row in selected:
            by_type[row["intervention_type"]].append(row)
        results[f"{budget}/{policy}"] = {
            **_aggregate(selected),
            "seen_vs_unseen": {label: _aggregate([r for r in selected
                                                 if (r["intervention_type"] in train_types) == seen])
                               for label, seen in (("seen", True), ("unseen", False))},
            "by_intervention_type": {t: {"seen_in_train": t in train_types, **_aggregate(rs)}
                                     for t, rs in sorted(by_type.items())}}
    return {"split": split, "held_out_templates": split == "test",
            "statement": "TRAIN arms fit the predictor and enter selection supervision. Reported TEST numbers "
                         "evaluate held-out scenario templates within represented families, with intervention "
                         "types seen or unseen in TRAIN. DEV supports model choice; all is a pooled diagnostic. "
                         "Probabilities refer to the intervention distribution, not natural frequencies. Positive "
                         "swap sensitivity alone does not establish accurate prediction or unseen-mechanism generalization.",
            "aggregation": "Mean queries per target group and pair; mean pairs per parent; equal-weight mean parents. "
                           "Matched combines the two branch group means with equal weight.",
            "bootstrap": {"cluster": "profile", "draws": 2000, "seed": 0,
                          "se": "Sample standard deviation of bootstrap parent means; null with fewer than two profiles."},
            "sensitivity": "NLL(branch labels | other actions) - NLL(branch labels | matched actions), separately "
                           "for arm|factual and factual|arm on common encodable queries. Numeric changes compare "
                           "quantized residuals; categorical changes compare classes. Censored labels are a separate "
                           "stratum and never treated as known changed or unchanged outcomes.",
            "calibration_inputs": {"file": "rows.jsonl", "field": "query_rows",
                                   "matched_conditions": ["arm|arm", "factual|factual"],
                                   "format": "Full probabilities and label/compatible indices for one-level domains; "
                                             "teacher-forced conditional step probabilities, candidate token IDs and "
                                             "target codes for all domains. Numeric coarse-root RPS and categorical "
                                             "Brier are aggregated; conditional steps are not a full numeric CDF."},
            "train_intervention_types": sorted(train_types), "diagnostics": diagnostics, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--selector", required=True, help="post-warmup selector checkpoint")
    args = parser.parse_args()
    if not args.name or "/" in args.name or args.name in (".", ".."):
        parser.error("name must be one directory name")
    selected = pairs(args.split)
    train_types = {p["intervention_type"] for p in pairs("train")}
    audit = {**arm_pairs.diagnostics, "split": args.split, "split_pairs": len(selected)}
    if not selected:
        raise ValueError(f"no ARM pairs selected for {args.split}")
    from Dataset.world_model.unified.predictor import Predictor
    from Dataset.world_model.unified.train_predictor import load_trainable

    model = Predictor(device="cuda", query_chunk=8)
    load_trainable(model, args.checkpoint)
    model.eval()
    enc = D.Encoder()
    from Dataset.world_model.unified.selector import Selector
    selector = Selector.load(args.selector)
    selector.eval()
    directory = OUT/args.name
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with (directory/"rows.jsonl").open("w") as handle:
        for pair in selected:
            budget = A.TOTAL_BUDGET
            arm = arm_item(enc, pair, "arm", budget, "selector", selector=selector)
            factual = arm_item(enc, pair, "factual", budget, "selector", kept=arm["kept"])
            common = _common_queries(arm, factual)
            inputs = {"arm|arm": arm, "factual|factual": factual,
                      "arm|factual": _swap_actions(_common_item(arm, common), factual),
                      "factual|arm": _swap_actions(_common_item(factual, common), arm)}
            query_rows = {name: _score_item(model, item) if item["encs"] else [] for name, item in inputs.items()}
            for scored in query_rows.values():
                for row in scored:
                    row["comparison"] = common.get(row["query_key"], "not_common")
            scores = {name: _summarize(scored) for name, scored in query_rows.items()}
            rec = {**pair, "budget": budget, "policy": "selector", "scores": scores,
                   "split": args.split, "seen_in_train": pair["intervention_type"] in train_types,
                   "query_rows": query_rows, "swaps": _swap_scores(query_rows, common),
                   "common_query_counts": {"all": len(common),
                                           **{b: sum(v == b for v in common.values())
                                              for b in ("changed", "unchanged", "censored")}},
                   "counts": {k: item["counts"] for k, item in inputs.items()},
                   "item_diagnostics": {"arm": arm["diagnostics"], "factual": factual["diagnostics"]}}
            handle.write(_json(rec)+"\n")
            handle.flush()
            rows.append({k: v for k, v in rec.items() if k != "query_rows"})
    report = _report(rows, audit, args.split, train_types)
    (directory/"report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False)+"\n")
    print(_json({"rows": len(rows), "report": str(directory/"report.json")}))


if __name__ == "__main__":
    main()
