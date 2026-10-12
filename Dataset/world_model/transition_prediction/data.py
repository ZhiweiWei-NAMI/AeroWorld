"""Source-backed prefix tensors and versioned observation-prefix censor labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[3]
GRAPH = ROOT / "Dataset/world_model/runs/belief_label_graph/formal"
OUT = ROOT / os.environ.get("AERO_PREDICATE_RUN_DIR", "Dataset/world_model/runs/predicate_transition_pilot")
PROFILES = ("L1-3_v1", "L1-3_v2", "L1-4_v1", "L1-4_v2")
TRUTH = {"false": 1, "true": 2, "unknown": 3, "out_of_scope": 4}
CATEGORIES = ("business", "pedestrian", "facility", "airspace_corridor", "uav", "vehicle", "compute_node")
LABEL_VIEW = "observation-prefix-censor-v2"
LEGACY_LABEL_VIEW = "complete-window-v1"
LABEL_VIEW_FILE = "samples.observation_prefix_censor_v2.parquet"
TARGET_COLUMNS = ("label_view", "event_observed", "event_bin", "observed_bins", "censor_reason",
                  "likelihood_valid", "horizon_outcome_known", "label")
METRIC_SELECTION = ("NLL uses events and prefixes with at least one observed whole bin. "
                    "Horizon metrics use observed events or all ten event-free bins. "
                    "Censoring may be informative, so these selected populations can be biased.")


def validate_targets(frame):
    """Reject stale or internally inconsistent targets before a likelihood consumer."""
    missing = set(TARGET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"censor target metadata missing: {sorted(missing)}")
    if not frame.label_view.eq(LABEL_VIEW).all():
        raise ValueError("likelihood consumers require the explicit v2 label view")
    event = frame.event_observed.to_numpy(dtype=bool)
    bins = frame.event_bin.to_numpy(dtype=np.int64)
    observed = frame.observed_bins.to_numpy(dtype=np.int64)
    if ((observed < 0) | (observed > 10)).any():
        raise ValueError("observed_bins must be in 0..10")
    if ((bins[event] < 0) | (bins[event] > 9)).any() or (bins[~event] != -1).any():
        raise ValueError("event_bin must be 0..9 for events and -1 otherwise")
    if not np.array_equal(observed[event], bins[event]):
        raise ValueError("event prefixes must stop before their event bin")
    if not np.array_equal(frame.likelihood_valid.to_numpy(dtype=bool), event | (observed > 0)):
        raise ValueError("likelihood_valid disagrees with observed prefix")
    if not np.array_equal(frame.horizon_outcome_known.to_numpy(dtype=bool), event | (observed == 10)):
        raise ValueError("horizon_outcome_known disagrees with observed prefix")
    expected = np.where(event, bins, np.where(observed == 10, 10, -1))
    if not np.array_equal(frame.label.to_numpy(dtype=np.int64), expected):
        raise ValueError("categorical label disagrees with censor metadata")


def require_v2_configuration(config):
    if "label_view" not in config or config["label_view"] != LABEL_VIEW:
        raise ValueError("this action requires an explicit observation-prefix-censor-v2 run configuration; preserve historical artifacts")


def person_period_targets(frame):
    """Observed hazard decisions only; an event includes its event-bin decision."""
    validate_targets(frame)
    event = frame.event_observed.to_numpy(dtype=bool)
    bins = frame.event_bin.to_numpy(dtype=np.int64)
    observed = frame.observed_bins.to_numpy(dtype=np.int64)
    limit = np.where(event, bins + 1, observed)
    person, period = np.where(np.arange(10)[None, :] < limit[:, None])
    target = (event[person] & (bins[person] == period)).astype(np.int64)
    return person, period, target


def attach_label_view(predictions, index):
    """Join immutable prediction row IDs to current metadata without refitting."""
    validate_targets(index)
    columns = [*TARGET_COLUMNS, "first_switch_tick", "source_sampling_step_ticks", "event_precision"]
    clean = predictions.drop(columns=[c for c in [*columns, "valid"] if c in predictions])
    attached = clean.merge(index[["row_id", *columns]], on="row_id", how="left", validate="many_to_one")
    if attached.label_view.isna().any():
        raise ValueError("prediction row ID absent from the immutable sample population")
    validate_targets(attached)
    return attached


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_episode(profile):
    path = GRAPH / (profile + "__seed00")
    scope = pq.read_table(path / "scope.parquet").to_pylist()[0]
    assert (scope["source_kind"], scope["start_tick"], scope["end_tick"]) == ("formal", 0, 900)
    states, streams, transitions = {}, {}, []
    for batch in pq.ParquetFile(path / "nodes.parquet").iter_batches(
            columns=["node_type", "node_id", "category_id", "tick", "attrs_json"]):
        for row in batch.to_pylist():
            kind, tick = row["node_type"], row["tick"]
            if kind not in ("EntityState", "PredicateAssertion", "PredicateTransition"):
                continue
            a = json.loads(row["attrs_json"])
            if kind == "PredicateTransition":
                transitions.append((row["category_id"], a))
                continue
            if tick % 5:
                continue
            if kind == "EntityState":
                name = a["entity_id"]
                pose = a["raw"]["truth_pose"]
                yaw = np.deg2rad(pose["rotation_deg"]["yaw_deg"])
                value = pose["position_enu_m"] + pose["velocity_enu_mps"] + [np.sin(yaw), np.cos(yaw)]
                assert len(value) == 8 and np.isfinite(value).all()
                if name not in states:
                    states[name] = {"category": a["entity_category"], "values": {}}
                assert tick not in states[name]["values"], (profile, name, tick)
                states[name]["values"][tick] = value
            else:
                if a.get("assertion_layer") is not None:
                    continue
                key = (a["predicate_id"], tuple(sorted(a["bindings"].items())))
                assert all(isinstance(v, str) for _, v in key[1]) and 0 < len(key[1]) <= 3
                if key not in streams:
                    streams[key] = {}
                assert tick not in streams[key], (profile, key, tick)
                streams[key][tick] = (TRUTH[a["truth_status"]], row["node_id"])
    # Identity strings are correspondence only; only prefix-present nodes reach the model.
    names = sorted(set(states) | {value for _, binding in streams for _, value in binding})
    entity_index = {name: i for i, name in enumerate(names)}
    keys = sorted(streams)
    physical = np.zeros((181, len(names), 8), np.float32)
    present = np.zeros((181, len(names)), bool)
    category = np.zeros(len(names), np.int64)
    for name, record in states.items():
        e = entity_index[name]
        category[e] = CATEGORIES.index(record["category"])
        for tick, value in record["values"].items():
            physical[tick // 5, e] = value
            present[tick // 5, e] = True
    truth = np.zeros((181, len(keys)), np.int64)
    pointers = np.zeros((len(keys), 3), np.int64)
    roles = [["<none>"] * 3 for _ in keys]
    role_present = np.zeros((len(keys), 3), bool)
    refs = {}
    for q, key in enumerate(keys):
        refs[q] = {tick: ref for tick, (_, ref) in streams[key].items()}
        for tick, (value, _) in streams[key].items():
            truth[tick // 5, q] = value
        for r, (role, name) in enumerate(key[1]):
            pointers[q, r] = entity_index[name]
            roles[q][r] = role
            role_present[q, r] = True
    transition_checked = 0
    for pred, a in transitions:
        if a.get("transition_authority") == "compute_comm_supplement_event":
            continue  # These projected event transitions have a different authority and no tuple binding.
        key = (pred, tuple(sorted(a["bindings"].items())))
        start, end = a["from_tick"], a["to_tick"]
        if key in streams and start in streams[key] and end in streams[key]:
            assert streams[key][start][0] == TRUTH[a["from_value"]]
            assert streams[key][end][0] == TRUTH[a["to_value"]]
            transition_checked += 1
    return dict(profile=profile, source=str(path.relative_to(ROOT)), keys=keys, names=names,
                physical=physical, present=present, category=category, truth=truth,
                pointers=pointers, roles=roles, role_present=role_present, refs=refs,
                transition_checked=transition_checked)


def split_anchors(profile):
    if profile.startswith("L1-3"):
        return [("train", t) for t in range(100, 501, 25)] + [("validation", t) for t in range(675, 851, 25)]
    return [("test", t) for t in range(100, 851, 25)]


def first_opposite_target(current, anchor, ticks, truth, *, source_sampling_step_ticks,
                          candidate_present=None, entity_exit=None):
    """First opposite truth on an explicit observation grid, stopped at first censor.

    Grid continuity does not establish continuous-time survival between observations.
    Extra candidate/exit masks must come from source evidence, never inferred from
    missing assertions or physical state disappearance.
    """
    step = source_sampling_step_ticks
    if step not in (1, 5) or current not in (1, 2):
        raise ValueError("target requires binary initial truth and an explicit one- or five-tick grid")
    ticks = np.asarray(ticks, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    if ticks.ndim != 1 or truth.shape != ticks.shape or (np.diff(ticks) <= 0).any():
        raise ValueError("truth observations require unique increasing ticks")
    if not np.isin(truth, [0, 1, 2, 3, 4]).all():
        raise ValueError("unknown truth code")
    masks = (candidate_present, entity_exit)
    if any(mask is not None and np.asarray(mask).shape != ticks.shape for mask in masks):
        raise ValueError("candidate/exit evidence must match observation ticks")
    candidates = None if candidate_present is None else np.asarray(candidate_present, dtype=bool)
    exits = None if entity_exit is None else np.asarray(entity_exit, dtype=bool)
    lookup = {int(tick): i for i, tick in enumerate(ticks)}
    event_tick, event_bin, last_offset, reason = None, -1, 0, ""
    for offset in range(step, 51, step):
        tick = anchor + offset
        if tick not in lookup:
            reason = "raw_tick_gap" if step == 1 else "observation_gap"
            break
        j = lookup[tick]
        value = int(truth[j])
        if exits is not None and exits[j]:
            reason = "entity_exit"
            break
        if candidates is not None and not candidates[j]:
            reason = "candidate_absent"
            break
        if value not in (1, 2):
            reason = {0: "missing_record", 3: "unknown", 4: "out_of_scope"}[value]
            break
        if value != current:
            event_tick, event_bin = tick, (offset-1)//5
            break
        last_offset = offset
    event = event_tick is not None
    observed = event_bin if event else last_offset//5
    return dict(label_view=LABEL_VIEW, event_observed=event, event_bin=event_bin,
        observed_bins=observed, censor_reason=reason, likelihood_valid=event or observed > 0,
        horizon_outcome_known=event or observed == 10,
        label=event_bin if event else (10 if observed == 10 else -1),
        first_switch_tick=event_tick, source_sampling_step_ticks=step,
        event_precision=f"first_opposite_binary_observed_on_{step}_tick_grid")


def labels(episode, *, source_sampling_step_ticks=5, candidate_present=None, entity_exit=None):
    """Labels on the supplied grid; sampled inputs remain the fixed 21-frame history."""
    if source_sampling_step_ticks != 5:
        raise ValueError("projected episodes use a five-tick observation grid")
    rows = []
    for split, anchor in split_anchors(episode["profile"]):
        index = anchor//5
        for q in np.flatnonzero(np.isin(episode["truth"][index], [1, 2])):
            current = int(episode["truth"][index, q])
            future = episode["truth"][index+1:index+11, q]
            target = first_opposite_target(current, anchor, anchor+5*np.arange(1, len(future)+1), future,
                source_sampling_step_ticks=5,
                candidate_present=None if candidate_present is None else candidate_present[index+1:index+11, q],
                entity_exit=None if entity_exit is None else entity_exit[index+1:index+11, q])
            pred, binding = episode["keys"][q]
            rows.append(dict(profile=episode["profile"], family=episode["profile"].split("_")[0],
                split=split, anchor=anchor, query=int(q), predicate=pred,
                bindings_json=json.dumps(dict(binding), sort_keys=True), current=current-1,
                source=episode["source"], **target))
    return rows


def observation_prefix_label_view(episodes, index, *, candidate_present=None, entity_exit=None):
    """Create an in-memory v2 view while preserving the immutable v1 row population.

    The saved projection distinguishes missing records, unknown and out-of-scope.
    It does not retain candidate-present or entity-exit evidence. Those causes are
    reported only when their explicit masks are supplied by the caller.
    """
    output = index.drop(columns=["valid", "invalid_reason"]).copy()
    output["legacy_complete_window_valid"] = index.valid.to_numpy(dtype=bool)
    targets = []
    for row in index.itertuples(index=False):
        episode = episodes[row.profile]
        j, q = int(row.anchor)//5, int(row.query)
        current = int(episode["truth"][j, q])
        if current not in (1, 2) or current-1 != row.current:
            raise ValueError(f"initial truth disagrees with immutable sample row {row.row_id}")
        future = episode["truth"][j+1:j+11, q]
        target = first_opposite_target(current, int(row.anchor),
            int(row.anchor)+5*np.arange(1, len(future)+1), future, source_sampling_step_ticks=5,
            candidate_present=None if candidate_present is None else candidate_present[row.profile][j+1:j+11, q],
            entity_exit=None if entity_exit is None else entity_exit[row.profile][j+1:j+11, q])
        targets.append(target)
    metadata = pd.DataFrame(targets, index=output.index)
    for column in metadata:
        output[column] = metadata[column]
    # An old node reference is usable only for exactly the same observed event.
    same_event = output.first_switch_tick.eq(index.first_switch_tick)
    output["switch_node"] = index.switch_node.where(same_event & output.event_observed, None)
    output["candidate_evidence_available"] = candidate_present is not None
    output["entity_exit_evidence_available"] = entity_exit is not None
    validate_targets(output)
    return output


def effective_sample_statistics(index):
    validate_targets(index)
    groups = []
    for split, group in index.groupby("split", sort=True):
        groups.append(dict(split=split, queries=len(group), events=int(group.event_observed.sum()),
            legacy_complete_window_queries=int(group.legacy_complete_window_valid.sum()),
            retained_events_before_later_censor=int((group.event_observed & ~group.legacy_complete_window_valid).sum()),
            likelihood_queries=int(group.likelihood_valid.sum()),
            horizon_known_queries=int(group.horizon_outcome_known.sum()),
            partial_censored=int((~group.event_observed & group.observed_bins.between(1, 9)).sum()),
            zero_bin_censored=int((~group.likelihood_valid).sum()),
            full_no_event=int((~group.event_observed & (group.observed_bins == 10)).sum()),
            observed_bin_counts=group.observed_bins.value_counts().sort_index().to_dict(),
            censor_reasons={reason: int(group.censor_reason.eq(reason).sum()) for reason in
                ("unknown", "out_of_scope", "missing_record", "candidate_absent", "entity_exit", "observation_gap", "raw_tick_gap")}))
    return dict(label_view=LABEL_VIEW, source_sampling_step_ticks=5, splits=groups,
        event_precision="First opposite binary observed on the five-tick assertion grid; between-grid flips are unidentifiable.",
        candidate_exit_limit="Legacy projection omits candidate and exit evidence; absent records do not establish either cause.",
        metric_selection=METRIC_SELECTION)


def _complete_window_trace(episode):
    """Preserve the original projection trace; v2 never consumes its validity mask."""
    rows = []
    for split, anchor in split_anchors(episode["profile"]):
        index = anchor//5
        for q in np.flatnonzero(np.isin(episode["truth"][index], [1, 2])):
            current = int(episode["truth"][index, q])
            future = episode["truth"][index+1:index+11, q]
            valid = bool(np.isin(future, [1, 2]).all())
            opposite = np.flatnonzero(future == 3-current)
            switch = int(opposite[0]+1) if len(opposite) else 0
            sequence = np.r_[current, future]
            binary = np.isin(sequence[:-1], [1, 2]) & np.isin(sequence[1:], [1, 2])
            flips = np.flatnonzero((sequence[:-1] != sequence[1:]) & binary)
            first_tick = anchor+switch*5 if switch else None
            pred, binding = episode["keys"][q]
            rows.append(dict(profile=episode["profile"], family=episode["profile"].split("_")[0],
                split=split, anchor=anchor, query=int(q), predicate=pred,
                bindings_json=json.dumps(dict(binding), sort_keys=True), current=current-1,
                valid=valid, label=(switch-1 if switch else 10) if valid else -1,
                observed_opposite=bool(len(opposite)), observed_adjacent_switches=len(flips),
                endpoint_same=bool(future[-1] == current), first_switch_tick=first_tick if valid else None,
                invalid_reason="" if valid else "nonbinary_or_absent_future_truth",
                source=episode["source"], anchor_node=episode["refs"][q][anchor],
                switch_node=episode["refs"][q][first_tick] if valid and switch else None))
    return rows


def prepare_projection():
    """Original source/vocabulary/budget preparation, restricted to a new run path."""
    if OUT.exists():
        raise FileExistsError(f"projection requires a new run directory: {OUT}")
    episodes, rows = {}, []
    for profile in PROFILES:
        episode = read_episode(profile)
        rows.extend(_complete_window_trace(episode))
        del episode["refs"]
        episodes[profile] = episode
        print(json.dumps(dict(profile=profile, entities=len(episode["names"]), predicates=len(episode["keys"]),
                              transition_checked=episode["transition_checked"])), flush=True)
    source_index = pd.DataFrame(rows)
    source_index.insert(0, "row_id", np.arange(len(source_index)))
    index = observation_prefix_label_view(episodes, source_index)
    train = index[(index.split == "train") & index.likelihood_valid]
    pred_names = sorted(train.predicate.unique())
    roles = sorted({r for p in PROFILES[:2] for rr in episodes[p]["roles"] for r in rr if r != "<none>"})
    vocab = dict(predicates=["<unseen>"]+pred_names, roles=["<none>"]+roles, categories=list(CATEGORIES))
    known = train[train.horizon_outcome_known]
    positive = known.groupby("predicate").event_observed.agg(["sum", "count"])
    primary = positive[(positive["sum"] >= 10) & (positive["count"]-positive["sum"] >= 10)].index.tolist()
    for episode in episodes.values():
        episode["predicate_type"] = np.array([vocab["predicates"].index(k[0]) if k[0] in pred_names else 0 for k in episode["keys"]])
        episode["role_type"] = np.array([[vocab["roles"].index(r) if r in vocab["roles"] else 0 for r in rr] for rr in episode["roles"]])
    physical = np.concatenate([e["physical"][:101][e["present"][:101]] for p, e in episodes.items() if p.startswith("L1-3")])
    mean, std = physical.mean(0), np.maximum(physical.std(0), 1.0)
    vocab["normalization"] = dict(mean=mean.tolist(), std=std.tolist(), source_ticks=[0, 500])
    config = dict(input_contract="ideal-graph-prefix-1", task="first_opposite_binary_observed_on_assertion_grid",
        label_view=LABEL_VIEW, source_sampling_step_ticks=5,
        history_ticks=100, sampling_step=5, horizon_ticks=50, bins=10, anchor_stride=25,
        train=dict(family="L1-3", profiles=list(PROFILES[:2]), anchors=[100, 500], exposure=[0, 550]),
        validation=dict(family="L1-3", profiles=list(PROFILES[:2]), anchors=[675, 850], exposure=[575, 900]),
        test=dict(family="L1-4", profiles=list(PROFILES[2:]), anchors=[100, 850], exposure=[0, 900]),
        source_seed="seed00", primary_predicates=primary, all_train_predicates=pred_names,
        loss="task-macro first-event or observed-prefix survival NLL via discrete hazard; no class reweighting",
        metric_selection=METRIC_SELECTION, models=["M0", "M1", "M2", "T", "G", "G_no_history"],
        width=24, learning_rates=[0.001, 0.003], epochs=35, patience=7, optimization_seeds=[0, 1, 2],
        weight_decay=0.0001, gradient_clip=5.0, reliability_tolerance=0.05, minimum_relative_nll_gain=0.02,
        decision="Engineering pilot only; G must beat M1, M2, T and boosted trees in primary macro NLL, without binary Brier regression; one test family cannot support a population significance claim.",
        calibration="Raw probabilities primary. No fitted calibration with this validation scope. ECE tolerance is descriptive, not a finite-sample or per-instance guarantee.",
        timestamp_policy="only fixed relative lags 100..0; no absolute ticks, IDs, profile or planned actions in inputs",
        source_paths=[str((GRAPH/(p+'__seed00')).relative_to(ROOT)) for p in PROFILES])
    dataset = GraphDataset(episodes, source_index, vocab)
    maxima = Counter()
    for sample in dataset.samples:
        x, _, _ = dataset.get(sample)
        for key, value in dict(entities=x["physical"].shape[1], predicates=x["truth"].shape[1],
                               role_edges=int(x["role_present"].sum()), queries=len(x["query"])).items():
            maxima[key] = max(maxima[key], value)
    n, p, e, t, d = maxima["entities"], maxima["predicates"], maxima["role_edges"], 21, config["width"]
    budget = dict(maximum_prefix_sizes=dict(maxima), frames=t, width=d,
        graph_two_layer_dense_matmul_flops_upper=2*t*(12*(n+p)*d*d+12*e*d*d),
        entity_attention_two_layer_flops_upper=2*t*(8*n*d*d+4*n*n*d),
        base_float32_node_activation_bytes=4*t*(n+p)*d, entity_attention_matrix_float32_bytes=4*t*n*n,
        allocation_budget_bytes=8*1024**3,
        assumptions="Per-graph batch, float32, two spatial layers, width 24, 21 sampled snapshots; bounds cover stated terms, not the complete framework/workspace allocation. GPU memory measured during execution; runtime cannot be inferred from arithmetic alone.")
    OUT.mkdir(parents=True, exist_ok=False)
    source_index.to_parquet(OUT/"samples.parquet", index=False)
    index.to_parquet(OUT/LABEL_VIEW_FILE, index=False)
    torch.save(episodes, OUT/"episodes.pt")
    write_json(OUT/"vocabulary.json", vocab)
    write_json(OUT/"config.json", config)
    write_json(OUT/"resource_budget.json", budget)
    support = index.groupby(["split", "profile", "predicate"], sort=True).agg(
        queries=("likelihood_valid", "size"), likelihood_queries=("likelihood_valid", "sum"),
        horizon_known_queries=("horizon_outcome_known", "sum"), events=("event_observed", "sum"))
    support.reset_index().to_csv(OUT/"support.csv", index=False)
    report = effective_sample_statistics(index)
    report.update(primary_predicates=primary, total_train_predicates=len(pred_names), budget=budget,
        source_files=[dict(path=str((GRAPH/(p+'__seed00')/'nodes.parquet').relative_to(ROOT)),
            bytes=(GRAPH/(p+'__seed00')/'nodes.parquet').stat().st_size,
            mtime_ns=(GRAPH/(p+'__seed00')/'nodes.parquet').stat().st_mtime_ns) for p in PROFILES])
    write_json(OUT/"data_summary.json", report)
    print(json.dumps(report), flush=True)


def prepare():
    """Persist only a new versioned label view; never rewrite the frozen projection."""
    path = OUT / LABEL_VIEW_FILE
    if path.exists():
        raise FileExistsError(f"label view already exists: {path}")
    dataset = GraphDataset(label_view=LABEL_VIEW)
    dataset.index.to_parquet(path, index=False)
    print(json.dumps(effective_sample_statistics(dataset.index)), flush=True)


class GraphDataset:
    def __init__(self, episodes=None, index=None, vocab=None, *, label_view=LABEL_VIEW,
                 candidate_present=None, entity_exit=None):
        self.episodes = torch.load(OUT / "episodes.pt", map_location="cpu", weights_only=False) if episodes is None else episodes
        source_index = pd.read_parquet(OUT / "samples.parquet") if index is None else index
        if label_view == LABEL_VIEW:
            self.index = observation_prefix_label_view(self.episodes, source_index,
                candidate_present=candidate_present, entity_exit=entity_exit)
        elif label_view == LEGACY_LABEL_VIEW:
            self.index = source_index.copy()
        else:
            raise ValueError(f"unrecognized explicit label view: {label_view}")
        self.label_view = label_view
        self.vocab = json.loads((OUT / "vocabulary.json").read_text()) if vocab is None else vocab
        self.samples = list(self.index.groupby(["profile", "anchor"], sort=True).indices)
        self.rows = self.index.groupby(["profile", "anchor"], sort=True).indices

    def get(self, sample, device="cpu"):
        profile, anchor = sample
        e = self.episodes[profile]
        end = anchor // 5
        span = slice(end-20, end+1)
        raw_truth = e["truth"][span]
        # Out-of-scope inventory rows never create nodes. Past binary/unknown records do.
        p = np.flatnonzero(np.isin(raw_truth, [1, 2, 3]).any(0))
        pointers = e["pointers"][p]
        role_present = e["role_present"][p]
        known_entities = e["present"][span].any(0)
        known_entities[pointers[role_present]] = True
        ent = np.flatnonzero(known_entities)
        ent_map = np.full(len(e["names"]), -1, np.int64)
        ent_map[ent] = np.arange(len(ent))
        remap = ent_map[pointers]
        remap[~role_present] = 0
        assert (remap[role_present] >= 0).all()
        frame = e["physical"][span][:, ent]
        present = e["present"][span][:, ent]
        norm = self.vocab["normalization"]
        physical = ((frame - np.array(norm["mean"], np.float32)) / np.array(norm["std"], np.float32)) * present[..., None]
        truth = raw_truth[:, p]
        # Entity presence at each time is current state OR a current in-scope predicate binding.
        visible = present.copy()
        for r in range(3):
            for j in range(21):
                active = np.isin(truth[j], [1, 2, 3]) & role_present[:, r]
                visible[j, remap[active, r]] = True
        rows = self.index.iloc[self.rows[sample]]
        pm = {int(old): new for new, old in enumerate(p)}
        q = np.array([pm[int(old)] for old in rows["query"]], np.int64)
        # No target, source identifier, profile, anchor or absolute timestamp is in x.
        x = dict(physical=physical, physical_present=present, entity_visible=visible,
            entity_type=np.where(present.any(0), e["category"][ent], 0), truth=truth, predicate_type=e["predicate_type"][p],
            pointers=remap, role_type=e["role_type"][p], role_present=role_present, query=q,
            relative_age=np.linspace(1, 0, 21, dtype=np.float32))
        x = {k: torch.as_tensor(v, device=device) for k, v in x.items()}
        assert x["physical"].shape[0] == 21 and x["truth"].shape[0] == 21
        assert torch.isfinite(x["physical"]).all()
        y = torch.as_tensor(rows.label.to_numpy(), dtype=torch.long, device=device)
        return x, y, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["labels", "projection", "statistics"], nargs="?", default="statistics")
    args = parser.parse_args()
    if args.action == "projection":
        prepare_projection()
    elif args.action == "labels":
        prepare()
    else:
        print(json.dumps(effective_sample_statistics(GraphDataset().index)), flush=True)
