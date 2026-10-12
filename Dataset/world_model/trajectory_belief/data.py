"""Immutable formal source adapter and family-component split for pair windows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from Dataset.world_model.transition_prediction.data import (
    LABEL_VIEW, TARGET_COLUMNS, TRUTH, first_opposite_target,
)
from .contracts import WindowDataset, HISTORY_STEPS, FUTURE_STEPS, STEP_TICKS
from .rules import evaluate_pair

ROOT = Path(__file__).resolve().parents[3]
PAIR_PREDICATE = "safety.aircraft_pair_below_separation"
PAIR_FAMILY = "predicate_contract_aircraft_pair_geometry"
GRAPH_RELATIVE = Path("Dataset/world_model/runs/belief_label_graph/formal")
SOURCE_RELATIVE = Path("aw_data/objective_semantic_truth")


class LabelContractMismatch(ValueError):
    """Frozen assertions disagree with geometry under prefix-known rule inputs."""


def family_of(profile):
    """Only remove a terminal variant suffix; '_violation' is part of a family."""
    return re.sub(r"_v\d+$", "", profile)


def select_budget_episodes(root=ROOT):
    """First lexicographic variant of each formal family, seed00 only."""
    graph = Path(root) / GRAPH_RELATIVE
    grouped = defaultdict(list)
    for directory in sorted(graph.iterdir()):
        if not directory.is_dir():
            continue
        profile, seed = directory.name.rsplit("__", 1)
        grouped[family_of(profile)].append((profile, seed, directory.name))
    selected, excluded = [], []
    for family in sorted(grouped):
        eligible = sorted(item for item in grouped[family] if item[1] == "seed00")
        if not eligible:
            raise ValueError(f"family has no seed00 formal episode: {family}")
        selected.append(eligible[0][2])
        for _, seed, episode in grouped[family]:
            if episode != eligible[0][2]:
                excluded.append(dict(episode=episode, family=family,
                                     reason="budget_other_seed" if seed != "seed00" else "budget_other_variant"))
    return selected, excluded


def _canonical_prefix(history, parameters):
    # Group at head-input float32 precision while preserving raw float64 arrays
    # for exact rule geometry. These are raw values, not a digest.
    normal = np.array(history, dtype="<f4", copy=True)
    normal[normal == 0] = 0
    reverse = normal[:, ::-1, :].copy()
    params = np.array(parameters, dtype="<f4", copy=True)
    params[params == 0] = 0
    if not np.isfinite(normal).all() or not np.isfinite(params).all():
        raise ValueError("head-precision prefix and rule inputs must be finite")
    return min(normal.tobytes(), reverse.tobytes()) + params.tobytes()


def assign_component_splits(history, parameters, rows, *, seed=20261002,
                            component_counts=None):
    """Keep families and exact shared prefixes together before any fitting.

    Components are sorted by their family names, then shuffled once with the
    declared seed. No target, future or event count participates in assignment.
    """
    rows = rows.reset_index(drop=True).copy()
    families = sorted(rows.family.unique())
    parent = {family: family for family in families}

    def find(family):
        while parent[family] != family:
            parent[family] = parent[parent[family]]
            family = parent[family]
        return family

    keys = defaultdict(list)
    for i, (prefix, rule) in enumerate(zip(history, parameters)):
        keys[_canonical_prefix(prefix, rule)].append(i)
    duplicates = [indices for indices in keys.values() if len(indices) > 1]
    for indices in duplicates:
        members = sorted(set(rows.family.iloc[indices]))
        for family in members[1:]:
            left, right = find(members[0]), find(family)
            parent[max(left, right)] = min(left, right)
    groups = defaultdict(list)
    for family in families:
        groups[find(family)].append(family)
    components = sorted(sorted(group) for group in groups.values())
    if component_counts is None:
        count = len(components)
        if count < 3:
            raise ValueError(f"at least three independent components required, observed {count}: {components}")
        train = min(count - 2, max(1, round(0.65 * count)))
        validation = max(1, (count - train) // 2)
        component_counts = (train, validation, count - train - validation)
    if sum(component_counts) != len(components):
        raise ValueError(f"component split requires {sum(component_counts)} components, observed {len(components)}; "
                         f"components={components}")
    permutation = np.random.default_rng(seed).permutation(len(components))
    split_for_component = {}
    start = 0
    for split, count in zip(("train", "validation", "test"), component_counts):
        for component in permutation[start:start + count]:
            split_for_component[int(component)] = split
        start += count
    component_for_family = {family: i for i, component in enumerate(components) for family in component}
    rows["family_component"] = rows.family.map(component_for_family).astype(np.int64)
    rows["split_group"] = rows.family_component
    rows["split"] = rows.family_component.map(split_for_component)
    rows["duplicate_prefix_group"] = -1
    for group, indices in enumerate(duplicates):
        rows.loc[indices, "duplicate_prefix_group"] = group
    # Direct equality of raw keys verifies that every exact shared prefix stays
    # in one split; labels and future values do not enter this check.
    for indices in duplicates:
        if rows.split.iloc[indices].nunique() != 1:
            raise ValueError("exact prefix leakage across splits")
    report = dict(split_seed=int(seed), split_unit="family_connected_by_exact_prefix_and_static_rule",
                  prefix_comparison_dtype="float32", raw_physical_array_dtype="float64",
                  component_allocation="round(0.65*G)_train_remaining_half_validation_and_test_minimum_one_each",
                  component_counts=dict(zip(("train", "validation", "test"), map(int, component_counts))),
                  components=[dict(component=i, families=component, split=split_for_component[i])
                              for i, component in enumerate(components)],
                  exact_duplicate_prefix_groups=len(duplicates),
                  exact_duplicate_prefix_rows=sum(map(len, duplicates)),
                  split_family_counts={split: int(rows.loc[rows.split.eq(split), "family"].nunique())
                                       for split in ("train", "validation", "test")})
    return rows, report


def mark_full_copies(history, future, present, parameters, rows):
    """Retain query windows and mark exact copies for unique training donors.

    Pair reversal canonicalizes history and future together. Identical histories
    with different futures remain distinct alternatives inside one component.
    """
    rows = rows.copy()
    copies = defaultdict(list)
    for i in range(len(rows)):
        h = np.array(history[i], dtype="<f8", copy=True)
        f = np.nan_to_num(future[i], nan=0.0).astype("<f8")
        h[h == 0] = 0
        f[f == 0] = 0
        mask = present[i]
        direct = h.tobytes() + f.tobytes() + mask.tobytes()
        reversed_pair = h[:, ::-1].copy().tobytes() + f[:, ::-1].copy().tobytes() + mask[:, ::-1].copy().tobytes()
        target = tuple(rows.iloc[i][["current", *TARGET_COLUMNS]].tolist())
        key = (int(rows.family_component.iloc[i]), min(direct, reversed_pair),
               np.asarray(parameters[i], dtype="<f8").tobytes(), target)
        copies[key].append(i)
    rows["full_copy_group"] = -1
    rows["full_copy_representative"] = False
    for group, indices in enumerate(copies.values()):
        rows.loc[indices, "full_copy_group"] = group
        rows.loc[indices[0], "full_copy_representative"] = True
    duplicate_groups = [group for group in copies.values() if len(group) > 1]
    return rows, dict(full_copy_policy="retain_query_population_unique_complete_donors",
                      exact_full_copy_groups=len(duplicate_groups),
                      exact_full_copy_rows=sum(map(len, duplicate_groups)),
                      unique_full_copy_windows=len(copies),
                      redundant_full_copy_windows=sum(len(group)-1 for group in duplicate_groups))


def _read_episode(root, episode):
    graph = root / GRAPH_RELATIVE / episode
    scope = pq.read_table(graph / "scope.parquet",
                          columns=["source_kind", "episode_id", "profile", "seed", "start_tick", "end_tick"]).to_pylist()
    if len(scope) != 1 or (scope[0]["source_kind"], scope[0]["start_tick"], scope[0]["end_tick"]) != ("formal", 0, 900):
        raise ValueError(f"formal 0..900 episode required, ARM excluded: {episode}")
    if scope[0]["episode_id"] != episode:
        raise ValueError(f"scope/directory episode mismatch: {episode}")
    states, state_nodes = defaultdict(dict), defaultdict(dict)
    selected = pq.read_table(graph / "nodes.parquet", columns=["tick", "node_id", "attrs_json"],
                             filters=[("node_type", "=", "EntityState"), ("category_id", "=", "uav")])
    for row in selected.to_pylist():
        tick = int(row["tick"])
        if tick % STEP_TICKS:
            continue
        attrs = json.loads(row["attrs_json"])
        name = attrs["entity_id"]
        pose = attrs["raw"]["truth_pose"]
        physical = np.asarray(pose["position_enu_m"] + pose["velocity_enu_mps"], dtype=np.float64)
        if physical.shape != (6,) or not np.isfinite(physical).all():
            raise ValueError(f"malformed physical state: {episode}:{name}:{tick}")
        if tick in states[name]:
            raise ValueError(f"duplicate physical state: {episode}:{name}:{tick}")
        states[name][tick] = physical
        state_nodes[name][tick] = row["node_id"]
    streams = defaultdict(dict)
    selected = pq.read_table(graph / "nodes.parquet", columns=["tick", "node_id", "attrs_json"],
                             filters=[("node_type", "=", "PredicateAssertion"), ("category_id", "=", PAIR_PREDICATE)])
    for row in selected.to_pylist():
        tick = int(row["tick"])
        if tick % STEP_TICKS:
            continue
        attrs = json.loads(row["attrs_json"])
        if attrs.get("assertion_layer") is not None:
            continue
        binding = attrs["bindings"]
        if set(binding) != {"first_aircraft", "second_aircraft"}:
            raise ValueError(f"pair binding contract mismatch: {episode}:{binding}")
        pair = (binding["first_aircraft"], binding["second_aircraft"])
        if pair[0] == pair[1] or tick in streams[pair]:
            raise ValueError(f"malformed or duplicate pair assertion: {episode}:{pair}:{tick}")
        streams[pair][tick] = (TRUTH[attrs["truth_status"]], row["node_id"])
    source = root / SOURCE_RELATIVE / episode
    margin = float(json.loads((source / "world_truth_summary.json").read_text())["governed_parameters"]["separation_margin_m"])
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError(f"invalid governed separation margin: {episode}")
    parameters = defaultdict(list)
    with (source / "l0_predicate_state.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            if PAIR_FAMILY not in line:
                continue
            record = json.loads(line)
            if record["observation_family"] != PAIR_FAMILY:
                continue
            values = record["values"]
            pair = (values["first_aircraft_id"], values["second_aircraft_id"])
            metric = values["pair_distance_metric"]
            if metric == "euclidean_3d":
                rule = np.array([0, margin, margin, margin], dtype=np.float64)
            elif metric == "anisotropic_safety_volume_equivalent_distance":
                rule = np.array([1, values["pair_horizontal_limit_m"], values["pair_vertical_limit_m"], margin], dtype=np.float64)
            else:
                raise ValueError(f"unsupported producer separation metric: {episode}:{metric}")
            if not np.isfinite(rule).all() or (rule[1:] <= 0).any():
                raise ValueError(f"invalid pair rule parameters: {episode}:{pair}:{record['tick']}")
            parameters[pair].append((int(record["tick"]), rule, line_number, record["observation_id"]))
    for pair in parameters:
        parameters[pair].sort(key=lambda item: item[0])
        if len({item[0] for item in parameters[pair]}) != len(parameters[pair]):
            raise ValueError(f"duplicate pair parameter observation tick: {episode}:{pair}")
    return scope[0], states, state_nodes, streams, parameters


def extract_episode(root, episode, *, anchors=range(100, 851, 25)):
    """Extract complete 21-step prefixes; future observation absence censors labels."""
    root = Path(root)
    scope, states, state_nodes, streams, parameters = _read_episode(root, episode)
    history, future, present, rules, records = [], [], [], [], []
    exclusions, checked, mismatches = Counter(), 0, []
    graph_path = str(GRAPH_RELATIVE / episode / "nodes.parquet")
    for pair in sorted(streams):
        stream = streams[pair]
        for anchor in anchors:
            if anchor not in stream or stream[anchor][0] not in (1, 2):
                exclusions["initial_truth_nonbinary_or_absent"] += 1
                continue
            history_ticks = anchor + STEP_TICKS * np.arange(-(HISTORY_STEPS - 1), 1)
            if any(int(tick) not in states[name] for name in pair for tick in history_ticks):
                exclusions["history_state_incomplete"] += 1
                continue
            candidates = [item for item in parameters[pair] if item[0] <= anchor]
            if not candidates:
                exclusions["static_rule_not_observed_by_cutoff"] += 1
                continue
            parameter_tick, rule, parameter_line, parameter_id = candidates[-1]
            h = np.stack([[states[name][int(tick)] for name in pair] for tick in history_ticks])
            future_ticks = anchor + STEP_TICKS * np.arange(1, FUTURE_STEPS + 1)
            f = np.full((FUTURE_STEPS, 2, 6), np.nan, dtype=np.float64)
            mask = np.zeros((FUTURE_STEPS, 2), dtype=bool)
            for j, tick in enumerate(future_ticks):
                for entity, name in enumerate(pair):
                    if int(tick) in states[name]:
                        f[j, entity] = states[name][int(tick)]
                        mask[j, entity] = True
            truth = np.array([stream[int(tick)][0] if int(tick) in stream else 0 for tick in future_ticks])
            current = stream[anchor][0] - 1
            rule_current = int(evaluate_pair(h[-1], rule))
            checked += 1
            if rule_current != current:
                mismatches.append(dict(episode=episode, pair=list(pair), tick=anchor,
                                       graph_truth=current, rule_truth=rule_current, check="initial"))
            for j in np.flatnonzero(mask.all(axis=1) & np.isin(truth, [1, 2])):
                expected = int(evaluate_pair(f[j], rule))
                checked += 1
                if expected != truth[j] - 1:
                    mismatches.append(dict(episode=episode, pair=list(pair), tick=int(future_ticks[j]),
                                           graph_truth=int(truth[j]-1), rule_truth=expected, check="future_label"))
            target = first_opposite_target(current + 1, anchor, future_ticks, truth, source_sampling_step_ticks=STEP_TICKS)
            binding = dict(first_aircraft=pair[0], second_aircraft=pair[1])
            switch = target["first_switch_tick"]
            records.append(dict(row_id=f"{episode}:{anchor}:{pair[0]}|{pair[1]}",
                                family=family_of(scope["profile"]), profile=scope["profile"], episode=episode,
                                seed=scope["seed"], cutoff_tick=int(anchor), anchor=int(anchor),
                                predicate=PAIR_PREDICATE, pair_json=json.dumps(binding, sort_keys=True),
                                first_aircraft_id=pair[0], second_aircraft_id=pair[1], current=int(current),
                                state_complete=bool(mask.all()), history_start_tick=int(history_ticks[0]),
                                source_kind="formal", graph_source=graph_path,
                                prefix_source=f"{graph_path}#EntityState;pair={pair[0]}|{pair[1]};ticks={history_ticks[0]}..{anchor}/5",
                                initial_assertion=stream[anchor][1],
                                switch_assertion=None if switch is None else stream[switch][1],
                                rule_parameter_tick=parameter_tick,
                                rule_source=f"{SOURCE_RELATIVE / episode / 'l0_predicate_state.jsonl'}#line={parameter_line}&observation_id={parameter_id}",
                                margin_source=f"{SOURCE_RELATIVE / episode / 'world_truth_summary.json'}#governed_parameters.separation_margin_m",
                                candidate_evidence_available=False, entity_exit_evidence_available=False,
                                **target))
            history.append(h)
            future.append(f)
            present.append(mask)
            rules.append(rule)
    if mismatches:
        raise LabelContractMismatch(json.dumps(dict(label_view=LABEL_VIEW, episode=episode,
            mismatch_count=len(mismatches), examples=mismatches[:12],
            action="preserve_frozen_sources_and_reconcile_experimental_label_authority"), sort_keys=True))
    n = len(records)
    return (np.asarray(history).reshape(n, HISTORY_STEPS, 2, 6),
            np.asarray(future).reshape(n, FUTURE_STEPS, 2, 6),
            np.asarray(present, dtype=bool).reshape(n, FUTURE_STEPS, 2),
            np.asarray(rules).reshape(n, 4), pd.DataFrame(records),
            dict(episode=episode, windows=n, exclusions=dict(exclusions),
                 rule_assertion_comparisons=checked, rule_assertion_mismatches=0))


def build_dataset(root=ROOT, *, split_seed=20261002, selected_episodes=None,
                  component_counts=None):
    root = Path(root)
    budget, excluded = select_budget_episodes(root)
    selected = budget if selected_episodes is None else list(selected_episodes)
    if any(episode not in budget for episode in selected):
        raise ValueError("this experiment requires declared first-variant seed00 formal budget episodes")
    extracted = [extract_episode(root, episode) for episode in selected]
    if not extracted or not sum(len(item[4]) for item in extracted):
        raise ValueError("selected formal cohort has no eligible windows")
    history, future, present, parameters = [np.concatenate([item[j] for item in extracted]) for j in range(4)]
    rows = pd.concat([item[4] for item in extracted], ignore_index=True)
    rows, split_report = assign_component_splits(history, parameters, rows,
                                                 seed=split_seed, component_counts=component_counts)
    rows, copy_report = mark_full_copies(history, future, present, parameters, rows)
    report = dict(selected_episodes=selected, budget_restriction="first_lexicographic_variant_per_family_seed00",
                  excluded_budget_episodes=excluded, source_authority="formal_graph_only_ARM_excluded",
                  physical_input_authority="EntityState.raw.truth_pose.position_enu_m_and_velocity_enu_mps_only",
                  static_rule_authority="latest_pair_geometry_observation_at_or_before_cutoff_plus_governed_margin",
                  label_authority="immutable_formal_assertions_observation_prefix_censor_v2",
                  pair_rule_assertion_agreement="checked_initial_and_observed_complete_future_states",
                  episodes=[item[5] for item in extracted], **split_report, **copy_report)
    report["split_counts"] = {split: dict(windows=int(len(group)), events=int(group.event_observed.sum()),
                                         complete_none=int((group.label == 10).sum()),
                                         partial_censored=int((~group.horizon_outcome_known).sum()),
                                         complete_state_donors=int(group.state_complete.sum()),
                                         unique_complete_state_donors=int((group.state_complete & group.full_copy_representative).sum()))
                              for split, group in rows.groupby("split", sort=True)}
    return WindowDataset(history, future, present, rows, parameters, report)


def load_window_dataset(directory):
    return WindowDataset.load(directory)


def support_component_counts(dataset):
    """Count non-test support before tuning, collapsing copied physical events.

    Physical identity is episode, unordered aircraft pair, predicate and observed
    switch tick. Multiple anchors of one event count once. Event identities from
    identical full-copy windows are unioned, even across episode/pair IDs. Test
    labels do not participate in this development-support audit.
    """
    rows = dataset.rows
    required = {"family_component", "split", "full_copy_group", "episode", "predicate",
                "first_aircraft_id", "second_aircraft_id", "first_switch_tick", "event_observed"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"support audit metadata missing: {sorted(missing)}")
    if not rows.split.isin(["train", "validation", "test"]).all():
        raise ValueError("support audit requires train/validation/test splits")
    if rows.groupby("family_component").split.nunique().gt(1).any():
        raise ValueError("a complete family component must belong to one split")
    if rows.groupby("full_copy_group").family_component.nunique().gt(1).any():
        raise ValueError("full-copy group crosses family components")
    development = rows.loc[rows.split.ne("test")]
    events = development.loc[development.event_observed]
    parent, copy_events, identities_for_component = {}, {}, defaultdict(set)

    def find(identity):
        if identity not in parent:
            parent[identity] = identity
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    for row in events.itertuples(index=False):
        tick = row.first_switch_tick
        if pd.isna(tick) or not np.isfinite(tick) or int(tick) != tick:
            raise ValueError(f"physical event requires an observed integer switch tick: {row.row_id}")
        identity = (row.episode, tuple(sorted((row.first_aircraft_id, row.second_aircraft_id))),
                    row.predicate, int(tick))
        find(identity)
        identities_for_component[int(row.family_component)].add(identity)
        copy_group = int(row.full_copy_group)
        if copy_group in copy_events:
            left, right = find(identity), find(copy_events[copy_group])
            parent[max(left, right)] = min(left, right)
        else:
            copy_events[copy_group] = identity
    components = []
    for component, group in development.groupby("family_component", sort=True):
        identities = identities_for_component[int(component)]
        components.append(dict(component=int(component), original_split=group.split.iloc[0],
                               windows=int(len(group)), event_windows=int(group.event_observed.sum()),
                               unique_physical_event_keys=len(identities),
                               copy_collapsed_physical_events=len({find(identity) for identity in identities}),
                               families=sorted(group.family.unique().tolist())))
    return dict(audit_scope="train_and_validation_only_before_tuning",
                physical_event_identity=["episode", "canonical_aircraft_pair", "predicate", "first_switch_tick"],
                event_copy_collapse="union_physical_event_keys_connected_by_exact_full_copy_window_group",
                development_windows=int(len(development)),
                development_event_windows=int(len(events)),
                development_unique_physical_event_keys=len(parent),
                development_copy_collapsed_physical_events=len({find(identity) for identity in parent}),
                locked_test_windows=int(rows.split.eq("test").sum()), components=components)


def _split_counts(rows):
    return {split: dict(windows=int(len(group)), events=int(group.event_observed.sum()),
                        complete_none=int((group.label == 10).sum()),
                        partial_censored=int((~group.horizon_outcome_known).sum()),
                        complete_state_donors=int(group.state_complete.sum()),
                        unique_complete_state_donors=int((group.state_complete & group.full_copy_representative).sum()))
            for split, group in rows.groupby("split", sort=True)}


def reassign_support_split(dataset, *, source_dataset=None, validation_components=5,
                           validation_event_components=4, max_event_component_share=Fraction(1, 2)):
    """Apply the authorized pretraining support rule to whole non-test components.

    Exactly four event-bearing components plus one zero-event component enter
    validation. Minimize squared deviation from one quarter of development
    windows and copied-event-collapsed physical events. Resolve ties by lexical
    component IDs. No model prediction, fitted parameter or test label enters
    selection. Original arrays, row order and all test rows remain exact.
    """
    if validation_components != 5 or validation_event_components != 4 or max_event_component_share != Fraction(1, 2):
        raise ValueError("support-v2 requires the declared five-component/four-event-bearing/50%-share rule")
    audit = support_component_counts(dataset)
    components = audit["components"]
    if len(components) <= validation_components:
        raise ValueError("support-v2 must retain at least one complete development component for training")
    windows = audit["development_windows"]
    events = audit["development_copy_collapsed_physical_events"]
    if windows == 0 or events == 0:
        raise ValueError("support-v2 requires positive development windows and physical events")
    eventful = [item for item in components if item["copy_collapsed_physical_events"] > 0]
    event_free = [item for item in components if item["copy_collapsed_physical_events"] == 0]
    if len(eventful) < 4 or not event_free:
        raise ValueError("support-v2 requires at least four event-bearing and one event-free development component")
    best, eligible = None, 0
    for event_group in combinations(eventful, 4):
        event_count = sum(item["copy_collapsed_physical_events"] for item in event_group)
        if max(item["copy_collapsed_physical_events"] for item in event_group) * 2 > event_count:
            continue
        for zero_group in event_free:
            candidate = (*event_group, zero_group)
            window_count = sum(item["windows"] for item in candidate)
            objective = (Fraction((4 * window_count - windows)**2, windows**2)
                         + Fraction((4 * event_count - events)**2, events**2))
            ids = tuple(sorted((item["component"] for item in candidate), key=lambda value: str(value)))
            tie = tuple(map(str, ids))
            rank = (objective, tie)
            eligible += 1
            if best is None or rank < best[0]:
                best = (rank, ids, window_count, event_count)
    if best is None:
        raise ValueError("no whole-component validation set meets the declared support rule")
    _, selected, selected_windows, selected_events = best
    selected = sorted(selected)
    rows = dataset.rows.copy()
    original_test = rows.split.eq("test").to_numpy()
    original_mapping = {int(component): group.split.iloc[0]
                        for component, group in rows.groupby("family_component", sort=True)}
    development = ~original_test
    rows.loc[development, "split"] = np.where(rows.loc[development, "family_component"].isin(selected),
                                               "validation", "train")
    if not rows.loc[original_test].equals(dataset.rows.loc[original_test]):
        raise ValueError("support reassignment changed locked test metadata")
    if not np.array_equal(rows.split.eq("test").to_numpy(), original_test):
        raise ValueError("support reassignment changed locked test row IDs")
    new_mapping = {int(component): group.split.iloc[0]
                   for component, group in rows.groupby("family_component", sort=True)}
    report = deepcopy(dataset.report)
    report["support_split_revision"] = dict(
        version="support-stratified-development-v2", source_dataset=None if source_dataset is None else str(source_dataset),
        reason="pretraining_support_audit_found_validation_events_concentrated_in_too_few_components",
        authorization_scope="whole_non_test_components_reassigned_before_model_tuning_test_locked",
        rule=dict(validation_components=5, event_bearing_components=4, zero_event_components=1,
                  maximum_single_component_event_share=0.5,
                  objective="((4*Wval-Wdev)/Wdev)^2+((4*Eval-Edev)/Edev)^2",
                  tie_break="lexicographically_sorted_component_ID_strings"),
        audit=audit, eligible_assignments=eligible, validation_component_ids=selected,
        validation_windows=int(selected_windows), validation_copy_collapsed_physical_events=int(selected_events),
        objective=float(best[0][0]), objective_numerator=int(best[0][0].numerator),
        objective_denominator=int(best[0][0].denominator),
        old_mapping=[dict(component=component, split=split) for component, split in original_mapping.items()],
        new_mapping=[dict(component=component, split=split) for component, split in new_mapping.items()],
        changed_components=[component for component in original_mapping
                            if original_mapping[component] != new_mapping[component]],
        locked_test_windows=int(original_test.sum()), locked_test_row_ids_unchanged=True,
        raw_arrays_unchanged=True, row_population_unchanged=True)
    if "components" in report:
        for component in report["components"]:
            component["split"] = new_mapping[component["component"]]
    report["component_allocation"] = "pretraining_support_stratified_development_v2_test_mapping_preserved"
    report["component_counts"] = rows.groupby("split").family_component.nunique().astype(int).to_dict()
    report["split_family_counts"] = rows.groupby("split").family.nunique().astype(int).to_dict()
    report["split_counts"] = _split_counts(rows)
    reassigned = WindowDataset(dataset.history, dataset.future, dataset.future_present,
                               rows, dataset.rule_parameters, report)
    for name in ("history", "future", "future_present", "rule_parameters"):
        if not np.array_equal(getattr(reassigned, name)[original_test], getattr(dataset, name)[original_test], equal_nan=True):
            raise ValueError(f"support reassignment changed locked test {name}")
    return reassigned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("extract", "support-counts", "support-split"), default="extract")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent.parent / "runs/trajectory_belief/inputs", help="existing WindowDataset directory for support actions")
    parser.add_argument("--output", type=Path, help="explicit new dataset directory; omission is read-only")
    args = parser.parse_args()
    if args.action == "extract":
        if args.input is not None:
            parser.error("extract reads immutable sources and does not accept --input")
        dataset = build_dataset()
    else:
        if args.input is None:
            parser.error("support actions require --input")
        dataset = WindowDataset.load(args.input)
        if args.action == "support-counts":
            if args.output is not None:
                parser.error("support-counts is read-only and does not accept --output")
            print(json.dumps(support_component_counts(dataset), indent=2, allow_nan=False))
            return
        dataset = reassign_support_split(dataset, source_dataset=args.input)
    if args.output is not None:
        dataset.save(args.output)
    print(json.dumps(dataset.report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
