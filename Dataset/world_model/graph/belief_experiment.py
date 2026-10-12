"""Finite grouped prediction experiments using the isolated observation input."""

from __future__ import annotations

import argparse
import json
import re
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from Dataset.world_model.graph.contract import REPO, CONTRACT_VERSION, iter_jsonl
from Dataset.world_model.graph.records import DEFAULT_ROOT, load_scope
from Dataset.world_model.graph.prediction import isolate_records, observation_nodes, STATE_FEATURES, INPUT_VERSION
from Dataset.world_model.graph.belief_dataset import source_split
from Dataset.world_model.graph.event_tasks import HORIZON_TICKS
from Dataset.semantic_truth.formal_clock import FORMAL_TICK_HZ

OUTPUT = REPO / "Dataset/world_model/runs/belief_experiment"
TASKS = ("safety.aircraft_pair_conflict_event", "safety.aircraft_structure_near_miss_event",
         "l4.intersection_braking_observed", "l4.uav_vehicle_contact_observed")
PROFILE_HOLDOUT = ("L4-3_v2", "L4-4_v2")


def features(view: dict, query: dict) -> dict[str, float]:
    """Only observed geometry, backward motion and cohort membership.

    Sample names, source paths, native IDs, predicates, results and graph labels
    never become columns. The query supplies roles and an opaque subject index.
    """
    p = view["input"]
    by_entity = defaultdict(list)
    for row in p["states"]:
        by_entity[row["entity_id"]].append(row)
    x, latest = {}, {}
    aliases = {entity: alias for alias, entity in view["trace"]["entity_references"].items()}
    for role, entity in sorted(query["bindings"].items()):
        alias = aliases[entity]
        members = p["observed_cohorts"].get(alias, [alias])
        rows = [r for member in members for r in by_entity[member]]
        if not rows:
            raise ValueError("experiment subject has no observed state in its prefix")
        last_tick = max(r["tick"] for r in rows)
        end_rows = [r for r in rows if r["tick"] == last_tick]
        for i, name in enumerate(STATE_FEATURES):
            values = [r["values"][i] for r in end_rows if r["values"][i] is not None]
            x[f"current.{role}.{name}.present"] = float(bool(values))
            if values:
                x[f"current.{role}.{name}"] = float(np.mean(values))
        positions = np.array([[np.nan if v is None else v for v in r["values"][:3]] for r in end_rows])
        if np.isfinite(positions).all():
            latest[role] = positions.mean(axis=0)
        for i, name in enumerate(STATE_FEATURES):
            values = [r["values"][i] for r in rows if r["values"][i] is not None]
            if values:
                x[f"history.{role}.{name}.mean"] = float(np.mean(values))
                x[f"history.{role}.{name}.range"] = float(max(values) - min(values))
        x[f"current.{role}.observation_age_ticks"] = p["cutoff_tick"] - last_tick
        x[f"current.{role}.observed_member_count"] = len(members)
    roles = sorted(latest)
    for i, first in enumerate(roles):
        for second in roles[i + 1:]:
            delta = latest[first] - latest[second]
            x[f"current.pair.{first}.{second}.distance_m"] = float(np.linalg.norm(delta))
            x[f"current.pair.{first}.{second}.horizontal_m"] = float(np.linalg.norm(delta[:2]))
            x[f"current.pair.{first}.{second}.vertical_m"] = float(abs(delta[2]))
    return x


_OBSERVATION_SCHEMA = pa.schema([
    ("sample_key", pa.string()), ("source_group", pa.string()), ("profile", pa.string()),
    ("split", pa.string()), ("task_id", pa.string()), ("query", pa.string()),
    ("anchor_tick", pa.int64()), ("y", pa.int64()), ("delay_ticks", pa.int64()),
    ("features_json", pa.string()), ("evidence_ids_json", pa.string()),
    ("positive_evidence_ids_json", pa.string()), ("bindings_json", pa.string()),
])


def _partition_queries(partition_text: str, target_text: str) -> dict:
    partition, path = Path(partition_text), Path(target_text)
    selected = [r for _, r in iter_jsonl(path) if r["task_id"] in TASKS and r["target_valid"]]
    scope = load_scope(partition)
    if scope["contract_version"] != CONTRACT_VERSION:
        raise ValueError("experiment graph version differs")
    split, group = source_split(scope, PROFILE_HOLDOUT), scope["episode_id"]
    rows, rejected, views = [], Counter(), {}
    nodes = observation_nodes(partition, observation="structured_physics") if selected else []
    for target in selected:
        tick = target["anchor_tick"]
        if tick not in views:
            views[tick] = isolate_records(nodes, cutoff_tick=tick,
                start_tick=max(scope["start_tick"], tick - 100), observation="structured_physics")
        view = views[tick]
        known = set(view["trace"]["entity_references"].values())
        if not set(target["bindings"].values()) <= known:
            rejected["subject_not_known_in_observed_prefix"] += 1
            continue
        try:
            x = features(view, target)
        except ValueError as error:
            if str(error) != "experiment subject has no observed state in its prefix":
                raise
            rejected["no_subject_state_under_observation_condition"] += 1
            continue
        delays = [t - tick for t in target.get("positive_ticks", []) if t > tick]
        rows.append({"sample_key": target["sample_key"], "source_group": group, "profile": scope["profile"],
            "split": split, "task_id": target["task_id"], "query": target["query"], "anchor_tick": tick,
            "y": int(target["target_value"]), "delay_ticks": min(delays) if delays else None,
            "features_json": json.dumps(x, separators=(",", ":")),
            "evidence_ids_json": json.dumps(target["evidence_ids"]),
            "positive_evidence_ids_json": json.dumps(target.get("positive_evidence_ids", [])),
            "bindings_json": json.dumps(target["bindings"], sort_keys=True)})
    return {"sample_key": scope["sample_key"], "source_group": group, "split": split,
            "eligible_targets": len(selected), "excluded_input_queries": dict(rejected), "rows": rows}


def _preparation_inputs(partition: Path, targets: Path) -> dict:
    files = [partition / "nodes.parquet", partition / "scope.parquet", targets,
             Path(__file__), Path(__file__).with_name("prediction.py"),
             Path(__file__).with_name("belief_dataset.py")]
    return {"graph_version": CONTRACT_VERSION, "input_version": INPUT_VERSION, "tasks": list(TASKS),
        "profile_holdout": list(PROFILE_HOLDOUT), "history_ticks": 100,
        "artifacts": [{"name": p.name, "bytes": p.stat().st_size,
                       "modified_ns": p.stat().st_mtime_ns} for p in files]}


def prepare(graph_root: Path, target_root: Path, output: Path = OUTPUT, *, workers: int = 8) -> dict:
    """Resume unchanged per-sample preparation; only this parent publishes files."""
    graph_root, target_root, output = graph_root.resolve(), target_root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs, products = [], []
    for path in sorted(target_root.rglob("targets.jsonl")):
        relative = path.parent.relative_to(target_root)
        partition, destination = graph_root / relative, output / "queries" / relative
        inputs = _preparation_inputs(partition, path)
        marker = destination / "scope.json"
        if marker.is_file():
            metadata = json.loads(marker.read_text())
            if metadata["artifact_inputs"] == inputs and (destination / "observations.parquet").is_file():
                products.append((destination, metadata))
                continue
        jobs.append((partition, path, destination, inputs))
    requested = len(jobs) + len(products)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(_partition_queries, str(p), str(t)): (d, inputs)
                   for p, t, d, inputs in jobs}
        for future in as_completed(pending):
            destination, inputs = pending[future]
            metadata = future.result()
            rows = metadata.pop("rows")
            metadata.update(rows=len(rows), artifact_inputs=inputs)
            destination.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=_OBSERVATION_SCHEMA),
                           destination / "observations.parquet", compression="zstd")
            (destination / "scope.json").write_text(json.dumps(metadata, ensure_ascii=False) + "\n")
            products.append((destination, metadata))
            print(json.dumps({"experiment_prepared": len(products), "requested": requested,
                              "sample": metadata["sample_key"], "queries": len(rows)}), flush=True)
    if not products or not sum(m["rows"] for _, m in products):
        raise ValueError("finite experiment has no eligible observation queries")
    rejected, groups = Counter(), {}
    with pq.ParquetWriter(output / "observations.parquet", _OBSERVATION_SCHEMA, compression="zstd") as writer:
        for destination, metadata in sorted(products, key=lambda item: item[1]["sample_key"]):
            group, split = metadata["source_group"], metadata["split"]
            if group in groups and groups[group] != split:
                raise ValueError("formal episode and same-source ARM cross experiment splits")
            groups[group] = split
            rejected.update(metadata["excluded_input_queries"])
            writer.write_table(pq.read_table(destination / "observations.parquet", schema=_OBSERVATION_SCHEMA))
    scope = {"graph_version": CONTRACT_VERSION, "input_version": INPUT_VERSION,
        "observation": "structured_physics", "history_ticks": 100, "tasks": list(TASKS),
        "profile_holdout": list(PROFILE_HOLDOUT), "source_group_splits": groups,
        "split_basis": "development versus reserved profiles; fit() groups development rows by scenario family, never by seed",
        "population": "every valid regular query for these tasks in the declared existing graph corpus; no class rebalancing or event oversampling",
        "rows": sum(m["rows"] for _, m in products), "partitions": len(products),
        "excluded_input_queries": dict(rejected),
        "feature_exclusions": ["native entity IDs", "sample/profile/source names", "predicate labels", "future nodes and edges", "outcomes", "source summaries"]}
    (output / "scope.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n")
    return scope


EXPERIMENT_VERSION = "structured-belief-experiment-3"
ABLATIONS = {"observed_history": (), "current_pose_and_backward_velocity": ("history.",)}
MIN_EVALUATED_FOLDS = 3
MIN_INTERVAL_GROUPS = 5
_VARIANT = re.compile(r"^(?P<family>.+)_v\d+$")


def scenario_family(profile: str) -> str:
    """Profile variants L4-4_v1/L4-4_v2 share one scenario specification."""
    match = _VARIANT.match(profile)
    return match["family"] if match else profile


def closest_approach(features: dict, roles: list[str]) -> dict | None:
    """Constant-velocity closest approach within the task horizon.

    Uses only the observed current pose and backward-difference velocity of
    the two bound roles. No separation radius is assumed; the fold model
    learns how separation maps to the label. Rows lacking any value are not
    covered and are never imputed.
    """
    if len(roles) != 2:
        return None
    first, second = roles
    try:
        p = [np.array([features[f"current.{role}.position_{a}_m"] for a in "enu"]) for role in roles]
        v = [np.array([features[f"current.{role}.velocity_{a}_mps"] for a in "enu"]) for role in roles]
    except KeyError:
        return None
    r, w = p[1] - p[0], v[1] - v[0]
    horizon_s = HORIZON_TICKS / FORMAL_TICK_HZ
    speed2 = float(w @ w)
    t = 0. if speed2 == 0. else float(np.clip(-(r @ w) / speed2, 0., horizon_s))
    d = r + w * t
    return {"closest.distance_m": float(np.linalg.norm(d)), "closest.horizontal_m": float(np.linalg.norm(d[:2])),
            "closest.vertical_m": float(abs(d[2])), "closest.time_ticks": t * FORMAL_TICK_HZ}


def probability_scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1. - 1e-6)
    curve = []
    for lower in np.linspace(0., .9, 10):
        indices = (p >= lower) & (p < lower + .1 if lower < .9 else p <= 1.)
        if indices.any():
            curve.append({"lower": float(lower), "count": int(indices.sum()),
                          "mean_probability": float(p[indices].mean()), "event_frequency": float(y[indices].mean())})
    positive = p >= .5
    tp, fp, fn = int(((y == 1) & positive).sum()), int(((y == 0) & positive).sum()), int(((y == 1) & ~positive).sum())
    negatives = int((y == 0).sum())
    return {"NLL": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1. - p))),
        "Brier": float(np.mean((p - y) ** 2)),
        "positive_queries": int(y.sum()), "queries": len(y), "reliability": curve,
        "precision_at_0_5": tp / (tp + fp) if tp + fp else None,
        "recall_at_0_5": tp / (tp + fn) if tp + fn else None,
        "false_alarms_at_0_5": fp, "false_alarm_rate_at_0_5": fp / negatives if negatives else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None}


def _unique_rows(rows: list[dict]) -> list[dict]:
    """Collapse exact (family, features, label, delay) repeats; keep multiplicity.

    Seeds and ARM branches that replay an identical observed prefix add no new
    model input, so each distinct input/label enters fitting and scoring once.
    """
    merged = {}
    for row in rows:
        key = (scenario_family(row["profile"]), row["features_json"], row["y"], row["delay_ticks"])
        item = merged.get(key)
        if item is None:
            item = merged[key] = {"family": key[0], "features_json": row["features_json"], "y": row["y"],
                "delay_ticks": row["delay_ticks"], "bindings": json.loads(row["bindings_json"]),
                "multiplicity": 0, "profiles": set(), "source_groups": set(), "positive_evidence_ids": set()}
        item["multiplicity"] += 1
        item["profiles"].add(row["profile"])
        item["source_groups"].add(row["source_group"])
        item["positive_evidence_ids"].update(json.loads(row["positive_evidence_ids_json"]))
    unique = list(merged.values())
    for item in unique:
        item["features"] = json.loads(item["features_json"])
        item["kinematics"] = closest_approach(item["features"], sorted(item["bindings"]))
    return unique


def _fit_fold(train: list[dict]) -> dict | None:
    y = np.array([r["y"] for r in train])
    if len(set(y)) < 2:
        return None
    fold = {"prior": (y.sum() + 1.) / (len(y) + 2.), "models": {}, "kinematic": None, "time": None,
            "mean_delay": None, "training_queries": len(train), "training_positive_queries": int(y.sum())}
    for name, prefix in ABLATIONS.items():
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(),
                              LogisticRegression(max_iter=200, random_state=0))
        model.fit([{k: v for k, v in r["features"].items() if not k.startswith(prefix)} for r in train], y)
        fold["models"][name] = (model, prefix)
    covered = [r for r in train if r["kinematics"] is not None]
    if len({r["y"] for r in covered}) == 2:
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(),
                              LogisticRegression(max_iter=200, random_state=0))
        model.fit([r["kinematics"] for r in covered], [r["y"] for r in covered])
        fold["kinematic"] = model
    positives = [r for r in train if r["delay_ticks"] is not None]
    if positives:
        fold["time"] = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), Ridge())
        fold["time"].fit([r["features"] for r in positives], [r["delay_ticks"] for r in positives])
        fold["mean_delay"] = float(np.mean([r["delay_ticks"] for r in positives]))
    return fold


def _predict(fold: dict, rows: list[dict], query: str) -> dict:
    out = {"event_prior": np.full(len(rows), fold["prior"])}
    if query != "current_hidden_predicate":
        out["state_persistence"] = np.zeros(len(rows))
    for name, (model, prefix) in fold["models"].items():
        out[name] = model.predict_proba([{k: v for k, v in r["features"].items()
                                          if not k.startswith(prefix)} for r in rows])[:, 1]
    covered = np.array([r["kinematics"] is not None for r in rows], dtype=bool)
    if fold["kinematic"] is not None and covered.any():
        p = np.full(len(rows), np.nan)
        p[covered] = fold["kinematic"].predict_proba([r["kinematics"] for r in rows if r["kinematics"] is not None])[:, 1]
        out["kinematic_closest_approach"] = p
    return out


def _references(query: str) -> tuple[str, ...]:
    return (("event_prior", "current_pose_and_backward_velocity") if query == "current_hidden_predicate"
            else ("event_prior", "state_persistence"))


def _score_population(rows: list[dict], predictions: dict, query: str, fold_time: list[tuple] | None = None) -> dict:
    """Score all held-out rows; compare with the motion baseline on its covered subset only."""
    y = np.array([r["y"] for r in rows])
    complete = predictions["observed_history"]
    scores = {name: probability_scores(y, p) for name, p in predictions.items()
              if name != "kinematic_closest_approach"}
    full = scores["observed_history"]
    hits = [r["delay_ticks"] for r, value in zip(rows, complete) if value >= .5 and r["y"] == 1 and r["delay_ticks"] is not None]
    full["matched_positive_prediction_lead_ticks"] = {"count": len(hits), "mean": float(np.mean(hits)) if hits else None}
    eligible = {i for r in rows for i in r["positive_evidence_ids"]}
    detected = {i for r, value in zip(rows, complete) if value >= .5 for i in r["positive_evidence_ids"]}
    full["eligible_event_or_predicate_onset_recall"] = {"eligible": len(eligible), "detected": len(detected),
        "recall": len(detected) / len(eligible) if eligible else None}
    wins = {baseline: all(full[key] < scores[baseline][key] for key in ("NLL", "Brier"))
            for baseline in _references(query)}
    covered = np.array([r["kinematics"] is not None for r in rows], dtype=bool)
    kinematic = {"covered_queries": int(covered.sum()), "queries": len(rows)}
    if "kinematic_closest_approach" in predictions and covered.any():
        kinematic["kinematic_closest_approach"] = probability_scores(y[covered], predictions["kinematic_closest_approach"][covered])
        kinematic["observed_history_same_subset"] = probability_scores(y[covered], complete[covered])
        wins["kinematic_closest_approach"] = all(kinematic["observed_history_same_subset"][key]
            < kinematic["kinematic_closest_approach"][key] for key in ("NLL", "Brier"))
    elif covered.any():
        kinematic["status"] = "no_kinematic_model_training_rows_lack_both_classes"
    result = {"scores": scores, "kinematic_comparison": kinematic, "beats_reference": wins}
    if fold_time is not None and query != "current_hidden_predicate":
        timing = [t for t in fold_time if t is not None]
        if timing:
            actual = np.array([t["actual"] for t in timing])
            ridge = np.array([t["ridge"] for t in timing])
            mean = np.array([t["mean_delay"] for t in timing])
            result["timing"] = {"conditioning": "held-out queries whose event/terminal occurred within the horizon",
                "queries": len(timing), "ridge_MAE_ticks": float(np.mean(abs(ridge - actual))),
                "training_mean_delay_MAE_ticks": float(np.mean(abs(mean - actual)))}
            closest = [(t["closest"], t["actual"]) for t in timing if t["closest"] is not None]
            if closest:
                result["timing"]["closest_approach_time_MAE_ticks"] = float(np.mean([abs(a - b) for a, b in closest]))
                result["timing"]["closest_approach_time_queries"] = len(closest)
        else:
            result["timing"] = {"status": "no_positive_time_training_or_evaluation"}
    return result


def _timing(fold: dict, rows: list[dict]) -> list:
    out = []
    for r in rows:
        if r["delay_ticks"] is None or fold["time"] is None:
            out.append(None)
            continue
        out.append({"actual": r["delay_ticks"],
            "ridge": float(np.clip(fold["time"].predict([r["features"]])[0], 0., HORIZON_TICKS)),
            "mean_delay": fold["mean_delay"],
            "closest": r["kinematics"]["closest.time_ticks"] if r["kinematics"] is not None else None})
    return out


def _retained_previous(output: Path) -> dict:
    """Keep earlier measured holdout failures verbatim; never overwrite them silently."""
    path = output / "results.json"
    if not path.is_file():
        return {"status": "no_previous_results_file"}
    previous = json.loads(path.read_text())
    if previous.get("experiment_version") == EXPERIMENT_VERSION:
        return previous["retained_previous_measurements"]
    return {"experiment_version": previous["experiment_version"],
        "status": "retained_verbatim_from_previous_run",
        "input_limitation": "earlier features included simulator dispatches without a receiver observation contract; those measurements do not evaluate the current physics-only input",
        "report": previous}


def fit(output: Path = OUTPUT) -> dict:
    """Leave-one-scenario-family-out over development profiles, plus the untouched profile holdout.

    Prepared rows must match the current input contract. Grouping is by
    scenario family: all variants, seeds, ARM branches and windows of one
    scenario leave together.
    """
    scope = json.loads((output / "scope.json").read_text())
    if scope["graph_version"] != CONTRACT_VERSION or scope["input_version"] != INPUT_VERSION:
        raise ValueError("experiment features require preparation under the current input contract")
    retained = _retained_previous(output)
    rows = pq.read_table(output / "observations.parquet").to_pylist()
    strata = defaultdict(list)
    for row in rows:
        strata[row["task_id"], row["query"]].append(row)
    results = []
    with threadpool_limits(limits=4):
        for (task, query), population in sorted(strata.items()):
            development = _unique_rows([r for r in population if r["profile"] not in PROFILE_HOLDOUT])
            holdout = _unique_rows([r for r in population if r["profile"] in PROFILE_HOLDOUT])
            families = sorted({r["family"] for r in development})
            item = {"task": task, "query": query,
                "raw_queries": len(population), "development_unique_queries": len(development),
                "development_raw_queries": sum(r["multiplicity"] for r in development),
                "maximum_multiplicity": max((r["multiplicity"] for r in development), default=0),
                "development_unique_positive_queries": sum(r["y"] for r in development),
                "development_families": {f: {"profiles": sorted({p for r in development if r["family"] == f for p in r["profiles"]}),
                    "unique_queries": sum(r["family"] == f for r in development),
                    "unique_positive_queries": sum(r["y"] for r in development if r["family"] == f)} for f in families},
                "kinematic_coverage": {"covered": sum(r["kinematics"] is not None for r in development), "queries": len(development)},
                "folds": []}
            pooled_rows, pooled_predictions, pooled_time = [], defaultdict(list), []
            for family in families:
                train = [r for r in development if r["family"] != family]
                held = [r for r in development if r["family"] == family]
                train_keys = {(r["features_json"], r["y"]) for r in train}
                train_inputs = {r["features_json"] for r in train}
                fold_item = {"heldout_family": family, "heldout_profiles": sorted({p for r in held for p in r["profiles"]}),
                    "heldout_unique_queries": len(held), "heldout_positive_queries": sum(r["y"] for r in held),
                    "exact_feature_label_in_training": sum((r["features_json"], r["y"]) in train_keys for r in held),
                    "exact_features_in_training": sum(r["features_json"] in train_inputs for r in held)}
                fold = _fit_fold(train)
                if fold is None:
                    fold_item["status"] = "no_other_development_family" if not train else "no_training_class_variation"
                    item["folds"].append(fold_item)
                    continue
                predictions = _predict(fold, held, query)
                timing = _timing(fold, held)
                fold_item.update(status="evaluated", training_queries=fold["training_queries"],
                    training_positive_queries=fold["training_positive_queries"],
                    **_score_population(held, predictions, query, timing))
                item["folds"].append(fold_item)
                pooled_rows.extend(held)
                pooled_time.extend(timing)
                predictions.setdefault("kinematic_closest_approach", np.full(len(held), np.nan))
                for name, values in predictions.items():
                    pooled_predictions[name].append(values)
            evaluated = [f for f in item["folds"] if f["status"] == "evaluated"]
            if pooled_rows:
                merged = {name: np.concatenate(parts) for name, parts in pooled_predictions.items()}
                scored = ~np.isnan(merged["kinematic_closest_approach"])
                if not scored.any():
                    merged.pop("kinematic_closest_approach")
                # Folds without a motion model leave their rows outside the motion comparison.
                pooled_view = [{**r, "kinematics": r["kinematics"] if keep else None} for r, keep in zip(pooled_rows, scored)]
                item["pooled_out_of_fold"] = _score_population(pooled_view, merged, query, pooled_time)
            references = sorted({b for f in evaluated for b in f["beats_reference"]})
            fold_wins = sum(all(f["beats_reference"].values()) for f in evaluated)
            positive_groups = sum(f["heldout_positive_queries"] > 0 for f in evaluated)
            item["claim"] = {"evaluated_folds": len(evaluated), "folds_beating_every_reference": fold_wins,
                "references": references, "heldout_families_with_positives": positive_groups,
                "group_interval": "not computed: fewer than %d held-out families with positives; per-fold results listed" % MIN_INTERVAL_GROUPS
                    if positive_groups < MIN_INTERVAL_GROUPS else "not computed; per-fold results listed"}
            if len(families) < 2:
                item["claim"]["status"] = "unsupported_single_development_family"
            elif len(evaluated) < MIN_EVALUATED_FOLDS or positive_groups < 2:
                item["claim"]["status"] = "unsupported_insufficient_group_support"
            elif (all(item["pooled_out_of_fold"]["beats_reference"].values())
                  and set(item["pooled_out_of_fold"]["beats_reference"]) >= set(references)
                  and fold_wins * 2 > len(evaluated)):
                item["claim"]["status"] = "supported_predictive_signal"
            else:
                item["claim"]["status"] = "no_demonstrated_signal_beyond_baselines"
            if holdout:
                fold = _fit_fold(development)
                train_keys = {(r["features_json"], r["y"]) for r in development}
                record = {"profiles": sorted({p for r in holdout for p in r["profiles"]}),
                    "unique_queries": len(holdout), "raw_queries": sum(r["multiplicity"] for r in holdout),
                    "exact_feature_label_in_training": sum((r["features_json"], r["y"]) in train_keys for r in holdout),
                    "training": "unique rows of every development profile; fixed configuration, no tuning on this holdout"}
                if fold is None:
                    record["status"] = "no_training_class_variation"
                else:
                    record.update(status="evaluated_untuned", training_queries=fold["training_queries"],
                        **_score_population(holdout, _predict(fold, holdout, query), query, _timing(fold, holdout)))
                item["predeclared_profile_holdout"] = record
            else:
                item["predeclared_profile_holdout"] = {"status": "no_evaluation_queries"}
            results.append(item)
            print(json.dumps({"fit": [task, query], "families": len(families), "evaluated_folds": len(evaluated),
                              "claim": item["claim"]["status"]}), flush=True)
    report = {"experiment_version": EXPERIMENT_VERSION, "graph_version": CONTRACT_VERSION,
        "prepared_features": {"reused": True, "observations": "observations.parquet from preparation recorded in scope.json",
            "note": "fit-only rerun under the current input contract; prepared features and labels are unchanged. Evaluation groups come from the profile column."},
        "scope": scope,
        "protocol": {"grouping": "leave one scenario family out (profile variants _vN of one scenario, all seeds, ARM branches and windows held out together) over development profiles",
            "development_profiles": "every profile not in profile_holdout", "profile_holdout": list(PROFILE_HOLDOUT),
            "duplicates": "exact (family, features, label, delay) repeats collapse to one row with multiplicity",
            "kinematic_closest_approach": "fold-trained logistic model on constant-velocity closest-approach separation (Euclidean, horizontal, vertical) and time within %d ticks, from observed current pose and backward-difference velocity; no radius or threshold; compared on its covered subset" % HORIZON_TICKS,
            "references": {"future_onset/future_terminal": ["event_prior", "state_persistence", "kinematic_closest_approach"],
                           "current_hidden_predicate": ["event_prior", "current_pose_and_backward_velocity", "kinematic_closest_approach"]},
            "claim_rule": "supported_predictive_signal only if observed_history has lower pooled out-of-fold NLL and Brier than every reference (motion baseline on the same covered rows) AND beats every reference on both in a strict majority of evaluated folds; at least %d evaluated folds and 2 held-out families with positives are required, otherwise unsupported_insufficient_group_support" % MIN_EVALUATED_FOLDS},
        "results": results, "retained_previous_measurements": retained,
        "limits": ["structured physical observations only; no RGB/depth/segmentation encoder",
                   "one fixed logistic model configuration; no validation-driven tuning",
                   "persistence means no new onset/terminal under a frozen state; it is not a hidden-label oracle",
                   "current_pose_and_backward_velocity keeps velocity and yaw derived from past positions; it is not a single-frame baseline",
                   "time regression is conditional on a positive horizon; event probability and false alarms are assessed separately",
                   "positive queries are windows over few distinct incidents; unique counts still overstate independent support"]}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-root", type=Path, default=REPO / "Dataset/world_model/runs/belief_event_tasks")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase", choices=("prepare", "fit", "all"), default="all")
    args = parser.parse_args()
    started = time.monotonic()
    if args.phase in {"prepare", "all"}:
        print(json.dumps(prepare(args.graph_root, args.target_root, args.output)))
    if args.phase in {"fit", "all"}:
        report = fit(args.output)
        print(json.dumps({"task_queries": len(report["results"]), "seconds": time.monotonic() - started}))
