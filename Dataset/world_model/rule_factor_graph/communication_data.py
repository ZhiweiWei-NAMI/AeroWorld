"""Cutoff-only communication inputs and separate real future supervision.

The six-state finite model uses ``(A, Q)`` with A indicating a reduced
``station_operational_state.availability`` in this actor/station/flow record.
Q is a threshold cell, not the producer's ``quality_level`` category.  A is
contextual to the actor and is not a global physical station state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import re
from typing import Mapping, Sequence

import numpy as np

from .contracts import GroundingError, RoleBinding
from .events import EventConstraint


PROFILE_PATH = "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"
SOURCE_ROOT = "aw_data/compute_comm_supplement"
MOTION_ROOT = "aw_data/render_ready_episodes_capture_filtered"
PREDICATES = ("communication.data_flow_latency_above_threshold",
              "communication.packet_loss_above_threshold")
HISTORY_FEATURE_NAMES = (
    "quality_score", "latency_over_threshold", "loss_over_threshold",
    "availability", "operational_factor", "range_factor", "weather_attenuation",
    "distance_over_nominal_range", "effective_range_over_nominal_range",
    "speed_over_10mps", "relative_station_x_over_nominal_range",
    "relative_station_y_over_nominal_range", "relative_station_z_over_nominal_range",
    "velocity_x_over_10mps", "velocity_y_over_10mps", "velocity_z_over_10mps",
)
STATE_DOMAIN = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
Q_PREDICATES = ((False, False), (False, True), (True, True))
DEFAULT_EPISODES = ("L5-1_v2__seed00", "L6-2_v2__seed00",
                    "L6-1_v1__seed00", "L6-4_v1__seed00")
DEFAULT_SPLIT = {"L5-1": "train", "L6-2": "train",
                 "L6-1": "validation", "L6-4": "test"}


def _frozen_array(value, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _number(value, field: str) -> float | None:
    if value in ("unknown", "source_required", None):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundingError(f"{field} must be numeric or explicitly unknown")
    if not np.isfinite(value):
        raise GroundingError(f"{field} must be finite")
    return float(value)


@dataclass(frozen=True)
class CommunicationOwner:
    actor_id: str
    station_id: str
    flow_id: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value or value == "unknown"
               for value in (self.actor_id, self.station_id, self.flow_id)):
            raise GroundingError("communication owner needs exact actor, station and flow identities")

    @property
    def owner_id(self) -> str:
        return "|".join((self.actor_id, self.station_id, self.flow_id))

    @property
    def role_bindings(self) -> tuple[RoleBinding, ...]:
        return (RoleBinding("actor", self.actor_id, "world:UnmannedAircraft"),
                RoleBinding("station", self.station_id, "world:CommunicationStation"),
                RoleBinding("flow", self.flow_id, "dom:DataFlow"))


@dataclass(frozen=True)
class CommunicationObservation:
    tick: int
    owner: CommunicationOwner
    quality_score: float | None
    latency_ms: float | None
    packet_loss_ratio: float | None
    latency_threshold_ms: float
    packet_loss_threshold: float
    availability: float | None
    operational_factor: float | None
    range_factor: float | None
    weather_attenuation: float | None
    distance_m: float | None
    effective_range_m: float | None
    requirement_status: str
    station_state_mode: str
    source_path: str
    source_line: int
    source_refs: tuple[str, ...]

    @property
    def predicates(self) -> tuple[bool | None, bool | None]:
        if self.requirement_status == "source_required":
            return (None, None)
        return (None if self.latency_ms is None else self.latency_ms > self.latency_threshold_ms,
                None if self.packet_loss_ratio is None else self.packet_loss_ratio > self.packet_loss_threshold)

    @property
    def availability_bin(self) -> int | None:
        if self.availability is None:
            return None
        if not 0 <= self.availability <= 1:
            raise GroundingError("availability must lie in [0,1]")
        return int(self.availability < 1)

    @property
    def joint_state(self) -> int | None:
        values = self.predicates
        if self.availability_bin is None or None in values:
            return None
        if values not in Q_PREDICATES:
            raise GroundingError("source violates the declared finite QoS support L implies P")
        return self.availability_bin * 3 + Q_PREDICATES.index(values)


@dataclass(frozen=True)
class CommunicationPredictionInput:
    """The public predictor input has no future labels or communication states."""
    window_id: str
    episode_id: str
    family: str
    component: str
    split: str
    cutoff_tick: int
    ticks: tuple[int, ...]
    owner: CommunicationOwner
    prefix: tuple[CommunicationObservation, ...]
    history: np.ndarray
    motion_history: np.ndarray
    station_position: np.ndarray
    nominal_range_m: float
    tick_hz: float
    initial_predicates: tuple[bool, bool]
    initial_joint_state: int
    profile_id: str
    profile_source_path: str
    motion_source_refs: tuple[str, ...]
    station_source_ref: str

    def __post_init__(self):
        if self.split not in ("train", "validation", "test") or type(self.cutoff_tick) is not int or self.cutoff_tick < 0:
            raise GroundingError("prediction needs a declared split and nonnegative integer cutoff")
        if not np.isfinite(self.tick_hz) or self.tick_hz <= 0 or not np.isfinite(self.nominal_range_m) or self.nominal_range_m <= 0:
            raise GroundingError("tick frequency and nominal range must be positive finite values")
        if not self.prefix or any(obs.tick > self.cutoff_tick for obs in self.prefix):
            raise GroundingError("prediction communication evidence must lie in the cutoff prefix")
        if len(self.prefix) < 2 or len(self.ticks) < 2 or any(type(tick) is not int for tick in self.ticks):
            raise GroundingError("prediction requires a complete prefix and future integer tick grid")
        if any(right.tick - left.tick != 5 for left, right in zip(self.prefix, self.prefix[1:])) or any(
                right - left != 5 for left, right in zip(self.ticks, self.ticks[1:])):
            raise GroundingError("prediction evidence and horizon must use the exact step-five grid")
        if self.prefix[-1].tick != self.cutoff_tick or self.ticks[0] != self.cutoff_tick:
            raise GroundingError("prediction anchor must equal the cutoff")
        if any(obs.owner != self.owner for obs in self.prefix):
            raise GroundingError("prefix communication owner changed")
        history = _frozen_array(self.history, np.float64)
        motion = _frozen_array(self.motion_history, np.float64)
        position = _frozen_array(self.station_position, np.float64)
        if history.shape != (len(self.prefix), len(HISTORY_FEATURE_NAMES)):
            raise GroundingError("history feature shape differs from the declared field order")
        if motion.shape != (len(self.prefix), 1, 6) or position.shape != (3,):
            raise GroundingError("motion history must contain one shared actor state per tick")
        if not (np.isfinite(history).all() and np.isfinite(motion).all() and np.isfinite(position).all()):
            raise GroundingError("selected prediction prefixes require complete finite observations")
        if self.prefix[-1].joint_state != self.initial_joint_state:
            raise GroundingError("initial joint state differs from cutoff evidence")
        if self.prefix[-1].predicates != self.initial_predicates or any(type(value) is not bool for value in self.initial_predicates):
            raise GroundingError("initial predicate truth must equal known cutoff evidence")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "motion_history", motion)
        object.__setattr__(self, "station_position", position)


@dataclass(frozen=True)
class CommunicationSupervision:
    """Labels, latent supervision and donor motion live outside prediction input."""
    ticks: tuple[int, ...]
    predicates: np.ndarray
    known: np.ndarray
    joint_state: np.ndarray
    availability_values: np.ndarray
    availability_known: np.ndarray
    future_motion: np.ndarray
    future_motion_known: np.ndarray
    source_refs: tuple[str, ...]
    label_source_refs: tuple[str, ...]
    motion_source_refs: tuple[str, ...]

    def __post_init__(self):
        n = len(self.ticks)
        shapes = {"predicates": ((n, 2), np.int64), "known": ((n, 2), bool),
                  "joint_state": ((n,), np.int64), "availability_values": ((n,), np.int64),
                  "availability_known": ((n,), bool), "future_motion": ((n - 1, 1, 6), np.float64),
                  "future_motion_known": ((n - 1, 1), bool)}
        for field, (shape, dtype) in shapes.items():
            value = _frozen_array(getattr(self, field), dtype)
            if value.shape != shape:
                raise GroundingError(f"{field} shape must be {shape}")
            object.__setattr__(self, field, value)
        if not np.array_equal(self.known, self.predicates >= 0):
            raise GroundingError("predicate unknown values must be explicit -1")
        if not np.isin(self.predicates, (-1, 0, 1)).all() or not np.isin(self.joint_state, (-1, 0, 1, 2, 3, 4, 5)).all():
            raise GroundingError("supervision values must use the declared discrete domains")
        if not np.array_equal(self.availability_known, self.availability_values >= 0):
            raise GroundingError("availability unknown values must be explicit -1")
        if not np.isin(self.availability_values, (-1, 0, 1)).all():
            raise GroundingError("availability supervision must use reduced/nominal bins or -1")
        if not np.isfinite(self.future_motion[self.future_motion_known]).all():
            raise GroundingError("known motion supervision must contain finite values")
        if not np.isnan(self.future_motion[~self.future_motion_known]).all():
            raise GroundingError("missing motion supervision must be explicit NaN")

    def event_constraints(self) -> dict[str, EventConstraint]:
        result = {}
        for column, name in enumerate(("L", "P")):
            if not self.known[0, column]:
                raise GroundingError("first-flip supervision needs known cutoff truth")
            initial = self.predicates[0, column]
            event = EventConstraint("none")
            for step in range(1, len(self.ticks)):
                if not self.known[step, column]:
                    event = EventConstraint("right_censored", step - 1)
                    break
                if self.predicates[step, column] != initial:
                    event = EventConstraint("first_flip", step)
                    break
            result[name] = event
        return result


@dataclass(frozen=True)
class CommunicationWindow:
    prediction: CommunicationPredictionInput
    supervision: CommunicationSupervision


def _observation(row: dict, source_path: str, line: int) -> CommunicationObservation:
    if (row["schema_name"], row["schema_version"], row["model_id"], row["model_version"],
            row["rule_id"], row["rule_version"]) != (
            "communication_state", "2.0.0", "aeroworld_discrete_compute_comm_sim", "1.6.0",
            "compute_comm_supplement.communication_state", "1.3.0"):
        raise GroundingError("communication source schema changed")
    q = row["link_quality"]
    threshold = row["quality_thresholds"]
    numeric = {field: _number(q[field], f"link_quality.{field}") for field in (
        "quality_score", "latency_ms", "packet_loss_ratio", "operational_factor", "range_factor",
        "weather_attenuation", "distance_m", "effective_range_m")}
    latency = _number(threshold["latency_ms"], "quality_thresholds.latency_ms")
    loss = _number(threshold["packet_loss_ratio"], "quality_thresholds.packet_loss_ratio")
    if latency is None or loss is None:
        raise GroundingError("the selected finite support requires known configured thresholds")
    return CommunicationObservation(
        tick=row["tick"], owner=CommunicationOwner(row["entity_id"], row["station_id"], row["flow_id"]),
        latency_threshold_ms=latency, packet_loss_threshold=loss,
        availability=_number(row["station_operational_state"]["availability"], "station_operational_state.availability"),
        requirement_status=row["requirement"]["status"],
        station_state_mode=row["station_operational_state"]["state_mode"],
        source_path=source_path, source_line=line, source_refs=tuple(row["source_refs"]), **numeric,
    )


def _profile(root: Path) -> tuple[dict, float]:
    profile = json.loads((root / PROFILE_PATH).read_text())
    channel = profile["communication"]["channel"]
    flow = profile["communication"]["flow_profiles"]["uav"]
    expected = (20.0, 220.0, 0.25, 120.0, 0.05)
    actual = (channel["nominal_latency_ms"], channel["worst_case_latency_ms"],
              channel["maximum_packet_loss_ratio"], flow["latency_threshold_ms"],
              flow["packet_loss_threshold_ratio"])
    if actual != expected:
        raise GroundingError(f"finite QoS support needs rederivation for profile values {actual}")
    if profile["model"]["model_version"] != "1.6.0":
        raise GroundingError("communication producer version changed")
    # Under score in [0,1], L implies score<.5 and then P>(.5)^2*.25=.0625>.05.
    return profile, float(profile["communication"]["station_profiles"]["base_station"]["nominal_range_m"])


def _features(obs: CommunicationObservation, state: np.ndarray, station: np.ndarray,
              nominal_range: float) -> np.ndarray | None:
    scalar = (obs.quality_score, obs.latency_ms, obs.packet_loss_ratio, obs.availability,
              obs.operational_factor, obs.range_factor, obs.weather_attenuation,
              obs.distance_m, obs.effective_range_m)
    if any(value is None for value in scalar):
        return None
    relative = state[:3] - station
    return np.array((obs.quality_score, obs.latency_ms / obs.latency_threshold_ms,
                     obs.packet_loss_ratio / obs.packet_loss_threshold, obs.availability,
                     obs.operational_factor, obs.range_factor, obs.weather_attenuation,
                     obs.distance_m / nominal_range, obs.effective_range_m / nominal_range,
                     np.linalg.norm(state[3:]) / 10.0, *(relative / nominal_range),
                     *(state[3:] / 10.0)), dtype=np.float64)


def load_communication_windows(
    repo_root: Path, episodes: Sequence[str] = DEFAULT_EPISODES,
    split_by_family: Mapping[str, str] = DEFAULT_SPLIT, *, history_steps: int = 8,
    future_steps: int = 10, step_ticks: int = 5, cutoff_stride: int = 25,
) -> tuple[list[CommunicationWindow], dict]:
    """Read a bounded episode list; grid selection uses cutoff evidence only.

    Every exact shared prefix is checked across families before any fit.  The
    explicit protocol fails if such a component is assigned to two splits.
    Future missing records and owner changes remain unknown supervision.
    """
    root = Path(repo_root)
    if history_steps < 2 or future_steps < 1 or step_ticks != 5 or cutoff_stride % step_ticks:
        raise ValueError("require history>=2, future>=1 and a step-five cutoff grid")
    profile, nominal_range = _profile(root)
    windows = []
    excluded = {"missing_prefix_communication": 0, "changed_prefix_owner": 0,
                "missing_prefix_motion": 0, "incomplete_prefix_state": 0,
                "missing_prefix_station": 0, "unsupported_actor_type": 0}
    episode_counts = {}
    for episode in episodes:
        family = re.sub(r"_v\d+$", "", episode.split("__", 1)[0])
        split = split_by_family[family]
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unsupported split {split}")
        source_path = f"{SOURCE_ROOT}/{episode}/communication_state.jsonl"
        manifest = json.loads((root / SOURCE_ROOT / episode / "simulation_manifest.json").read_text())
        if manifest["profile_id"] != profile["profile_id"] or manifest["model_version"] != profile["model"]["model_version"]:
            raise GroundingError("source manifest differs from the declared communication profile")
        communication = {}
        actors = set()
        with (root / source_path).open() as stream:
            for line, text in enumerate(stream, 1):
                row = json.loads(text)
                if row["episode_id"] != episode:
                    raise GroundingError("communication source episode identity changed")
                obs = _observation(row, source_path, line)
                if (obs.latency_threshold_ms, obs.packet_loss_threshold) != (120.0, 0.05):
                    raise GroundingError("source thresholds differ from the selected finite model")
                key = (obs.tick, obs.owner.actor_id)
                if key in communication:
                    raise GroundingError("duplicate actor communication record at a tick")
                communication[key] = obs
                actors.add(obs.owner.actor_id)
        motion_path = f"{MOTION_ROOT}/{episode}/truth_frames.jsonl"
        motion = {}
        station_positions = {}
        actor_categories = {}
        tick_hz = None
        with (root / motion_path).open() as stream:
            for line, text in enumerate(stream, 1):
                row = json.loads(text)
                if row["episode_id"] != episode:
                    raise GroundingError("motion source episode identity changed")
                if row["tick"] % step_ticks:
                    continue
                if tick_hz is None:
                    tick_hz = float(row["tick_hz"])
                elif tick_hz != row["tick_hz"]:
                    raise GroundingError("motion source tick frequency changed")
                for entity in row["entities"]:
                    if entity["entity_id"] in actor_categories and actor_categories[entity["entity_id"]] != entity["entity_category"]:
                        raise GroundingError("an exact actor identity changed entity category")
                    actor_categories[entity["entity_id"]] = entity["entity_category"]
                    pose = entity["truth_pose"]
                    state = np.asarray(pose["position_enu_m"] + pose["velocity_enu_mps"], dtype=np.float64)
                    if state.shape != (6,) or not np.isfinite(state).all():
                        raise GroundingError("motion source must contain finite six-dimensional states")
                    key = (row["tick"], entity["entity_id"])
                    if key in motion:
                        raise GroundingError("duplicate entity motion at a tick")
                    motion[key] = (state, f"{motion_path}#line={line}#entity={entity['entity_id']}")
                    if entity["entity_category"] == "facility":
                        station_positions[key] = (state[:3], motion[key][1])
        labels_path = f"{SOURCE_ROOT}/{episode}/predicate_truth.jsonl"
        labels = {}
        with (root / labels_path).open() as stream:
            for line, text in enumerate(stream, 1):
                row = json.loads(text)
                if row["episode_id"] != episode:
                    raise GroundingError("saved predicate episode identity changed")
                if row["predicate_id"] not in PREDICATES:
                    continue
                if row["bindings"] != {"flow": row["event_participants"]["flow"]}:
                    raise GroundingError("communication label flow binding changed")
                if row["binding_ontology_classes"] != {"flow": "world:DataFlow"}:
                    raise GroundingError("saved runtime DataFlow class changed; formal dom alias needs review")
                if (row["schema_name"], row["schema_version"]) != ("compute_comm_predicate_truth", "2.0.0"):
                    raise GroundingError("saved communication predicate schema changed")
                key = (row["tick"], row["bindings"]["flow"], row["predicate_id"])
                if key in labels:
                    raise GroundingError("duplicate saved communication predicate")
                labels[key] = (row["value"], f"{labels_path}#line={line}",
                               row["event_participants"]["actor"], row["event_participants"]["station"])
        before = len(windows)
        for actor in sorted(actors):
            if actor in actor_categories and actor_categories[actor] != "uav":
                excluded["unsupported_actor_type"] += 1
                continue
            # Scope is chosen from typed source identity, never actor name similarity.
            for cutoff in range(history_steps * step_ticks, 901 - future_steps * step_ticks, cutoff_stride):
                history_ticks = tuple(range(cutoff - (history_steps - 1) * step_ticks, cutoff + 1, step_ticks))
                prefix = tuple(communication.get((tick, actor)) for tick in history_ticks)
                if any(obs is None for obs in prefix):
                    excluded["missing_prefix_communication"] += 1
                    continue
                owner = prefix[-1].owner
                if any(obs.owner != owner for obs in prefix):
                    excluded["changed_prefix_owner"] += 1
                    continue
                # This exact prefix is the producer's declared synthetic-station ID mapping.
                station_actor = owner.station_id
                synthetic = "simulated_communication_station:"
                if station_actor.startswith(synthetic):
                    station_actor = station_actor[len(synthetic):]
                station_record = station_positions.get((cutoff, station_actor))
                if station_record is None:
                    excluded["missing_prefix_station"] += 1
                    continue
                station_position, station_ref = station_record
                history_motion = tuple(motion.get((tick, actor)) for tick in history_ticks)
                if any(item is None for item in history_motion):
                    excluded["missing_prefix_motion"] += 1
                    continue
                feature_rows = tuple(_features(obs, item[0], station_position, nominal_range)
                                     for obs, item in zip(prefix, history_motion))
                if any(item is None for item in feature_rows) or prefix[-1].joint_state is None:
                    excluded["incomplete_prefix_state"] += 1
                    continue
                ticks = tuple(cutoff + step * step_ticks for step in range(future_steps + 1))
                prediction = CommunicationPredictionInput(
                    window_id=f"{episode}|{owner.owner_id}|cutoff={cutoff}", episode_id=episode,
                    family=family, component=family, split=split, cutoff_tick=cutoff, ticks=ticks,
                    owner=owner, prefix=prefix, history=np.stack(feature_rows),
                    motion_history=np.stack([item[0] for item in history_motion])[:, None],
                    station_position=station_position, nominal_range_m=nominal_range,
                    tick_hz=tick_hz, initial_predicates=prefix[-1].predicates,
                    initial_joint_state=prefix[-1].joint_state, profile_id=profile["profile_id"],
                    profile_source_path=PROFILE_PATH, motion_source_refs=tuple(item[1] for item in history_motion),
                    station_source_ref=station_ref,
                )
                y = np.full((len(ticks), 2), -1, dtype=np.int64)
                a = np.full(len(ticks), -1, dtype=np.int64)
                joint = np.full(len(ticks), -1, dtype=np.int64)
                source_refs, label_refs = [], []
                for step, tick in enumerate(ticks):
                    obs = communication.get((tick, actor))
                    if obs is None or obs.owner != owner:
                        continue
                    source_refs.append(f"{obs.source_path}#line={obs.source_line}")
                    for column, predicate in enumerate(PREDICATES):
                        saved = labels.get((tick, owner.flow_id, predicate))
                        if saved is None:
                            continue
                        value, ref, label_actor, label_station = saved
                        if (label_actor, label_station) != (owner.actor_id, owner.station_id):
                            raise GroundingError(f"saved label producer owner differs from communication record at {ref}")
                        computed = obs.predicates[column]
                        expected = "unknown" if computed is None else ("true" if computed else "false")
                        if value != expected:
                            raise GroundingError(f"saved predicate differs from strict threshold replay at {ref}")
                        label_refs.append(ref)
                        if value in ("true", "false"):
                            y[step, column] = int(value == "true")
                        elif value not in ("unknown", "out_of_scope"):
                            raise GroundingError(f"invalid saved predicate value {value}")
                    if obs.availability_bin is not None:
                        a[step] = obs.availability_bin
                    if obs.joint_state is not None:
                        joint[step] = obs.joint_state
                future_motion = np.full((future_steps, 1, 6), np.nan)
                motion_known = np.zeros((future_steps, 1), dtype=bool)
                future_motion_refs = []
                for step, tick in enumerate(ticks[1:]):
                    item = motion.get((tick, actor))
                    if item is not None:
                        future_motion[step, 0] = item[0]
                        motion_known[step, 0] = True
                        future_motion_refs.append(item[1])
                supervision = CommunicationSupervision(ticks, y, y >= 0, joint, a, a >= 0,
                    future_motion, motion_known, tuple(source_refs), tuple(label_refs), tuple(future_motion_refs))
                if not supervision.known[0].all():
                    raise GroundingError("selected prefix has no saved initial predicate labels")
                windows.append(CommunicationWindow(prediction, supervision))
        episode_counts[episode] = {"communication_records": len(communication), "actors": len(actors),
                                   "windows": len(windows) - before}
    components = _shared_prefix_components(windows, split_by_family)
    windows = [CommunicationWindow(replace(window.prediction, component=components[window.prediction.family]),
                                   window.supervision) for window in windows]
    manifest = {"episodes": episode_counts, "excluded_cutoffs": excluded,
                "family_splits": dict(split_by_family), "components": components,
                "history_features": list(HISTORY_FEATURE_NAMES), "history_steps": history_steps,
                "future_steps": future_steps, "step_ticks": step_ticks, "cutoff_stride": cutoff_stride,
                "profile_source": PROFILE_PATH, "source_root": SOURCE_ROOT, "motion_root": MOTION_ROOT,
                "nominal_range_source": PROFILE_PATH + "#communication.station_profiles.base_station.nominal_range_m",
                "flow_class_alias": "saved runtime world:DataFlow mapped to formal dom:DataFlow",
                "state_domain": [list(state) for state in STATE_DOMAIN],
                "availability_owner": "actor|station|flow producer context",
                "availability_bin": "reduced [0,1) versus nominal {1}",
                "q_cells": ["L=false,P=false", "L=false,P=true", "L=true,P=true"],
                "selection_policy": "regular cutoff grid; prefix completeness only; no future event selection",
                "claim": "finite-model feasibility on four grouped scenario families; no physical latent identification"}
    return windows, manifest


def _shared_prefix_components(windows: Sequence[CommunicationWindow], split_by_family: Mapping[str, str]) -> dict[str, str]:
    families = sorted({window.prediction.family for window in windows})
    parent = {family: family for family in families}

    def find(family):
        while parent[family] != family:
            parent[family] = parent[parent[family]]
            family = parent[family]
        return family

    # Keys are raw equality data, kept in memory. They are not digests or artifacts.
    seen = {}
    for window in windows:
        query = window.prediction
        key = (tuple(query.history.ravel()), tuple(query.motion_history.ravel()),
               tuple(query.station_position), query.initial_joint_state,
               query.prefix[-1].latency_threshold_ms, query.prefix[-1].packet_loss_threshold)
        if key in seen:
            left, right = find(query.family), find(seen[key])
            parent[max(left, right)] = min(left, right)
        else:
            seen[key] = query.family
    groups = {}
    for family in families:
        groups.setdefault(find(family), []).append(family)
    for group in groups.values():
        if len({split_by_family[family] for family in group}) != 1:
            raise GroundingError(f"exact shared prefixes connect families assigned to different splits: {group}")
    return {family: "+".join(groups[find(family)]) for family in families}
