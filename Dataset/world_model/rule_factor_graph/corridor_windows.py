"""Frozen-prefix corridor candidates and separate observation-prefix labels.

The bounded protocol below was selected from static geometry before any model
metrics. Missing future rows censor a fixed candidate; they never remove its
prediction window or become false labels.
"""
from __future__ import annotations

from Dataset.semantic_truth.provenance import open_artifact_text
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .contracts import GroundingError, RoleBinding
from .events import EventConstraint
from .corridor_state_space import CorridorBox, inside_points, signed_distance_points


SOURCE_ROOT = "aw_data/objective_semantic_truth"
MOTION_ROOT = "aw_data/render_ready_episodes_capture_filtered"
PREDICATES = ("airspace.aircraft_occupies_airspace_corridor",
              "airspace.aircraft_moving_in_corridor",
              "airspace.aircraft_stationary_in_corridor")
EVENT_NAMES = ("O1", "M1", "S1", "O2", "M2", "S2")
HISTORY_FEATURE_NAMES = (
    "speed_over_tau", "speed_delta_over_tau", "velocity_x_over_tau",
    "velocity_y_over_tau", "velocity_z_over_tau", "inside_c1", "inside_c2",
    "signed_distance_c1_over_extent_scale", "signed_distance_c2_over_extent_scale",
    "history_fraction",
)
DEFAULT_EPISODES = ("L4-3_v2__seed00", "L4-5_v2__seed00",
                    "L4-2_v2__seed01", "L4-4_v1__seed00")
DEFAULT_SPLIT = {"L4-3": "train", "L4-5": "train",
                 "L4-2": "validation", "L4-4": "test"}
# Exact identities, fixed from their scene configuration, not a future scan.
DEFAULT_SCOPES = {
    "L4-3_v2": ("uav_forced_landing_l4_3_v2", ("corridor_l4_3_v2_uav_forced_landing_l4_3_v2_01",
                                             "corridor_l4_3_v2_uav_forced_landing_l4_3_v2_02")),
    "L4-5_v2": ("uav_ped_nearmiss_l4_5_v2", ("corridor_l4_5_v2_uav_ped_nearmiss_l4_5_v2_01",
                                           "corridor_l4_5_v2_uav_ped_nearmiss_l4_5_v2_02")),
    "L4-2_v2": ("uav_facade_l4_2_v2", ("corridor_l4_2_v2_uav_facade_l4_2_v2_02",
                                     "corridor_l4_2_v2_uav_facade_l4_2_v2_04")),
    "L4-4_v1": ("uav_falls_vehicle_l4_4_v1", ("corridor_l4_4_v1_uav_falls_vehicle_l4_4_v1_01",
                                             "corridor_l4_4_v1_uav_falls_vehicle_l4_4_v1_02")),
}


def frozen_array(value, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CorridorObservation:
    tick: int
    actor_id: str
    corridor_id: str
    inside: bool | None
    speed_mps: float | None
    source_ref: str
    source_refs: tuple[str, ...]

    @property
    def role_bindings(self) -> tuple[RoleBinding, ...]:
        return (RoleBinding("actor", self.actor_id, "world:UnmannedAircraft"),
                RoleBinding("corridor", self.corridor_id, "world:AirspaceCorridor"))

    def predicates(self, tau_mps: float) -> tuple[int, int, int]:
        if self.inside is None:
            return (-1, -1, -1)
        if not self.inside:
            return (0, 0, 0)
        if self.speed_mps is None:
            return (1, -1, -1)
        moving = int(self.speed_mps > tau_mps)
        return (1, moving, 1 - moving)


@dataclass(frozen=True)
class CorridorPredictionInput:
    window_id: str
    episode_id: str
    family: str
    component: str
    split: str
    cutoff_tick: int
    ticks: tuple[int, ...]
    actor_id: str
    corridor_ids: tuple[str, str]
    boxes: tuple[CorridorBox, CorridorBox]
    prefix: tuple[tuple[CorridorObservation, CorridorObservation], ...]
    history: np.ndarray
    motion_history: np.ndarray
    latent_history: np.ndarray
    tick_hz: float
    tau_mps: float
    initial_latent: tuple[int, int, int]
    initial_predicates: tuple[bool, ...]
    prefix_source_refs: tuple[str, ...]
    motion_source_refs: tuple[str, ...]
    geometry_source_refs: tuple[str, ...]
    threshold_source_ref: str

    def __post_init__(self):
        if self.split not in ("train", "validation", "test"):
            raise GroundingError("corridor split must be explicit")
        if not self.actor_id or len(set(self.corridor_ids)) != 2:
            raise GroundingError("corridor query needs one exact actor and two distinct corridors")
        if tuple(box.corridor_id for box in self.boxes) != self.corridor_ids:
            raise GroundingError("geometry owner differs from the fixed candidate")
        if len(self.prefix) < 2 or self.ticks[0] != self.cutoff_tick:
            raise GroundingError("prefix and horizon require the same anchor")
        if any(b - a != 5 for a, b in zip(self.ticks, self.ticks[1:])):
            raise GroundingError("corridor event grid must use step five")
        prefix_ticks = [pair[0].tick for pair in self.prefix]
        if prefix_ticks[-1] != self.cutoff_tick or any(b - a != 5 for a, b in zip(prefix_ticks, prefix_ticks[1:])):
            raise GroundingError("corridor prefix must be complete on the step-five grid")
        for pair in self.prefix:
            if len(pair) != 2 or any(obs.tick > self.cutoff_tick or obs.actor_id != self.actor_id or
                    obs.corridor_id != corridor for obs, corridor in zip(pair, self.corridor_ids)):
                raise GroundingError("corridor prefix has inconsistent typed owners or future evidence")
            if pair[0].tick != pair[1].tick or pair[0].speed_mps != pair[1].speed_mps:
                raise GroundingError("same actor speed must agree across corridor evidence")
            if any(obs.inside is None or obs.speed_mps is None for obs in pair):
                raise GroundingError("selected prefix latent state must be known")
        h = len(self.prefix)
        fields = {"history": ((h, len(HISTORY_FEATURE_NAMES)), np.float64),
                  "motion_history": ((h, 6), np.float64), "latent_history": ((h, 3), np.int64)}
        for name, (shape, dtype) in fields.items():
            value = frozen_array(getattr(self, name), dtype)
            if value.shape != shape or not np.isfinite(value).all():
                raise GroundingError(f"{name} needs finite declared shape {shape}")
            object.__setattr__(self, name, value)
        if not np.isin(self.latent_history, (0, 1)).all() or tuple(self.latent_history[-1]) != self.initial_latent:
            raise GroundingError("latent prefix and anchor differ")
        if len(self.initial_predicates) != 6 or any(type(v) is not bool for v in self.initial_predicates):
            raise GroundingError("anchor predicate order must be O1,M1,S1,O2,M2,S2")
        if not np.isfinite(self.tick_hz) or self.tick_hz <= 0 or not np.isfinite(self.tau_mps) or self.tau_mps <= 0:
            raise GroundingError("tick frequency and configured speed threshold must be positive")

    @property
    def boxes_array(self) -> np.ndarray:
        """[corridor,center/halfextent,xyz], always in source ENU metres."""
        return frozen_array([[box.center_enu_m, box.extent_m] for box in self.boxes])


@dataclass(frozen=True)
class CorridorSupervision:
    ticks: tuple[int, ...]
    predicates: np.ndarray
    known: np.ndarray
    latent: np.ndarray
    latent_known: np.ndarray
    future_motion: np.ndarray
    future_motion_known: np.ndarray
    censor_reasons: tuple[tuple[str, ...], ...]
    source_refs: tuple[str, ...]
    label_source_refs: tuple[str, ...]
    motion_source_refs: tuple[str, ...]

    def __post_init__(self):
        t = len(self.ticks)
        fields = {"predicates": ((t, 6), np.int64), "known": ((t, 6), bool),
                  "latent": ((t, 3), np.int64), "latent_known": ((t, 3), bool),
                  "future_motion": ((t - 1, 6), np.float64), "future_motion_known": ((t - 1,), bool)}
        for name, (shape, dtype) in fields.items():
            value = frozen_array(getattr(self, name), dtype)
            if value.shape != shape:
                raise GroundingError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, value)
        if not np.array_equal(self.known, self.predicates >= 0) or not np.array_equal(self.latent_known, self.latent >= 0):
            raise GroundingError("supervision unknown values must be explicit -1")
        if not np.isin(self.predicates, (-1, 0, 1)).all() or not np.isin(self.latent, (-1, 0, 1)).all():
            raise GroundingError("corridor labels must be Boolean or -1 unknown")
        if not self.known[0].all() or not self.latent_known[0].all():
            raise GroundingError("selected prediction anchor must have saved source truth")
        if len(self.censor_reasons) != t or any(len(row) != 6 for row in self.censor_reasons):
            raise GroundingError("censor reasons must accompany every predicate observation")
        if not np.isfinite(self.future_motion[self.future_motion_known]).all() or not np.isnan(
                self.future_motion[~self.future_motion_known]).all():
            raise GroundingError("unknown oracle motion must be explicit NaN")

    def event_constraints(self) -> dict[str, EventConstraint]:
        result = {}
        for column, name in enumerate(EVENT_NAMES):
            initial = self.predicates[0, column]
            observed = 0
            event = EventConstraint("none")
            for step in range(1, len(self.ticks)):
                if not self.known[step, column]:
                    break
                if self.predicates[step, column] != initial:
                    event = EventConstraint("first_flip", step)
                    break
                observed = step
            if event.kind == "none" and observed > 0:
                event = EventConstraint("right_censored", observed)
            result[name] = event
        return result

    @property
    def event_informative(self) -> bool:
        return any(event.kind != "none" for event in self.event_constraints().values())

    def event_metadata(self) -> dict[str, dict]:
        result = {}
        for column, (name, event) in enumerate(self.event_constraints().items()):
            observed = event.step - 1 if event.kind == "first_flip" else (event.step or 0)
            first_missing = next((step for step in range(1, len(self.ticks))
                                  if not self.known[step, column]), None)
            reason = "" if first_missing is None else self.censor_reasons[first_missing][column]
            result[name] = {"kind": event.kind, "step": event.step,
                            "observed_future_bins": observed,
                            "likelihood_valid": event.kind != "none",
                            "horizon_outcome_known": event.kind == "first_flip" or observed == len(self.ticks) - 1,
                            "censor_reason": reason if event.kind != "first_flip" else ""}
        return result


@dataclass(frozen=True)
class CorridorWindow:
    prediction: CorridorPredictionInput
    supervision: CorridorSupervision


def _observation(row: dict, path: str, line: int) -> CorridorObservation:
    if (row["schema_name"], row["schema_version"], row["annotation_layer"],
            row["model_id"], row["model_version"], row["rule_id"], row["rule_version"]) != (
            "domain_state_observation", "3.0.0", "L0", "geometric_computer", "3.0.0",
            "geometric_computer.aircraft_corridor_membership", "3.0.0"):
        raise GroundingError("corridor observation source contract changed")
    v = row["values"]
    if v["aircraft_ontology_class_id"] != "world:UnmannedAircraft" or v["corridor_ontology_class_id"] != "world:AirspaceCorridor":
        raise GroundingError("corridor observation roles differ from declared ontology")
    if row["subject_id"] != f"{v['aircraft_id']}|{v['corridor_id']}":
        raise GroundingError("corridor observation subject differs from typed role join")
    inside, speed = v["aircraft_inside_corridor"], v["speed_mps"]
    if inside is not None and type(inside) is not bool:
        raise GroundingError("inside must be Boolean or explicitly missing")
    if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float)) or not np.isfinite(speed) or speed < 0):
        raise GroundingError("speed must be finite nonnegative or explicitly missing")
    return CorridorObservation(row["tick"], v["aircraft_id"], v["corridor_id"], inside,
                               speed, f"{path}#line={line}", tuple(row["source_refs"]))


def _geometry(root: Path, episode: str, actor: str, corridors: tuple[str, str]):
    manifest_path = f"{MOTION_ROOT}/{episode}/episode_manifest.json"
    manifest = json.loads((root / manifest_path).read_text())
    if manifest["episode_id"] != episode or manifest["duration_ticks"] != 900:
        raise GroundingError("selected formal episode identity/duration changed")
    scene_path = manifest["source_scene_setup_path"]
    scene = json.loads((root / scene_path).read_text())
    entities = {entity["entity_id"]: entity for entity in scene["entities"]}
    if entities[actor]["category"] != "uav":
        raise GroundingError("fixed actor is not a typed UAV in the source scene")
    boxes = []
    for corridor in corridors:
        entity = entities[corridor]
        placement = entity["placement"]
        if entity["category"] != "airspace_corridor" or entity["placement_mode"] != "box_volume":
            raise GroundingError("selected corridor does not have rectangular box authority")
        if entity["activation_tick"] != 0 or entity["enabled"] is not True or "deactivation_tick" in entity:
            raise GroundingError("bounded protocol requires static enabled tick-zero geometry")
        if placement["rotation_deg"]["pitch_deg"] != 0 or placement["rotation_deg"]["roll_deg"] != 0:
            raise GroundingError("source geometry is not a yaw-only box")
        if placement["source_uav_entity_id"] != actor:
            raise GroundingError("selected box source actor ownership differs")
        boxes.append(CorridorBox(corridor, tuple(placement["center_enu_m"]),
                                  tuple(placement["extent_m"]), placement["rotation_deg"]["yaw_deg"],
                                  f"{scene_path}#entity={corridor}"))
    return tuple(boxes), scene_path, manifest_path


def _load_episode(root: Path, episode: str, actor: str, corridors: tuple[str, str], boxes):
    source = f"{SOURCE_ROOT}/{episode}"
    summary_path = f"{source}/world_truth_summary.json"
    summary = json.loads((root / summary_path).read_text())
    if summary["episode_id"] != episode or summary["governed_parameters"]["formal_step_ticks"] != 5:
        raise GroundingError("corridor summary episode/grid changed")
    tau = float(summary["governed_parameters"]["aircraft_stationary_speed_threshold_mps"])
    observations = {}
    observation_path = f"{source}/domain_state_observations.jsonl"
    with (root / observation_path).open() as stream:
        for line, text in enumerate(stream, 1):
            if "predicate_contract_aircraft_corridor_geometry" not in text or actor not in text:
                continue
            row = json.loads(text)
            if row["episode_id"] != episode:
                raise GroundingError("corridor observation episode identity changed")
            if row["tick"] % 5 or row["values"]["aircraft_id"] != actor or row["values"]["corridor_id"] not in corridors:
                continue
            obs = _observation(row, observation_path, line)
            key = (obs.tick, obs.corridor_id)
            if key in observations:
                raise GroundingError("duplicate exact actor/corridor observation at a tick")
            if boxes[corridors.index(obs.corridor_id)].source_ref not in obs.source_refs:
                raise GroundingError("record geometry provenance differs from selected static box")
            observations[key] = obs
    motion = {}
    motion_path = f"{MOTION_ROOT}/{episode}/truth_frames.jsonl"
    tick_hz = None
    with (root / motion_path).open() as stream:
        for line, text in enumerate(stream, 1):
            row = json.loads(text)
            if row["episode_id"] != episode:
                raise GroundingError("motion episode identity changed")
            if row["tick"] % 5:
                continue
            hz = float(row["tick_hz"])
            if tick_hz is not None and hz != tick_hz:
                raise GroundingError("motion tick frequency changed")
            tick_hz = hz
            for entity in row["entities"]:
                if entity["entity_id"] != actor:
                    continue
                if entity["entity_category"] != "uav" or entity["truth_pose"]["coordinate_contract_id"] != "coord.external_enu_m.v1":
                    raise GroundingError("motion typed owner or coordinate frame changed")
                state = np.asarray(entity["truth_pose"]["position_enu_m"] + entity["truth_pose"]["velocity_enu_mps"], dtype=np.float64)
                if state.shape != (6,) or not np.isfinite(state).all() or row["tick"] in motion:
                    raise GroundingError("motion state malformed or duplicated")
                motion[row["tick"]] = (state, f"{motion_path}#line={line}#entity={actor}")
    labels = {}
    label_path = f"{source}/world_truth_graph_deltas.jsonl"
    with open_artifact_text(root / label_path) as stream:
        for line, text in enumerate(stream, 1):
            if actor not in text or not any(predicate in text for predicate in PREDICATES):
                continue
            row = json.loads(text)
            if row["episode_id"] != episode:
                raise GroundingError("saved label episode changed")
            for operation in row["operations"]:
                predicate = operation.get("predicate_id")
                bindings = operation.get("bindings")
                if predicate not in PREDICATES or not isinstance(bindings, dict) or bindings.get("actor") != actor or bindings.get("corridor") not in corridors:
                    continue
                if set(bindings) != {"actor", "corridor"} or operation["binding_ontology_classes"] != {"actor": "world:UnmannedAircraft", "corridor": "world:AirspaceCorridor"}:
                    raise GroundingError("saved label typed bindings changed")
                key = (row["tick"], bindings["corridor"], predicate)
                kind = operation["operation"]
                if kind == "set_predicate_value":
                    value = operation["to_value"]
                elif kind == "refresh_predicate_evidence":
                    value = operation["value"]
                elif kind == "remove_predicate_assertion":
                    value = "out_of_scope"
                else:
                    raise GroundingError(f"unsupported corridor label operation {kind}")
                if key in labels:
                    raise GroundingError("duplicate exact saved corridor predicate at a tick")
                if value not in ("true", "false", "unknown", "out_of_scope"):
                    raise GroundingError("saved corridor predicate value changed")
                labels[key] = (value, f"{label_path}#line={line}#tuple={operation['tuple_id']}")
    return observations, motion, labels, tau, tick_hz, summary_path


def _prefix_features(motion: np.ndarray, latent: np.ndarray, boxes, tau: float) -> np.ndarray:
    speed = np.linalg.norm(motion[:, 3:], axis=-1)
    speed_delta = np.concatenate(([0.0], np.diff(speed)))
    distances = np.stack([signed_distance_points(motion[:, :3], box) / max(box.extent_m) for box in boxes], axis=-1)
    return np.column_stack((speed / tau, speed_delta / tau, motion[:, 3:] / tau,
                            latent[:, 1:], distances, np.linspace(0.0, 1.0, len(motion))))


def load_corridor_windows(
    repo_root: Path, episodes: Sequence[str] = DEFAULT_EPISODES,
    split_by_family: Mapping[str, str] = DEFAULT_SPLIT, *, history_steps: int = 8,
    future_steps: int = 10, step_ticks: int = 5, cutoff_stride: int = 25,
    scopes: Mapping[str, tuple[str, tuple[str, str]]] = DEFAULT_SCOPES,
) -> tuple[list[CorridorWindow], dict]:
    """Fixed prefix candidates; future missing truth remains partial supervision."""
    if history_steps < 2 or future_steps < 1 or step_ticks != 5 or cutoff_stride % 5:
        raise ValueError("corridor windows need H>=2, future>=1 and step-five grids")
    root = Path(repo_root)
    windows, episode_counts = [], {}
    excluded = {"missing_prefix_candidate": 0, "incomplete_prefix_latent": 0,
                "missing_prefix_motion": 0, "missing_anchor_labels": 0}
    for episode in episodes:
        scenario = episode.split("__", 1)[0]
        family = re.sub(r"_v\d+$", "", scenario)
        split = split_by_family[family]
        actor, corridors = scopes[scenario]
        boxes, scene_path, manifest_path = _geometry(root, episode, actor, corridors)
        observations, motion, labels, tau, tick_hz, summary_path = _load_episode(root, episode, actor, corridors, boxes)
        before = len(windows)
        for cutoff in range((history_steps - 1) * 5, 901, cutoff_stride):
            history_ticks = tuple(range(cutoff - (history_steps - 1) * 5, cutoff + 1, 5))
            pairs = tuple(tuple(observations.get((tick, corridor)) for corridor in corridors) for tick in history_ticks)
            if any(obs is None for pair in pairs for obs in pair):
                excluded["missing_prefix_candidate"] += 1
                continue
            if any(obs.inside is None or obs.speed_mps is None for pair in pairs for obs in pair):
                excluded["incomplete_prefix_latent"] += 1
                continue
            if any(tick not in motion for tick in history_ticks):
                excluded["missing_prefix_motion"] += 1
                continue
            for pair in pairs:
                if pair[0].speed_mps != pair[1].speed_mps:
                    raise GroundingError("same actor motion evidence conflicts across corridors")
            motion_history = np.stack([motion[tick][0] for tick in history_ticks])
            latent_history = np.asarray([(int(pair[0].speed_mps > tau), *(int(obs.inside) for obs in pair)) for pair in pairs])
            for step, pair in enumerate(pairs):
                if int(np.linalg.norm(motion_history[step, 3:]) > tau) != latent_history[step, 0]:
                    raise GroundingError("motion speed regime differs from exact corridor evidence")
                if any(bool(inside_points(motion_history[step, :3], box)) != obs.inside for obs, box in zip(pair, boxes)):
                    raise GroundingError("motion prefix does not replay producer box membership")
            ticks = tuple(cutoff + step * 5 for step in range(future_steps + 1))
            y = np.full((len(ticks), 6), -1, dtype=np.int64)
            latent = np.full((len(ticks), 3), -1, dtype=np.int64)
            reasons = [["missing_record"] * 6 for _ in ticks]
            refs, label_refs = [], []
            for step, tick in enumerate(ticks):
                pair = tuple(observations.get((tick, corridor)) for corridor in corridors)
                speeds = [obs.speed_mps for obs in pair if obs is not None and obs.speed_mps is not None]
                if speeds:
                    if any(speed != speeds[0] for speed in speeds[1:]):
                        raise GroundingError("future same actor speed evidence conflicts across corridors")
                    latent[step, 0] = int(speeds[0] > tau)
                for corridor_index, (corridor, obs) in enumerate(zip(corridors, pair)):
                    if obs is None:
                        for column in range(corridor_index * 3, corridor_index * 3 + 3):
                            reasons[step][column] = "candidate_absent"
                        continue
                    refs.append(obs.source_ref)
                    if obs.inside is not None:
                        latent[step, corridor_index + 1] = int(obs.inside)
                    computed = obs.predicates(tau)
                    for predicate_index, predicate in enumerate(PREDICATES):
                        column = corridor_index * 3 + predicate_index
                        saved = labels.get((tick, corridor, predicate))
                        if saved is None:
                            continue
                        value, ref = saved
                        label_refs.append(ref)
                        if value in ("true", "false"):
                            value_int = int(value == "true")
                            if computed[predicate_index] != value_int:
                                raise GroundingError(f"saved label differs from exact hard rule replay at {ref}")
                            y[step, column] = value_int
                            reasons[step][column] = ""
                        else:
                            reasons[step][column] = value
            if not (y[0] >= 0).all():
                excluded["missing_anchor_labels"] += 1
                continue
            prediction = CorridorPredictionInput(
                f"{episode}|actor={actor}|corridors={','.join(corridors)}|cutoff={cutoff}",
                episode, family, family, split, cutoff, ticks, actor, corridors, boxes, pairs,
                _prefix_features(motion_history, latent_history, boxes, tau), motion_history, latent_history,
                tick_hz, tau, tuple(int(v) for v in latent_history[-1]), tuple(bool(v) for v in y[0]),
                tuple(obs.source_ref for pair in pairs for obs in pair),
                tuple(motion[tick][1] for tick in history_ticks), tuple(box.source_ref for box in boxes),
                f"{summary_path}#governed_parameters.aircraft_stationary_speed_threshold_mps",
            )
            oracle = np.full((future_steps, 6), np.nan)
            oracle_known = np.zeros(future_steps, dtype=bool)
            oracle_refs = []
            for step, tick in enumerate(ticks[1:]):
                if tick in motion:
                    oracle[step] = motion[tick][0]
                    oracle_known[step] = True
                    oracle_refs.append(motion[tick][1])
            supervision = CorridorSupervision(ticks, y, y >= 0, latent, latent >= 0, oracle, oracle_known,
                                               tuple(tuple(row) for row in reasons), tuple(refs),
                                               tuple(label_refs), tuple(oracle_refs))
            windows.append(CorridorWindow(prediction, supervision))
        episode_counts[episode] = {
            "actor_id": actor, "corridor_ids": list(corridors), "windows": len(windows) - before,
            "step_five_candidate_records": len(observations), "step_five_motion_records": len(motion),
            "scene_source": scene_path, "episode_manifest_source": manifest_path,
            "geometry_known_at_tick": 0, "geometry_policy": "enabled_static_scene_yaw_boxes",
            "boxes": [{"corridor_id": box.corridor_id, "center_enu_m": list(box.center_enu_m),
                        "half_extent_m": list(box.extent_m), "yaw_deg": box.yaw_deg, "source_ref": box.source_ref} for box in boxes],
            "dualinside_observed_ticks": [tick for tick in range(0, 901, 5)
                if all((tick, corridor) in observations and observations[tick, corridor].inside is True for corridor in corridors)],
        }
    components, duplicate_pairs = _prefix_components(windows, split_by_family)
    windows = [CorridorWindow(replace(window.prediction, component=components[window.prediction.family]), window.supervision) for window in windows]
    event_counts = {name: {"first_flip": 0, "partial_censor": 0, "complete_event_free": 0,
                          "zero_observed_bins": 0} for name in EVENT_NAMES}
    for window in windows:
        for name, event in window.supervision.event_constraints().items():
            key = "first_flip" if event.kind == "first_flip" else ("zero_observed_bins" if event.kind == "none" else
                  "complete_event_free" if event.step == future_steps else "partial_censor")
            event_counts[name][key] += 1
    manifest = {
        "protocol_id": "corridor_joint_shared_actor_prefix_v3", "protocol_frozen_before_metrics": True,
        "episodes": episode_counts, "family_splits": dict(split_by_family), "components": components,
        "exact_shared_motion_prefix_pairs": duplicate_pairs, "excluded_cutoffs": excluded,
        "history_steps": history_steps, "future_steps": future_steps, "step_ticks": 5,
        "cutoff_stride": cutoff_stride, "first_cutoff": (history_steps - 1) * 5,
        "history_features": list(HISTORY_FEATURE_NAMES), "event_names": list(EVENT_NAMES),
        "event_support_counts": event_counts,
        "selection_policy": "fixed static pair, regular cutoffs, complete observed prefix; future rows never select inputs",
        "label_policy": "first opposite on step-five grid, censor at first unknown/out_of_scope/missing candidate; zero observed bins not counted",
        "source_root": SOURCE_ROOT, "motion_root": MOTION_ROOT,
        "latent_domain": "one B(actor,t)=speed>episode_tau; I(actor,corridor,t)=source closed oriented box membership",
        "predicates": "O=I; M=I&B; S=I&notB; order O1,M1,S1,O2,M2,S2",
        "geometry_boundary": "abs(local)<=halfextent+1e-9 ENU metres, yaw in degrees",
        "claim": "bounded four-family feasibility; observation-prefix censoring may be informative",
    }
    return windows, manifest


def _prefix_components(windows: Sequence[CorridorWindow], split_by_family: Mapping[str, str]):
    families = sorted({window.prediction.family for window in windows})
    parent = {family: family for family in families}
    def find(family):
        while parent[family] != family:
            parent[family] = parent[parent[family]]
            family = parent[family]
        return family
    seen, duplicate_pairs = {}, []
    for window in windows:
        query = window.prediction
        # Raw numerical equality ignores actor strings, seed labels and family IDs.
        key = tuple(query.motion_history.ravel())
        if key in seen and seen[key].prediction.family != query.family:
            other = seen[key].prediction
            left, right = find(query.family), find(other.family)
            parent[max(left, right)] = min(left, right)
            duplicate_pairs.append([other.window_id, query.window_id])
        else:
            seen[key] = window
    components = {family: find(family) for family in families}
    for component in set(components.values()):
        splits = {split_by_family[family] for family in families if components[family] == component}
        if len(splits) != 1:
            raise GroundingError(f"exact shared prefix component {component} crosses frozen splits")
    return components, duplicate_pairs
