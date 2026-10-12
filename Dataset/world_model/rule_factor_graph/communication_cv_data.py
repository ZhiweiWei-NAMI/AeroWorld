"""Source-traced communication windows and fixed six-component retrospective CV.

Motion is read from the existing formal EntityState Parquet extraction. The
communication adapter remains the authority for observation, feature and label
semantics. This module never writes the original corpus or historical results.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from types import MappingProxyType
import argparse
import json
from pathlib import Path
import pickle
import re
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq

from Dataset.semantic_truth.formal_clock import FORMAL_TICK_HZ, validate_formal_frame_clock
from .communication_data import (
    HISTORY_FEATURE_NAMES, MOTION_ROOT, PREDICATES, PROFILE_PATH, SOURCE_ROOT,
    CommunicationPredictionInput, CommunicationSupervision, CommunicationWindow,
    _features, _observation, _profile,
)
from .contracts import GroundingError


RUN_ROOT = Path("Dataset/world_model/runs/communication_component_cv")
GRAPH_ROOT = Path("Dataset/world_model/runs/belief_label_graph/formal")
CACHE_SCHEMA = "communication_component_cv_data_v1"
EXPECTED_GROUP_COUNTS = {"G00": (33, 180, 34712), "G01": (1, 3, 291),
                         "G02": (1, 6, 897), "G03": (2, 12, 1928),
                         "G04": (1, 6, 723), "G05": (1, 3, 475)}


@dataclass(frozen=True)
class ReceiverNumericField:
    field_id: str
    artifact: str
    field_path: str
    data_type: str
    unit: str
    owner_keys: tuple[str, ...]
    coordinate_keys: tuple[str, ...]
    predicate_ids: tuple[str, ...]
    source_refs: tuple[str, ...]


def receiver_numeric_fields(predicate_ids=PREDICATES):
    """Discover published numeric operands from executable direct contracts."""
    from .state_dependencies import direct_monitor_fields
    from .unified_contracts import load_unified_contracts
    contracts, fields = load_unified_contracts(), {}
    for pid in predicate_ids:
        _, nodes = direct_monitor_fields(contracts[pid])
        for node in nodes:
            cell = node.source_cell
            if cell.form != "published_monitor":
                continue
            if (node.data_type != "numeric" or node.unit_status != "established"
                    or node.owner_status != "established" or node.owner_keys != ("actor", "station")):
                raise GroundingError("receiver field lacks established numeric/unit/actor-station authority")
            key = cell.artifact, cell.field_path
            spec = ReceiverNumericField(node.node_id, cell.artifact, cell.field_path, node.data_type,
                node.unit, node.owner_keys, cell.coordinate_keys, (pid,), node.provenance)
            if key in fields:
                old = fields[key]
                if replace(old, predicate_ids=(pid,)) != spec:
                    raise GroundingError("predicates disagree on a shared receiver field")
                spec = replace(old, predicate_ids=old.predicate_ids + (pid,))
            fields[key] = spec
    if not fields:
        raise GroundingError("no receiver numeric fields were discovered")
    return tuple(fields[key] for key in sorted(fields))


def _receiver_array(values, dtype):
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ReceiverNumericRecord:
    key: tuple[str, int, str, str, str]
    values: np.ndarray
    known: np.ndarray
    missing: tuple[str | None, ...]
    owner_types: tuple[tuple[str, str, str], ...]
    station_authority: str
    observation_ref: str
    source_refs: tuple[str, ...]
    requirement_status: str


@dataclass(frozen=True)
class ReceiverNumericSequence:
    window_id: str
    episode_id: str
    component: str
    split: str
    ticks: tuple[int, ...]
    owner: tuple[str, str]
    flow_id: str
    field_specs: tuple[ReceiverNumericField, ...]
    values: np.ndarray
    known: np.ndarray
    owner_types: tuple[tuple[str, str, str], ...]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class ReceiverSequenceSelection:
    sequence: ReceiverNumericSequence | None
    exclusion_reason: str | None
    missing_ticks: tuple[int, ...] = ()


def _unique_receiver_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GroundingError(f"receiver JSON duplicates key {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ReceiverNumericRecordBank:
    """Immutable owner/time records; prefix selection never examines futures."""
    field_specs: tuple[ReceiverNumericField, ...]
    records: object
    absent: object
    source_authority: tuple[str, ...]

    @classmethod
    def from_source_dictionaries(cls, episode_id, observations, roster, source_rows,
                                 *, communication_source, roster_source, field_specs=None):
        """Validate one episode against physical-line source rows and roster."""
        from Dataset.semantic_simulation.compute_comm import _is_station_entity, _simulated_station_from_facility_anchor
        from Dataset.world_model.graph.observation_logs import DIAGNOSTIC_FIELDS, _record
        from .state_dependencies import communication_row_actor_category
        specs = receiver_numeric_fields() if field_specs is None else tuple(field_specs)
        predicate_ids = tuple(dict.fromkeys(pid for spec in specs for pid in spec.predicate_ids))
        if specs != receiver_numeric_fields(predicate_ids):
            raise GroundingError("supplied receiver field metadata differs from executable authority")
        if any(spec.artifact != "communication_state.jsonl" or DIAGNOSTIC_FIELDS[spec.field_path] != spec.unit for spec in specs):
            raise GroundingError("receiver publisher artifact/field/unit differs from direct authority")
        entities = {}
        for entity in roster["entities"]:
            identity = entity["entity_id"]
            if not isinstance(identity, str) or not identity or identity in entities:
                raise GroundingError("roster entity identity is absent or duplicated")
            entities[identity] = entity
        stations = {key: (entity, f"{roster_source}#entity={key}")
                    for key, entity in entities.items() if _is_station_entity(entity)}
        if not stations:
            for entity in _simulated_station_from_facility_anchor(entities):
                stations[entity["entity_id"]] = (entity,
                    f"{roster_source}#_simulated_station_from_facility_anchor")
        records, absent, owners = {}, {}, {}
        actor_category = communication_row_actor_category()
        for index, observation in enumerate(observations):
            if (observation["kind"], observation["channel"], observation["receiver_kind"]) != (
                    "gateway_diagnostic", "communication", "local_gateway_monitor"):
                raise GroundingError("receiver bank accepts gateway diagnostic rows only")
            tick = observation["sample_tick"]
            if type(tick) is not int or tick < 0 or tick % 5 or tick != observation["availability_tick"]:
                raise GroundingError("receiver sample/availability clock differs from the five-tick grid")
            parsed = {name: json.loads(observation[name + "_json"], object_pairs_hook=_unique_receiver_json)
                      for name in ("references", "values", "missing", "source_refs")}
            refs, values, missing, sources = (parsed[name] for name in ("references", "values", "missing", "source_refs"))
            if not isinstance(refs, dict) or any(not isinstance(ids, list) or len(ids) != 1
                    or type(ids[0]) is not str or not ids[0] or ids[0] in ("unknown", "none") for ids in refs.values()):
                raise GroundingError("receiver references must be exact unique singleton identities")
            if not isinstance(values, dict) or not isinstance(missing, dict) or set(values) != set(DIAGNOSTIC_FIELDS):
                raise GroundingError("receiver diagnostic field group differs from the publisher")
            if (not isinstance(sources, list) or not sources or any(not isinstance(ref, dict)
                    or set(ref) != {"path", "line"} or type(ref["path"]) is not str or not ref["path"]
                    or type(ref["line"]) is not int or ref["line"] < 1 for ref in sources)):
                raise GroundingError("receiver source references require exact path and physical line")
            actor = refs["subject"][0]
            if actor not in entities or entities[actor]["entity_category"] != actor_category:
                raise GroundingError("receiver actor differs from the producer's roster category")
            observation_ref = f"{episode_id}/gateway_diagnostic[{index}]/{observation['observation_id']}"
            if set(refs) == {"subject"}:
                raise GroundingError(
                    f"owner-less receiver row {episode_id}/{tick}/{actor}: "
                    "formal communication_source is always declared; "
                    "an owner-less absent-diagnostic receiver row is unproducible and carries no evidence authority"
                )
            if not {"subject", "station", "flow"} <= set(refs):
                raise GroundingError("receiver row lacks exact actor/station/flow query identity")
            station, flow = refs["station"][0], refs["flow"][0]
            if station not in stations:
                raise GroundingError("receiver station lacks roster or re-executed simulated station authority")
            if len(sources) != 1 or sources[0]["path"] != communication_source:
                raise GroundingError("receiver source artifact differs from the episode authority")
            raw = source_rows[sources[0]["line"]]
            if (raw["episode_id"], raw["tick"], raw["entity_id"], raw["station_id"], raw["flow_id"]) != (
                    episode_id, tick, actor, station, flow):
                raise GroundingError("receiver source row episode/time/owner/query-role drift")
            expected_refs = {"subject": [raw["entity_id"]]}
            for field, role in (("station_id", "station"), ("session_id", "session"), ("flow_id", "flow"),
                                ("retransmission_id", "retransmission_monitor"), ("handover_id", "handover_monitor")):
                if raw[field] not in (None, "unknown", "none"):
                    expected_refs[role] = [raw[field]]
            status = raw["requirement"]["status"]
            if status not in ("source_required", "not_required"):
                raise GroundingError("receiver source requirement status drift")
            replay = _record("gateway_diagnostic", tick, expected_refs, raw, DIAGNOSTIC_FIELDS,
                sources, missing_reason="source_required" if status == "source_required" else None)
            if any(parsed[name] != replay[name] for name in parsed) or observation["observation_id"] != replay["observation_id"]:
                raise GroundingError("receiver values/missing/references differ from publisher replay")
            vector, known, reasons = [], [], []
            for spec in specs:
                value = values[spec.field_path]
                reason = missing.get(spec.field_path)
                if value is None:
                    if reason is None:
                        raise GroundingError("receiver missing numeric value lacks an explicit reason")
                    vector.append(np.nan); known.append(False)
                else:
                    if type(value) not in (int, float) or not np.isfinite(float(value)) or reason is not None:
                        raise GroundingError("receiver numeric value is nonnumeric, overflowing or inconsistent with missingness")
                    vector.append(float(value)); known.append(True)
                reasons.append(reason)
            key = episode_id, tick, actor, station, flow
            owner_time = episode_id, tick, actor
            if owner_time in owners and owners[owner_time] != (station, flow):
                raise GroundingError("receiver actor/time has conflicting station or flow identities")
            if owner_time in absent:
                raise GroundingError("receiver actor/time is both absent and published")
            owners[owner_time] = station, flow
            entity, station_authority = stations[station]
            record = ReceiverNumericRecord(key, _receiver_array(vector, np.float64), _receiver_array(known, bool),
                tuple(reasons), (("actor", actor, actor_category), ("station", station, entity["entity_category"])),
                station_authority, observation_ref, tuple(f"{ref['path']}#line={ref['line']}" for ref in sources), status)
            if key in records:
                old = records[key]
                if (not np.array_equal(old.values, record.values, equal_nan=True) or old.missing != record.missing
                        or old.source_refs != record.source_refs or old.requirement_status != record.requirement_status):
                    raise GroundingError("conflicting duplicate receiver numeric records")
            else:
                records[key] = record
        if set(absent) & set(owners):
            raise GroundingError("receiver actor/time is both absent and published")
        return cls(specs, MappingProxyType(records), MappingProxyType(absent), (communication_source, roster_source))

    def receiver_prefix(self, window):
        q = window.prediction
        if len(q.prefix) != 8 or tuple(obs.tick for obs in q.prefix) != tuple(range(q.cutoff_tick - 35, q.cutoff_tick + 1, 5)):
            raise GroundingError("receiver query needs exactly eight cutoff-ending prefix rows")
        if q.window_id != f"{q.episode_id}|{q.owner.owner_id}|cutoff={q.cutoff_tick}":
            raise GroundingError("receiver query window identity differs from episode/owner/cutoff")
        if any(obs.owner != q.owner for obs in q.prefix):
            raise GroundingError("historical prefix owner changed")
        selected = self._sequence(q, tuple(obs.tick for obs in q.prefix))
        if selected.sequence is None:
            return selected
        expected_fields = {"link_quality.latency_ms": "latency_ms", "link_quality.packet_loss_ratio": "packet_loss_ratio"}
        for column, spec in enumerate(self.field_specs):
            if spec.field_path in expected_fields:
                expected = np.asarray([getattr(obs, expected_fields[spec.field_path]) for obs in q.prefix], dtype=np.float64)
                if not np.array_equal(selected.sequence.values[:, column], expected):
                    raise GroundingError("receiver numeric prefix differs from historical cutoff inputs")
        return selected

    def donor_sequence(self, window):
        """Read receiver futures only for an explicitly TRAIN donor candidate."""
        q = window.prediction
        if q.split != "train":
            raise GroundingError("receiver donor sequence requires a TRAIN-only window")
        prefix = self.receiver_prefix(window)
        if prefix.sequence is None:
            return prefix
        if q.ticks != tuple(range(q.cutoff_tick, q.cutoff_tick + 51, 5)):
            raise GroundingError("receiver donor requires anchor plus ten future grid ticks")
        return self._sequence(q, prefix.sequence.ticks + q.ticks[1:])

    def _sequence(self, query, ticks):
        keys = tuple((query.episode_id, tick, query.owner.actor_id, query.owner.station_id, query.owner.flow_id) for tick in ticks)
        absent = tuple(key[1] for key in keys if key not in self.records)
        if absent:
            return ReceiverSequenceSelection(None, "receiver_record_absent_or_owner_changed", absent)
        records = tuple(self.records[key] for key in keys)
        missing = tuple(record.key[1] for record in records if not record.known.all())
        if missing:
            return ReceiverSequenceSelection(None, "receiver_numeric_field_missing", missing)
        values = _receiver_array([record.values for record in records], np.float64)
        known = _receiver_array([record.known for record in records], bool)
        refs = tuple(dict.fromkeys(ref for record in records for ref in (*record.source_refs, record.station_authority)))
        sequence = ReceiverNumericSequence(query.window_id, query.episode_id, query.component, query.split, ticks,
            (query.owner.actor_id, query.owner.station_id), query.owner.flow_id, self.field_specs,
            values, known, records[0].owner_types, refs)
        return ReceiverSequenceSelection(sequence, None)


def load_receiver_numeric_bank(repo_root, episode_ids, *, predicate_ids=PREDICATES):
    """Read immutable formal receiver observations and replay their sources."""
    root, specs = Path(repo_root), receiver_numeric_fields(predicate_ids)
    records, absent, authority = {}, {}, []
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise GroundingError("receiver bank requires unique explicit episode identities")
    for episode in episode_ids:
        scope = pq.read_table(root / GRAPH_ROOT / episode / "scope.parquet",
            columns=["source_kind", "episode_id", "source_root"]).to_pylist()
        if len(scope) != 1 or (scope[0]["source_kind"], scope[0]["episode_id"]) != ("formal", episode):
            raise GroundingError("receiver episode differs from its formal source scope")
        artifacts = {spec.artifact for spec in specs}
        if len(artifacts) != 1:
            raise GroundingError("gateway numeric fields must share one publisher artifact")
        source = str(Path(scope[0]["source_root"]) / next(iter(artifacts)))
        with (root / source).open() as stream:
            source_rows = {line: json.loads(text, object_pairs_hook=_unique_receiver_json) for line, text in enumerate(stream, 1)}
        roster_source = f"{MOTION_ROOT}/{episode}/global_entity_roster.json"
        roster = json.loads((root / roster_source).read_text(), object_pairs_hook=_unique_receiver_json)
        path = root / "Dataset/world_model/runs/belief_observation_logs/formal" / episode / "observations.parquet"
        observations = pq.read_table(path, filters=[("kind", "=", "gateway_diagnostic")]).to_pylist()
        bank = ReceiverNumericRecordBank.from_source_dictionaries(episode, observations, roster, source_rows,
            communication_source=source, roster_source=roster_source, field_specs=specs)
        records.update(bank.records); absent.update(bank.absent); authority.extend(bank.source_authority)
    return ReceiverNumericRecordBank(specs, MappingProxyType(records), MappingProxyType(absent), tuple(authority))


def family_id(episode_id: str) -> str:
    return re.sub(r"_v\d+$", "", episode_id.split("__", 1)[0])


def window_support(window: CommunicationWindow) -> dict:
    """Compatible 11-way event sets and coverage; neither creates a false label."""
    masks = np.zeros((2, 11), dtype=bool)
    events = window.supervision.event_constraints()
    for column, name in enumerate(("L", "P")):
        event = events[name]
        if event.kind == "first_flip":
            masks[column, event.step - 1] = True
        elif event.kind == "none":
            masks[column, -1] = True
        else:
            masks[column, event.step:] = True
    informative = not bool(masks.all())
    return {"informative": informative,
            "complete_future_observation": bool(window.supervision.known[1:].all()),
            "complete_future_joint_state": bool((window.supervision.joint_state[1:] >= 0).all()),
            "compatible_event_masks": masks,
            "event_constraints": events}


def with_heldout_group(windows: Sequence[CommunicationWindow],
                       heldout_group: str) -> list[CommunicationWindow]:
    groups = {window.prediction.component for window in windows}
    if groups != set(EXPECTED_GROUP_COUNTS) or heldout_group not in groups:
        raise GroundingError("fold assignment requires the verified six G00..G05 components")
    return [CommunicationWindow(replace(window.prediction,
             split="test" if window.prediction.component == heldout_group else "train"),
             window.supervision) for window in windows]


def _motion_from_parquet(root: Path, episode: str, actors: set[str]):
    path = GRAPH_ROOT / episode / "nodes.parquet"
    expected_source = f"{MOTION_ROOT}/{episode}/truth_frames.jsonl"
    table = pq.read_table(root / path,
        columns=["sample_key", "category_id", "tick", "attrs_json", "source_file", "source_line"],
        filters=[("node_type", "=", "EntityState"),
                 ("category_id", "in", ["uav", "facility"]),
                 ("tick", "in", list(range(0, 901, 5)))])
    motion, stations, categories, replay = {}, {}, {}, {}
    for row in table.to_pylist():
        if row["sample_key"] != f"formal:{episode}" or row["source_file"] != expected_source:
            raise GroundingError(f"Parquet motion source identity differs: {path}")
        tick, line = row["tick"], row["source_line"]
        if type(line) is not int or line != tick + 1:
            raise GroundingError(f"formal motion source line differs from clock: {path}:{tick}")
        attrs = json.loads(row["attrs_json"])
        raw = attrs["raw"]
        entity, category = raw["entity_id"], raw["entity_category"]
        if category != row["category_id"] or category != attrs["entity_category"]:
            raise GroundingError(f"Parquet motion category differs: {path}:{tick}:{entity}")
        if entity in categories and categories[entity] != category:
            raise GroundingError(f"motion actor changed category: {path}:{entity}")
        categories[entity] = category
        if entity not in actors and category != "facility":
            continue
        pose = raw["truth_pose"]
        state = np.asarray(pose["position_enu_m"] + pose["velocity_enu_mps"], dtype=np.float64)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise GroundingError(f"nonfinite six-dimensional source motion: {path}:{tick}:{entity}")
        key = tick, entity
        if key in motion:
            raise GroundingError(f"duplicate motion row: {path}:{tick}:{entity}")
        ref = f"{expected_source}#line={line}#entity={entity}"
        motion[key] = state, ref
        if category == "facility":
            stations[key] = state[:3], ref
        if "uav_global_flow" in raw:
            flow, segment = raw["uav_global_flow"], raw["uav_segment"]
            required_flow = ("policy", "uav_id", "task_id", "source_prev_time_s",
                             "source_next_time_s", "source_alpha")
            required_segment = ("seed_index", "seed_label", "segment_start_s", "segment_end_s")
            route = raw["planned_route_waypoints_enu_m"]
            if not route or not all(name in flow for name in required_flow):
                raise GroundingError(f"global replay record lacks route/source clock: {ref}")
            replay[key] = {"flow": {name: flow[name] for name in required_flow},
                           "segment": {name: segment[name] for name in required_segment},
                           "route": route, "source_ref": ref,
                           "parquet_ref": str(path) + "#EntityState=" + entity + "@" + str(tick)}
    return motion, stations, categories, replay, str(path)


def _read_episode(root: Path, episode: str, profile: dict, nominal_range: float):
    family = family_id(episode)
    source_path = f"{SOURCE_ROOT}/{episode}/communication_state.jsonl"
    clock_source_path = f"{MOTION_ROOT}/{episode}/truth_frames.jsonl"
    with (root / clock_source_path).open() as stream:
        clock_frame = json.loads(next(stream))
    if clock_frame["episode_id"] != episode:
        raise GroundingError("formal clock frame episode identity differs")
    tick_hz = validate_formal_frame_clock(clock_frame, expected_tick=0,
                                         source=clock_source_path + ":1")
    sim_manifest_path = f"{SOURCE_ROOT}/{episode}/simulation_manifest.json"
    manifest = json.loads((root / sim_manifest_path).read_text())
    if (manifest["episode_id"] != episode or manifest["profile_id"] != profile["profile_id"]
            or manifest["model_version"] != profile["model"]["model_version"]):
        raise GroundingError("communication manifest differs from the fixed source profile")
    communication, actors = {}, set()
    with (root / source_path).open() as stream:
        for line, text in enumerate(stream, 1):
            row = json.loads(text)
            if row["episode_id"] != episode:
                raise GroundingError("communication source episode identity changed")
            obs = _observation(row, source_path, line)
            if (obs.latency_threshold_ms, obs.packet_loss_threshold) != (120.0, 0.05):
                raise GroundingError("communication threshold differs from the fixed finite model")
            key = obs.tick, obs.owner.actor_id
            if key in communication:
                raise GroundingError("duplicate actor communication record at a tick")
            communication[key] = obs
            actors.add(obs.owner.actor_id)
    motion, stations, categories, replay, parquet_path = _motion_from_parquet(root, episode, actors)
    labels_path = f"{SOURCE_ROOT}/{episode}/predicate_truth.jsonl"
    labels = {}
    checked_labels = 0
    with (root / labels_path).open() as stream:
        for line, text in enumerate(stream, 1):
            # Other predicate families are not CV inputs or supervision.
            if not any(predicate in text for predicate in PREDICATES):
                continue
            row = json.loads(text)
            if row["predicate_id"] not in PREDICATES:
                continue
            if (row["episode_id"] != episode or row["bindings"] != {"flow": row["event_participants"]["flow"]}
                    or row["binding_ontology_classes"] != {"flow": "world:DataFlow"}
                    or (row["schema_name"], row["schema_version"], row["rule_version"]) !=
                    ("compute_comm_predicate_truth", "2.0.0", "1.0.0")
                    or row["rule_id"] != "compute_comm_supplement." + row["predicate_id"]):
                raise GroundingError(f"saved communication label contract differs: {labels_path}:{line}")
            key = row["tick"], row["bindings"]["flow"], row["predicate_id"]
            if key in labels:
                raise GroundingError("duplicate saved communication predicate")
            labels[key] = (row["value"], f"{labels_path}#line={line}",
                           row["event_participants"]["actor"], row["event_participants"]["station"])
            obs = communication.get((row["tick"], row["event_participants"]["actor"]))
            if obs is None or (obs.owner.station_id, obs.owner.flow_id) != (
                    row["event_participants"]["station"], row["bindings"]["flow"]):
                raise GroundingError(f"saved label owner lacks its exact communication source: {labels_path}:{line}")
            column = PREDICATES.index(row["predicate_id"])
            computed = obs.predicates[column]
            expected = "unknown" if computed is None else ("true" if computed else "false")
            if row["value"] != expected:
                raise GroundingError(f"saved label differs from threshold replay: {labels_path}:{line}")
            checked_labels += 1
    if checked_labels != 2 * len(communication):
        raise GroundingError(f"communication records lack two saved LP labels: {episode}")
    excluded = Counter()
    windows, window_replay, missing_reasons = [], {}, []
    for actor in sorted(actors):
        if actor in categories and categories[actor] != "uav":
            excluded["unsupported_actor_type"] += 1
            continue
        for cutoff in range(40, 851, 25):
            history_ticks = tuple(range(cutoff - 35, cutoff + 1, 5))
            prefix = tuple(communication.get((tick, actor)) for tick in history_ticks)
            if any(obs is None for obs in prefix):
                excluded["missing_prefix_communication"] += 1
                continue
            owner = prefix[-1].owner
            if any(obs.owner != owner for obs in prefix):
                excluded["changed_prefix_owner"] += 1
                continue
            station_actor = owner.station_id
            synthetic = "simulated_communication_station:"
            if station_actor.startswith(synthetic):
                station_actor = station_actor[len(synthetic):]
            station = stations.get((cutoff, station_actor))
            if station is None:
                excluded["missing_prefix_station"] += 1
                continue
            station_position, station_ref = station
            history_motion = tuple(motion.get((tick, actor)) for tick in history_ticks)
            if any(item is None for item in history_motion):
                excluded["missing_prefix_motion"] += 1
                continue
            feature_rows = tuple(_features(obs, item[0], station_position, nominal_range)
                                 for obs, item in zip(prefix, history_motion))
            if any(item is None for item in feature_rows) or prefix[-1].joint_state is None:
                excluded["incomplete_prefix_state"] += 1
                continue
            ticks = tuple(cutoff + step * 5 for step in range(11))
            prediction = CommunicationPredictionInput(
                window_id=f"{episode}|{owner.owner_id}|cutoff={cutoff}", episode_id=episode,
                family=family, component=family, split="train", cutoff_tick=cutoff, ticks=ticks,
                owner=owner, prefix=prefix, history=np.stack(feature_rows),
                motion_history=np.stack([item[0] for item in history_motion])[:, None],
                station_position=station_position, nominal_range_m=nominal_range,
                tick_hz=tick_hz, initial_predicates=prefix[-1].predicates,
                initial_joint_state=prefix[-1].joint_state, profile_id=profile["profile_id"],
                profile_source_path=PROFILE_PATH,
                motion_source_refs=tuple(item[1] for item in history_motion), station_source_ref=station_ref)
            y, a, joint = np.full((11, 2), -1, dtype=np.int64), np.full(11, -1, dtype=np.int64), np.full(11, -1, dtype=np.int64)
            source_refs, label_refs = [], []
            # 0 observed, 1 record missing, 2 exact actor changed station/flow.
            reason = np.zeros(10, dtype=np.uint8)
            for step, tick in enumerate(ticks):
                obs = communication.get((tick, actor))
                if obs is None or obs.owner != owner:
                    if step:
                        reason[step - 1] = 1 if obs is None else 2
                    continue
                source_refs.append(f"{obs.source_path}#line={obs.source_line}")
                for column, predicate in enumerate(PREDICATES):
                    saved = labels.get((tick, owner.flow_id, predicate))
                    if saved is None:
                        raise GroundingError(f"missing verified communication predicate: {episode}:{tick}:{owner.flow_id}")
                    value, ref, label_actor, label_station = saved
                    if (label_actor, label_station) != (owner.actor_id, owner.station_id):
                        raise GroundingError(f"saved label owner differs at {ref}")
                    label_refs.append(ref)
                    if value in ("true", "false"):
                        y[step, column] = int(value == "true")
                    elif value != "unknown":
                        raise GroundingError(f"invalid saved communication predicate value: {value}")
                if obs.availability_bin is not None:
                    a[step] = obs.availability_bin
                if obs.joint_state is not None:
                    joint[step] = obs.joint_state
            future_motion = np.full((10, 1, 6), np.nan)
            motion_known = np.zeros((10, 1), dtype=bool)
            future_refs = []
            for step, tick in enumerate(ticks[1:]):
                item = motion.get((tick, actor))
                if item is not None:
                    future_motion[step, 0] = item[0]
                    motion_known[step, 0] = True
                    future_refs.append(item[1])
            supervision = CommunicationSupervision(ticks, y, y >= 0, joint, a, a >= 0,
                future_motion, motion_known, tuple(source_refs), tuple(label_refs), tuple(future_refs))
            if not supervision.known[0].all():
                raise GroundingError("selected prefix lacks saved initial communication labels")
            windows.append(CommunicationWindow(prediction, supervision))
            missing_reasons.append(reason)
            if (cutoff, actor) in replay:
                window_replay[prediction.window_id] = replay[cutoff, actor]
    episode_info = {"communication_records": len(communication), "checked_lp_labels": checked_labels,
                    "actors": len(actors), "windows": len(windows), "source_manifest": sim_manifest_path,
                    "motion_parquet": parquet_path, "clock_source_ref": clock_source_path + "#line=1",
                    "tick_hz": tick_hz}
    return windows, episode_info, excluded, window_replay, missing_reasons


class _Components:
    def __init__(self, families):
        self.parent = {family: family for family in families}

    def find(self, family):
        while self.parent[family] != family:
            self.parent[family] = self.parent[self.parent[family]]
            family = self.parent[family]
        return family

    def connect(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)

    def groups(self):
        groups = {}
        for family in sorted(self.parent):
            groups.setdefault(self.find(family), []).append(family)
        return sorted(groups.values())


def _derive_groups(windows, replay, evidence_path: Path):
    families = sorted({window.prediction.family for window in windows})
    exact, combined = _Components(families), _Components(families)
    seen_full, seen_replay = {}, {}
    counts = Counter()
    with evidence_path.open("w") as stream:
        for window in windows:
            q = window.prediction
            key = (tuple(q.history.ravel()), tuple(q.motion_history.ravel()), tuple(q.station_position),
                   q.initial_joint_state, q.prefix[-1].latency_threshold_ms, q.prefix[-1].packet_loss_threshold)
            earlier = seen_full.get(key)
            if earlier is None:
                seen_full[key] = q
            elif earlier.family != q.family:
                exact.connect(earlier.family, q.family)
                combined.connect(earlier.family, q.family)
                counts["full_input_cross_family_matches"] += 1
                stream.write(json.dumps({"kind": "exact_full_prediction_input", "left_window_id": earlier.window_id,
                    "right_window_id": q.window_id, "left_family": earlier.family, "right_family": q.family,
                    "left_communication_refs": [f"{o.source_path}#line={o.source_line}" for o in earlier.prefix],
                    "right_communication_refs": [f"{o.source_path}#line={o.source_line}" for o in q.prefix],
                    "left_motion_refs": earlier.motion_source_refs, "right_motion_refs": q.motion_source_refs}, separators=(",", ":")) + "\n")
            stationary = bool(np.all(q.motion_history[:, 0, :3] == q.motion_history[0, 0, :3])
                              and np.all(q.motion_history[:, 0, 3:] == 0))
            if stationary:
                counts["stationary_windows_excluded_actor_replay_links"] += 1
                continue
            provenance = replay.get(q.window_id)
            if provenance is None:
                counts["moving_windows_without_global_replay_declaration"] += 1
                continue
            flow = provenance["flow"]
            key = (q.owner.actor_id, flow["policy"], flow["uav_id"], flow["task_id"],
                   tuple(provenance["segment"].items()), tuple(tuple(p) for p in provenance["route"]),
                   q.cutoff_tick, tuple(q.motion_history.ravel()))
            earlier = seen_replay.get(key)
            if earlier is None:
                seen_replay[key] = q
            elif earlier.family != q.family:
                combined.connect(earlier.family, q.family)
                counts["moving_actor_replay_cross_family_matches"] += 1
                stream.write(json.dumps({"kind": "same_actor_replay_and_exact_motion_prefix",
                    "left_window_id": earlier.window_id, "right_window_id": q.window_id,
                    "left_family": earlier.family, "right_family": q.family,
                    "left_motion_refs": earlier.motion_source_refs, "right_motion_refs": q.motion_source_refs,
                    "left_replay": replay[earlier.window_id], "right_replay": provenance}, separators=(",", ":")) + "\n")
    exact_groups, groups = exact.groups(), combined.groups()
    expected_small = {"G01": ["L1-2"], "G02": ["L2-1"], "G03": ["L2-3", "L2-4"],
                      "G04": ["L3-3"], "G05": ["L5-5"]}
    small_families = {family for group in expected_small.values() for family in group}
    expected = {"G00": sorted(set(families) - small_families), **expected_small}
    if sorted(groups) != sorted(expected.values()) or len(exact_groups) != 33:
        raise GroundingError(f"source-derived components differ from the approved six groups: {groups}; exact={len(exact_groups)}")
    mapping = {family: name for name, group in expected.items() for family in group}
    return mapping, exact_groups, counts


def build_communication_cv_cache(repo_root: Path, output_root: Path | None = None,
                                 groups_path: Path | None = None):
    """Build once, without replacing old evidence or relying on cached counts."""
    root = Path(repo_root)
    output = root / RUN_ROOT / "data" if output_root is None else Path(output_root)
    groups_file = root / RUN_ROOT / "groups.json" if groups_path is None else Path(groups_path)
    if any((output / name).exists() for name in ("windows.pkl", "manifest.json", "support.npz", "component_links.jsonl")) or groups_file.exists():
        raise FileExistsError("CV data products already exist; inspect them before an explicit rebuild")
    output.mkdir(parents=True, exist_ok=True)
    profile, nominal_range = _profile(root)
    episodes = sorted(path.name for path in (root / SOURCE_ROOT).iterdir()
                      if path.is_dir() and (path / "communication_state.jsonl").is_file())
    if len(episodes) != 210 or len({family_id(ep) for ep in episodes}) != 39:
        raise GroundingError("current formal corpus differs from 210 episodes / 39 families")
    windows, episode_info, replay, reasons = [], {}, {}, []
    exclusions = Counter()
    for index, episode in enumerate(episodes):
        found, info, excluded, provenance, missing = _read_episode(root, episode, profile, nominal_range)
        windows.extend(found)
        episode_info[episode] = info
        replay.update(provenance)
        reasons.extend(missing)
        exclusions.update(excluded)
        if (index + 1) % 10 == 0:
            print(json.dumps({"episodes_loaded": index + 1, "windows": len(windows)}), flush=True)
    mapping, exact_groups, link_counts = _derive_groups(windows, replay, output / "component_links.jsonl")
    windows = [CommunicationWindow(replace(w.prediction, component=mapping[w.prediction.family]), w.supervision) for w in windows]
    support = [window_support(window) for window in windows]
    informative = np.array([item["informative"] for item in support], dtype=bool)
    complete = np.array([item["complete_future_observation"] for item in support], dtype=bool)
    complete_joint = np.array([item["complete_future_joint_state"] for item in support], dtype=bool)
    compatible = np.stack([item["compatible_event_masks"] for item in support])
    reason_array = np.stack(reasons)
    totals = {"episodes": len(episodes), "families": len(mapping), "windows": len(windows),
              "communication_records": sum(item["communication_records"] for item in episode_info.values()),
              "checked_lp_labels": sum(item["checked_lp_labels"] for item in episode_info.values()),
              "informative_windows": int(informative.sum()), "zero_information_windows": int((~informative).sum()),
              "complete_future_observation_windows": int(complete.sum()),
              "unknown_future_joint_slots": sum(int((w.supervision.joint_state[1:] < 0).sum()) for w in windows),
              "future_record_missing_slots": int((reason_array == 1).sum()),
              "future_owner_changed_slots": int((reason_array == 2).sum()),
              "known_future_joint_state_counts": [sum(int((w.supervision.joint_state[1:] == state).sum()) for w in windows) for state in range(6)]}
    required = {"windows": 39026, "communication_records": 226676, "checked_lp_labels": 453352,
                "informative_windows": 38586, "zero_information_windows": 440,
                "unknown_future_joint_slots": 19794, "future_record_missing_slots": 17017,
                "future_owner_changed_slots": 2777}
    if any(totals[name] != count for name, count in required.items()):
        raise GroundingError(f"current data extraction differs from the approved scan counts: {totals}")
    if totals["known_future_joint_state_counts"] != [351823, 6632, 7799, 161, 0, 4051]:
        raise GroundingError(f"observed six-state support differs: {totals['known_future_joint_state_counts']}")
    groups = {}
    for group, expected_count in EXPECTED_GROUP_COUNTS.items():
        indices = [i for i, w in enumerate(windows) if w.prediction.component == group]
        group_windows = [windows[i] for i in indices]
        group_families = sorted({w.prediction.family for w in group_windows})
        group_episodes = sorted({w.prediction.episode_id for w in group_windows})
        if (len(group_families), len(group_episodes), len(indices)) != expected_count:
            raise GroundingError(f"group count differs: {group}")
        groups[group] = {"families": group_families, "episodes": group_episodes,
            "family_count": len(group_families), "episode_count": len(group_episodes), "windows": len(indices),
            "informative_windows": int(informative[indices].sum()), "zero_information_windows": int((~informative[indices]).sum()),
            "complete_future_observation_windows": int(complete[indices].sum()),
            "first_flip_windows_L": sum(w.supervision.event_constraints()["L"].kind == "first_flip" for w in group_windows),
            "first_flip_windows_P": sum(w.supervision.event_constraints()["P"].kind == "first_flip" for w in group_windows),
            "heldout_fold_training_windows": len(windows) - len(indices),
            "heldout_fold_informative_training_windows": int(informative.sum() - informative[indices].sum())}
    grouping = {"schema": CACHE_SCHEMA, "groups": groups, "family_to_group": mapping,
        "exact_full_input_components": exact_groups, "source_link_counts": dict(link_counts),
        "component_link_evidence": str((output / "component_links.jsonl").relative_to(root)),
        "grouping_policy": "Versions/seeds remain within families; exact full predictor inputs and nonstationary same-actor/replay-policy/seed/source-time-segment/route/motion prefixes connect families.",
        "stationary_policy": "Zero positional span and all-zero velocity prefixes do not create actor replay links.",
        "holdout_status": "Retrospective exploratory CV. All six groups overlap prior pair training/validation/test usage; no new holdout.",
        "totals": totals}
    manifest = {"schema": CACHE_SCHEMA, "episodes": episode_info, "totals": totals,
        "groups_path": str(groups_file.relative_to(root)), "excluded_cutoffs": dict(exclusions),
        "source_root": SOURCE_ROOT, "motion_root": MOTION_ROOT, "motion_parquet_root": str(GRAPH_ROOT),
        "profile_source": PROFILE_PATH, "history_features": list(HISTORY_FEATURE_NAMES),
        "history_steps": 8, "future_steps": 10, "step_ticks": 5, "cutoff_stride": 25,
        "cutoff_ticks": list(range(40, 851, 25)), "tick_hz": FORMAL_TICK_HZ,
        "support_array_order": "Same as windows.pkl; future_reason: 0 observed, 1 record missing, 2 owner changed.",
        "selection_policy": "Regular cutoff grid and complete cutoff prefix only; future labels never select windows.",
        "nll_policy": "Joint compatible-event-set sum; exclude 440 all-events-compatible windows from training weights and NLL denominator; retain coverage.",
        "clock_source": "One real first frame per episode validated by Dataset/semantic_truth/formal_clock.py; EntityState tick/source_line=tick+1 validated throughout extraction.",
        "requested_worker_model": "gpt-6.1-sol", "requested_worker_effort": "xhigh",
        "effective_worker_model": None, "effective_configuration_evidence": "Not exposed by callable tools; requested configuration only."}
    with (output / "windows.pkl").open("wb") as stream:
        pickle.dump(windows, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(output / "support.npz", informative=informative,
                        complete_future_observation=complete, complete_future_joint_state=complete_joint,
                        compatible_event_masks=compatible, future_reason=reason_array)
    groups_file.parent.mkdir(parents=True, exist_ok=True)
    groups_file.write_text(json.dumps(grouping, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"totals": totals, "source_link_counts": dict(link_counts)}, indent=2), flush=True)
    return windows, manifest


def load_communication_cv_cache(repo_root: Path, cache_root: Path | None = None):
    """Load this module's trusted local cache; absent/drifting products fail."""
    root = Path(repo_root)
    cache = root / RUN_ROOT / "data" if cache_root is None else Path(cache_root)
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["schema"] != CACHE_SCHEMA:
        raise GroundingError("CV data cache schema changed")
    with (cache / "windows.pkl").open("rb") as stream:
        windows = pickle.load(stream)
    if len(windows) != manifest["totals"]["windows"] or len(windows) != 39026:
        raise GroundingError("CV cache window count differs from its source manifest")
    if len({window.prediction.window_id for window in windows}) != len(windows):
        raise GroundingError("CV cache contains duplicate window identity")
    grouping = json.loads((root / manifest["groups_path"]).read_text())
    if grouping["schema"] != CACHE_SCHEMA or grouping["totals"] != manifest["totals"]:
        raise GroundingError("CV cache and source grouping manifests disagree")
    for group, expected in EXPECTED_GROUP_COUNTS.items():
        selected = [window for window in windows if window.prediction.component == group]
        families = sorted({window.prediction.family for window in selected})
        episodes = sorted({window.prediction.episode_id for window in selected})
        if (len(families), len(episodes), len(selected)) != expected:
            raise GroundingError(f"CV cache component count differs: {group}")
        if grouping["groups"][group]["families"] != families or grouping["groups"][group]["episodes"] != episodes:
            raise GroundingError(f"CV cache component identity differs: {group}")
    if any(grouping["family_to_group"][window.prediction.family] != window.prediction.component
           for window in windows):
        raise GroundingError("CV cache family-to-component mapping differs")
    with np.load(cache / "support.npz", allow_pickle=False) as support:
        if support["future_reason"].shape != (39026, 10) or not np.isin(support["future_reason"], (0, 1, 2)).all():
            raise GroundingError("CV cache future missingness reason contract differs")
        required_reason_counts = ((support["future_reason"] == 1).sum(), (support["future_reason"] == 2).sum())
        if required_reason_counts != (17017, 2777):
            raise GroundingError("CV cache future missingness reason counts differ")
        recomputed = [window_support(window) for window in windows]
        for name in ("informative", "complete_future_observation", "complete_future_joint_state", "compatible_event_masks"):
            if not np.array_equal(support[name], np.array([item[name] for item in recomputed])):
                raise GroundingError(f"CV cache support metadata differs from saved real labels: {name}")
        if int(support["informative"].sum()) != 38586:
            raise GroundingError("CV cache informative-window count differs")
    for window in windows:
        for array in (window.prediction.history, window.prediction.motion_history, window.prediction.station_position,
                      window.supervision.predicates, window.supervision.known, window.supervision.joint_state,
                      window.supervision.availability_values, window.supervision.availability_known,
                      window.supervision.future_motion, window.supervision.future_motion_known):
            array.setflags(write=False)
    return windows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    build_communication_cv_cache(args.repo_root)


if __name__ == "__main__":
    main()
