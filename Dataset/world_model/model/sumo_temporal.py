"""Causal SUMO context, independent of the typed prediction/field registry.

Only a caller's *input* prefix is accepted; this module never opens an episode
or reads target records.  The canonical raw inputs are
``fields['archive.truth_frames']['sumo_vehicle']`` and
``fields['archive.scene.sumo_traffic_light_states']``.  Dotted projections of
those two paths are also accepted.  Roster entries supply identity aliases and
map references, NOT initial observations (rosters can describe later entrants).

Coordinate contract: the project's road-derived SUMO XY metres are truth ENU
XY metres, with identity transform and ground Z=0 for 2-D lane geometry.  This
is the upstream road_signal_context.py contract, not a general SUMO geodesic
transform.  SUMO positions are front-bumper centres, whereas truth vehicle
poses/center_lane_position_m are body centres; these references stay distinct.

``records`` contains an as-of boundary snapshot plus observed updates, including
``emit=False`` freshness refreshes.  Consumers should suppress those refreshes
from serialization, NOT from replay.  State fields contain numerical values at
every available sample and discrete values only on change.  ``freshness`` is a
value-free sidecar for unchanged observed fields.  Interval closures appear
only on the record observing the change/removal; older records are never
backfilled with a future endpoint.  Replay methods ignore records after their
query tick, including future removals and freshness updates.

Grouped canonical truth-frame rows are a complete snapshot of the *admitted
observed subset*, not the whole SUMO world.  Disappearance means observation
missing, not physical departure.  No observations are interpolated across
missing source ticks.  Following derivation uses only vehicles present in the
same tick: its leader/gap is explicitly observed-subset geometry, not a sensor
measurement or an assertion of the true global leader.
"""
from __future__ import annotations

from collections import ChainMap, Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET


__all__ = ["DYNAMIC_NUMERIC_FIELDS", "TemporalContext", "build_sumo_context"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_NET = (_PROJECT_ROOT / "Plugins/SumoImporter/Maps/donghu_road_topo"
                / "source/road_derived/road_derived.net.xml")
_MISSING = {"kind": "missing"}
_PRESENT = {"kind": "present", "value": True}
_COORDINATE_FRAME = {
    "source": "sumo_xy_m", "destination": "truth_world_enu_m",
    "xy_transform": "identity", "two_dimensional_lane_z_m": 0.0,
    "contract": "road_derived_network_identity_xy_not_general_sumo_projection",
}
# Deliberately exclude schedule/event metadata, source_next_time_s,
# next_switch_s, and vehicle controlling_signal_state.  Phases come from the
# scene TLS samples only, never from a controller's cached state or timetable.
_NUMERIC_VEHICLE = (
    "lane_position_m", "center_lane_position_m", "speed_mps", "accel_mps2",
    "allowed_speed_mps", "sumo_angle_deg",
)
_VECTOR_VEHICLE = (
    ("sumo_xy_m", "sumo.xy_m", 2),
    ("sumo_front_bumper_xy_m", "sumo.front_bumper_xy_m", 2),
    ("truth_front_bumper_enu_m", "sumo.truth_front_bumper_enu_m", 3),
)
# Per-tick vehicle numerics that duplicate the admitted same-tick pose (and the
# previous tick) to source precision.  Measured on all L4-3_v1__seed00 frames
# (32,428): |v| vs speed <= 6.9e-7, 90-yaw vs angle <= 6e-14, pose+L/2*heading
# vs front bumper <= 1e-6, lane_pos-L/2 vs centre <= 2e-14, prev+speed*dt vs
# lane position <= 1e-6, d|v|/dt vs accel <= 1e-5.  Each value is checked per
# record (``_reconstruction``) and serialized whenever the check fails; replay
# keeps every raw value in ``derived_fields``.
DYNAMIC_NUMERIC_FIELDS = frozenset({
    "sumo.lane_position_m", "sumo.center_lane_position_m", "sumo.speed_mps", "sumo.accel_mps2",
    "sumo.sumo_angle_deg", "sumo.xy_m", "sumo.front_bumper_xy_m", "sumo.truth_front_bumper_enu_m",
    "sumo.following_distance_m", "sumo.observed_following_distance_m",
})
_POSITION_TOLERANCE_M = 2e-6
_SPEED_TOLERANCE_MPS = 1e-6
_ANGLE_TOLERANCE_DEG = 1e-6
_ACCEL_TOLERANCE_MPS2 = 2e-5
_DERIVATION = "observed_same_lane_center_order"
_COVERAGE = "observed_subset_not_full_world"
_VEHICLE_INPUT_KEYS = frozenset(_NUMERIC_VEHICLE) | {
    "vehicle_id", "vehicle_type", "route_id", "sumo_lane_id", "lane_id",
    "sumo_edge_id", "lane_instance_source_ref", "dimensions_m", "signals",
    "sumo_position_reference", "truth_position_reference",
    "sumo_xy_m", "sumo_front_bumper_xy_m", "truth_front_bumper_enu_m",
    "leading_vehicle_id", "leader_id", "following_distance_m",
}


def _missing() -> dict:
    return dict(_MISSING)


def _is_missing(value: Any) -> bool:
    return isinstance(value, dict) and value.get("kind") == "missing"


def _string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _cell(value: Any) -> Any:
    """JSON-safe values; unavailable is explicit, never null/nonfinite."""
    if value is None:
        return _missing()
    if isinstance(value, float) and not math.isfinite(value):
        return _missing()
    if isinstance(value, Mapping):
        return {str(k): _cell(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cell(v) for v in value]
    return value


def _text_cell(value: Any) -> Any:
    return _string(value) or _missing()


def _numeric_cell(value: Any) -> Any:
    number = _number(value)
    return number if number is not None else _missing()


def _key(kind: str, bindings: Mapping[str, str]) -> str:
    # JSON escaping avoids ambiguous IDs when native names contain |, :, etc.
    return kind + ":" + json.dumps(dict(bindings), sort_keys=True,
                                  ensure_ascii=False, separators=(",", ":"))


def _source(record: Mapping[str, Any], path: str) -> dict:
    source = record.get("source") or {}
    return {"record_id": _cell(record.get("id")), "field_path": path,
            "path": _cell(source.get("path")), "line": _cell(source.get("line"))}


def _available(record: Mapping[str, Any], at: int) -> bool:
    """Mirror content_available_at at the observation tick, not the cutoff."""
    availability = ((record.get("provenance") or {}).get("available_time") or {})
    rule = availability.get("rule")
    if rule is None:
        return True
    if rule in ("capture_bound", "hindsight_truth"):
        tick = availability.get("tick", record.get("tick"))
        return type(tick) is int and tick <= at
    return False


def _vehicle_payload(fields: Mapping[str, Any], *, roster: bool = False) -> Any:
    family = "archive.global_entity_roster" if roster else "archive.truth_frames"
    dotted = fields.get(family + ".sumo_vehicle")
    if isinstance(dotted, Mapping):
        return dotted
    body = fields.get(family)
    return body.get("sumo_vehicle") if isinstance(body, Mapping) else None


def _shape(value: str | None) -> Any:
    if not value:
        return _missing()
    points = []
    for token in value.split():
        parts = [_number(p) for p in token.split(",")]
        if len(parts) not in (2, 3) or any(p is None for p in parts):
            raise ValueError(f"malformed SUMO lane shape: {token!r}")
        points.append(parts)
    return points if len(points) >= 2 else _missing()


def _static_node(identity: str, node_type: str, fields: dict, path: str,
                 version: str, fragment: str) -> dict:
    return {"identity": identity, "node_type": node_type, "fields": fields,
            "source_path": path, "source_ref": path + "#" + fragment,
            "map_version": version}


@lru_cache(maxsize=4)
def _parse_static(path: str, mtime_ns: int, size: int, digest: str,
                  contents: bytes) -> tuple[dict, dict]:
    """Cache by exact file version, including bytes hash even if stat repeats."""
    root = ET.fromstring(contents)
    if root.tag != "net":
        raise ValueError(f"not a SUMO <net> XML document: {path}")
    version = "sha256:" + digest
    nodes: dict[str, dict] = {}
    lanes: dict[str, dict] = {}
    edges: dict[str, list[str]] = {}
    lane_indices: dict[tuple[str, int], str] = {}
    for edge in root.findall("edge"):
        edge_id = _string(edge.get("id"))
        if edge_id is None:
            raise ValueError(f"SUMO edge without native id: {path}")
        if edge_id in nodes:
            raise ValueError(f"duplicate SUMO native id: {edge_id}")
        edge_lanes = []
        for lane in edge.findall("lane"):
            lane_id = _string(lane.get("id"))
            if lane_id is None or lane_id in nodes or lane_id == edge_id:
                raise ValueError(f"missing/duplicate SUMO lane id: {lane_id!r}")
            index = _integer(lane.get("index"))
            if index is not None:
                if (edge_id, index) in lane_indices:
                    raise ValueError(f"duplicate SUMO lane index: {edge_id}/{index}")
                lane_indices[edge_id, index] = lane_id
            shape = _shape(lane.get("shape"))
            enu_shape = ([[p[0], p[1], p[2] if len(p) == 3 else 0.0]
                          for p in shape] if isinstance(shape, list) else _missing())
            fields = {
                "sumo.node_type": "lane", "sumo.lane_id": lane_id,
                "sumo.edge_id": edge_id,
                "sumo.lane_index": index if index is not None else _missing(),
                "sumo.edge_function": edge.get("function", "normal"),
                "sumo.shape_xy_m": ([p[:2] for p in shape]
                                    if isinstance(shape, list) else _missing()),
                "sumo.shape_enu_m": enu_shape,
                "sumo.length_m": _numeric_cell(lane.get("length")),
                "sumo.speed_mps": _numeric_cell(lane.get("speed")),
                "sumo.width_m": _numeric_cell(lane.get("width")),
                "sumo.allow": _text_cell(lane.get("allow")),
                "sumo.disallow": _text_cell(lane.get("disallow")),
                "sumo.adjacent_lane_ids": [], "sumo.left_lane_ids": [],
                "sumo.right_lane_ids": [], "sumo.successor_lane_ids": [],
                "sumo.predecessor_lane_ids": [],
                "sumo.connected_destination_lane_ids": [],
                "sumo.connections": [], "sumo.signal_links": [],
                "sumo.coordinate_frame": deepcopy(_COORDINATE_FRAME),
            }
            nodes[lane_id] = _static_node(lane_id, "lane", fields, path,
                                          version, "lane=" + lane_id)
            lanes[lane_id] = fields
            edge_lanes.append(lane_id)
        edges[edge_id] = edge_lanes
        nodes[edge_id] = _static_node(edge_id, "edge", {
            "sumo.node_type": "edge", "sumo.edge_id": edge_id,
            "sumo.lane_ids": edge_lanes, "sumo.edge_function": edge.get("function", "normal"),
            "sumo.from_junction_id": _text_cell(edge.get("from")),
            "sumo.to_junction_id": _text_cell(edge.get("to")),
            "sumo.priority": _numeric_cell(edge.get("priority")),
        }, path, version, "edge=" + edge_id)
    for (edge_id, index), lane_id in lane_indices.items():
        fields = lanes[lane_id]
        # SUMO index 0 is rightmost; index adjacency is NOT permission to change.
        fields["sumo.left_lane_ids"] = ([lane_indices[edge_id, index + 1]]
                                        if (edge_id, index + 1) in lane_indices else [])
        fields["sumo.right_lane_ids"] = ([lane_indices[edge_id, index - 1]]
                                         if (edge_id, index - 1) in lane_indices else [])
        fields["sumo.adjacent_lane_ids"] = (fields["sumo.right_lane_ids"]
                                             + fields["sumo.left_lane_ids"])
    connections = []
    signals: dict[str, dict] = {}
    for item in root.findall("connection"):
        from_index, to_index = _integer(item.get("fromLane")), _integer(item.get("toLane"))
        from_lane = lane_indices.get((item.get("from"), from_index))
        to_lane = lane_indices.get((item.get("to"), to_index))
        if from_lane is None or to_lane is None:
            raise ValueError(f"SUMO connection references unknown lanes: {dict(item.attrib)}")
        via = (item.get("via") or "").split()
        if any(lane_id not in lanes for lane_id in via):
            raise ValueError(f"SUMO connection references unknown via lane: {via}")
        tls_id, link_index = _string(item.get("tl")), _integer(item.get("linkIndex"))
        connection = {
            "from_lane_id": from_lane, "to_lane_id": to_lane, "via_lane_ids": via,
            "tls_id": tls_id if tls_id is not None else _missing(),
            "link_index": link_index if link_index is not None else _missing(),
            "direction": _text_cell(item.get("dir")),
            "source_ref": path + f"#connection={from_lane}->{to_lane}",
        }
        connections.append(connection)
        lanes[from_lane]["sumo.connections"].append(connection)
        lanes[from_lane]["sumo.connected_destination_lane_ids"].append(to_lane)
        if tls_id is not None:
            signals.setdefault(tls_id, {"programs": [], "links": []})["links"].append(connection)
            lanes[from_lane]["sumo.signal_links"].append(connection)
    successors: dict[str, set[str]] = defaultdict(set)
    # Explicit connections own internal continuation (some turns have several
    # internal lanes).  A via->destination completion is used ONLY where that
    # via has no explicit outgoing connection in the XML.
    outgoing = {c["from_lane_id"] for c in connections}
    for connection in connections:
        chain = [connection["from_lane_id"]] + connection["via_lane_ids"]
        target = connection["to_lane_id"]
        if len(chain) == 1:
            successors[chain[0]].add(target)
        else:
            for a, b in zip(chain, chain[1:]):
                successors[a].add(b)
            if chain[-1] not in outgoing:
                successors[chain[-1]].add(target)
    predecessors: dict[str, set[str]] = defaultdict(set)
    for lane_id, targets in successors.items():
        for target in targets:
            predecessors[target].add(lane_id)
    for lane_id, fields in lanes.items():
        fields["sumo.successor_lane_ids"] = sorted(successors[lane_id])
        fields["sumo.predecessor_lane_ids"] = sorted(predecessors[lane_id])
        fields["sumo.connected_destination_lane_ids"] = sorted(set(fields["sumo.connected_destination_lane_ids"]))
    for logic in root.findall("tlLogic"):
        tls_id = _string(logic.get("id"))
        if tls_id is None:
            raise ValueError(f"SUMO tlLogic without native id: {path}")
        phases = []
        for index, phase in enumerate(logic.findall("phase")):
            phases.append({"phase_index": index,
                           "state": _text_cell(phase.get("state")),
                           "duration_s": _numeric_cell(phase.get("duration")),
                           "min_duration_s": _numeric_cell(phase.get("minDur")),
                           "max_duration_s": _numeric_cell(phase.get("maxDur")),
                           "next_phase_indices": ([int(v) for v in phase.get("next", "").split()]
                                                  if phase.get("next") else _missing())})
        signals.setdefault(tls_id, {"programs": [], "links": []})["programs"].append({
            "program_id": _text_cell(logic.get("programID")),
            "type": _text_cell(logic.get("type")),
            "offset_s": _numeric_cell(logic.get("offset")), "phases": phases,
        })
    for tls_id, definition in signals.items():
        if tls_id in nodes:
            raise ValueError(f"lane/edge/TLS native identity collision: {tls_id}")
        nodes[tls_id] = _static_node(tls_id, "signal", {
            "sumo.node_type": "signal", "sumo.tls_id": tls_id,
            "sumo.programs": definition["programs"] or _missing(),
            "sumo.controlled_links": definition["links"],
        }, path, version, "tlLogic=" + tls_id)
    location = root.find("location")
    metadata = {
        "map_version": version, "net_xml": path,
        "file_version": {"mtime_ns": mtime_ns, "size_bytes": size, "sha256": digest},
        "lane_count": len(lanes), "edge_count": len(edges), "signal_count": len(signals),
        "connection_count": len(connections), "coordinate_frame": deepcopy(_COORDINATE_FRAME),
        "sumo_location": _cell(dict(location.attrib)) if location is not None else _missing(),
        "adjacency_contract": "same_edge_adjacent_index_not_lane_change_permission",
        "connectivity_contract": "explicit_first_hop_via_internal_and_logical_destination",
    }
    return nodes, metadata


def _load_static(path: Path) -> tuple[dict, dict]:
    path = path.expanduser().resolve()
    # Hash on every call.  Stat-only caches can return stale geometry after a
    # same-size write with restored mtime.  fstat binds version to the bytes read.
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        contents = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError(f"SUMO map changed while being read: {path}")
    digest = hashlib.sha256(contents).hexdigest()
    nodes, metadata = _parse_static(str(path), after.st_mtime_ns, len(contents), digest, contents)
    # The parsed map is shared read-only by every window of a process (one
    # copy, not one per window); window-local additions go to a ChainMap layer.
    return nodes, deepcopy(metadata)


@dataclass
class TemporalContext:
    """Standalone generic context; it does not extend FIELD_REGISTRY/heads.

    ``state_at(tick)`` returns entity -> {fields, freshness, observation_tick,
    changed_tick, node_type, lifecycle}.  Missing entities/fields remain explicit
    missing cells.  ``relations_at(tick)`` returns active key -> context_relation
    with evidence replayed and age/freshness; removed relations are not active.
    Neither method infers phase changes, interpolates values, or uses endpoints
    observed after ``tick``.  Queries are restricted to this context's window.
    """

    static_nodes: dict[str, dict] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    source_bindings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # (tick, vehicle) -> (lane_id, lane position, length) as a decoder rebuilds
    # it from serialized records and poses only (window ticks).
    decoder_lane_positions: dict = field(default_factory=dict)

    def __getstate__(self):
        # Pickle only the window-local definitions; the shared map is reloaded
        # (and version-checked) from its file by the receiving process.
        state = dict(self.__dict__)
        if isinstance(self.static_nodes, ChainMap):
            state["static_nodes"] = ("shared_map", self.metadata["net_xml"], self.metadata["map_version"],
                                     self.static_nodes.maps[0])
        return state

    def __setstate__(self, state):
        nodes = state["static_nodes"]
        if isinstance(nodes, tuple) and nodes[0] == "shared_map":
            shared, metadata = _load_static(Path(nodes[1]))
            if metadata["map_version"] != nodes[2]:
                raise ValueError(f"SUMO map {nodes[1]} changed since this context was built")
            state["static_nodes"] = ChainMap(nodes[3], shared)
        self.__dict__.update(state)

    def _check_tick(self, tick: int) -> None:
        if type(tick) is not int:
            raise TypeError("replay tick must be an integer source tick")
        if self.metadata and not self.metadata["window_start"] <= tick <= self.metadata["cutoff"]:
            raise ValueError("replay tick is outside this context's causal window")

    def state_at(self, tick: int) -> dict[str, dict]:
        self._check_tick(tick)
        active: dict[str, dict] = {}
        for record in self.records:
            if record["tick"] > tick:
                continue
            if record["kind"] != "sumo_state":
                continue
            entity = record["entity"]
            state = active.setdefault(entity, {"fields": {}, "freshness": {}})
            state["fields"].update(deepcopy(record["fields"]))
            state["fields"].update(deepcopy(record.get("derived_fields", {})))
            for name in record.get("observed", ()):
                prior = state["freshness"].get(name)
                if prior is not None and name not in record["freshness"]:
                    state["freshness"][name] = {**{k: v for k, v in prior.items() if not k.startswith("previous_valid")},
                                                "observation_tick": record["tick"], "unchanged": True}
            state["freshness"].update(deepcopy(record["freshness"]))
            for name in ("observation_tick", "changed_tick", "node_type", "lifecycle"):
                state[name] = record[name]
        for state in active.values():
            for info in state["freshness"].values():
                info["age_ticks"] = tick - info["observation_tick"]
                info["age_seconds"] = info["age_ticks"] * self.metadata.get("tick_seconds", 1.0)
            state["as_of_tick"] = tick
        return active

    def relations_at(self, tick: int) -> dict[str, dict]:
        self._check_tick(tick)
        active: dict[str, dict] = {}
        for record in self.records:
            if record["tick"] > tick or record["kind"] != "sumo_relation":
                continue
            relation = record["context_relation"]
            key = relation["key"]
            if record["operation"] == "remove":
                active.pop(key, None)
                continue
            previous_fields = active.get(key, {}).get("fields", {})
            merged = {**deepcopy(previous_fields), **deepcopy(relation["fields"]),
                      **deepcopy(relation.get("derived", {}))}
            active[key] = {**deepcopy(relation), "fields": merged}
        for relation in active.values():
            relation["age_ticks"] = tick - relation["observed_tick"]
            relation["age_seconds"] = relation["age_ticks"] * self.metadata.get("tick_seconds", 1.0)
            relation["as_of_tick"] = tick
        return active


@dataclass
class _State:
    node_type: str
    values: dict = field(default_factory=dict)
    freshness: dict = field(default_factory=dict)
    observed_tick: int = -1
    changed_tick: int = -1
    present: bool = False
    lifecycle: str = "entry"


class _Builder:
    def __init__(self, context: TemporalContext):
        self.context = context
        self.states: dict[str, _State] = {}
        self.relations: dict[str, dict] = {}
        # Ordered key dictionaries preserve global insertion/removal order when
        # scanning only one endpoint's incident relations (sets would not).
        self.relation_keys_by_entity: dict[str, dict[str, None]] = {}
        self.ever_seen: set[str] = set()
        self.previous_vehicle_tick: dict[str, int] = {}
        self.previous_lanes: dict[str, str] = {}
        # Per vehicle: the decoder-side reconstruction of the previous tick.
        self.reconstruction: dict[str, dict] = {}
        self.native_aliases: dict[str, str] = {}
        self.last_snapshot_tick: int | None = None
        self.sequence = 0
        self.retaining = False
        self.counts: Counter = Counter()
        self.observed_sets: dict[tuple, tuple] = {}

    def _append(self, record: dict, sources: list) -> None:
        self.counts[record["kind"] + "." + (record.get("operation") or "observation")] += 1
        if not self.retaining:
            return
        self.context.records.append(record)
        self.context.source_bindings.append({
            "id": record["id"], "tick": record["tick"], "sources": deepcopy(sources),
        })

    def state(self, entity: str, node_type: str, values: dict, numeric: set[str],
              tick: int, sources: list, lifecycle: str, *, present: bool = True,
              provenance: dict | None = None, reconstructed: dict | None = None) -> None:
        reconstructed = reconstructed or {}
        old = self.states.setdefault(entity, _State(node_type))
        if old.node_type != node_type:
            raise ValueError(f"vehicle/TLS native identity collision: {entity}")
        changed_fields, derived_fields, freshness, full = {}, {}, {}, {}
        for name, value in values.items():
            changed = name not in old.values or value != old.values[name]
            prior = old.freshness.get(name)
            changed_tick = tick if changed else prior["changed_tick"]
            info = {"kind": "missing" if _is_missing(value) else "present",
                    "observation_tick": tick, "changed_tick": changed_tick,
                    "valid_from_tick": changed_tick, "unchanged": not changed}
            if provenance and name in provenance:
                info["provenance"] = deepcopy(provenance[name])
            if changed and prior is not None:
                info["previous_valid_from_tick"] = prior["valid_from_tick"]
                info["previous_valid_to_tick"] = tick
            # Records carry freshness only where it is not implied: an unchanged
            # observed field keeps its prior entry with this observation tick
            # (``observed`` lists every field observed; replay rebuilds it).
            full[name] = info
            if changed or prior is None or info.get("provenance") != prior.get("provenance"):
                freshness[name] = info
            if changed or (name in numeric and not _is_missing(value)):
                # A value recomputable from serialized inputs is replay-only.
                (derived_fields if name in reconstructed else changed_fields)[name] = deepcopy(value)
            if changed:
                old.changed_tick = tick
        old.values.update(deepcopy(values))
        old.freshness.update(deepcopy(full))
        old.observed_tick, old.present, old.lifecycle = tick, present, lifecycle
        self.sequence += 1
        record = {"kind": "sumo_state", "tick": tick, "entity": entity,
                  "id": _key("sumo_state", {"entity": entity}) + f":{tick}:{self.sequence}",
                  "node_type": node_type, "fields": changed_fields,
                  "derived_fields": derived_fields, "observed": self._observed(values),
                  "observation_tick": tick, "changed_tick": old.changed_tick,
                  "emit": bool(changed_fields), "freshness": freshness,
                  "lifecycle": lifecycle}
        self._append(record, sources)

    def _observed(self, values: dict) -> tuple:
        # One shared tuple per distinct field set (records reference it).
        names = tuple(values)
        return self.observed_sets.setdefault(names, names)

    def missing_state(self, entity: str, tick: int, sources: list, reason: str) -> None:
        old = self.states[entity]
        values = {name: _missing() for name in old.values}
        self.state(entity, old.node_type, values, set(), tick, sources, reason, present=False)

    def relation(self, kind: str, bindings: dict[str, str], fields: dict, tick: int,
                 sources: list, derived: dict | None = None) -> None:
        """``fields`` are serialized on change; ``derived`` is replay-only payload
        recomputable from serialized states/static map, never a change trigger."""
        key = _key(kind, bindings)
        old = self.relations.get(key)
        operation = "add" if old is None else "update" if old["fields"] != fields else "refresh"
        changed = tick if operation != "refresh" else old["changed_tick"]
        valid_from = tick if old is None else old["valid_from_tick"]
        body = {"relation_type": kind, "key": key, "bindings": dict(bindings),
                "fields": deepcopy(fields), "valid_from_tick": valid_from,
                "observed_tick": tick, "changed_tick": changed, "value": dict(_PRESENT),
                "derived": deepcopy(derived or {})}
        if operation == "update":
            body["previous_value_valid_from_tick"] = old["changed_tick"]
            body["previous_value_valid_to_tick"] = tick
        self.relations[key] = deepcopy(body)
        if old is None:
            for identity in bindings.values():
                self.relation_keys_by_entity.setdefault(identity, {})[key] = None
        if operation == "refresh":
            body["fields"] = {}
        else:
            # Updates are discrete evidence changes; suppress identical values.
            if old is not None:
                body["fields"] = {k: v for k, v in body["fields"].items()
                                  if k not in old["fields"] or v != old["fields"][k]}
                body["fields"].update({k: _missing() for k in old["fields"] if k not in fields})
        self._append({"kind": "sumo_relation", "tick": tick, "entity": None,
                      "id": key, "fields": {}, "operation": operation,
                      "emit": operation != "refresh", "context_relation": body}, sources)

    def remove(self, key: str, tick: int, sources: list, reason: str,
               transition: str | None = None) -> None:
        old = self.relations.pop(key)
        for identity in dict.fromkeys(old["bindings"].values()):
            keys = self.relation_keys_by_entity[identity]
            del keys[key]
            if not keys:
                del self.relation_keys_by_entity[identity]
        body = {**deepcopy(old), "fields": {"sumo.invalidation_reason": reason},
                "observed_tick": tick, "changed_tick": tick,
                "valid_to_tick": tick, "value": _missing()}
        if transition:
            body["fields"]["sumo.transition"] = transition
        self._append({"kind": "sumo_relation", "tick": tick, "entity": None,
                      "id": key, "fields": {}, "operation": "remove",
                      "emit": True, "context_relation": body}, sources)

    def invalidate(self, entity: str, tick: int, sources: list, reason: str) -> None:
        for key in list(self.relation_keys_by_entity.get(entity, {})):
            self.remove(key, tick, sources, reason)

    def boundary(self, tick: int) -> None:
        self.retaining = True
        # The boundary serializes raw values; reconstruction restarts from them.
        for entity, carried in self.reconstruction.items():
            value = _number(self.states[entity].values.get("sumo.lane_position_m")) if entity in self.states else None
            if value is not None:
                carried["lane_position"] = value
                self.context.decoder_lane_positions[tick, entity] = (
                    carried["lane"], value, _number(self.states[entity].values.get("sumo.length_m")))
        for entity, state in sorted(self.states.items()):
            self._append({"kind": "sumo_state", "tick": tick, "entity": entity,
                          "id": _key("sumo_boundary", {"entity": entity}) + f":{tick}",
                          "node_type": state.node_type, "fields": deepcopy(state.values),
                          "observation_tick": state.observed_tick,
                          "changed_tick": state.changed_tick, "emit": True,
                          "freshness": deepcopy(state.freshness), "lifecycle": "window_boundary"},
                         [{"boundary_as_of_tick": tick, "observation_tick": state.observed_tick}])
        for key, relation in sorted(self.relations.items()):
            self._append({"kind": "sumo_relation", "tick": tick, "entity": None,
                          "id": key, "fields": {}, "operation": "add", "emit": True,
                          "context_relation": deepcopy(relation)},
                         [{"boundary_as_of_tick": tick, "observation_tick": relation["observed_tick"]}])

    def transition(self, old_lane: str, new_lane: str) -> str:
        old = self.context.static_nodes.get(old_lane, {}).get("fields", {})
        new = self.context.static_nodes.get(new_lane, {}).get("fields", {})
        old_edge, new_edge = old.get("sumo.edge_id"), new.get("sumo.edge_id")
        if isinstance(old_edge, str) and old_edge == new_edge:
            return "lane_change"
        successors = old.get("sumo.successor_lane_ids", [])
        successors = successors if isinstance(successors, list) else []
        destinations = old.get("sumo.connected_destination_lane_ids", [])
        destinations = destinations if isinstance(destinations, list) else []
        if new_lane in successors or new_lane in destinations:
            return "segment_transition"
        # Only internal continuations may be skipped.  Arbitrary downstream
        # reachability is not evidence of a connected adjacent segment change.
        frontier = list(successors)
        visited = set()
        while frontier:
            lane = frontier.pop()
            if lane in visited:
                continue
            visited.add(lane)
            fields = self.context.static_nodes.get(lane, {}).get("fields", {})
            if fields.get("sumo.edge_function") != "internal":
                continue
            targets = fields.get("sumo.successor_lane_ids", [])
            targets = targets if isinstance(targets, list) else []
            if new_lane in targets:
                return "segment_transition"
            frontier.extend(targets)
        return "unconnected_transition"


def _vehicle_values(raw: Mapping[str, Any], observed: dict, nodes: dict) -> tuple[dict, set[str], dict]:
    lane = _string(raw.get("sumo_lane_id")) or _string(raw.get("lane_id"))
    if _string(raw.get("sumo_lane_id")) and _string(raw.get("lane_id")) and raw["sumo_lane_id"] != raw["lane_id"]:
        raise ValueError("raw lane_id and sumo_lane_id disagree")
    edge = _string(raw.get("sumo_edge_id"))
    values = {"sumo.presence": True, "sumo.lane_id": lane or _missing(),
              "sumo.edge_id": edge or _missing(),
              "sumo.vehicle_id": _text_cell(raw.get("vehicle_id")),
              "sumo.vehicle_type": _text_cell(raw.get("vehicle_type")),
              "sumo.route_id": _text_cell(raw.get("route_id")),
              "sumo.signals": (_integer(raw.get("signals"))
                               if _integer(raw.get("signals")) is not None else _missing()),
              "sumo.position_reference": _text_cell(raw.get("sumo_position_reference")),
              "sumo.truth_position_reference": _text_cell(raw.get("truth_position_reference")),
              "sumo.leading_vehicle_id": (_string(raw.get("leading_vehicle_id"))
                                           or _string(raw.get("leader_id")) or _missing()),
              "sumo.following_distance_m": _numeric_cell(raw.get("following_distance_m")),
              "sumo.observed_leading_vehicle_id": observed.get("leader", _missing()),
              "sumo.observed_following_distance_m": observed.get("distance", _missing())}
    provenance = {}
    if edge is None and lane in nodes:
        values["sumo.edge_id"] = deepcopy(nodes[lane]["fields"]["sumo.edge_id"])
        provenance["sumo.edge_id"] = {"derivation": "static_lane_edge_membership"}
    for name in ("sumo.observed_leading_vehicle_id", "sumo.observed_following_distance_m"):
        provenance[name] = {"derivation": _DERIVATION, "coverage": _COVERAGE,
                            "position_reference": observed.get("position_reference", _missing())}
    for name in _NUMERIC_VEHICLE:
        values["sumo." + name] = _numeric_cell(raw.get(name))
    for raw_name, name, size in _VECTOR_VEHICLE:
        vector = raw.get(raw_name)
        values[name] = ([float(v) for v in vector]
                        if isinstance(vector, (list, tuple)) and len(vector) == size
                        and all(_number(v) is not None for v in vector) else _missing())
    dimensions = raw.get("dimensions_m") or {}
    for name in ("length", "width", "height"):
        values["sumo." + name + "_m"] = _numeric_cell(dimensions.get(name))
    # Dimensions and allowed speed are change-detected definitions, not
    # numerical time series.  Dynamic kinematics/gap remain available each tick.
    numeric = {"sumo." + name for name in _NUMERIC_VEHICLE if name != "allowed_speed_mps"}
    numeric.update(name for _, name, _ in _VECTOR_VEHICLE)
    numeric.update({"sumo.following_distance_m", "sumo.observed_following_distance_m"})
    return values, numeric, provenance


def _present(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value") if value.get("kind", "present") == "present" else None
    return value


def _pose(fields: Mapping[str, Any]) -> dict | None:
    """Same-tick admitted pose (model input fields), or None."""
    position, velocity = _present(fields.get("pose.position_enu_m")), _present(fields.get("pose.velocity_enu_mps"))
    yaw = _number(_present(fields.get("pose.yaw_deg")))
    if not (isinstance(position, list) and len(position) == 3 and isinstance(velocity, list) and len(velocity) == 3):
        return None
    if any(_number(v) is None for v in position+velocity):
        return None
    return {"position": [float(v) for v in position], "velocity": [float(v) for v in velocity], "yaw": yaw}


def _reconstruction(values: dict, pose: dict | None, previous: dict | None,
                    tick_seconds: float) -> tuple[dict, dict]:
    """Per-record check of which dynamic numerics follow from serialized inputs.

    Returns ``(reconstructed, state)``: field -> rule for values a decoder
    recomputes within tolerance from the same-tick pose, the vehicle length and
    its own reconstruction of the previous tick; and the reconstruction to carry
    forward.  Lane position integrates |v| from the last *serialized* value, so
    errors never accumulate past the tolerance.  The leader identity/separation
    duplicate the vehicle_following relation and the serialized centre positions.
    """
    reconstructed = {"sumo.observed_leading_vehicle_id": "vehicle_following_relation_binding",
                     "sumo.observed_following_distance_m": "leader_centre_minus_follower_centre"}
    lane, raw_lane_position = values.get("sumo.lane_id"), _number(values.get("sumo.lane_position_m"))
    length = _number(values.get("sumo.length_m"))
    state = {"lane": lane, "lane_position": raw_lane_position, "speed": None}
    if pose is None:
        return reconstructed, state
    speed = math.sqrt(sum(v*v for v in pose["velocity"]))
    state["speed"] = speed
    raw = {name: _number(values.get(name)) for name in ("sumo.speed_mps", "sumo.sumo_angle_deg",
                                                        "sumo.accel_mps2", "sumo.center_lane_position_m")}
    if raw["sumo.speed_mps"] is not None and abs(raw["sumo.speed_mps"]-speed) <= _SPEED_TOLERANCE_MPS:
        reconstructed["sumo.speed_mps"] = "norm(pose.velocity_enu_mps)"
    yaw = pose["yaw"]
    if yaw is not None and raw["sumo.sumo_angle_deg"] is not None and \
            abs((raw["sumo.sumo_angle_deg"]-(90.-yaw)+180.) % 360.-180.) <= _ANGLE_TOLERANCE_DEG:
        reconstructed["sumo.sumo_angle_deg"] = "90 - pose.yaw_deg"
    if yaw is not None and length is not None:
        x, y, z = pose["position"]
        front = [x+length/2*math.cos(math.radians(yaw)), y+length/2*math.sin(math.radians(yaw)), z]
        for name, size in (("sumo.xy_m", 2), ("sumo.front_bumper_xy_m", 2), ("sumo.truth_front_bumper_enu_m", 3)):
            vector = values.get(name)
            if isinstance(vector, list) and len(vector) == size and \
                    max(abs(a-b) for a, b in zip(vector, front)) <= _POSITION_TOLERANCE_M:
                reconstructed[name] = "pose.position_enu_m + length/2 * heading(pose.yaw_deg)"
    if previous is not None and previous["lane"] == lane and previous["lane_position"] is not None \
            and raw_lane_position is not None:
        estimate = previous["lane_position"]+speed*tick_seconds
        if abs(estimate-raw_lane_position) <= _POSITION_TOLERANCE_M:
            reconstructed["sumo.lane_position_m"] = "previous lane position + norm(velocity) * dt"
            state["lane_position"] = estimate
    if length is not None and state["lane_position"] is not None and raw["sumo.center_lane_position_m"] is not None \
            and abs(raw["sumo.center_lane_position_m"]-(state["lane_position"]-length/2)) <= _POSITION_TOLERANCE_M:
        reconstructed["sumo.center_lane_position_m"] = "lane position - length/2"
    if previous is not None and previous["speed"] is not None and raw["sumo.accel_mps2"] is not None and \
            abs(raw["sumo.accel_mps2"]-(speed-previous["speed"])/tick_seconds) <= _ACCEL_TOLERANCE_MPS2:
        reconstructed["sumo.accel_mps2"] = "(norm(velocity) - previous norm(velocity)) / dt"
    return reconstructed, state


def _observed_following(vehicles: dict) -> dict[str, dict]:
    lanes: dict[str, list] = defaultdict(list)
    for entity, (raw, _) in vehicles.items():
        lane = _string(raw.get("sumo_lane_id")) or _string(raw.get("lane_id"))
        position = _number(raw.get("center_lane_position_m"))
        reference = "center_lane_position_m"
        if position is None:
            position, reference = _number(raw.get("lane_position_m")), "lane_position_m_fallback"
        if lane and position is not None:
            lanes[lane].append((position, entity, raw, reference))
    following = {}
    for lane, items in sorted(lanes.items()):
        ordered = sorted(items, key=lambda item: (item[0], item[1]))
        for follower, leader in zip(ordered, ordered[1:]):
            p, entity, raw, reference = follower
            q, leader_id, leader_raw, leader_reference = leader
            distance = max(0.0, q - p)
            length = _number((raw.get("dimensions_m") or {}).get("length"))
            leader_length = _number((leader_raw.get("dimensions_m") or {}).get("length"))
            following[entity] = {
                "leader": leader_id, "distance": distance,
                "lane": lane, "position_reference": [reference, leader_reference],
                "follower_position": p, "leader_position": q,
                "follower_length": length if length is not None else _missing(),
                "leader_length": leader_length if leader_length is not None else _missing(),
            }
    return following


def _signal_values(raw: Mapping[str, Any]) -> dict:
    phase = _integer(raw.get("phase_index"))
    return {"sumo.presence": True, "sumo.signal_state": _text_cell(raw.get("state")),
            "sumo.phase_index": phase if phase is not None else _missing(),
            "sumo.program_id": _text_cell(raw.get("program_id"))}


def _resolve_map(refs: set[str], explicit: Path | None) -> tuple[Path, str]:
    paths = {str(Path(ref.split("#", 1)[0]).expanduser().resolve()) for ref in refs}
    if len(paths) > 1:
        raise ValueError(f"SUMO input prefix refers to different authoritative maps: {sorted(paths)}")
    if paths:
        path = Path(next(iter(paths)))
        if explicit is not None and explicit.expanduser().resolve() != path:
            raise ValueError("explicit net_xml conflicts with authoritative lane_instance_source_ref")
        return path, "lane_instance_source_ref"
    if explicit is not None:
        return explicit.expanduser().resolve(), "explicit_net_xml_without_source_ref"
    return _DEFAULT_NET.resolve(), "project_default_no_authoritative_ref"


def build_sumo_context(input_records: Iterable[Mapping[str, Any]], *,
                       window_start: int, cutoff: int, tick_seconds: float,
                       net_xml: Path | None = None) -> TemporalContext:
    """Build a generic, causal temporal context from an admitted raw prefix.

    Pass records before ``window_start`` as well as the window itself.  They
    initialize the boundary/freshness and establish entry/reentry history, but
    are not emitted as older context slots.  Future records are discarded by
    tick before inspecting fields.  Availability is additionally gated at each
    record's own observation tick.  A missing authoritative XML is an error;
    it never silently selects another map.  Static XML is parsed with stdlib.
    """
    if type(window_start) is not int or type(cutoff) is not int or not 0 <= window_start <= cutoff:
        raise ValueError("window_start/cutoff must be nonnegative ordered integer ticks")
    if isinstance(tick_seconds, bool) or _number(tick_seconds) is None or tick_seconds <= 0:
        raise ValueError("tick_seconds must be a finite positive source duration")
    if net_xml is not None:
        net_xml = Path(net_xml)
    snapshots: dict[int, dict] = {}
    refs: set[str] = set()
    roster_aliases: dict[str, str] = {}
    episode_ids: set[str] = set()
    skipped_unavailable = 0
    for record in input_records:
        tick = record.get("tick")
        if tick is not None and type(tick) is not int:
            raise ValueError("canonical SUMO record tick must be an integer or static None")
        if tick is not None and tick > cutoff:
            continue
        if tick is not None and tick < 0:
            raise ValueError("canonical SUMO frame tick is negative")
        if not _available(record, 0 if tick is None else tick):
            skipped_unavailable += 1
            continue
        episode = _string(record.get("episode_id"))
        if episode:
            episode_ids.add(episode)
        fields = record.get("fields") or {}
        roster = _vehicle_payload(fields, roster=True)
        if isinstance(roster, Mapping):
            ref = _string(roster.get("lane_instance_source_ref"))
            if ref:
                refs.add(ref)
            native = _string(roster.get("vehicle_id"))
            entity = _string(fields.get("roster.entity_id")) or _string(record.get("entity_id"))
            if native and entity:
                roster_aliases[native] = entity
        if tick is None or record.get("record_kind") != "frame":
            continue
        raw = _vehicle_payload(fields)
        lights = fields.get("archive.scene.sumo_traffic_light_states")
        # An archive truth frame, a vehicle projection, or a scene TLS dict is
        # evidence that the admitted snapshot exists.  Empty TLS dicts count.
        if not ("archive.truth_frames" in fields
                or "archive.truth_frames.sumo_vehicle" in fields
                or "archive.scene.sumo_traffic_light_states" in fields):
            continue
        snapshot = snapshots.setdefault(tick, {"vehicles": {}, "lights": None, "sources": [], "poses": {}})
        source = _source(record, "archive.truth_frames.sumo_vehicle")
        if source not in snapshot["sources"]:
            snapshot["sources"].append(source)
        if isinstance(raw, Mapping) and raw:
            ref = _string(raw.get("lane_instance_source_ref"))
            if ref:
                refs.add(ref)
            entity = _string(record.get("entity_id")) or _string(raw.get("vehicle_id"))
            if entity is None:
                raise ValueError("observed SUMO vehicle lacks a stable identity")
            physical = {k: v for k, v in raw.items() if k in _VEHICLE_INPUT_KEYS}
            old = snapshot["vehicles"].get(entity)
            if old is not None and old[0] != physical:
                raise ValueError(f"conflicting SUMO vehicle observations: {tick}/{entity}")
            snapshot["vehicles"][entity] = (physical, source)
            pose = _pose(fields)
            if pose is not None:
                snapshot["poses"][entity] = pose
        if isinstance(lights, Mapping):
            light_values = {str(identity): {k: value[k] for k in ("state", "phase_index", "program_id")
                                          if k in value}
                            for identity, value in lights.items() if isinstance(value, Mapping)}
            if snapshot["lights"] is not None and snapshot["lights"] != light_values:
                raise ValueError(f"conflicting scene TLS samples at tick {tick}")
            snapshot["lights"] = light_values
            snapshot["light_source"] = _source(record, "archive.scene.sumo_traffic_light_states")
    if len(episode_ids) > 1:
        raise ValueError("cross-episode records in SUMO input prefix")
    path, map_source = _resolve_map(refs, net_xml)
    shared, metadata = _load_static(path)
    nodes = ChainMap({}, shared)
    metadata.update({
        "schema": "sumo_temporal_v1", "window_start": window_start, "cutoff": cutoff,
        "tick_seconds": float(tick_seconds), "map_source": map_source,
        "episode_id": next(iter(episode_ids)) if episode_ids else _missing(),
        "following_derivation": _DERIVATION, "following_coverage": _COVERAGE,
        "reconstructed_numeric_fields": sorted(DYNAMIC_NUMERIC_FIELDS | {"sumo.observed_leading_vehicle_id"}),
        "reconstruction_contract": "derived_fields recomputed within source precision from same-tick pose, "
                                   "vehicle length and the previous tick (_reconstruction)",
        "following_distance_contract": "observed_longitudinal_center_separation_not_sensor_bumper_gap",
        "following_position_fallback": "raw_lane_position_with_explicit_reference",
        "raw_leader_policy": "prefer_explicit_observed_leader_else_subset_derivation",
        "signal_phase_policy": "actual_scene_samples_only_no_timetable_inference",
        "interval_policy": "close_only_on_observed_change_or_removal_never_backfill",
        "boundary_policy": "last_observed_prefix_state_with_original_freshness_no_interpolation",
        "source_snapshot_policy": "complete_canonical_admitted_subset_not_full_sumo_world",
        "skipped_unavailable_records": skipped_unavailable,
        "source_gaps": [], "lifecycle_events": [],
    })
    context = TemporalContext(nodes, [], [], metadata)
    builder = _Builder(context)
    builder.native_aliases.update(roster_aliases)
    boundary_emitted = False
    raw_leader_frames = derived_following_frames = vehicle_frames = signal_samples = 0
    for tick, snapshot in sorted(snapshots.items()):
        # The boundary is the state AS OF window_start: a snapshot at
        # window_start is applied first (unretained), so the boundary carries
        # its values; otherwise the last earlier observation with its freshness.
        if tick > window_start and not boundary_emitted:
            builder.boundary(window_start)
            boundary_emitted = True
        vehicles, lights = snapshot["vehicles"], snapshot["lights"]
        sources = snapshot["sources"]
        gap = builder.last_snapshot_tick is not None and tick > builder.last_snapshot_tick + 1
        if gap:
            if tick >= window_start:
                metadata["source_gaps"].append({"after_tick": builder.last_snapshot_tick,
                                               "detected_tick": tick,
                                               "unobserved_tick_count": tick - builder.last_snapshot_tick - 1})
            for entity, state in list(builder.states.items()):
                if state.present:
                    builder.missing_state(entity, tick, sources, "observation_gap")
                    builder.invalidate(entity, tick, sources, "observation_gap")
        for entity, (raw, _) in sorted(vehicles.items()):
            native = _string(raw.get("vehicle_id"))
            if native:
                existing = builder.native_aliases.get(native)
                if existing is not None and existing != entity:
                    raise ValueError(f"native SUMO vehicle identity is aliased inconsistently: {native}")
                builder.native_aliases[native] = entity
        # Disappearance is detected at an authentic snapshot, never at a made-up
        # intermediate tick or the last frame's future endpoint.
        present_signals = set(lights or {})
        for entity, state in list(builder.states.items()):
            if not state.present:
                continue
            observed = entity in vehicles if state.node_type == "vehicle" else entity in present_signals
            if not observed:
                reason = "source_disappearance" if state.node_type == "vehicle" or lights is not None else "signal_observation_missing"
                builder.missing_state(entity, tick, sources, reason)
                builder.invalidate(entity, tick, sources, reason)
                if tick >= window_start:
                    metadata["lifecycle_events"].append({"entity": entity, "tick": tick, "event": reason})
        for tls_id, raw in sorted((lights or {}).items()):
            if not _string(tls_id):
                raise ValueError("scene TLS sample has an empty native identity")
            old = builder.states.get(tls_id)
            lifecycle = "observation_gap" if gap else "entry" if old is None else "reentry" if not old.present else "observed"
            light_source = [snapshot["light_source"]]
            builder.state(tls_id, "signal", _signal_values(raw), set(), tick, light_source, lifecycle)
            signal_samples += 1
            if tls_id not in nodes:
                nodes[tls_id] = _static_node(tls_id, "signal", {
                    "sumo.node_type": "signal", "sumo.tls_id": tls_id,
                    "sumo.programs": _missing(), "sumo.controlled_links": _missing(),
                }, str(path), metadata["map_version"], "unknown_tlLogic=" + tls_id)
                nodes[tls_id].update({"known_from_tick": tick,
                                      "definition_status": _missing(),
                                      "source_observation": deepcopy(snapshot["light_source"])})
        derived = _observed_following(vehicles)
        desired_keys: set[str] = set()
        for entity, (raw, source) in sorted(vehicles.items()):
            previous_tick = builder.previous_vehicle_tick.get(entity)
            old_state = builder.states.get(entity)
            discontinuous = previous_tick is not None and previous_tick != tick - 1
            lifecycle = ("observation_gap" if gap else "entry" if entity not in builder.ever_seen
                         else "reentry" if old_state is not None and not old_state.present
                         else "observation_gap" if discontinuous else "observed")
            observed = derived.get(entity, {})
            values, numeric, provenance = _vehicle_values(raw, observed, nodes)
            consecutive = previous_tick == tick-1 and not gap
            reconstructed, builder.reconstruction[entity] = _reconstruction(
                values, snapshot["poses"].get(entity), builder.reconstruction.get(entity) if consecutive else None,
                float(tick_seconds))
            builder.state(entity, "vehicle", values, numeric, tick, [source], lifecycle,
                          provenance=provenance, reconstructed=reconstructed)
            if tick > window_start and builder.reconstruction[entity]["lane_position"] is not None:
                context.decoder_lane_positions[tick, entity] = (
                    builder.reconstruction[entity]["lane"], builder.reconstruction[entity]["lane_position"],
                    _number(values.get("sumo.length_m")))
            vehicle_frames += 1
            if lifecycle != "observed" and tick >= window_start:
                metadata["lifecycle_events"].append({"entity": entity, "tick": tick, "event": lifecycle})
            lane = _string(values["sumo.lane_id"])
            old_lane = builder.previous_lanes.get(entity)
            transition = ("observation_gap" if gap or discontinuous else
                          builder.transition(old_lane, lane) if old_lane and lane and old_lane != lane
                          else "entry" if previous_tick is None else "observed")
            membership_bindings = {"vehicle": entity, "lane": lane} if lane else None
            membership_key = _key("lane_membership", membership_bindings) if membership_bindings else None
            for key in list(builder.relation_keys_by_entity.get(entity, {})):
                relation = builder.relations[key]
                if (relation["relation_type"] == "lane_membership"
                        and relation["bindings"]["vehicle"] == entity and key != membership_key):
                    builder.remove(key, tick, [source], "lane_membership_changed", transition)
            if lane:
                if lane not in nodes:
                    nodes[lane] = _static_node(lane, "lane", {
                        "sumo.node_type": "lane", "sumo.lane_id": lane,
                        "sumo.edge_id": deepcopy(values["sumo.edge_id"]),
                        "sumo.shape_xy_m": _missing(), "sumo.shape_enu_m": _missing(),
                        "sumo.length_m": _missing(), "sumo.speed_mps": _missing(),
                        "sumo.lane_index": _missing(), "sumo.adjacent_lane_ids": _missing(),
                        "sumo.successor_lane_ids": _missing(), "sumo.predecessor_lane_ids": _missing(),
                        "sumo.connected_destination_lane_ids": _missing(),
                        "sumo.connections": _missing(), "sumo.signal_links": _missing(),
                    }, str(path), metadata["map_version"], "unknown_lane=" + lane)
                    nodes[lane].update({"known_from_tick": tick,
                                        "definition_status": _missing(),
                                        "source_observation": deepcopy(source)})
                membership_fields = {"sumo.lane_id": lane, "sumo.edge_id": values["sumo.edge_id"],
                                     "sumo.provenance": "archive.truth_frames.sumo_vehicle"}
                if membership_key not in builder.relations:
                    membership_fields["sumo.transition"] = transition
                    if old_lane:
                        membership_fields["sumo.previous_lane_id"] = old_lane
                else:
                    # Entry/transition labels persist as event provenance; they
                    # do not induce a spurious next-tick value update.
                    for name in ("sumo.transition", "sumo.previous_lane_id"):
                        if name in builder.relations[membership_key]["fields"]:
                            membership_fields[name] = builder.relations[membership_key]["fields"][name]
                builder.relation("lane_membership", membership_bindings, membership_fields, tick, [source])
                desired_keys.add(membership_key)
                lane_links = nodes[lane]["fields"].get("sumo.signal_links", [])
                by_signal: dict[str, list] = defaultdict(list)
                for link in lane_links if isinstance(lane_links, list) else []:
                    if isinstance(link["tls_id"], str):
                        by_signal[link["tls_id"]].append(link)
                for tls_id, links in sorted(by_signal.items()):
                    signal = (lights or {}).get(tls_id)
                    if signal is None:
                        continue
                    indices = sorted({l["link_index"] for l in links if type(l["link_index"]) is int})
                    state = _string(signal.get("state"))
                    bindings = {"signal": tls_id, "vehicle": entity, "lane": lane}
                    phase = _integer(signal.get("phase_index"))
                    # The vehicle's own link states are serialized on change; the
                    # whole-signal phase string belongs to the signal's own state.
                    fields = {"sumo.link_indices": indices,
                              "sumo.link_states": {str(i): state[i] if state and 0 <= i < len(state) else _missing()
                                                   for i in indices},
                              "sumo.provenance": "static_tls_link_map+actual_scene_tls_sample"}
                    derived_signal = {"sumo.signal_state": state or _missing(),
                                      "sumo.phase_index": phase if phase is not None else _missing(),
                                      "sumo.program_id": _text_cell(signal.get("program_id"))}
                    builder.relation("signal_control", bindings, fields, tick,
                                     [source, snapshot["light_source"]], derived_signal)
                    desired_keys.add(_key("signal_control", bindings))
            leader = (_string(raw.get("leading_vehicle_id")) or _string(raw.get("leader_id")))
            if leader and leader not in ("none", "unknown"):
                leader_identity = builder.native_aliases.get(leader, leader)
                if leader_identity == entity:
                    raise ValueError("SUMO raw leader observation binds a vehicle to itself")
                bindings = {"following_vehicle": entity, "leading_vehicle": leader_identity}
                fields = {"sumo.following_distance_m": _numeric_cell(raw.get("following_distance_m")),
                          "sumo.provenance": "explicit_raw_leader_observation",
                          "sumo.coverage": "source_declared_leader"}
                follow_derived = {}
                raw_leader_frames += 1
            elif observed:
                bindings = {"following_vehicle": entity, "leading_vehicle": observed["leader"]}
                # The separation is max(0, leader centre - follower centre) of
                # two serialized/reconstructible lane positions: replay-only.
                fields = {
                    "sumo.lane_id": observed["lane"],
                    "sumo.position_reference": observed["position_reference"],
                    "sumo.derivation": _DERIVATION, "sumo.coverage": _COVERAGE,
                }
                follow_derived = {
                    "sumo.observed_following_distance_m": observed["distance"],
                    "sumo.follower_position_m": observed["follower_position"],
                    "sumo.leader_position_m": observed["leader_position"],
                    "sumo.follower_length_m": observed["follower_length"],
                    "sumo.leader_length_m": observed["leader_length"],
                }
                derived_following_frames += 1
            else:
                bindings = None
            if bindings:
                key = _key("vehicle_following", bindings)
                # Remove the old binding before adding the current leader.
                for old_key in list(builder.relation_keys_by_entity.get(entity, {})):
                    relation = builder.relations[old_key]
                    if (relation["relation_type"] == "vehicle_following"
                            and relation["bindings"]["following_vehicle"] == entity and old_key != key):
                        builder.remove(old_key, tick, [source], "observed_leader_changed")
                leader_sources = [source]
                if bindings["leading_vehicle"] in vehicles:
                    leader_sources.append(vehicles[bindings["leading_vehicle"]][1])
                builder.relation("vehicle_following", bindings, fields, tick, leader_sources, follow_derived)
                desired_keys.add(key)
            builder.previous_vehicle_tick[entity] = tick
            if lane:
                builder.previous_lanes[entity] = lane
            builder.ever_seen.add(entity)
        # Membership missing, controlling signal unavailable, frontmost
        # follower, and disappeared endpoints all invalidate at THIS tick.
        for key, relation in list(builder.relations.items()):
            if key not in desired_keys:
                builder.remove(key, tick, sources, "relation_not_observed_in_current_snapshot")
        builder.last_snapshot_tick = tick
        if tick == window_start:
            builder.boundary(window_start)
            boundary_emitted = True
    if not boundary_emitted:
        builder.boundary(window_start)
    metadata.update({
        "observed_snapshot_count": len(snapshots), "vehicle_frame_count": vehicle_frames,
        "signal_sample_count": signal_samples, "raw_leader_frame_count": raw_leader_frames,
        "derived_following_frame_count": derived_following_frames,
        "record_count": len(context.records),
        "emitted_record_count": sum(r["emit"] for r in context.records),
        "operation_counts_including_prefix_and_boundary": dict(builder.counts),
        "context_status": "present" if snapshots else "missing_raw_sumo_prefix",
    })
    return context
