"""Causal finite-window typed world graphs and physical query supervision."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from Dataset.world_model.contracts.schema import FIELD_REGISTRY
from Dataset.world_model.data.provenance import content_available_at, resolve_source_split
from Dataset.world_model.data.windows import build_supervision_window
from Dataset.semantic_truth.core_semantic_registry import load_core_semantic_registry
from Dataset.semantic_truth.provenance import open_artifact_text
from Dataset.world_model.model.derivations import PRIMARY_FIELDS
from Dataset.world_model.model.sumo_temporal import DYNAMIC_NUMERIC_FIELDS, build_sumo_context
from Dataset.world_model.model.query_head import FieldTarget, QueryAddress, fit_input_scale, modelable_field_targets

COMM = {
    "communication.latency_ms": ("real", "ms", (0., None), ()),
    "communication.packet_loss_ratio": ("real", "ratio", (0., 1.), ()),
    "communication.handover_active": ("bool", None, (None, None), (False, True)),
    "communication.quality_level": ("enum", None, (None, None), ("excellent", "good", "fair", "poor", "down")),
}
COMM_KEYS = {"communication.latency_ms": ("link_quality", "latency_ms"),
             "communication.packet_loss_ratio": ("link_quality", "packet_loss_ratio"),
             "communication.handover_active": ("handover", "active"),
             "communication.quality_level": ("link_quality", "quality_level")}
MODEL_INPUT_EXCLUDED = frozenset({"annotations.network_latency_ms", "annotations.network_packet_loss", "annotations.network_status"})
STATUS = ("present", "missing", "inapplicable", "absent", "source_unknown")
# Anchor atom semantics "source_resolution_v2": the stored target is consistent
# with zero change from its forecast anchor at source precision.  Pose position
# and velocity are stored on a 1e-6 decimal grid (convert_to_render_ready.py:973,
# 979); observer pose is recomputed deterministically from capture calibration
# and is derived, not predicted (derivations.py).  The previous semantics, abs(target-anchor) <= 1e-5 in
# float64 ("tolerance_1e-5_v1"), applies to results produced before this change.
ATOM_SEMANTICS = "source_resolution_v2"
SOURCE_DECIMALS = {"pose.position_enu_m": 6, "pose.velocity_enu_mps": 6}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def cell(value, component):
    if isinstance(value, dict) and "kind" in value and value["kind"] in STATUS and value["kind"] != "present":
        return value["kind"], None
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    return "present", value if component is None else value[component]


def model_payload(value):
    """The meaningful payload actually read by _value, recursively clock-free."""
    if isinstance(value, dict):
        return {k: model_payload(v) for k, v in sorted(value.items()) if not TypedGraphEncoder._clock_key(k)}
    if isinstance(value, list):
        return [model_payload(v) for v in value]
    return value


def evidence_payload(observations):
    """Model evidence, excluding changing provenance pointers/maintenance clocks."""
    return sorted(({"path": o["path"], "value": model_payload(o["value"])} for o in observations
                   if not TypedGraphEncoder._clock_key(o["path"])), key=lambda o: o["path"])


def relation_payload(body):
    return {"predicate_id": body["predicate_id"], "bindings": body["bindings"],
            "binding_ontology_classes": body["binding_ontology_classes"], "value": body["value"],
            "observations": evidence_payload(body.get("observations", []))}


# Static-map fields not encoded: duplicates of encoded fields (XY copy of the ENU
# shape, adjacency = left + right, signal links/destinations within connections,
# TLS links within lane connections) or one constant coordinate contract.
STATIC_ENCODING_EXCLUDED = frozenset({"sumo.shape_xy_m", "sumo.coordinate_frame", "sumo.adjacent_lane_ids",
                                      "sumo.signal_links", "sumo.connected_destination_lane_ids",
                                      "sumo.controlled_links"})
# Context fields whose change times enter as freshness (time in lane, since a
# route change, since a signal phase change, since last observation).
FRESHNESS_FIELDS = ("sumo.presence", "sumo.lane_id", "sumo.route_id", "sumo.signal_state")


def context_numeric_leaves(path, value):
    """``(scale key, number)`` for every numeric leaf of an input-only context value.

    Numeric vectors are keyed per axis (``path[i]``); lists of vectors (lane
    shapes) share the per-axis key; dictionary members extend the path.
    """
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [(path, float(value))]
    if isinstance(value, list):
        if value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return [(f"{path}[{i}]", float(v)) for i, v in enumerate(value)]
        return [leaf for item in value for leaf in context_numeric_leaves(path, item)]
    if isinstance(value, dict):
        return [leaf for k, v in sorted(value.items()) if not TypedGraphEncoder._clock_key(k)
                for leaf in context_numeric_leaves(f"{path}.{k}", v)]
    return []


def encoded_static_fields(fields):
    return {k: v for k, v in fields.items() if k not in STATIC_ENCODING_EXCLUDED}


@dataclass
class EpisodeIndex:
    episode: str
    cutoff: int
    target_ticks: tuple[int, ...]
    entities: dict
    states: dict
    semantic_records: list
    fields: tuple
    scales: dict
    forecast_scales: dict
    source_bindings: list
    graph_nodes: dict
    graph_predicates: list
    identity_aliases: dict
    tick_seconds: float
    window_start: int
    observer_ids: tuple[str, ...]
    graph_parameters: dict
    local_identities: dict[str, int]
    split: str
    # State keyframe spacing; active relations are incremental at every observation.
    keyframe_ticks: int | None = None
    temporal_context: Any = None

    def addresses(self, tick):
        return tuple(QueryAddress(self.episode, e, f.family, tick, f.component)
                     for e, meta in sorted(self.entities.items()) for f in self.fields
                     if (not f.family.startswith("communication.") or meta["roster.entity_category"] == "uav")
                     )


def read_tick_seconds(canonical: Path) -> float:
    """Read the one authentic archive.scene.dt_s clock before fitting a window."""
    ticks = set()
    with canonical.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("tick") is not None and "archive.scene.dt_s" in record["fields"]:
                ticks.add(float(record["fields"]["archive.scene.dt_s"]))
    if len(ticks) != 1:
        raise ValueError("history must declare one authentic archive.scene.dt_s clock")
    seconds = ticks.pop()
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("invalid source tick duration")
    return seconds


def load_index(canonical: Path, communication: Path, episode: str, cutoff: int, targets,
               graph_base: Path, history_s=.5, observer_ids=(), keyframe_s=None, sumo_net_xml=None):
    """Read immutable sources once; labels never determine causal membership.

    Normalization is not fitted here: ``fit_state_normalization`` fits it once
    on the TRAIN windows and the caller assigns it to every window.
    """
    split = resolve_source_split(episode)
    if history_s <= 0 or not math.isfinite(history_s):
        raise ValueError("history_s must be a positive physical duration")
    clocks, sumo_prefix = set(), []
    def source_records():
        with canonical.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("tick") is not None and record["tick"] <= cutoff and "archive.scene.dt_s" in record["fields"]:
                    clocks.add(float(record["fields"]["archive.scene.dt_s"]))
                # A narrow, explicitly causal observation projection, not an
                # exception to windows.py's blanket archive-input prohibition.
                # Admission at the observation tick prevents backdating late data.
                tick = record.get("tick")
                if isinstance(tick, int) and tick <= cutoff and content_available_at(record, cutoff=tick):
                    raw = record["fields"].get("archive.truth_frames", {})
                    context_fields = {}
                    if isinstance(raw, dict) and "sumo_vehicle" in raw:
                        context_fields["archive.truth_frames"] = {"sumo_vehicle": raw["sumo_vehicle"]}
                    if "archive.scene.sumo_traffic_light_states" in record["fields"]:
                        context_fields["archive.scene.sumo_traffic_light_states"] = record["fields"]["archive.scene.sumo_traffic_light_states"]
                    if context_fields:
                        # Same-tick admitted pose: decides which SUMO numerics are
                        # reconstructible (sumo_temporal._reconstruction).
                        context_fields.update({k: record["fields"][k] for k in
                                               ("pose.position_enu_m", "pose.velocity_enu_mps", "pose.yaw_deg")
                                               if k in record["fields"]})
                        sumo_prefix.append({**record, "fields": context_fields})
                # Retain context/history and only requested labels; no full archive in RAM.
                if tick is None or tick <= cutoff or tick in targets:
                    yield record
    sample = build_supervision_window(source_records(), episode_id=episode, split=split, cutoff=cutoff, target_times=targets)
    if len(clocks) != 1:
        raise ValueError("history must declare one authentic archive.scene.dt_s clock")
    tick_seconds = clocks.pop()
    if tick_seconds <= 0 or not math.isfinite(tick_seconds):
        raise ValueError("invalid source tick duration")
    history_ticks = history_s/tick_seconds
    if not math.isclose(history_ticks, round(history_ticks), abs_tol=1e-8):
        raise ValueError("history duration must align to source ticks")
    left = cutoff-int(round(history_ticks))
    if left < 0:
        raise ValueError(f"cutoff={cutoff} cannot supply requested history_s={history_s}: authentic window starts at negative tick {left}")
    history = [r for r in sample.input_records if r.get("tick") is None or r["tick"] >= left]
    roster = {r["fields"]["roster.entity_id"]: r["fields"] for r in history if r["record_kind"] == "entity"}
    observed_ticks = {r["tick"] for r in history if r["record_kind"] == "frame"}
    missing_ticks = sorted(set(range(left, cutoff+1))-observed_ticks)
    if missing_ticks:
        raise ValueError(f"cutoff={cutoff} cannot supply requested history_s={history_s}: missing authentic frame ticks {missing_ticks}")
    actors = {r["entity_id"] for r in history if r["record_kind"] == "frame"}
    baseline = json.loads(graph_base.read_text())
    if baseline["episode_id"] != episode or baseline["tick"] != 0:
        raise ValueError("graph initial source does not match episode/tick0")
    if any(a["truth_state_update_tick"] != 0 or a["evidence_update_tick"] != 0 for a in baseline["initial_assertions"]):
        raise ValueError("initial graph contains later source evidence")
    delta_path = graph_base.with_name("world_truth_graph_deltas.jsonl")
    for family, native_path in (("world_truth_graph_base", graph_base), ("world_truth_graph_deltas", delta_path)):
        if any(Path(r["source"]["path"]).resolve() != native_path.resolve()
               for r in sample.input_records if r.get("source_family") == family):
            raise ValueError(f"canonical {family} source differs from native graph")
    active = {(a["predicate_id"], a["tuple_id"]): dict(a) for a in baseline["initial_assertions"]}
    deltas = []
    with open_artifact_text(delta_path) as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["tick"] > cutoff:
                break
            if row["episode_id"] != episode:
                raise ValueError("cross-episode graph delta")
            deltas.append((line_no, row))
    def apply(operation, observed_tick):
        body = operation["assertion"] if operation["operation"] == "add_predicate_assertion" else operation
        key = body["predicate_id"], body["tuple_id"]
        op = operation["operation"]
        if op == "add_predicate_assertion":
            if key in active:
                raise ValueError("duplicate active relation")
            active[key] = {**body, "evidence_update_tick": observed_tick}
        elif op == "remove_predicate_assertion":
            if active[key]["value"] != body["from_value"]:
                raise ValueError("relation removal diverges from current value")
            del active[key]
        elif op == "set_predicate_value":
            if active[key]["value"] != body["from_value"]:
                raise ValueError("relation value update diverges")
            active[key] = {**active[key], **body, "value": body["to_value"],
                           "truth_state_update_tick": observed_tick, "evidence_update_tick": observed_tick}
        elif op == "refresh_predicate_evidence":
            if active[key]["value"] != body["value"]:
                raise ValueError("evidence refresh diverges")
            active[key] = {**active[key], **body, "evidence_update_tick": observed_tick}
        else:
            raise ValueError(f"unsupported graph operation {op}")
    for _, row in deltas:
        if row["tick"] <= left:
            for op in row["operations"]:
                apply(op, row["tick"])
    semantic, bindings, graph_nodes = [], [], {}
    def register(body, source):
        for role, identity in body["bindings"].items():
            ontology = body["binding_ontology_classes"][role]
            if identity not in graph_nodes:
                graph_nodes[identity] = {"identity": identity, "ontology_classes": [], "role_bindings": []}
            node = graph_nodes[identity]
            if ontology not in node["ontology_classes"]:
                node["ontology_classes"].append(ontology)
            node["role_bindings"].append({"role": role, "predicate_id": body["predicate_id"], "source": source})
    for body in active.values():
        register(body, {"path": str(graph_base), "boundary_as_of": left})
        semantic.append({"kind": "relation", "tick": left, "entity": None,
                         "id": body["assertion_id"], "fields": {}, "relation": body})
        bindings.append({"id": body["assertion_id"], "tick": left, "paths": [str(graph_base), str(delta_path)], "source": body})
    for line_no, row in deltas:
        if row["tick"] <= left:
            continue
        for op_index, op in enumerate(row["operations"]):
            body = op["assertion"] if op["operation"] == "add_predicate_assertion" else op
            source = {"path": str(delta_path), "line": line_no, "operation_index": op_index}
            register(body, source)
            semantic.append({"kind": "update", "tick": row["tick"], "entity": None,
                             "id": body["assertion_id"], "fields": {}, "operation": op["operation"], "relation": body})
            bindings.append({**source, "tick": row["tick"], "source": op})
            apply(op, row["tick"])
    entities = {e: roster[e] for e in sorted(actors)}
    for r in history:
        if r.get("source_family") in ("world_truth_graph_base", "world_truth_graph_deltas") or r["record_kind"] == "episode":
            continue
        kind = r["record_kind"]
        if kind == "entity" and r["fields"]["roster.entity_id"] not in entities:
            continue
        fields = {k: v for k, v in r["fields"].items() if not k.startswith("archive.") and k not in MODEL_INPUT_EXCLUDED}
        semantic.append({"kind": kind, "tick": left if r.get("tick") is None else r["tick"],
                         "entity": r.get("entity_id", r["fields"].get("roster.entity_id")), "id": r["id"], "fields": fields})
        bindings.append({"id": r["id"], "tick": r.get("tick"), "source": r.get("source"), "fields": r["fields"]})
    states = {(r["tick"], r["entity_id"]): dict(r["fields"])
              for r in history + sample.target_records if r["record_kind"] == "frame"}
    comm_rows, comm_lines = {}, {}
    for line_no, line in enumerate(communication.open(), 1):
        row = json.loads(line)
        if row["episode_id"] != episode:
            raise ValueError("cross-episode communication")
        key = row["tick"], row["entity_id"]
        if key in comm_rows:
            raise ValueError(f"duplicate communication row {key}")
        comm_rows[key], comm_lines[key] = row, line_no
    for (tick, e), state in states.items():
        if e not in entities or entities[e]["roster.entity_category"] != "uav" or tick % 5:
            continue
        address_context = f"episode={episode} tick={tick} entity={e} source={communication}"
        if (tick, e) not in comm_rows:
            raise ValueError(f"missing authentic communication row: {address_context}")
        row = comm_rows[tick, e]
        if "scope_active" not in row or "model_version" not in row:
            raise ValueError(f"missing authentic communication scope/version: {address_context}")
        if row["scope_active"] is not True or row["model_version"] != "1.6.0":
            raise ValueError(f"unsupported communication scope/version: {address_context} scope_active={row['scope_active']} model_version={row['model_version']}")
        values = {}
        for name, (parent, leaf) in COMM_KEYS.items():
            if parent not in row or leaf not in row[parent]:
                raise ValueError(f"missing authentic communication field {name}: {address_context}")
            value = row[parent][leaf]
            if value == "unknown":
                values[name] = {"kind": "source_unknown"}
                continue
            dtype, _, bounds, vocab = COMM[name]
            if dtype == "real":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"invalid communication value {name}")
                if (bounds[0] is not None and value < bounds[0]) or (bounds[1] is not None and value > bounds[1]):
                    raise ValueError(f"out-of-support {name}")
            elif value not in vocab or (dtype == "bool" and type(value) is not bool):
                raise ValueError(f"undeclared communication value {name}")
            values[name] = value
        state.update(values)
        bindings.append({"tick": tick, "entity": e, "path": str(communication), "line": comm_lines[tick, e], "source": row})
        if tick <= cutoff:
            semantic.append({"kind": "communication", "tick": tick, "entity": e, "id": f"communication:{tick}:{e}", "fields": values})
    observer_ids = tuple(observer_ids)
    if any(observer not in entities for observer in observer_ids):
        raise ValueError("observer query identity lacks causal actor membership")
    # The camera (observer) position is the observer UAV's own pose; no separate
    # camera state enters the graph (derivations.py).
    # Only primary fields are query targets; derived fields (derivations.py) stay
    # history inputs and are computed from predicted primaries.
    fields = tuple(f for f in modelable_field_targets()[0] if f.family in PRIMARY_FIELDS)
    fields += tuple(FieldTarget(name, None, dtype, unit, None, None, 1, False, bounds[0], bounds[1], vocab, ("source_unknown",))
                    for name, (dtype, unit, bounds, vocab) in COMM.items() if name in PRIMARY_FIELDS)
    temporal_context = build_sumo_context(sumo_prefix, window_start=left, cutoff=cutoff,
                                          tick_seconds=tick_seconds, net_xml=sumo_net_xml)
    semantic.extend(temporal_context.records)
    bindings.extend(temporal_context.source_bindings)
    if keyframe_s is not None:
        semantic = compress_history(semantic, cutoff=cutoff, tick_seconds=tick_seconds, keyframe_s=keyframe_s,
                                    states={k: v for k, v in states.items() if k[0] <= cutoff},
                                    lane_positions=temporal_context.decoder_lane_positions)
    # Static native roster definitions are available at the window boundary, even
    # for actors without a state query. Admission to queries never reorders them.
    local_identities = {}
    for record in history:
        if record["record_kind"] == "entity" and record.get("tick") is None:
            identity = record["fields"]["roster.entity_id"]
            if identity not in local_identities:
                local_identities[identity] = len(local_identities)
    # Map identities are static and available at the boundary, not admitted by
    # future vehicle transitions. They do not become state prediction queries.
    for identity, definition in sorted(temporal_context.static_nodes.items()):
        if "known_from_tick" not in definition and identity not in local_identities:
            local_identities[identity] = len(local_identities)
    for record in sorted(semantic, key=lambda row: row["tick"]):
        identities = []
        if record["entity"] is not None:
            identities.append(record["entity"])
        if "relation" in record:
            identities.extend(record["relation"]["bindings"].values())
        if "context_relation" in record:
            identities.extend(record["context_relation"]["bindings"].values())
        for identity in identities:
            if identity not in local_identities:
                local_identities[identity] = len(local_identities)
    aliases = {f"local:{ordinal}": identity for identity, ordinal in local_identities.items()}
    return EpisodeIndex(episode, cutoff, tuple(targets), entities, states, semantic, fields, {}, {},
                        bindings, graph_nodes, baseline["predicate_vocabulary"], aliases, tick_seconds, left,
                        observer_ids, baseline["governed_parameters"], local_identities, split,
                        None if keyframe_s is None else round(keyframe_s/tick_seconds), temporal_context)


def _applicable(index, f):
    return [e for e in index.entities if not f.family.startswith("observer.") or e in index.observer_ids]


# A category's own residual rate is used once it has this many TRAIN off-atom
# residuals; sparser categories fall back to the pooled rate.
MIN_CATEGORY_RESIDUALS = 50


def entity_category(index, entity):
    return index.entities[entity]["roster.entity_category"]


def fit_state_normalization(indices, unit_grouping="pooled"):
    """Fit the one state normalization on TRAIN windows and return it.

    Input scales and categorical priors come from history cells (tick <= cutoff),
    each (episode, tick, entity) cell counted once across overlapping windows.
    ``residual_rate`` is the RMS of (future value - forecast anchor)/elapsed
    seconds over every supervised TRAIN target; it only sets the physical unit
    of the predicted mean offset and standard deviation.  Unbounded fields have
    an anchor atom, so their unit is fitted on the residuals off the atom.  A
    component with no such residual has no Gaussian unit; it is returned in
    ``constant`` and stays an input field but not a query target.

    ``unit_grouping="category"`` also fits the rate per roster entity category
    (``residual_rate_by_category``): residual scales differ by orders of
    magnitude between pedestrians, vehicles and UAVs, and a pooled unit makes
    every pedestrian Gaussian as wide as a UAV manoeuvre.
    """
    if unit_grouping not in ("pooled", "category"):
        raise ValueError(f"unknown unit grouping {unit_grouping}")
    if not indices:
        raise ValueError("state normalization needs at least one TRAIN window")
    if any(index.split != "TRAIN" for index in indices):
        raise ValueError("state normalization must be fitted on declared TRAIN windows only")
    fields = indices[0].fields
    if any(index.fields != fields for index in indices):
        raise ValueError("TRAIN windows declare different query field sets")
    scales, forecast_scales = {}, {}
    for f in fields:
        observed = {}
        for index in indices:
            applicable = set(_applicable(index, f))
            for (t, e), state in index.states.items():
                if t <= index.cutoff and e in applicable and f.family in state:
                    kind, value = cell(state[f.family], f.component)
                    if kind == "present":
                        observed[index.episode, t, e] = value
        if not observed:
            raise ValueError(f"TRAIN windows have no scale/prior authority for {f.family}:{f.component}")
        if f.logical_dtype != "real":
            counts = [0]*len(f.enum_values)
            for v in observed.values():
                counts[f.enum_values.index(v)] += 1
            forecast_scales[f.family, f.component] = {"categorical_counts": counts}
            continue
        scales[f.family, f.component] = fit_input_scale(f.family, f.component, observed.values())
        positive = [float(v)-f.lower for v in observed.values() if f.lower is not None and float(v) > f.lower]
        forecast_scales[f.family, f.component] = {"positive_mean": sum(positive)/len(positive) if positive else None}
    for index in indices:
        index.scales, index.forecast_scales = scales, forecast_scales
    residuals = {key: [] for key in scales}
    by_category = {key: {} for key in scales}
    atom_counts = {key: [0, 0] for key in scales}
    for index in indices:
        for tick in index.target_ticks:
            for address in index.addresses(tick):
                key = address.field_family, address.component
                if key not in scales:
                    continue
                state = index.states.get((tick, address.entity_id))
                if state is None or address.field_family not in state:
                    continue
                kind, truth = cell(state[address.field_family], address.component)
                terms = anchor_terms(index, address)
                if kind != "present" or not terms["observed"]:
                    continue
                anchor, elapsed = terms["anchor"], terms["elapsed"]
                f = next(f for f in fields if (f.family, f.component) == key)
                change = (float(truth)-anchor+180.) % 360.-180. if f.circular else float(truth)-anchor
                if unbounded_field(f):
                    atom = anchor_atom(index, address, float(truth), terms)[0]
                    atom_counts[key][atom] += 1
                    if atom:
                        continue
                residuals[key].append(change/elapsed)
                by_category[key].setdefault(entity_category(index, address.entity_id), []).append(change/elapsed)
    constant = []
    for key, values in residuals.items():
        if not values and not sum(atom_counts[key]):
            raise ValueError(f"TRAIN windows supervise no present target for {key[0]}:{key[1]}")
        rate = math.sqrt(sum(v*v for v in values)/len(values)) if values else 0.
        forecast_scales[key].update({"residual_rate": rate, "residual_count": len(values),
                                     "anchor_atom_fraction": atom_counts[key][1]/sum(atom_counts[key]) if sum(atom_counts[key]) else None,
                                     "atom_semantics": ATOM_SEMANTICS,
                                     "source": "TRAIN targets: RMS of (future value - forecast anchor)/elapsed seconds"
                                               ", off the anchor atom for unbounded fields"})
        if unit_grouping == "category":
            forecast_scales[key]["residual_rate_by_category"] = {
                category: math.sqrt(sum(v*v for v in group)/len(group))
                for category, group in sorted(by_category[key].items()) if len(group) >= MIN_CATEGORY_RESIDUALS}
            forecast_scales[key]["residual_count_by_category"] = {c: len(g) for c, g in sorted(by_category[key].items())}
        if rate == 0:
            constant.append(key)
    return scales, forecast_scales, tuple(constant)


def restrict_query_fields(index, constant):
    """Remove TRAIN-constant components from one window's query targets."""
    index.fields = tuple(f for f in index.fields if (f.family, f.component) not in constant)


def fit_observation_scales(indices):
    """Relation-evidence numeric scales, fitted once over TRAIN window records."""
    def collect(path, value, component=()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return [((path, component), float(value))]
        if isinstance(value, list):
            return [item for i, v in enumerate(value) for item in collect(path, v, component+(i,))]
        return []
    seen = {}
    for index in indices:
        if index.split != "TRAIN":
            raise ValueError("observation scales must be fitted on declared TRAIN windows only")
        for record in index.semantic_records:
            if "relation" not in record:
                continue
            for observation in record["relation"].get("observations", []):
                if TypedGraphEncoder._clock_key(observation["path"]):
                    continue
                for key, value in collect(observation["path"], observation["value"]):
                    # The same relation evidence recurs in overlapping windows.
                    seen[index.episode, record["tick"], record["id"], key] = (key, value)
    # Input-only SUMO/map context: serialized state/relation fields of TRAIN
    # windows, and every static map definition once per map version (the map is
    # time-invariant and carries no labels).
    maps = {}
    for index in indices:
        context = index.temporal_context
        if context is None:
            continue
        maps.setdefault(context.metadata["map_version"], context.static_nodes)
        for record in index.semantic_records:
            fields = (record["fields"] if record["kind"] == "sumo_state" else
                      record["context_relation"].get("fields", {}) if record["kind"] == "sumo_relation" else {})
            for name, value in fields.items():
                for key, number in context_numeric_leaves(name, value):
                    seen[index.episode, record["tick"], record["id"], ("context", key)] = (("context", key), number)
    for version, nodes in maps.items():
        for identity, node in nodes.items():
            for name, value in encoded_static_fields(node["fields"]).items():
                for i, (key, number) in enumerate(context_numeric_leaves(name, value)):
                    seen["map", version, identity, i, ("context", key)] = (("context", key), number)
    observations = {}
    for key, value in seen.values():
        if key not in observations:
            observations[key] = []
        observations[key].append(value)
    scales = {}
    for key, values in observations.items():
        mean = sum(values)/len(values)
        std = math.sqrt(sum((v-mean)**2 for v in values)/len(values))
        scales[key] = (mean, std if std else 1.)
    return scales



class _IncidenceAttention(nn.Module):
    """Eight-head attention on explicit directed incidence edges, no dense NxN mask."""
    def __init__(self, width):
        super().__init__()
        self.width, self.heads = width, 8
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x, edges):
        n, d = x.shape[0], self.width//self.heads
        q, k, v = self.qkv(self.norm1(x)).reshape(n, 3, self.heads, d).unbind(1)
        # Self edges make every node's attention well-defined.
        dest = torch.cat((edges[0], torch.arange(n, device=x.device)))
        source = torch.cat((edges[1], torch.arange(n, device=x.device)))
        score = (q[dest]*k[source]).sum(-1)/math.sqrt(d)
        groups = dest[:, None].expand(-1, self.heads)
        maxima = torch.full((n, self.heads), -torch.inf, device=x.device, dtype=score.dtype)
        maxima.scatter_reduce_(0, groups, score, reduce="amax", include_self=True)
        weights = (score-maxima[dest]).exp()
        denominator = torch.zeros_like(maxima).scatter_add_(0, groups, weights)
        weighted = v[source]*(weights/denominator[dest]).unsqueeze(-1)
        aggregate = torch.zeros_like(v).scatter_add_(0, dest[:, None, None].expand_as(weighted), weighted)
        x = x+self.out(aggregate.reshape(n, self.width))
        return x+self.ff(self.norm2(x))


class TypedGraphEncoder(nn.Module):
    """Typed field readouts, role ports, graph interaction and append-only time memory.

    Parameters are window-independent; ``encode_history`` binds one window's
    index, so a single module serves every window.
    """
    def __init__(self, observation_scales, hidden=2048, width=256, mode="graph_full"):
        super().__init__()
        if width != 256 or mode not in ("graph_full", "graph_summary", "record_ablation"):
            raise ValueError("unsupported structured encoder configuration")
        self.index, self.width, self.mode = None, width, mode
        registry = load_core_semantic_registry()
        self.templates = {t["id"]: t for t in registry["templates"]}
        self.field_names = sorted(FIELD_REGISTRY.keys() | COMM.keys())
        self.field_ids = {name: i for i, name in enumerate(self.field_names)}
        self.pred_ids = {name: i for i, name in enumerate(sorted(self.templates))}
        roles = sorted({role["key"] for t in self.templates.values() for role in t["argument_roles"]})
        self.role_ids = {name: i for i, name in enumerate(roles)}
        evidence = {r["field"] for t in self.templates.values() if t["implementation_status"] == "executable_l1" for r in t["state_contract"]["required_fields"]}
        # world_truth.py declares l0.scope_active as its scope-removal observation.
        evidence.add("l0.scope_active")
        self.evidence_ids = {name: i for i, name in enumerate(sorted(evidence))}
        self.evidence_units = {r["field"]: r["unit"] for t in self.templates.values() if t["implementation_status"] == "executable_l1" for r in t["state_contract"]["required_fields"]}
        self.evidence_units["l0.scope_active"] = "bool"
        self.observation_scales = observation_scales
        parameter_names = sorted(registry["parameter_governance"]["governed_defaults"])
        self.parameter_ids = {name: i for i, name in enumerate(parameter_names)}
        self.field_embedding = nn.Embedding(len(self.field_names), width)
        self.predicate_embedding = nn.Embedding(len(self.pred_ids), width)
        self.role_embedding = nn.Embedding(len(roles), width)
        self.evidence_embedding = nn.Embedding(len(evidence), width)
        self.parameter_embedding = nn.Embedding(len(parameter_names), width)
        self.status_embedding = nn.Embedding(len(STATUS), width)
        self.type_embedding = nn.Embedding(8, width)
        self.component_embedding = nn.Embedding(16, width)
        categories = {"true", "false", "unknown", "out_of_scope", "add_predicate_assertion", "remove_predicate_assertion", "set_predicate_value", "refresh_predicate_evidence"}
        for declaration in FIELD_REGISTRY.values():
            categories.update(str(value) for value in declaration.get("enum") or ())
        for _, _, _, enum in COMM.values():
            categories.update(str(value) for value in enum)
        for template in self.templates.values():
            categories.update("world:"+role["class"] for role in template["argument_roles"])
        categories.update(self.evidence_units.values())
        for name in self.field_names:
            declaration = self._declaration(name)
            categories.add(":".join(str(declaration.get(k)) for k in ("dtype", "unit", "frame", "axis_role", "vector_length", "circular")))
        self.category_ids = {value: i for i, value in enumerate(sorted(categories))}
        self.category_embedding = nn.Embedding(len(categories), width)
        self.byte_embedding = nn.Embedding(256, width)
        self.numeric = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.identity = nn.Linear(32, width)
        self.time = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.metadata = nn.Linear(32, width)
        self.readouts = nn.Parameter(torch.randn(2, width)*.02)
        # One slot for a state record that follows exactly from the entity's
        # previous record (compress_history).
        self.same_embedding = nn.Parameter(torch.randn(width)*.02)
        self.read_attention = nn.MultiheadAttention(width, 8, dropout=0, batch_first=True)
        self.read_ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.graph_layers = nn.ModuleList([_IncidenceAttention(width) for _ in range(2)])
        # Learned, zero-initialised prior-state link.  The link from a state slot
        # to the same entity's previous emitted slot starts neutral (zero weight
        # and bias) and is trained, instead of the previous magnitude-accumulating
        # raw addition.
        self.prior_link = nn.Linear(width, width)
        nn.init.zeros_(self.prior_link.weight)
        nn.init.zeros_(self.prior_link.bias)
        self.temporal_layers = nn.ModuleList([nn.MultiheadAttention(width, 8, dropout=0, batch_first=True) for _ in range(2)])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.temporal_ffs = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width)) for _ in range(2)])
        self.global_readouts = nn.Parameter(torch.randn(2, width)*.02)
        self.projection = nn.Linear(width, hidden)
        self.reconstruction_projection = nn.Linear(hidden, width)
        maximum_classes = max(len(self._declaration(name).get("enum") or ()) for name in self.field_names)
        self.reconstruction_heads = nn.ModuleDict({"numeric": nn.Linear(width, 2), "status": nn.Linear(width, len(STATUS)),
                                                  "categorical": nn.Linear(width, maximum_classes), "role": nn.Linear(width, 32),
                                                  "truth": nn.Linear(width, 4)})

    @property
    def device(self):
        return self.readouts.device

    def _bind(self, index):
        # One source-built causal table is shared with the address-query module.
        self.index = index
        self.local_ids = index.local_identities
        self.pointer_ids = {}
        # Pure functions of the current parameters, evaluated once per encode call.
        self._memo = {}

    def _memoized(self, key, compute):
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def _tag(self, ordinal, length=32):
        def compute():
            phase = torch.tensor(float(ordinal+1), device=self.device)*torch.exp(-torch.arange(length//2, device=self.device)*math.log(10000)/(length//2))
            return torch.cat((phase.sin(), phase.cos()))
        return self._memoized(("tag", ordinal, length), compute)

    def _identity(self, name):
        return self._memoized(("identity", name), lambda: self.identity(self._tag(self.local_ids[name])))

    def _text(self, value):
        return self._memoized(("text", value), lambda: self._text_embedding(value))

    def _text_embedding(self, value):
        if value in self.category_ids:
            return self.category_embedding.weight[self.category_ids[value]]
        if value in self.local_ids:
            return self._identity(value)
        if re.search(r"[0-9a-f]{16,}|(?:tick|frame)[_:]?[0-9]+", value) or value.startswith("/") or value.endswith((".json", ".jsonl", ".png", ".npz")):
            if value not in self.pointer_ids:
                self.pointer_ids[value] = len(self.pointer_ids)
                self.index.identity_aliases[f"pointer:{self.pointer_ids[value]}"] = value
            return self.identity(self._tag(self.pointer_ids[value]))+self.type_embedding.weight[7]
        encoded = list(value.encode("utf-8"))
        if not encoded:
            raise ValueError("empty semantic category")
        weights = torch.arange(1, len(encoded)+1, device=self.device, dtype=self.readouts.dtype)
        return (self.byte_embedding(torch.tensor(encoded, device=self.device))*weights[:, None]).sum(0)/weights.sum()

    def _time(self, tick, predicted):
        tau = (tick-self.index.cutoff)*self.index.tick_seconds
        return self._memoized(("time", tau, bool(predicted)), lambda: self.time(
            torch.tensor([tau, math.sin(tau), float(predicted)], device=self.device, dtype=self.readouts.dtype)))

    def _declaration(self, name):
        if name in FIELD_REGISTRY:
            return FIELD_REGISTRY[name]
        if name in COMM:
            dtype, unit, _, enum = COMM[name]
            return {"dtype": dtype, "unit": unit, "enum": enum, "frame": None, "axis_role": None, "vector_length": 1, "circular": False}
        raise ValueError(f"undeclared model field {name}")

    def _field(self, name, value, component=None):
        declaration = self._declaration(name)
        kind, actual = cell(value, None if isinstance(value, dict) and "value" in value and not isinstance(value["value"], list) else component)
        if declaration.get("unit") == "simulation_tick":
            raise ValueError(f"source clock field {name} belongs to the sidecar, not model fields")
        result = self.field_embedding.weight[self.field_ids[name]]+self.status_embedding.weight[STATUS.index(kind)]
        if isinstance(value, dict) and "record_probability" in value:
            probability = float(value["record_probability"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid predicted record probability")
            result = result+self.numeric(torch.tensor([0., 0., probability, 1.], device=self.device, dtype=self.readouts.dtype))
        metadata = ":".join(str(declaration.get(k)) for k in ("dtype", "unit", "frame", "axis_role", "vector_length", "circular"))
        # Unit/frame vocabulary is schema metadata, independent of source-window sizes.
        result = result+self._text(metadata)
        result = result+self.component_embedding.weight[0 if component is None else component+1]
        if kind != "present":
            return result
        if isinstance(actual, bool):
            return result+self.type_embedding.weight[1]+self.numeric(torch.tensor([float(actual), 0., 0., 1.], device=self.device, dtype=self.readouts.dtype))
        if isinstance(actual, (int, float)):
            scale = self.index.scales.get((name, component))
            v = scale.transform(actual) if scale is not None else float(actual)
            if not math.isfinite(v):
                raise ValueError(f"nonfinite field {name}")
            angle = math.radians(actual) if declaration.get("circular") else 0.
            values = [math.sin(angle), math.cos(angle), 0., 1.] if declaration.get("circular") else [v, 0., 0., 1.]
            return result+self.type_embedding.weight[0]+self.numeric(torch.tensor(values, device=self.device, dtype=self.readouts.dtype))
        if isinstance(actual, str):
            return result+self.type_embedding.weight[2]+self._text(actual)
        if isinstance(actual, dict):
            parts = [self._text(str(k))+self._value(v) for k, v in sorted(actual.items()) if not self._clock_key(k)]
            if not parts:
                return result+self.type_embedding.weight[3]
            return result+self.type_embedding.weight[3]+torch.stack(parts).mean(0)
        if isinstance(actual, list):
            return result+self.type_embedding.weight[4]+(self._value(actual))
        if actual is None:
            raise ValueError(f"undeclared null field {name}")
        raise TypeError(f"unsupported semantic value {name}")

    @staticmethod
    def _clock_key(name):
        # Source-maintenance clocks and opaque record addresses are sidecar-only.
        if name in FIELD_REGISTRY and FIELD_REGISTRY[name]["unit"] == "simulation_tick":
            return True
        leaf = name.rsplit(".", 1)[-1]
        return leaf in {"tick", "raw_tick", "sim_time_s", "absolute_time_s", "elapsed_s", "elapsed_ticks", "timestamp_ns", "frame_seq", "truth_state_update_tick", "evidence_update_tick", "event_id", "tuple_id", "assertion_id", "occurrence_id", "episode_id", "scenario_id", "source_refs", "source_ref", "path"}

    def _value(self, value):
        if isinstance(value, str):
            return self._text(value)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return self.numeric(torch.tensor([float(value), 0., 0., 1.], device=self.device, dtype=self.readouts.dtype))
        if isinstance(value, bool):
            return self.type_embedding.weight[int(value)]
        if isinstance(value, list):
            if not value:
                return self.type_embedding.weight[5]
            positioned = []
            for component, item in enumerate(value):
                tag = self._memoized(("metadata", component), lambda: self.metadata(self._tag(component)))
                positioned.append(self._value(item)*(1+tag.tanh())+tag)
            return torch.stack(positioned).mean(0)
        if isinstance(value, dict):
            parts = [self._text(str(k))+self._value(v) for k, v in sorted(value.items()) if not self._clock_key(k)]
            return torch.stack(parts).mean(0) if parts else self.type_embedding.weight[5]
        raise TypeError("untyped relation evidence value")

    def _observation(self, path, value, component=()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mean, std = self.observation_scales[path, component]
            return self.numeric(torch.tensor([(float(value)-mean)/std, 0., 0., 1.], device=self.device, dtype=self.readouts.dtype))
        if isinstance(value, list):
            values = []
            for i, item in enumerate(value):
                tag = self._memoized(("metadata", i), lambda: self.metadata(self._tag(i)))
                values.append(self._observation(path, item, component+(i,))*(1+tag.tanh())+tag)
            return torch.stack(values).mean(0) if values else self.type_embedding.weight[5]
        return self._value(value)

    def _fields(self, fields):
        result, names = [], []
        for name, value in sorted(fields.items()):
            if name.startswith("archive.") or name in MODEL_INPUT_EXCLUDED or self._clock_key(name):
                continue
            if name not in self.field_ids:
                raise ValueError(f"unregistered complete field key {name}")
            length = self._declaration(name).get("vector_length") or 1
            for component in (range(length) if length > 1 else (None,)):
                result.append(self._memoized(("field", name, component, dumps(value)),
                                             lambda: self._field(name, value, component)))
                names.append(name)
        return result, sorted(set(names))

    def _read_batch(self, requests):
        """Two readout slots per ``(values, identity, time)`` request, in one padded attention call."""
        if not requests:
            return []
        values = [items if items else [self.type_embedding.weight[5]] for items, _, _ in requests]
        lengths = torch.tensor([len(items) for items in values], device=self.device)
        flat = torch.stack([item for items in values for item in items])
        rows = torch.repeat_interleave(torch.arange(len(values), device=self.device), lengths)
        columns = torch.cat([torch.arange(len(items), device=self.device) for items in values])
        kv = flat.new_zeros((len(values), int(lengths.max()), self.width)).index_put((rows, columns), flat)
        padding = torch.arange(kv.shape[1], device=self.device)[None] >= lengths[:, None]
        q = self.readouts[None]+torch.stack([identity+time for _, identity, time in requests])[:, None]
        read = self.read_attention(q, kv, kv, key_padding_mask=padding, need_weights=False)[0]
        return list(q+read+self.read_ff(read))

    def _role_tag(self, predicate, role):
        unordered = self.templates[predicate]["grounding_spec"]["unordered_role_sets"]
        canonical = next((min(group) for group in unordered if role in group), role)
        return self.role_ids[canonical]

    def _relation_parts(self, body, operation=None):
        """Time-independent relation parts, reusable for an unchanged body.

        A relation body that is not updated keeps the same object across the
        window, so its role ports, evidence and predicate embedding are
        registered once and only re-dated per tick.  Returns
        ``(predicate, port_bases, evidence, names)``.
        """
        predicate = body["predicate_id"]
        template = self.templates[predicate]
        expected = tuple(role["key"] for role in template["argument_roles"])
        if set(body["bindings"]) != set(expected):
            raise ValueError(f"relation role binding diverges for {predicate}")
        pred = self.predicate_embedding.weight[self.pred_ids[predicate]]
        ports, names = [], []
        for role in expected:
            name = body["bindings"][role]
            # Symmetric role sets use the same role tag and invariant attention reduction.
            role_tag = self._role_tag(predicate, role)
            ports.append(self._identity(name)+self.role_embedding.weight[role_tag]+pred)
            names.append(name)
        evidence = []
        for observation in body.get("observations", []):
            path = observation["path"]
            if self._clock_key(path):
                continue
            if path == "l0.predicate_source_unavailability":
                # Native world_truth.py emits this explicit availability/reason
                # payload outside template required_fields; it is not absent truth.
                evidence.append(self._text(path)+self._value(observation["value"]))
                continue
            if path not in self.evidence_ids:
                raise ValueError(f"undeclared graph observation field {path}")
            evidence.append(self.evidence_embedding.weight[self.evidence_ids[path]]+self._text(self.evidence_units[path])+self._observation(path, observation["value"]))
        for parameter in template["state_contract"]["threshold_parameters"]:
            evidence.append(self.parameter_embedding.weight[self.parameter_ids[parameter]]+self._value(self.index.graph_parameters[parameter]))
        truth = body["to_value"] if "to_value" in body else body.get("value", body.get("from_value"))
        if truth not in ("true", "false", "unknown", "out_of_scope"):
            raise ValueError("undeclared relation truth state")
        evidence.append(self._text(truth))
        if operation is not None:
            evidence.append(self._text(operation))
        return pred, ports, evidence, names

    def _dated_relation(self, parts, tick, predicted):
        """Re-date cached relation parts; returns the read request and the dated ports."""
        pred, ports, evidence, names = parts
        time = self._time(tick, predicted)
        ports = [port+time for port in ports]
        return (ports+evidence, pred, time), ports, evidence, names

    def _context_fields(self, fields):
        """Input-only SUMO/map context; no forecast families or new head parameters."""
        return [self._memoized(("context_field", name, dumps(value)),
                              lambda name=name, value=value: self._text(name)+self._context_value(name, value))
                for name, value in sorted(fields.items()) if not self._clock_key(name)]

    def _context_value(self, path, value):
        """``_value`` with TRAIN-fitted standardization of every numeric leaf
        (context_numeric_leaves keys); raw ENU metres never enter the MLP."""
        if isinstance(value, bool) or isinstance(value, str):
            return self._value(value)
        if isinstance(value, (int, float)):
            key = ("context", path)
            if key not in self.observation_scales:
                raise ValueError(f"context numeric {path} has no TRAIN scale")
            mean, std = self.observation_scales[key]
            return self.numeric(torch.tensor([(float(value)-mean)/std, 0., 0., 1.], device=self.device, dtype=self.readouts.dtype))
        if isinstance(value, list):
            if not value:
                return self.type_embedding.weight[5]
            vector = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            if vector:
                # One batched MLP call per numeric vector (lane shape points).
                rows = []
                for i, item in enumerate(value):
                    key = ("context", f"{path}[{i}]")
                    if key not in self.observation_scales:
                        raise ValueError(f"context numeric {key[1]} has no TRAIN scale")
                    mean, std = self.observation_scales[key]
                    rows.append([(float(item)-mean)/std, 0., 0., 1.])
                numeric = self.numeric(torch.tensor(rows, device=self.device, dtype=self.readouts.dtype))
                tags = torch.stack([self._memoized(("metadata", i), lambda i=i: self.metadata(self._tag(i))) for i in range(len(value))])
                return (numeric*(1+tags.tanh())+tags).mean(0)
            positioned = []
            for i, item in enumerate(value):
                tag = self._memoized(("metadata", i), lambda: self.metadata(self._tag(i)))
                positioned.append(self._context_value(f"{path}[{i}]" if vector else path, item)*(1+tag.tanh())+tag)
            return torch.stack(positioned).mean(0)
        if isinstance(value, dict):
            parts = [self._text(str(k))+self._context_value(f"{path}.{k}", v)
                     for k, v in sorted(value.items()) if not self._clock_key(k)]
            return torch.stack(parts).mean(0) if parts else self.type_embedding.weight[5]
        raise TypeError("untyped context value")

    def _context_relation_parts(self, body, operation=None):
        pred = self._text("sumo."+body["relation_type"])
        names, ports = [], []
        for role, name in sorted(body["bindings"].items()):
            names.append(name)
            ports.append(self._identity(name)+self._text("sumo.role."+role)+pred)
        evidence = self._context_fields(body.get("fields", {}))+[self._value(body["value"])]
        if operation is not None:
            evidence.append(self._text("sumo.operation."+operation))
        return pred, ports, evidence, names

    def _freshness(self, tick, valid_from, observed):
        """Relative validity/freshness, never a future validity endpoint."""
        if valid_from > tick or observed > tick:
            raise ValueError("relation context contains future clocks")
        key = ("freshness", tick-valid_from, tick-observed)
        # log1p seconds: validity ages reach the episode length (~90 s).
        return self._memoized(key, lambda: self.numeric(torch.tensor(
            [math.log1p((tick-valid_from)*self.index.tick_seconds), math.log1p((tick-observed)*self.index.tick_seconds), 0., 1.],
            device=self.device, dtype=self.readouts.dtype)))

    def _reconstruction(self, projected, specifications, records, record_keys):
        """Shared auxiliary losses from emitted slots and historical source cells."""
        losses = {name: [] for name in ("field_numeric", "field_status", "field_categorical", "relation_roles", "relation_truth")}
        decoded = self.reconstruction_projection(projected)
        for ordinal, record in enumerate(records):
            if record.get("predicted") or record["kind"] in ("entity", "sumo_state", "sumo_relation"):
                continue
            key = record_keys[id(record)]
            support = [i for i, specification in enumerate(specifications) if specification[4] == key]
            if not support and self.mode == "graph_summary":
                support = [i for i, specification in enumerate(specifications) if specification[3] == "global"]
            if not support:
                continue
            context = decoded[support].mean(0)
            if "relation" in record:
                body = record["relation"]
                context = context+self.predicate_embedding.weight[self.pred_ids[body["predicate_id"]]]+self.metadata(self._tag(ordinal))
                for role, entity in body["bindings"].items():
                    role_context = context+self.role_embedding.weight[self.role_ids[role]]
                    losses["relation_roles"].append(F.mse_loss(self.reconstruction_heads["role"](role_context), self._tag(self.local_ids[entity])))
                truth = body["to_value"] if "to_value" in body else body.get("value", body.get("from_value"))
                truth_index = ("true", "false", "unknown", "out_of_scope").index(truth)
                losses["relation_truth"].append(F.cross_entropy(self.reconstruction_heads["truth"](context)[None], torch.tensor([truth_index], device=self.device)))
                continue
            for name, value in record["fields"].items():
                if name not in self.field_ids or self._clock_key(name) or name in MODEL_INPUT_EXCLUDED:
                    continue
                declaration = self._declaration(name)
                length = declaration.get("vector_length") or 1
                for component in (range(length) if length > 1 else (None,)):
                    kind, actual = cell(value, component)
                    field_context = context+self.field_embedding.weight[self.field_ids[name]]+self.component_embedding.weight[0 if component is None else component+1]
                    losses["field_status"].append(F.cross_entropy(self.reconstruction_heads["status"](field_context)[None], torch.tensor([STATUS.index(kind)], device=self.device)))
                    if kind != "present":
                        continue
                    if (name, component) in self.index.scales:
                        scale = self.index.scales[name, component]
                        target = [math.sin(math.radians(actual)), math.cos(math.radians(actual))] if declaration.get("circular") else [scale.transform(actual), 0.]
                        losses["field_numeric"].append(F.mse_loss(self.reconstruction_heads["numeric"](field_context), torch.tensor(target, device=self.device, dtype=field_context.dtype)))
                    elif declaration.get("enum") or declaration["dtype"] == "bool":
                        vocabulary = tuple(declaration["enum"]) if declaration.get("enum") else (False, True)
                        losses["field_categorical"].append(F.cross_entropy(self.reconstruction_heads["categorical"](field_context)[:len(vocabulary)][None], torch.tensor([vocabulary.index(actual)], device=self.device)))
        return losses

    def encode_history(self, index, modality_contexts=None, reconstruct=True, *, return_state=False):
        return self._encode(index, (), modality_contexts, reconstruct, return_state=return_state)

    def encode_future(self, index, feedback, history_state, modality_contexts=None, reconstruct=False):
        """Branch after one causal history encoding, retaining its autograd graph.

        Containers are copied for each branch; shared tensors are read-only.
        A state is valid only for this index and these encoder parameters.
        """
        return self._encode(index, feedback, modality_contexts, reconstruct, resume=history_state)

    def _encode(self, index, feedback, modality_contexts, reconstruct=True, *, resume=None, return_state=False):
        self._bind(index)
        records = [] if resume is not None else list(self.index.semantic_records)
        if resume is not None:
            if resume["owner"] != id(self) or resume["index"] != id(index):
                raise ValueError("graph continuation belongs to a different encoder or causal window")
            if resume["parameter_versions"] != tuple(p._version for p in self.parameters()):
                raise ValueError("graph continuation was invalidated by a parameter update")
            self.pointer_ids = dict(resume["pointer_ids"])
            self._memo = dict(resume["memo"])
        for tick, values in feedback:
            if resume is not None and tick <= index.cutoff:
                raise ValueError("a future branch cannot rewrite the encoded causal history")
            for entity, cells in values.items():
                if entity not in self.index.entities:
                    raise ValueError("feedback introduces a noncausal entity")
                groups = {"frame": {}, "communication": {}}
                for (family, component), payload in cells.items():
                    # Feedback re-enters as history input: any registered field,
                    # predicted (primary) or derived, is admissible.
                    if family not in self.field_ids:
                        raise ValueError("feedback address is not registered")
                    kind = payload["kind"]
                    if kind not in STATUS:
                        raise ValueError("feedback status is not declared")
                    group = groups["communication" if family.startswith("communication.") else "frame"]
                    length = self._declaration(family).get("vector_length") or 1
                    if length == 1:
                        group[family] = dict(payload)
                    else:
                        # Component statuses are kept separately until field encoding.
                        if family not in group:
                            group[family] = {}
                        group[family][component] = payload
                for category, fields in groups.items():
                    if fields:
                        # "Predicted" comes from provenance (this rollout feedback
                        # path), never from tick > cutoff.
                        records.append({"kind": category, "tick": tick, "entity": entity,
                                        "id": "predicted", "fields": fields, "predicted": True,
                                        "source": "rollout_feedback"})
        by_tick = {}
        for record in records:
            tick = record["tick"]
            if tick not in by_tick:
                by_tick[tick] = []
            by_tick[tick].append(record)
        if modality_contexts:
            for tick in modality_contexts:
                if resume is not None and tick <= index.cutoff:
                    continue
                if self.index.window_start <= tick <= self.index.cutoff or tick in [t for t, _ in feedback]:
                    if tick not in by_tick:
                        by_tick[tick] = []
        relation_cache, context_active, context_states = {}, {}, {}
        entity_relations, context_freshness = {}, {}
        static_nodes = {}
        emitted, entity_labels, field_labels, times, memory = [], [], [], [], []
        active, seen, append_lengths = {}, set(), []
        prior_state, seen_entities, seen_state_ticks = {}, set(), set()
        if resume is not None:
            relation_cache = dict(resume["relation_cache"])
            context_active = dict(resume["context_active"])
            context_states = {k: dict(v) for k, v in resume["context_states"].items()}
            entity_relations = {k: set(v) for k, v in resume["entity_relations"].items()}
            context_freshness = {k: dict(v) for k, v in resume["context_freshness"].items()}
            static_nodes = dict(resume["static_nodes"])
            memory = list(resume["memory"])
            active = dict(resume["active"])
            seen = set(resume["seen"])
            prior_state = dict(resume["prior_state"])
            seen_entities = set(resume["seen_entities"])
            seen_state_ticks = set(resume["seen_state_ticks"])
        reconstruction = {name: [] for name in ("field_numeric", "field_status", "field_categorical", "relation_roles", "relation_truth")}
        ledger = {"entity": 0, "state": 0, "same": 0, "communication": 0, "relation": 0, "update": 0,
                  "global": 0, "field_cells": 0, "new_entity": 0, "new_state": 0,
                  "sumo_state": 0, "sumo_relation": 0, "relation_cache_reads": 0,
                  "graph_sizes": [], "mode": self.mode,
                  "temporal_input_version": "sumo_incremental_v1"}
        for tick in sorted(by_tick):
            current = by_tick[tick]
            # "Predicted" is provenance: a record is predicted only when the
            # feedback path tagged it, never merely because tick > cutoff.
            predicted = any(bool(r.get("predicted")) for r in current)
            # Exact addresses select emitted slots; opaque tuple/source keys never
            # enter the model embedding or replace declared role bindings.
            record_keys = {}
            for ordinal, record in enumerate(current):
                if "relation" in record:
                    body = record["relation"]
                    key = record["kind"], body["predicate_id"], body["tuple_id"]
                    if record["kind"] == "update":
                        key = key+(tick, ordinal)
                else:
                    key = record["kind"], tick, ordinal
                record_keys[id(record)] = key
            # Replay EVERY delta, including non-emitting unchanged observations.
            # Only changed payloads are read again; clocks advance independently.
            touched = set()
            def unbind(key, body):
                for name in body["bindings"].values():
                    entity_relations.get(name, set()).discard(key)
            def bind(key, body):
                for name in body["bindings"].values():
                    entity_relations.setdefault(name, set()).add(key)
            for r in current:
                if r["kind"] == "sumo_state":
                    context_states.setdefault(r["entity"], {}).update(r["fields"])
                    # A value that became replay-only (decoder-recomputable) makes
                    # the last serialized copy stale; it leaves the context.
                    for name in r.get("derived_fields", {}):
                        context_states[r["entity"]].pop(name, None)
                    fresh = context_freshness.setdefault(r["entity"], {})
                    for name in FRESHNESS_FIELDS:
                        # Observed unchanged: same validity, new observation tick.
                        if name in fresh and name in r.get("observed", ()) and name not in r.get("freshness", {}):
                            fresh[name] = {**fresh[name], "observation_tick": r["tick"]}
                    fresh.update({k: v for k, v in r.get("freshness", {}).items() if k in FRESHNESS_FIELDS})
                if "context_relation" in r:
                    body = r["context_relation"]
                    key = ("sumo", body["key"])
                    if r["operation"] == "remove":
                        old = context_active.pop(key)
                        unbind(key, old); relation_cache.pop(key, None)
                    else:
                        old = context_active.get(key)
                        if old is not None:
                            unbind(key, old)
                        context_active[key] = {**(old or {}), **body,
                                               "fields": {**(old or {}).get("fields", {}), **body.get("fields", {})}}
                        bind(key, context_active[key]); touched.add(key)
                if "relation" not in r:
                    continue
                body = r["relation"]
                key = body["predicate_id"], body["tuple_id"]
                if r["kind"] == "relation" or r["operation"] == "add_predicate_assertion":
                    active[key] = {k: v for k, v in body.items() if k != "derived_evidence_paths"}
                    if r["kind"] != "relation":
                        active[key]["evidence_update_tick"] = tick
                    bind(key, active[key]); touched.add(key)
                elif r["operation"] == "remove_predicate_assertion":
                    old = active.pop(key)
                    unbind(key, old); relation_cache.pop(key, None)
                else:
                    old = active[key]
                    value = body["to_value"] if "to_value" in body else body["value"]
                    active[key] = {**old, **body, "value": value, "evidence_update_tick": tick,
                                   "truth_state_update_tick": tick if r["operation"] == "set_predicate_value"
                                   else old["truth_state_update_tick"]}
                    # Operation metadata is not part of the current assertion.
                    for leaf in ("to_value", "from_value", "operation"):
                        active[key].pop(leaf, None)
                    # Evidence recomputable from serialized states is read as a
                    # "derived" token, so its per-tick value change is not
                    # re-encoded (compress_history.derivable_evidence).
                    if r["operation"] == "refresh_predicate_evidence" and "changed_evidence_paths" in r:
                        derived = set(old.get("derived_evidence_paths", ()))
                        derived = (derived-set(r["changed_evidence_paths"]))|set(r["derived_evidence_paths"])
                    else:
                        # A truth-value update or unannotated refresh carries its
                        # own raw evidence: nothing stays masked.
                        derived = set()
                    active[key]["derived_evidence_paths"] = sorted(derived)
                    touched.add(key)
            cache_requests, cache_keys = [], []
            for key in sorted(touched):
                body = context_active.get(key) if key[0] == "sumo" else active.get(key)
                if body is None:
                    continue  # add+remove within one observation tick
                if key[0] != "sumo" and body.get("derived_evidence_paths"):
                    masked = set(body["derived_evidence_paths"])
                    body = {**body, "observations": [{**o, "value": "derived_from_serialized_state"} if o["path"] in masked else o
                                                     for o in body.get("observations", [])]}
                payload = ({name: body[name] for name in ("relation_type", "bindings", "fields", "value")}
                           if key[0] == "sumo" else relation_payload(body))
                signature = json.dumps(model_payload(payload), sort_keys=True, ensure_ascii=False, allow_nan=False)
                cached = relation_cache.get(key)
                if cached is not None and cached["signature"] == signature:
                    continue
                parts = self._context_relation_parts(body) if key[0] == "sumo" else self._relation_parts(body)
                pred, ports, evidence, names = parts
                cache_requests.append((ports+evidence, pred, torch.zeros_like(pred)))
                cache_keys.append((key, signature, parts))
            for (key, signature, parts), slots in zip(cache_keys, self._read_batch(cache_requests)):
                relation_cache[key] = {"signature": signature, "slots": slots, "parts": parts}
            ledger["relation_cache_reads"] += len(cache_keys)
            involved = set()
            ontology_by_identity = {}
            def bind_types(body):
                for role, identity in body["bindings"].items():
                    if identity not in ontology_by_identity:
                        ontology_by_identity[identity] = set()
                    ontology_by_identity[identity].add(body["binding_ontology_classes"][role])
            for body in active.values():
                involved.update(body["bindings"].values())
                bind_types(body)
            for body in context_active.values():
                involved.update(body["bindings"].values())
            for r in current:
                if r["entity"] is not None:
                    involved.add(r["entity"])
                if "relation" in r:
                    involved.update(r["relation"]["bindings"].values())
                    bind_types(r["relation"])
                if "context_relation" in r:
                    involved.update(r["context_relation"]["bindings"].values())
            contexts = None if modality_contexts is None else modality_contexts.get(tick)
            if contexts is not None:
                for names in contexts["entity_ids"]:
                    involved.update(names)
            unknown = involved-set(self.local_ids)
            if unknown:
                raise ValueError(f"modality/graph binding lacks causal identity: {sorted(unknown)}")
            nodes, edges, identities, output_specs = [], [], {}, []
            def add(value):
                nodes.append(value)
                return len(nodes)-1
            def connect(a, b):
                edges.extend(((a, b), (b, a)))
            # Static information (roster attributes) is registered once per
            # entity and referenced per tick; relation-only identities may
            # accrete ontology classes, so their static part is rebuilt.  The
            # node index stays stable because entities are visited in sorted
            # order.
            for name in sorted(involved, key=self.local_ids.__getitem__):
                value = self._identity(name)+self._time(tick, predicted)
                map_nodes = {} if self.index.temporal_context is None else self.index.temporal_context.static_nodes
                definition = map_nodes.get(name, {})
                known_from = definition.get("known_from_tick", self.index.window_start)
                # Relation-only identities may accrete ontology classes; their
                # static part is rebuilt only when the class set changes, and a
                # late-known map definition only from its first observation.
                classes = () if name in self.index.entities else tuple(sorted(ontology_by_identity.get(name, ())))
                version = (classes, "known_from_tick" not in definition or known_from <= tick)
                cached_static = static_nodes.get(name)
                static = cached_static[1] if cached_static is not None and cached_static[0] == version else None
                if static is None:
                    static = self._identity(name)
                    if name in self.index.entities:
                        roster, _ = self._fields(self.index.entities[name])
                        if roster:
                            static = static+torch.stack(roster).mean(0)
                    elif name in self.index.graph_nodes:
                        if classes:
                            static = static+torch.stack([self.category_embedding.weight[self.category_ids[cls]] for cls in classes]).mean(0)
                    if name in map_nodes and known_from <= tick:
                        values = self._context_fields(encoded_static_fields(map_nodes[name]["fields"]))
                        if values:
                            static = static+torch.stack(values).mean(0)
                    static_nodes[name] = (version, static)
                value = value+static
                identities[name] = add(value)
                if name not in seen:
                    output_specs.append((identities[name], [name], list(self.index.entities[name]) if name in self.index.entities else [], "entity", None, "new_entity", None))
                    seen.add(name)
            # Entity-specific, one-hop context: unchanged relation reads and map
            # definitions are reused, with current validity/observation ages.
            # These are pre-incidence values and never overwritten by graph
            # outputs; no cached tensor can ingest a later state or lose gradients.
            # Each recipient's context is the mean of its support elements.  The
            # sum is gathered from a per-tick bank of distinct tensors with one
            # index_add (instead of per-element adds); an element "slot + age +
            # role" contributes three bank rows but counts once.
            bank, bank_ids, rows, owners, counts = [], {}, [], [], {}
            def ref(key, compute):
                if key not in bank_ids:
                    bank_ids[key] = len(bank)
                    bank.append(compute())
                return bank_ids[key]
            def discrete(fields):
                return {k: v for k, v in fields.items() if k not in DYNAMIC_NUMERIC_FIELDS}
            def context_refs(entity):
                fields = discrete(context_states.get(entity, {}))
                return [ref(("context_field", k, dumps(v)), lambda k=k, v=v: self._context_fields({k: v})[0])
                        for k, v in sorted(fields.items()) if not self._clock_key(k)]
            recipients = sorted({r["entity"] for r in current if r["entity"] is not None and r["kind"] != "entity"},
                                key=self.local_ids.__getitem__)
            for slot_index, name in enumerate(recipients):
                elements = []  # each element: bank rows summed into one support value
                # Discrete context only: per-tick numerics are reconstructible from
                # the state slots (sumo_temporal.DYNAMIC_NUMERIC_FIELDS) and a
                # last-serialized copy would be stale.
                elements.extend([row] for row in context_refs(name))
                for field in FRESHNESS_FIELDS:
                    clock = context_freshness.get(name, {}).get(field)
                    if clock is not None:
                        valid, observed = clock.get("changed_tick", tick), clock.get("observation_tick", tick)
                        elements.append([ref(("text", field), lambda field=field: self._text(field)),
                                         ref(("freshness", tick-valid, tick-observed), lambda v=valid, o=observed: self._freshness(tick, v, o))])
                for key in sorted(entity_relations.get(name, ())):
                    body = context_active[key] if key[0] == "sumo" else active[key]
                    slots = relation_cache[key]["slots"]
                    valid = body.get("valid_from_tick", body.get("truth_state_update_tick", tick))
                    observed = body.get("observed_tick", body.get("evidence_update_tick", tick))
                    age = [ref(("freshness", tick-valid, tick-observed), lambda v=valid, o=observed: self._freshness(tick, v, o)),
                           ref(("time", tick), lambda: self._time(tick, predicted))]
                    roles = [ref(("sumo_role", r), lambda r=r: self._text("sumo.role."+r)) if key[0] == "sumo" else
                             ref(("role", self._role_tag(body["predicate_id"], r)),
                                 lambda r=r: self.role_embedding.weight[self._role_tag(body["predicate_id"], r)])
                             for r, identity in body["bindings"].items() if identity == name]
                    for i in range(len(slots)):
                        elements.append([ref(("relation_slot", key, i), lambda i=i: slots[i])]+age+roles)
                    for bound in body["bindings"].values():
                        if bound != name:
                            elements.append([ref(("static", bound), lambda bound=bound: static_nodes[bound][1])])
                            elements.extend([row] for row in context_refs(bound))
                if elements:
                    counts[name] = len(elements)
                    for element in elements:
                        rows.extend(element); owners.extend([len(counts)-1]*len(element))
            context_nodes = {}
            if counts:
                # means = W @ bank with W[recipient, row] = multiplicity/count; a
                # sparse product keeps only W (not the gathered rows) for backward.
                owner = torch.tensor(owners, device=self.device)
                weight = 1/torch.tensor(list(counts.values()), device=self.device, dtype=self.readouts.dtype)
                mixing = torch.sparse_coo_tensor(torch.stack((owner, torch.tensor(rows, device=self.device))), weight[owner],
                                                 (len(counts), len(bank))).coalesce()
                means = torch.sparse.mm(mixing, torch.stack(bank))
                context_nodes = {name: add(means[i]) for i, name in enumerate(counts)}
            def attach_context(slot, entity):
                if entity in context_nodes and self.mode != "record_ablation":
                    # Edge tensor is (destination, source). Keep context read-only.
                    edges.append((slot, context_nodes[entity]))
            boundary_keys = {(r["relation"]["predicate_id"], r["relation"]["tuple_id"]) for r in current if r["kind"] == "relation"}
            # All of this tick's slot reads run as one batch; nodes are then
            # added in the original relation/record order.
            requests, relation_items, record_items = [], [], []
            # Full relation incidence is emitted only at the boundary. Later
            # states/SAME slots read the incrementally maintained cache directly,
            # even between keyframes; keyframes now govern state compression only.
            for key in sorted(boundary_keys):
                body, cached = active[key], relation_cache[key]
                _, ports, evidence, names = self._dated_relation(cached["parts"], tick, predicted)
                slots = cached["slots"]+self._time(tick, predicted)
                relation_items.append((key, body, ports, evidence, names, slots))
            for r in current:
                if r["kind"] in ("relation", "entity", "same") or r.get("emit") is False:
                    continue
                if r["kind"] in ("update", "sumo_relation"):
                    parts = (self._context_relation_parts(r["context_relation"], r["operation"])
                             if r["kind"] == "sumo_relation" else self._relation_parts(r["relation"], r["operation"]))
                    request, ports, evidence, names = self._dated_relation(parts, tick, predicted)
                    requests.append(request); record_items.append((ports, evidence, names))
                    continue
                if r["kind"] == "sumo_state":
                    field_values, names = self._context_fields(r["fields"]), sorted(r["fields"])
                elif r.get("source") == "rollout_feedback":
                    # Feedback vectors store one typed payload per component.
                    # Pass each cell directly to _field: cell(vector, axis)
                    # otherwise returns the nested payload as a present dict,
                    # bypassing native scaling and losing unknown statuses.
                    field_values, names = [], []
                    for name, value in sorted(r["fields"].items()):
                        length = self._declaration(name).get("vector_length") or 1
                        for component in range(length) if length > 1 else (None,):
                            payload = value[component] if component is not None else value
                            field_values.append(self._memoized(("feedback_field", name, component, dumps(payload)),
                                                lambda n=name, p=payload, c=component: self._field(n, p, c)))
                            names.append(name)
                    names = sorted(set(names))
                else:
                    field_values, names = self._fields(r["fields"])
                identity = self._identity(r["entity"]) if r["entity"] is not None else self.type_embedding.weight[6]
                # In record_ablation, leave state context isolated by contract.
                # One compact context tensor avoids padding all tiny update
                # reads to the largest entity's full active-relation support.
                context_node = context_nodes.get(r["entity"])
                support = [nodes[context_node]] if context_node is not None and self.mode != "record_ablation" else []
                requests.append((field_values+support, identity, self._time(tick, predicted)))
                record_items.append((field_values, names))
            reads = iter(self._read_batch(requests))
            record_items = iter(record_items)
            for key, body, ports, evidence, names, slots in relation_items:
                slot_ids = [add(s) for s in slots]
                for port, name in zip(ports, names):
                    p = add(port)
                    connect(p, identities[name])
                    for s in slot_ids:
                        connect(p, s)
                for value in evidence:
                    p = add(value)
                    for s in slot_ids:
                        connect(p, s)
                if key in boundary_keys:
                    for s in slot_ids:
                        output_specs.append((s, names, ["relation."+body["predicate_id"]], "relation", ("relation", *key), "new_state", None))
            for r in current:
                kind, entity = r["kind"], r["entity"]
                if kind in ("relation", "entity") or r.get("emit") is False:
                    continue
                if kind == "same":
                    # Identity, time and the elapsed gap; the entity's previous
                    # slot enters through prior_link in the temporal layer.
                    elapsed = torch.tensor([r["elapsed_ticks"]*self.index.tick_seconds, 0., 0., 1.],
                                           device=self.device, dtype=self.readouts.dtype)
                    s = add(self._identity(entity)+self._time(tick, predicted)+self.same_embedding+self.numeric(elapsed))
                    connect(s, identities[entity])
                    attach_context(s, entity)
                    if r.get("position_unchanged"):
                        nodes[s] = nodes[s]+self._text("same.position_unchanged")
                    correction = r.get("position_rounding_residual_units")
                    if correction is not None:
                        # Encode source-grid units, not sub-micro metre features
                        # that disappear when the prefix is projected to BF16.
                        residual = [float(Fraction(v)) for v in correction]
                        nodes[s] = nodes[s]+self._text("same.rounding_residual_source_units")+self._value(residual)
                    output_specs.append((s, [entity], r["field_names"], "same", record_keys[id(r)], "new_state", entity))
                    seen_entities.add(entity); seen_state_ticks.add((entity, tick))
                    continue
                if kind in ("update", "sumo_relation"):
                    slots = next(reads)
                    ports, evidence, names = next(record_items)
                    slot_ids = [add(s) for s in slots]
                    for port, name in zip(ports, names):
                        p = add(port)
                        connect(p, identities[name])
                        for s in slot_ids:
                            connect(p, s)
                    for value in evidence:
                        p = add(value)
                        for s in slot_ids:
                            connect(p, s)
                    family = ("sumo.relation."+r["context_relation"]["relation_type"] if kind == "sumo_relation"
                              else "relation."+r["relation"]["predicate_id"])
                    for s in slot_ids:
                        output_specs.append((s, names, [family], kind, record_keys[id(r)], "update", None))
                    continue
                slots = next(reads)
                field_values, names = next(record_items)
                ledger["field_cells"] += len(field_values)
                # Explicit entity/state identity, classified per appended
                # record: new_entity when the actor first appears in the
                # window, new_state for a first slot of a known actor at a new
                # tau, update for any later slot at an already-emitted
                # (entity, tau).  Every slot carries its entity and the
                # actor's prior emitted slot, consumed by the temporal layer.
                category = ("sumo_state" if kind == "sumo_state" else "communication" if kind == "communication"
                            else "state" if kind == "frame" else "update")
                if entity is not None and (entity, tick) in seen_state_ticks:
                    classification = "update"
                elif entity is not None and entity in seen_entities:
                    classification = "new_state"
                else:
                    classification = "new_entity"
                # spec[6] is the entity whose prior emitted slot this
                # slot links to (None for a first-seen entity).
                prior_link = None if classification == "new_entity" or kind == "sumo_state" else entity
                for slot in slots:
                    s = add(slot)
                    output_specs.append((s, [entity] if entity is not None else [], names, category,
                                         record_keys[id(r)], classification, prior_link))
                    if entity is not None:
                        connect(s, identities[entity])
                        attach_context(s, entity)
                    for value in field_values:
                        p = add(value)
                        connect(s, p)
                ledger["new_entity" if classification == "new_entity" else
                       "new_state" if classification == "new_state" else "update"] += len(slots)
                if entity is not None:
                    seen_entities.add(entity)
                    seen_state_ticks.add((entity, tick))
            modality_ids = []
            if contexts is not None:
                features = contexts["features"].to(device=self.device, dtype=self.readouts.dtype)
                if features.ndim != 2 or features.shape[1] != self.width or len(features) != len(contexts["entity_ids"]):
                    raise ValueError("invalid modality graph context shape")
                for feature, names in zip(features, contexts["entity_ids"]):
                    p = add(feature+self._time(tick, predicted))
                    modality_ids.append(p)
                    for name in names:
                        connect(p, identities[name])
            graph = torch.stack(nodes)
            edge_tensor = torch.tensor(edges, device=self.device, dtype=torch.long).T if edges else torch.empty(2, 0, device=self.device, dtype=torch.long)
            ledger["graph_sizes"].append({"tick": tick, "nodes": len(nodes), "directed_edges": len(edges),
                                          "active_relations": len(active), "active_sumo_relations": len(context_active),
                                          "context_nodes": len(context_nodes), "relation_reads": len(cache_keys),
                                          "modality_nodes": len(modality_ids)})
            if self.mode != "record_ablation":
                for layer in self.graph_layers:
                    # Edge-sized attention activations are recomputed in backward.
                    graph = (checkpoint(layer, graph, edge_tensor, use_reentrant=False)
                             if torch.is_grad_enabled() else layer(graph, edge_tensor))
            if self.mode == "graph_summary":
                output_specs = [spec for spec in output_specs if spec[3] in ("entity", "state", "communication")]
                globals_ = self.read_attention(self.global_readouts[None], graph[None], graph[None], need_weights=False)[0][0]
                start = len(graph)
                graph = torch.cat((graph, globals_), dim=0)
                output_specs.extend((start+i, sorted(involved), [], "global", None, "new_state", None) for i in range(2))
            selected = graph[[spec[0] for spec in output_specs]]
            # The temporal layer consumes the explicit prior-state link stored
            # per state slot.  A slot whose single entity has a previous emitted
            # field-state slot is mixed with that prior embedding through a
            # learned zero-initialised projection, so the (entity, prior_state)
            # relation is used rather than merely stored and the link starts
            # neutral (no magnitude accumulation).
            if self.mode != "record_ablation":
                if prior_state:
                    rows = []
                    for index, spec in enumerate(output_specs):
                        prior = prior_state.get(spec[6]) if spec[6] is not None else None
                        rows.append(selected[index] if prior is None else
                                    selected[index]+self.prior_link(prior.to(selected.dtype)))
                    selected = torch.stack(rows)
                for layer, norm, ff in zip(self.temporal_layers, self.temporal_norms, self.temporal_ffs):
                    if memory:
                        kv = torch.cat(memory, dim=0)[None]
                        selected = selected+layer(norm(selected)[None], norm(kv), norm(kv), need_weights=False)[0][0]
                    selected = selected+ff(selected)
            memory.append(selected)
            # Key the prior state only from that single entity's own
            # field-state slots (state/communication/update).  Relation slots
            # carry multiple role entities and global specs are appended last,
            # so either would overwrite the alphabetically first involved
            # entity's prior state.
            prior_state.update({spec[1][0]: selected[index] for index, spec in enumerate(output_specs)
                                if len(spec[1]) == 1 and spec[3] in ("state", "same", "communication", "update")
                                and not (spec[2] and spec[2][0].startswith("relation."))})
            projected = self.projection(selected)
            if reconstruct:
                auxiliary = self._reconstruction(projected, output_specs, current, record_keys)
                for category, values in auxiliary.items():
                    reconstruction[category].extend(values)
            emitted.append(projected)
            append_lengths.append(len(selected))
            tau = (tick-self.index.cutoff)*self.index.tick_seconds
            times.extend([tau]*len(selected))
            for spec in output_specs:
                names, fields, category = spec[1], spec[2], spec[3]
                entity_labels.append(names)
                field_labels.append(fields)
                ledger[category] += 1
        result = torch.cat(emitted, dim=0)[None]
        length = result.shape[1]
        positions = torch.arange(length, device=self.device).view(1, 1, length).expand(3, 1, length)
        ledger["total_slots"] = length
        ledger["records"] = {kind: sum(r["kind"] == kind for r in records) for kind in sorted({r["kind"] for r in records})}
        reconstruction = {name: torch.stack(values).mean() if values else result.sum()*0. for name, values in reconstruction.items()}
        reconstruction["total"] = sum(reconstruction.values())
        output = {"embeddings": result, "ledger": ledger, "reconstruction_losses": reconstruction, "token_entities": entity_labels,
                  "token_fields": field_labels, "times": torch.tensor(times, device=self.device),
                  "positions": positions, "append_lengths": append_lengths}
        if return_state:
            output["continuation_state"] = {
                "owner": id(self), "index": id(self.index),
                "parameter_versions": tuple(p._version for p in self.parameters()),
                "pointer_ids": self.pointer_ids, "memo": self._memo,
                "relation_cache": relation_cache, "context_active": context_active,
                "context_states": context_states, "entity_relations": entity_relations,
                "context_freshness": context_freshness, "static_nodes": static_nodes,
                "memory": memory, "active": active, "seen": seen, "prior_state": prior_state,
                "seen_entities": seen_entities, "seen_state_ticks": seen_state_ticks,
            }
        return output


# No blanket "state-derivable" evidence suppression: exact synchronous
# longitudinal lane/leader coverage is not guaranteed by observed poses/maps.
def _same_value(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def record_predictable(previous, current, elapsed_ticks, tick_seconds):
    """Whether a state record follows exactly from the entity's previous record.

    All fields except position are equal and position equals the previous
    position plus the (unchanged) velocity times the elapsed time within the
    source rounding bound (anchor_atom rule).  Values off the source grid fall
    back to exact equality.
    """
    if set(previous) != set(current):
        return False
    if any(not _same_value(previous[k], current[k]) for k in current if k != "pose.position_enu_m"):
        return False
    if "pose.position_enu_m" not in current:
        return True
    p0, p1 = cell(previous["pose.position_enu_m"], None), cell(current["pose.position_enu_m"], None)
    if p0[0] != "present" or p1[0] != "present":
        return _same_value(previous["pose.position_enu_m"], current["pose.position_enu_m"])
    velocity = cell(current["pose.velocity_enu_mps"], None) if "pose.velocity_enu_mps" in current else ("absent", None)
    elapsed = Fraction(elapsed_ticks)*Fraction(tick_seconds).limit_denominator(10**6)
    bound = 1+elapsed/2
    try:
        for i in range(3):
            start = _decimal_units(p0[1][i], 6, "history"); end = _decimal_units(p1[1][i], 6, "history")
            step = _decimal_units(velocity[1][i], 6, "history")*elapsed if velocity[0] == "present" else 0
            if abs(end-start-step) > bound:
                return False
    except ValueError:
        return _same_value(previous["pose.position_enu_m"], current["pose.position_enu_m"])
    return True


# Formal evidence a decoder recomputes from the same tick's serialized inputs.
# Each rule is checked per record (``derivable_evidence``); a refresh is silent
# only when every changed path passes.  Measured on L4-1@310 and L4-3@545:
# geometry.speed_mps = |v| (max error 9e-16), geometry.position_z_m = z (0),
# truth.speed_mps = SUMO speed (0; |v| within 7e-7), truth.following_distance_m
# = max(0, leader centre - follower centre) lane position (0).  The following
# check uses the DECODER's lane positions (serialized or integrated, each within
# 2e-6 m of the source, sumo_temporal._reconstruction), hence 5e-6 m.
# derived.nearest_*_distance_m is the 3-D distance between the relation's two
# bound entities at that tick: within 1e-6 for 11,711 of 12,137 changed values
# in L4-8@660 and L4-1@310; the rest (an endpoint without a same-tick pose, the
# aircraft separation model) fail the per-record check and stay serialized.
PAIR_DISTANCE_EVIDENCE = frozenset({"derived.nearest_pedestrian_distance_m", "derived.nearest_ground_vehicle_distance_m",
                                    "derived.nearest_aircraft_distance_m"})
DERIVABLE_EVIDENCE_TOLERANCE = {"geometry.speed_mps": 1e-9, "geometry.position_z_m": 1e-9,
                                "truth.speed_mps": 1e-6, "truth.following_distance_m": 5e-6,
                                **{path: 1e-6 for path in PAIR_DISTANCE_EVIDENCE}}


def derivable_evidence(path, value, bindings, tick, states, lane_positions):
    """Whether one formal evidence value follows from same-tick serialized inputs.

    ``lane_positions``: (tick, vehicle) -> (lane, decoder lane position, length).
    """
    if path not in DERIVABLE_EVIDENCE_TOLERANCE or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    def present(entity, family):
        state = states.get((tick, entity))
        if state is None or family not in state:
            return None
        kind, actual = cell(state[family], None)
        return actual if kind == "present" else None
    if path == "truth.following_distance_m":
        follower, leader = (lane_positions.get((tick, bindings.get(role))) for role in ("following_vehicle", "leading_vehicle"))
        if not follower or not leader or follower[0] != leader[0] or follower[2] is None or leader[2] is None:
            return False
        estimate = max(0., (leader[1]-leader[2]/2)-(follower[1]-follower[2]/2))
    elif path in PAIR_DISTANCE_EVIDENCE:
        ends = [present(entity, "pose.position_enu_m") for entity in bindings.values()]
        if len(ends) != 2 or any(not isinstance(end, list) or len(end) != 3 for end in ends):
            return False
        estimate = math.dist([float(v) for v in ends[0]], [float(v) for v in ends[1]])
    else:
        roles = ("vehicle",) if path == "truth.speed_mps" else ("actor", "aircraft")
        subjects = [bindings[role] for role in roles if role in bindings]
        if len(subjects) != 1:
            return False
        family = "pose.position_enu_m" if path == "geometry.position_z_m" else "pose.velocity_enu_mps"
        vector = present(subjects[0], family)
        if not isinstance(vector, list) or len(vector) != 3:
            return False
        estimate = vector[2] if path == "geometry.position_z_m" else math.sqrt(sum(float(v)**2 for v in vector))
    return abs(float(value)-estimate) <= DERIVABLE_EVIDENCE_TOLERANCE[path]


def compress_history(semantic, *, cutoff, tick_seconds, keyframe_s, states=None, lane_positions=None):
    """Compress a long history window without losing reconstructible information.

    Consecutive observed, predictable state records become one ``same`` slot
    except at keyframes. Exact rational rounding residuals make reconstruction
    lossless; an observation gap can never become SAME. Evidence refreshes are
    replayed but emit no slot when identical, or when every changed path is
    recomputable from the same tick's serialized inputs (``derivable_evidence``,
    given ``states`` (tick, entity) -> frame fields and the decoder's
    ``lane_positions``); the refresh records its changed and derived paths.
    """
    states, lane_positions = states or {}, lane_positions or {}
    spacing = keyframe_s/tick_seconds
    if not math.isfinite(spacing) or spacing < 1 or not math.isclose(spacing, round(spacing), abs_tol=1e-8):
        raise ValueError("keyframe duration must be positive and align to source ticks")
    keyframe = round(spacing)
    last_state, last_evidence, out = {}, {}, []
    for record in sorted(semantic, key=lambda r: r["tick"]):
        kind = record["kind"]
        if kind == "frame" and record["entity"] is not None:
            previous = last_state.get(record["entity"])
            last_state[record["entity"]] = (record["tick"], record["fields"])
            if previous is not None and record["tick"]-previous[0] == 1 and (cutoff-record["tick"]) % keyframe and \
                    record_predictable(previous[1], record["fields"], 1, tick_seconds):
                tag = {**record, "kind": "same", "fields": {}, "field_names": sorted(record["fields"]),
                       "elapsed_ticks": 1, "previous_tick": previous[0]}
                p0 = cell(previous[1].get("pose.position_enu_m", {"kind": "missing"}), None)
                p1 = cell(record["fields"].get("pose.position_enu_m", {"kind": "missing"}), None)
                velocity = cell(previous[1].get("pose.velocity_enu_mps", {"kind": "missing"}), None)
                if p0[0] == p1[0] == "present":
                    if _same_value(p0[1], p1[1]):
                        tag["position_unchanged"] = True
                    else:
                        elapsed = Fraction(tick_seconds).limit_denominator(10**6)
                        residual = [_decimal_units(p1[1][i], 6, "history")-_decimal_units(p0[1][i], 6, "history")
                                    -(_decimal_units(velocity[1][i], 6, "history")*elapsed if velocity[0] == "present" else 0)
                                    for i in range(3)]
                        if any(residual):
                            tag["position_rounding_residual_units"] = [str(v) for v in residual]
                out.append(tag)
                continue
        elif "relation" in record:
            body = record["relation"]
            key = body["predicate_id"], body["tuple_id"]
            evidence = evidence_payload(body.get("observations", []))
            if kind == "update" and record["operation"] == "refresh_predicate_evidence":
                old = {o["path"]: o["value"] for o in last_evidence.get(key, [])}
                changed = sorted(o["path"] for o in evidence if o["path"] not in old or not _same_value(o["value"], old[o["path"]]))
                derived = [path for path in changed if derivable_evidence(
                    path, next(o["value"] for o in evidence if o["path"] == path), body["bindings"],
                    record["tick"], states, lane_positions)]
                record = {**record, "changed_evidence_paths": changed, "derived_evidence_paths": derived}
                if key in last_evidence and len(derived) == len(changed) and \
                        {o["path"] for o in evidence} == set(old):
                    record["emit"] = False
            last_evidence[key] = evidence
        out.append(record)
    return out


def restore_history_records(semantic, tick_seconds):
    """Reconstruct exact observed frame fields from keyframes and SAME tags.

    Used for replay/validation, not to enlarge the model's compressed prefix.
    Rational arithmetic retains the authentic decimal-grid rounding innovations.
    """
    previous, restored = {}, []
    for record in sorted(semantic, key=lambda r: r["tick"]):
        if record["kind"] == "same":
            entity = record["entity"]
            prior = previous.get(entity)
            if prior is None or prior[0] != record["previous_tick"] or record["tick"]-prior[0] != 1:
                raise ValueError("SAME lacks a consecutive authentic predecessor")
            fields = dict(prior[1])
            p = cell(fields.get("pose.position_enu_m", {"kind": "missing"}), None)
            v = cell(fields.get("pose.velocity_enu_mps", {"kind": "missing"}), None)
            if p[0] == "present" and not record.get("position_unchanged"):
                elapsed = Fraction(tick_seconds).limit_denominator(10**6)
                correction = record.get("position_rounding_residual_units", ["0"]*3)
                position = [float((_decimal_units(p[1][i], 6, "replay")
                                   +(_decimal_units(v[1][i], 6, "replay")*elapsed if v[0] == "present" else 0)
                                   +Fraction(correction[i]))/10**6) for i in range(3)]
                old = fields["pose.position_enu_m"]
                fields["pose.position_enu_m"] = {**old, "value": position} if isinstance(old, dict) else position
            record = {k: v for k, v in record.items() if k not in
                      ("field_names", "elapsed_ticks", "previous_tick", "position_unchanged", "position_rounding_residual_units")}
            record = {**record, "kind": "frame", "fields": fields}
        if record["kind"] == "frame" and record["entity"] is not None:
            previous[record["entity"]] = (record["tick"], record["fields"])
        restored.append(record)
    return restored


def unbounded_field(f):
    return f.logical_dtype == "real" and not f.circular and f.lower is None and f.upper is None


def forecast_anchor(index, address, *, rollout_values=None, cutoff=None):
    """Physical anchor for one real query: ``(anchor, elapsed_seconds, observed)``."""
    terms = anchor_terms(index, address, rollout_values=rollout_values, cutoff=cutoff)
    return terms["anchor"], terms["elapsed"], terms["observed"]


def _decimal_units(value, decimals, context):
    scaled = float(value)*10**decimals
    units = round(scaled)
    if abs(scaled-units) > 1e-3:
        raise ValueError(f"{context}: value {value!r} is not on the declared 1e-{decimals} source grid")
    return Fraction(units)


def anchor_atom(index, address, value, terms=None):
    """Whether ``value`` is the anchor atom, and the atom's half-width in physical units.

    Pose position: |T - (P + V*elapsed)| <= q*(1 + elapsed/2) in exact rational
    arithmetic on the q=1e-6 grid (rounding of T, P and V).  Persistence anchors
    (pose velocity, position without velocity): stored values equal on the grid.
    Unobserved anchors (TRAIN mean) never form the atom.
    """
    family = address.field_family
    terms = anchor_terms(index, address) if terms is None else terms
    if not terms["observed"]:
        return False, 0.
    if terms["rollout"]:
        raise ValueError("the anchor atom is defined for source operands, not rollout predictions")
    context = f"{address.episode_id}:{address.entity_id}:{family}[{address.component}]@{address.target_tick}"
    if family not in SOURCE_DECIMALS:
        raise ValueError(f"no declared atom rule for unbounded field {family}")
    decimals = SOURCE_DECIMALS[family]
    target, latest = _decimal_units(value, decimals, context), _decimal_units(terms["latest"], decimals, context)
    if terms["velocity"] is None:
        return target == latest, 0.
    elapsed = _elapsed_seconds(index, terms)
    velocity = _decimal_units(terms["velocity"], SOURCE_DECIMALS["pose.velocity_enu_mps"], context+" velocity")
    bound = 1+elapsed/2
    return abs(target-latest-velocity*elapsed) <= bound, float(bound)*10.**-decimals


def _elapsed_seconds(index, terms):
    return Fraction(terms["elapsed_ticks"])*Fraction(index.tick_seconds).limit_denominator(10**6)


def atom_band(index, address, terms):
    """Half-width (physical units) of the anchor atom, known before the target."""
    if not terms["observed"] or terms["rollout"] or address.field_family not in SOURCE_DECIMALS or terms["velocity"] is None:
        return 0.
    return float(1+_elapsed_seconds(index, terms)/2)*10.**-SOURCE_DECIMALS[address.field_family]


def anchor_terms(index, address, *, rollout_values=None, cutoff=None):
    """Forecast anchor and the source operands it is built from.

    The anchor is the latest present value, with pose position advanced by the
    latest velocity times the elapsed time; an unobserved address anchors at the
    TRAIN scale mean.
    """
    family, component = address.field_family, address.component
    cutoff = index.cutoff if cutoff is None else cutoff
    tick = address.target_tick
    history = sorted((t, cell(v[family], component)) for (t, e), v in index.states.items()
                     if e == address.entity_id and t <= index.cutoff and family in v)
    present = [(t, v) for t, (kind, v) in history if kind == "present"]
    if rollout_values is not None and address.entity_id in rollout_values and (family, component) in rollout_values[address.entity_id]:
        predicted = rollout_values[address.entity_id][family, component]
        present = present+[(cutoff, predicted["value"])] if predicted["kind"] == "present" else []
    rollout = rollout_values is not None
    if not present:
        return {"anchor": index.scales[family, component].mean, "elapsed": (tick-cutoff)*index.tick_seconds,
                "observed": False, "rollout": rollout}
    elapsed = (tick-present[-1][0])*index.tick_seconds
    anchor = float(present[-1][1])
    velocity = None
    if family == "pose.position_enu_m":
        velocity_key = "pose.velocity_enu_mps", component
        velocity_prediction = None if rollout_values is None else rollout_values.get(address.entity_id, {}).get(velocity_key)
        if velocity_prediction is not None and velocity_prediction["kind"] == "present":
            velocity = float(velocity_prediction["value"])
        else:
            states_with_velocity = sorted((t, cell(v[velocity_key[0]], component)) for (t, e), v in index.states.items()
                                          if e == address.entity_id and t <= index.cutoff and velocity_key[0] in v)
            velocities = [v for _, (k, v) in states_with_velocity if k == "present"]
            velocity = float(velocities[-1]) if velocities else None
        if velocity is not None:
            anchor += velocity*elapsed
    return {"anchor": anchor, "elapsed": elapsed, "observed": True, "rollout": rollout, "latest": float(present[-1][1]),
            "elapsed_ticks": tick-present[-1][0], "velocity": velocity}


def history_atom_features(index, address, f, present):
    """Exact anchor-atom history of one queried field: [atom rate, last transition atom, available].

    Each consecutive pair of present source values in the window history is
    scored with ``anchor_atom`` (same rule as the target), so the gate sees the
    resolution-level persistence that standardized inputs cannot carry.
    """
    if not unbounded_field(f) or len(present) < 2:
        return [0., 0., 0.]
    velocities = None
    if f.family == "pose.position_enu_m":
        velocities = sorted((t, cell(v["pose.velocity_enu_mps"], f.component)) for (t, e), v in index.states.items()
                            if e == address.entity_id and t <= index.cutoff and "pose.velocity_enu_mps" in v)
        velocities = [(t, float(v)) for t, (kind, v) in velocities if kind == "present"]
    flags = []
    for (t0, v0), (t1, v1) in zip(present, present[1:]):
        velocity = None
        if velocities is not None:
            prior = [v for t, v in velocities if t <= t0]
            velocity = prior[-1] if prior else None
        terms = {"observed": True, "rollout": False, "latest": float(v0), "elapsed_ticks": t1-t0, "velocity": velocity}
        step = QueryAddress(address.episode_id, address.entity_id, f.family, t1, f.component)
        flags.append(anchor_atom(index, step, float(v1), terms)[0])
    return [sum(flags)/len(flags), float(flags[-1]), 1.]


def query_tensors(index, tick, *, rollout_values=None, cutoff=None):
    if rollout_values is not None:
        if cutoff is None:
            raise ValueError("rollout_values requires explicit prediction cutoff/timestamp")
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or not index.cutoff < cutoff < tick:
            raise ValueError(f"prediction cutoff must be an integer satisfying original cutoff={index.cutoff} < prediction cutoff={cutoff} < target tick={tick}")
    cutoff = index.cutoff if cutoff is None else cutoff
    addresses = index.addresses(tick)
    field_map = {(f.family, f.component): f for f in index.fields}
    rows, allowed = [], []
    anchors, forecast_stds, positive_means, category_prior, bands = [], [], [], [], []
    for address in addresses:
        f = field_map[address.field_family, address.component]
        history = sorted((t, cell(v[f.family], f.component)) for (t, e), v in index.states.items()
                         if e == address.entity_id and t <= index.cutoff and f.family in v)
        present = [(t, v) for t, (kind, v) in history if kind == "present"]
        history_atoms = history_atom_features(index, address, f, present)
        scale = index.scales.get((f.family, f.component))
        historical_values = [scale.transform(v) if scale else float(f.enum_values.index(v)) for _, v in present]
        historical_count = len(historical_values)
        delta = (historical_values[-1]-historical_values[-2])/((present[-1][0]-present[-2][0])*index.tick_seconds) if historical_count > 1 else 0.
        if f.circular and historical_count > 1:
            physical_delta = (float(present[-1][1])-float(present[-2][1])+180.) % 360.-180.
            delta = physical_delta/(scale.std*(present[-1][0]-present[-2][0])*index.tick_seconds)
        predicted = None
        if rollout_values is not None and address.entity_id in rollout_values and (f.family, f.component) in rollout_values[address.entity_id]:
            predicted = rollout_values[address.entity_id][f.family, f.component]
            if predicted["kind"] == "present":
                present.append((cutoff, predicted["value"]))
            else:
                # A non-present latest prediction cannot reuse a masked physical value.
                present = []
        values = [scale.transform(v) if scale else float(f.enum_values.index(v)) for _, v in present]
        observed = bool(values)
        current = values[-1] if observed else 0.  # masked tensor storage
        confidence = float(observed) if predicted is None else predicted["record_probability"] if predicted["kind"] == "present" else 0.
        # Query horizon stays relative to the ORIGINAL origin. Transition dt is only a law input.
        tau = (tick-index.cutoff)*index.tick_seconds
        rows.append([current, delta, confidence, math.log1p(historical_count), tau,
                     float(f.logical_dtype == "real"), float(f.logical_dtype == "enum"), float(f.logical_dtype == "bool"),
                     float(f.circular), float(f.lower is not None), float(f.upper is not None),
                     -1. if f.component is None else float(f.component), *history_atoms])
        physical_latest = 0. if not observed or scale else float(f.enum_values.index(present[-1][1]))
        if scale is not None:
            temporal = index.forecast_scales[f.family, f.component]
            terms = anchor_terms(index, address, rollout_values=rollout_values, cutoff=cutoff)
            anchor, elapsed = terms["anchor"], terms["elapsed"]
            bands.append(atom_band(index, address, terms) if unbounded_field(f) else 0.)
            if elapsed <= 0:
                raise ValueError(f"query {address} does not lie after its latest present value")
            # Physical unit of the learned mean offset and standard deviation.
            rate = temporal.get("residual_rate_by_category", {}).get(entity_category(index, address.entity_id),
                                                                    temporal["residual_rate"])
            unit = rate*elapsed
            if unit <= 0:
                raise ValueError(f"TRAIN residual rate of {f.family}:{f.component} is zero; its Gaussian unit is undefined")
            anchors.append(anchor)
            forecast_stds.append(unit)
            positive_means.append(temporal["positive_mean"] if temporal["positive_mean"] is not None else scale.std)
            category_prior.append(0)
        else:
            counts = index.forecast_scales[f.family, f.component]["categorical_counts"]
            prior = int(physical_latest) if observed else max(range(len(counts)), key=counts.__getitem__)
            anchors.append(float(prior))
            forecast_stds.append(1.)
            positive_means.append(1.)
            bands.append(0.)
            category_prior.append(prior)
        allowed.append([s == "present" or s in f.allowed_non_present for s in STATUS])
    return {"addresses": addresses, "fields": [field_map[a.field_family, a.component] for a in addresses],
            "features": torch.tensor(rows), "allowed_status": torch.tensor(allowed),
            # Physical anchors and units stay float64: float32 cannot resolve the
            # 1e-6 source rounding at map-scale coordinates.
            "anchor": torch.tensor(anchors, dtype=torch.float64), "forecast_std": torch.tensor(forecast_stds, dtype=torch.float64),
            "positive_prior": torch.tensor(positive_means, dtype=torch.float64),
            "atom_band": torch.tensor(bands, dtype=torch.float64), "rollout_anchor": rollout_values is not None,
            "category_prior": torch.tensor(category_prior),
            "entity_ordinals": torch.tensor([index.local_identities[a.entity_id] for a in addresses])}


def labels(index, queries):
    if queries["rollout_anchor"]:
        raise ValueError("anchor-atom labels are defined for source anchors; this query anchors on rollout feedback")
    recorded, entity_recorded, status, status_mask, numeric, categorical, atoms = [], [], [], [], [], [], []
    for address, f in zip(queries["addresses"], queries["fields"]):
        state = index.states.get((address.target_tick, address.entity_id))
        found = state is not None and f.family in state
        present_record = found
        entity_recorded.append(state is not None)
        kind, value = cell(state[f.family], f.component) if found else ("present", None)
        recorded.append(present_record)
        status_mask.append(found)
        status.append(STATUS.index(kind))
        numeric.append(float(value) if found and kind == "present" and f.logical_dtype == "real" else 0.)
        categorical.append(f.enum_values.index(value) if found and kind == "present" and f.logical_dtype != "real" else 0)
        atoms.append(found and kind == "present" and unbounded_field(f) and anchor_atom(index, address, float(value))[0])
    return {"recorded": torch.tensor(recorded, dtype=torch.float32), "entity_recorded": torch.tensor(entity_recorded), "status": torch.tensor(status),
            "status_mask": torch.tensor(status_mask), "numeric": torch.tensor(numeric, dtype=torch.float64),
            "categorical": torch.tensor(categorical), "atom": torch.tensor(atoms, dtype=torch.bool)}
