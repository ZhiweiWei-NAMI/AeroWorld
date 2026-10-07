"""One TRAIN-prefix index, reversible semantic projection, source-bound queries."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from Dataset.world_model.contracts.schema import FIELD_REGISTRY
from Dataset.world_model.data.provenance import resolve_source_split
from Dataset.world_model.data.windows import build_supervision_window
from Dataset.world_model.model.query_head import FieldTarget, InputScale, QueryAddress, modelable_field_targets

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
EXCLUDED_TARGETS = {"annotations.network_latency_ms", "annotations.network_packet_loss", "annotations.visibility_state"}
MODEL_INPUT_EXCLUDED = frozenset({"annotations.network_latency_ms", "annotations.network_packet_loss", "annotations.network_status"})
STATUS = ("present", "missing", "inapplicable", "absent", "source_unknown")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def cell(value, component):
    if isinstance(value, dict) and set(value) == {"kind"}:
        return value["kind"], None
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    return "present", value if component is None else value[component]


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

    def addresses(self, tick):
        # Frame producer declares pose/motion for every captured actor.
        # Communication producer declares only UAV flows. A missing latest
        # cell does not remove a field from this applicability declaration.
        return tuple(QueryAddress(self.episode, e, f.family, tick, f.component)
                     for e, meta in sorted(self.entities.items()) for f in self.fields
                     if not f.family.startswith("communication.") or meta["roster.entity_category"] == "uav")


def load_index(canonical: Path, communication: Path, episode: str, cutoff: int, targets, graph_base: Path):
    if resolve_source_split(episode) != "TRAIN":
        raise ValueError("the demonstration fits normalization on TRAIN only")
    with canonical.open() as stream:
        sample = build_supervision_window((json.loads(x) for x in stream if x.strip()),
                                         episode_id=episode, split="TRAIN", cutoff=cutoff, target_times=targets)
    prefix_actors = {r["entity_id"] for r in sample.input_records if r["record_kind"] == "frame"}
    roster = {r["fields"]["roster.entity_id"]: r["fields"] for r in sample.input_records if r["record_kind"] == "entity"}
    entities = {e: roster[e] for e in sorted(prefix_actors)}
    states = {(r["tick"], r["entity_id"]): dict(r["fields"])
              for r in sample.input_records + sample.target_records if r["record_kind"] == "frame"}
    comm_rows, comm_lines, bindings = {}, {}, []
    with communication.open() as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["episode_id"] != episode:
                raise ValueError("cross-episode communication row")
            key = row["tick"], row["entity_id"]
            if key in comm_rows:
                raise ValueError(f"duplicate communication row {key}")
            comm_rows[key] = row
            comm_lines[key] = line_no
    semantic = []
    baseline = json.loads(graph_base.read_text())
    if baseline["episode_id"] != episode or baseline["tick"] != 0:
        raise ValueError("graph initial source does not match the episode/tick0")
    # Native graph base is the model's initial-state channel. Canonical base
    # edges are the importer projection of this same authority, not extra facts.
    # Bind that projection once while building the fixed index, never per query.
    initial_records = [record for record in sample.input_records
                       if record.get("source_family") == "world_truth_graph_base"]
    if any(record["tick"] != baseline["tick"] for record in initial_records):
        raise ValueError("canonical initial assertion is bound to a different native initial tick")
    for family, native_path in (("world_truth_graph_base", graph_base),
                                ("world_truth_graph_deltas", graph_base.with_name("world_truth_graph_deltas.jsonl"))):
        paths = {record["source"]["path"] for record in sample.input_records
                 if record.get("source_family") == family}
        expected = native_path.resolve()
        if any(Path(path).resolve() != expected for path in paths):
            raise ValueError(f"canonical {family} is bound to a different native graph source")
    # This is an independent worldtruth channel, not observer visibility.
    # Initial tick0 values and cutoff-legal deltas do not expand query actors.
    graph_nodes = {}
    def node_roles(assertion, source):
        for role, identity in assertion["bindings"].items():
            ontology = assertion["binding_ontology_classes"][role]
            if identity not in graph_nodes:
                graph_nodes[identity] = {"identity": identity, "ontology_classes": [], "role_bindings": []}
            node = graph_nodes[identity]
            if ontology not in node["ontology_classes"]:
                node["ontology_classes"].append(ontology)
                node["ontology_classes"].sort()
            binding = {"role": role, "ontology_class": ontology,
                       "predicate_id": assertion["predicate_id"], "tuple_id": assertion["tuple_id"], "source": dict(source)}
            if binding not in node["role_bindings"]:
                node["role_bindings"].append(binding)
    for assertion in baseline["initial_assertions"]:
        if assertion["truth_state_update_tick"] != 0 or assertion["evidence_update_tick"] != 0:
            raise ValueError("initial assertion contains later evidence")
        semantic.append({"kind": "edge", "tick": 0, "entity": None, "id": assertion["assertion_id"],
                         "fields": {"relation.predicate_id": assertion["predicate_id"], "relation.tuple_id": assertion["tuple_id"],
                                    "relation.bindings": assertion["bindings"], "relation.value": assertion["value"], "relation.asserted": True}})
        bindings.append({"id": assertion["assertion_id"], "kind": "edge", "tick": 0,
                         "path": str(graph_base), "source_pointer": "initial_assertions", "source_refs": assertion["source_refs"]})
        node_roles(assertion, {"path": str(graph_base), "assertion_id": assertion["assertion_id"]})
    delta_path = graph_base.with_name("world_truth_graph_deltas.jsonl")
    graph_state = {(a["predicate_id"], a["tuple_id"]): dict(a) for a in baseline["initial_assertions"]}
    with delta_path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            delta = json.loads(line)
            if delta["tick"] > cutoff: break
            for op_index, operation in enumerate(delta["operations"]):
                body = operation["assertion"] if operation["operation"] == "add_predicate_assertion" else operation
                node_roles(body, {"path": str(delta_path), "line": line_no, "operation_index": op_index})
                key = body["predicate_id"], body["tuple_id"]
                action = operation["operation"]
                if action == "add_predicate_assertion":
                    if key in graph_state: raise ValueError("graph add duplicates a current tuple")
                    graph_state[key] = dict(body)
                elif action == "remove_predicate_assertion":
                    if graph_state[key]["value"] != body["from_value"]: raise ValueError("graph remove source value diverges")
                    del graph_state[key]
                elif action == "set_predicate_value":
                    if graph_state[key]["value"] != body["from_value"]: raise ValueError("graph set source value diverges")
                    graph_state[key]["value"] = body["to_value"]
                elif action == "refresh_predicate_evidence":
                    if graph_state[key]["value"] != body["value"]: raise ValueError("graph evidence value diverges")
                else: raise ValueError(f"unsupported current graph operation {action}")
    for r in sample.input_records:
        if r.get("source_family") == "world_truth_graph_base":
            continue  # Already consumed above through its verified native channel.
        kind = r["record_kind"]
        if kind == "entity" and r["fields"]["roster.entity_id"] not in prefix_actors:
            continue  # Full-episode capture-filter roster is hindsight-selected.
        fields = {k: v for k, v in r["fields"].items() if not k.startswith("archive.") and k not in MODEL_INPUT_EXCLUDED}
        semantic.append({"kind": kind, "tick": r.get("tick"), "entity": r.get("entity_id"), "id": r["id"], "fields": fields})
        bindings.append({"id": r["id"], "kind": kind, "tick": r.get("tick"), "entity": r.get("entity_id"), "source": r.get("source")})
    for (tick, e), fields in states.items():
        if e not in entities or entities[e]["roster.entity_category"] != "uav":
            continue
        if tick % 5:
            continue  # This source is sampled on the declared capture clock.
        if (tick, e) not in comm_rows:
            raise ValueError(f"active captured UAV has no authentic communication row {(tick, e)}")
        row = comm_rows[tick, e]
        if row["scope_active"] is not True or row["model_version"] != "1.6.0":
            raise ValueError("unsupported communication scope/version")
        values = {}
        for name, (parent, leaf) in COMM_KEYS.items():
            v = row[parent][leaf]
            if v == "unknown":
                values[name] = {"kind": "source_unknown"}
            else:
                dtype, _, bounds, vocab = COMM[name]
                if dtype == "real" and (isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)):
                    raise ValueError(f"invalid actual {name}")
                if dtype == "real" and ((bounds[0] is not None and v < bounds[0]) or (bounds[1] is not None and v > bounds[1])):
                    raise ValueError(f"out-of-support actual {name}")
                if dtype != "real" and v not in vocab:
                    raise ValueError(f"undeclared actual {name}")
                if dtype == "bool" and type(v) is not bool:
                    raise ValueError(f"actual {name} is not a Boolean")
                values[name] = v
        fields.update(values)
        bindings.append({"tick": tick, "entity": e, "path": str(communication), "line": comm_lines[tick, e],
                         "source_class": row["source_class"], "model_version": row["model_version"],
                         "station_source": row["station_source"], "values": values})
        if tick <= cutoff:
            semantic.append({"kind": "communication", "tick": tick, "entity": e,
                             "id": f"communication:{tick}:{e}", "fields": values})
    fields = tuple(f for f in modelable_field_targets()[0] if f.family not in EXCLUDED_TARGETS)
    fields += tuple(FieldTarget(name, None, dtype, unit, None, None, 1, False,
                                bounds[0], bounds[1], vocab, ("source_unknown",))
                    for name, (dtype, unit, bounds, vocab) in COMM.items())
    scales = {}
    for f in fields:
        if f.logical_dtype != "real":
            continue
        values = [float(cell(row[f.family], f.component)[1]) for (tick, e), row in states.items()
                  if tick <= cutoff and e in entities and f.family in row
                  and cell(row[f.family], f.component)[0] == "present"]
        if not values:
            raise ValueError(f"TRAIN prefix has no scale authority for {f.family}:{f.component}")
        mean = sum(values) / len(values)
        std = math.sqrt(sum((x-mean)**2 for x in values)/len(values))
        scales[f.family, f.component] = InputScale(f.family, f.component, mean, std if std else 1., std, len(values), cutoff)
    forecast_scales = {}
    for f in fields:
        if f.logical_dtype != "real":
            counts = [0]*len(f.enum_values)
            for (t, e), state in states.items():
                if t <= cutoff and e in entities and f.family in state:
                    k, v = cell(state[f.family], f.component)
                    if k == "present": counts[f.enum_values.index(v)] += 1
            if not sum(counts): raise ValueError(f"no TRAIN categorical prior for {f.family}")
            forecast_scales[f.family, f.component] = {"categorical_counts": counts, "max_tick": cutoff}
            continue
        rates, positive = [], []
        for e in entities:
            history = sorted((t, float(cell(v[f.family], f.component)[1]))
                             for (t, actor), v in states.items() if actor == e and t <= cutoff
                             and f.family in v and cell(v[f.family], f.component)[0] == "present")
            positive.extend(v-f.lower for _, v in history if f.lower is not None and v > f.lower)
            for (t0, v0), (t1, v1) in zip(history, history[1:]):
                change = (v1-v0+180.) % 360.-180. if f.circular else v1-v0
                rates.append(change/((t1-t0)*.1))
        rms_rate = math.sqrt(sum(v*v for v in rates)/len(rates)) if rates else None
        forecast_scales[f.family, f.component] = {
            "rms_rate": rms_rate, "rate_count": len(rates), "max_tick": cutoff,
            "constant_rate_scale": 1. if rms_rate == 0 else None,
            "positive_mean": sum(positive)/len(positive) if positive else None,
            "source": "TRAIN prefix per-entity consecutive present increments divided by physical dt"}
    return EpisodeIndex(episode, cutoff, tuple(targets), entities, states, semantic, fields, scales, forecast_scales, bindings,
                        graph_nodes, baseline["predicate_vocabulary"], {})


def semantic_text(index: EpisodeIndex):
    """Dictionary + exact per-frame changes; source bookkeeping is a sidecar.

    Round-trip retains every admitted semantic field, including edge/event
    bindings and all historical frame membership. Absent keys are explicit
    removals, distinct from a source cell's non-present value.
    """
    names = sorted({k for r in index.semantic_records for k in r["fields"]})
    fmap = {f: i for i, f in enumerate(names)}
    # Keep causal semantic string values exactly; symbols shorten repeated
    # identifiers without replacing their meanings with free-form prose.
    strings = set()
    def collect(x):
        if isinstance(x, str): strings.add(x)
        elif isinstance(x, list):
            for v in x: collect(v)
        elif isinstance(x, dict):
            for k, v in x.items(): collect(k); collect(v)
    for r in index.semantic_records: collect(r)
    symbols = sorted(strings); smap = {s: i for i, s in enumerate(symbols)}
    def encode(x):
        if isinstance(x, str): return "@"+str(smap[x])
        if isinstance(x, list): return [encode(v) for v in x]
        if isinstance(x, dict): return {"@"+str(smap[k]): encode(v) for k, v in sorted(x.items())}
        return x
    def decode(x):
        if isinstance(x, str): return symbols[int(x[1:])]
        if isinstance(x, list): return [decode(v) for v in x]
        if isinstance(x, dict): return {symbols[int(k[1:])]: decode(v) for k, v in x.items()}
        return x
    # Opaque producer keys have no numeric/hash semantics. Preserve equality
    # and joins with short identities; exact source names remain in a
    # reversible sidecar, while meaningful entity/predicate/field names stay.
    wire_symbols = []
    for i, value in enumerate(symbols):
        if re.search(r"[0-9a-f]{20,}", value):
            alias = "identity:"+str(i)
            index.identity_aliases[alias] = value
            wire_symbols.append(alias)
        else: wire_symbols.append(value)
    lines = ["SEMANTIC_SCHEMA units/frame/support apply to values; renderer visibility is a submission-scope label, not pixel visibility or world existence. Renderer hardcoded network latency/loss/status defaults are omitted; communication.* comes from the controlled simulated communication product. Opaque producer identities are reversible sidecar aliases. Full worldtruth is independent from observer visibility. Row=[kind_symbol,tick,entity,id_symbol,changed_field_pairs,removed_field_indices]; @n references STRINGS[n], including dictionary keys; field indices reference FIELD_NAMES; frame changes inherit only the same entity's previous frame, other rows are complete; numeric/bool/null values are literal.",
             "STRINGS="+dumps(wire_symbols), "FIELD_NAMES="+dumps(names)]
    spans = {}; offset = sum(len(s)+1 for s in lines)
    for name in names:
        if name in FIELD_REGISTRY:
            declaration = FIELD_REGISTRY[name]
        elif name in COMM:
            dtype, unit, bounds, vocab = COMM[name]
            declaration = {"dtype": dtype, "unit": unit, "bounds": bounds, "enum": vocab,
                           "meaning": "controlled simulated communication, UAV-only; not ns3/hardware telemetry"}
        else:
            raise ValueError(f"unregistered semantic field {name}")
        line = "F"+str(fmap[name])+"="+dumps({"name": name, **declaration})
        spans["field", name] = (offset, offset+len(line)); lines.append(line); offset += len(line)+1
    previous = {}; recovered = []
    for r in index.semantic_records:
        prior = previous.get(r["entity"], {}) if r["kind"] == "frame" else {}
        changed = {k: v for k, v in r["fields"].items() if k not in prior or prior[k] != v}
        removed = sorted(set(prior)-set(r["fields"]))
        row = [smap[r["kind"]], r["tick"], encode(r["entity"]), smap[r["id"]],
               [[fmap[k], encode(v)] for k, v in sorted(changed.items())], [fmap[k] for k in removed]]
        line = dumps(row)
        spans["record", r["id"]] = (offset, offset+len(line))
        if r["entity"] is not None:
            spans["entity", r["entity"]] = (offset, offset+len(line))
        lines.append(line); offset += len(line)+1
        restored = dict(prior)
        for k in removed: del restored[k]
        for k, v in row[4]: restored[names[k]] = decode(v)
        recovered.append({"kind": symbols[row[0]], "tick": row[1], "entity": decode(row[2]), "id": symbols[row[3]], "fields": restored})
        if r["kind"] == "frame": previous[r["entity"]] = restored
    if recovered != index.semantic_records:
        raise ValueError("semantic change encoding did not round-trip exactly")
    return "\n".join(lines)+"\n", spans


def query_tensors(index, tick, *, rollout_values=None, cutoff=None):
    cutoff = index.cutoff if cutoff is None else cutoff
    addresses = index.addresses(tick)
    field_map = {(f.family, f.component): f for f in index.fields}
    rows, latest, means, stds, allowed = [], [], [], [], []
    anchors, forecast_stds, positive_means, latest_present, category_prior = [], [], [], [], []
    for a in addresses:
        f = field_map[a.field_family, a.component]
        history = [(t, cell(v[f.family], f.component)) for (t, e), v in index.states.items()
                   if e == a.entity_id and t <= index.cutoff and f.family in v]
        history.sort()
        present = [(t, v) for t, (kind, v) in history if kind == "present"]
        predicted = None
        if rollout_values is not None and a.entity_id in rollout_values and (f.family, f.component) in rollout_values[a.entity_id]:
            predicted = rollout_values[a.entity_id][f.family, f.component]
            if predicted["kind"] == "present":
                present.append((cutoff, predicted["value"]))
        scale = index.scales.get((f.family, f.component))
        values = [(float(v)-scale.mean)/scale.std if scale else float(f.enum_values.index(v)) for _, v in present]
        observed = bool(values)
        current = values[-1] if observed else 0.  # masked storage, never a world value
        delta = (values[-1]-values[-2])/((present[-1][0]-present[-2][0])*.1) if len(values) > 1 else 0.
        confidence = float(observed) if predicted is None else (predicted["record_probability"] if predicted["kind"] == "present" else 0.)
        rows.append([current, delta, confidence, math.log1p(len(values)), (tick-cutoff)*.1,
                     float(f.logical_dtype == "real"), float(f.logical_dtype == "enum"), float(f.logical_dtype == "bool"),
                     float(f.circular), float(f.lower is not None), float(f.upper is not None),
                     -1. if f.component is None else float(f.component)])
        physical_latest = float(present[-1][1]) if observed and scale else (0. if not observed else float(f.enum_values.index(present[-1][1])))
        latest.append(physical_latest)
        latest_present.append(observed)
        if scale is not None:
            temporal = index.forecast_scales[f.family, f.component]
            rate = temporal["rms_rate"]
            # If no temporal pair is recorded, use the admitted TRAIN state
            # dispersion as an explicit distribution prior, never storage0.
            elapsed = (tick-present[-1][0])*.1 if observed else (tick-cutoff)*.1
            forecast_std = (rate if rate else (1. if rate == 0 else scale.std))*elapsed
            anchor = physical_latest if observed else scale.mean
            if f.family == "pose.position_enu_m" and observed:
                velocity_key = "pose.velocity_enu_mps", f.component
                velocity_prediction = None if rollout_values is None else rollout_values.get(a.entity_id, {}).get(velocity_key)
                if velocity_prediction is not None and velocity_prediction["kind"] == "present":
                    velocity = float(velocity_prediction["value"])
                else:
                    states_with_velocity = [(t, cell(v[velocity_key[0]], f.component)) for (t, e), v in index.states.items()
                                            if e == a.entity_id and t <= index.cutoff and velocity_key[0] in v]
                    states_with_velocity.sort()
                    velocities = [v for _, (k, v) in states_with_velocity if k == "present"]
                    velocity = float(velocities[-1]) if velocities else None
                if velocity is not None: anchor += velocity*elapsed
            anchors.append(anchor); forecast_stds.append(forecast_std)
            positive_means.append(temporal["positive_mean"] if temporal["positive_mean"] is not None else scale.std)
            category_prior.append(0)
        else:
            counts = index.forecast_scales[f.family, f.component]["categorical_counts"]
            prior = int(physical_latest) if observed else max(range(len(counts)), key=counts.__getitem__)
            anchors.append(float(prior)); forecast_stds.append(1.); positive_means.append(1.); category_prior.append(prior)
        means.append(0. if scale is None else scale.mean); stds.append(1. if scale is None else scale.std)
        allowed.append([s == "present" or s in f.allowed_non_present for s in STATUS])
    return {"addresses": addresses, "fields": [field_map[a.field_family, a.component] for a in addresses],
            "features": torch.tensor(rows), "latest": torch.tensor(latest), "mean": torch.tensor(means),
            "std": torch.tensor(stds), "allowed_status": torch.tensor(allowed),
            "anchor": torch.tensor(anchors), "forecast_std": torch.tensor(forecast_stds),
            "positive_prior": torch.tensor(positive_means), "latest_present": torch.tensor(latest_present),
            "category_prior": torch.tensor(category_prior)}


def labels(index, queries):
    recorded, status, status_mask, numeric, categorical = [], [], [], [], []
    for a, f in zip(queries["addresses"], queries["fields"]):
        state = index.states.get((a.target_tick, a.entity_id))
        present_record = state is not None
        found = present_record and f.family in state
        k, v = cell(state[f.family], f.component) if found else ("present", None)
        recorded.append(present_record); status_mask.append(found); status.append(STATUS.index(k))
        numeric.append(float(v) if found and k == "present" and f.logical_dtype == "real" else 0.)
        categorical.append(f.enum_values.index(v) if found and k == "present" and f.logical_dtype != "real" else 0)
    return {"recorded": torch.tensor(recorded, dtype=torch.float32), "status": torch.tensor(status),
            "status_mask": torch.tensor(status_mask), "numeric": torch.tensor(numeric), "categorical": torch.tensor(categorical)}
