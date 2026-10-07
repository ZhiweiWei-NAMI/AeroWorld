"""Read-only source episode import into canonical records.

Imports one complete source episode into p01v4 canonical records with full
per-class coverage counts and source provenance (path + line) preserved on
every record.  Sources are opened read-only; every write goes to a new
versioned derivative under the results root.

Imported streams (episode L4-1_v1__seed00 engineering pilot):
  truth_frames.jsonl              -> frame records (pose/motion/annotations)
  global_entity_roster.json       -> entity records (roster)
  world_truth_graph_base.json     -> edge records (native initial assertions)
  world_truth_graph_deltas.jsonl  -> edge records (relation; per-operation)
  event_occurrences.jsonl         -> event records (event)
  rgb/lidar capture json/npz      -> observation records (observation)
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from Dataset.world_model.contracts.schema import (
    SchemaError,
    validate_record,
)

REPO = Path(__file__).resolve().parents[3]

# Non-present markers: only emitted where the source itself distinguishes them.
# "absent" is emitted only for relation removals (v2): the removed assertion
# carries no current truth value; its from_value stays in the op payload.
V = {"missing": {"kind": "missing"},
     "absent": {"kind": "absent"},
     "inapplicable": {"kind": "inapplicable"}}


def _present(value: Any) -> Any:
    return value


def _bump(mapping: dict[str, Counter], stream: str, key: str) -> None:
    mapping.setdefault(stream, Counter())[key] += 1


def _frame_vec3(values: Any) -> Any:
    if values is None:
        return V["missing"]
    if len(values) != 3:
        return V["missing"]
    return {"frame": "world_enu_m", "value": [float(c) for c in values]}


@dataclass
class ImportReport:
    """Per-class coverage of one episode import."""

    episode_id: str
    sources: dict[str, str] = dc_field(default_factory=dict)
    source_records: Counter = dc_field(default_factory=Counter)
    canonical_records: Counter = dc_field(default_factory=Counter)
    ticks_covered: list[int] = dc_field(default_factory=list)
    modalities: Counter = dc_field(default_factory=Counter)
    missing_modalities: list[str] = dc_field(default_factory=list)
    event_traceability: dict[str, Any] = dc_field(default_factory=dict)
    #: v2 (coordinator gap 4): per-stream lossless-archive inventory.  For
    #: every source stream: how many source keys were mapped into typed
    #: families, which exact keys those were, the bytes archived verbatim, and
    #: which top-level keys were archived-only or are genuinely residual.
    #: Residual keys are tracked, never silently dropped and never labelled
    #: with a generic UNKNOWN semantics.
    archive_streams: dict[str, dict[str, Any]] = dc_field(default_factory=dict)
    errors: list[str] = dc_field(default_factory=list)

    def stream_inventory(self, stream: str) -> dict[str, Any]:
        return self.archive_streams.setdefault(stream, {
            "source_keys": set(), "mapped_keys": set(), "archived_keys": set(),
            "residual_keys": set(),
            "records_archived": 0, "bytes_archived": 0,
            "mapped_occurrences": Counter(), "archived_occurrences": Counter(),
            "residual_occurrences": Counter(),
        })

    def note_archive(self, stream: str, *, records: int = 0, bytes_: int = 0,
                     archived_keys: Iterable[str] = ()) -> None:
        inv = self.stream_inventory(stream)
        inv["records_archived"] += records
        inv["bytes_archived"] += bytes_
        for k in archived_keys:
            inv["archived_keys"].add(k)
            inv["archived_occurrences"][k] += 1

    def note_mapped(self, stream: str, keys: Iterable[str]) -> None:
        inv = self.stream_inventory(stream)
        for k in keys:
            inv["mapped_keys"].add(k)
            inv["mapped_occurrences"][k] += 1

    def note_residual(self, stream: str, keys: Iterable[str]) -> None:
        inv = self.stream_inventory(stream)
        for k in keys:
            inv["residual_keys"].add(k)
            inv["residual_occurrences"][k] += 1

    def finalize_inventories(self) -> None:
        """A key is residual iff it is neither mapped nor archived anywhere."""
        for inv in self.archive_streams.values():
            inv["residual_keys"] -= inv["mapped_keys"] | inv["archived_keys"]

    def to_dict(self) -> dict[str, Any]:
        streams: dict[str, Any] = {}
        for stream, inv in self.archive_streams.items():
            streams[stream] = {
                "records_archived": inv["records_archived"],
                "bytes_archived": inv["bytes_archived"],
                "source_keys": sorted(inv["source_keys"]),
                "mapped_keys": sorted(inv["mapped_keys"]),
                "archived_keys": sorted(inv["archived_keys"]),
                "residual_keys": sorted(inv["residual_keys"]),
                "mapped_occurrences": dict(inv["mapped_occurrences"]),
                "archived_occurrences": dict(inv["archived_occurrences"]),
                "residual_occurrences": dict(inv["residual_occurrences"]),
            }
        return {
            "episode_id": self.episode_id,
            "sources": self.sources,
            "source_records": dict(self.source_records),
            "canonical_records": dict(self.canonical_records),
            "ticks_covered": [self.ticks_covered[0], self.ticks_covered[-1],
                              len(self.ticks_covered)],
            "modalities": dict(self.modalities),
            "missing_modalities": self.missing_modalities,
            "event_traceability": self.event_traceability,
            "archive_streams": streams,
            "errors": self.errors,
        }


class CanonicalEpisode:
    """In-memory canonical record set for one episode."""

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.records: list[dict[str, Any]] = []
        self._entity_ids: set[str] = set()

    def add(self, record: dict[str, Any], *, check_id: bool = False) -> None:
        rec_episode = record.get("episode_id")
        if rec_episode is None:
            record["episode_id"] = self.episode_id
        elif rec_episode != self.episode_id:
            raise SchemaError(
                f"cross-episode record rejected: record episode_id '{rec_episode}' "
                f"!= canonical episode '{self.episode_id}'")
        if check_id:
            eid = record["fields"].get("roster.entity_id")
            if eid is not None:
                if eid in self._entity_ids:
                    raise SchemaError(f"duplicate entity_id '{eid}' in episode index")
                self._entity_ids.add(eid)
        validate_record(record)
        self.records.append(record)

    def by_kind(self, kind: str) -> Iterator[dict[str, Any]]:
        for r in self.records:
            if r["record_kind"] == kind:
                yield r


def _open_lines(path: Path) -> tuple[list[str], int]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return lines, len(lines)


# v2: every frame-level key of truth_frames.jsonl maps to a registered typed
# family (archive.scene.*) or is already consumed by the canonical header
# (episode_id, schema_name, schema_version).  Typed scalar families receive
# the plain source scalar; object families must receive a mapping (a
# divergence surfaces as SchemaError, never a guessed value).
_FRAME_SCENE_FAMILIES = {
    "active_roi_id": "archive.scene.active_roi_id",
    "active_site_id": "archive.scene.active_site_id",
    "capture_boundary_id": "archive.scene.capture_boundary_id",
    "map_id": "archive.scene.map_id",
    "dt_s": "archive.scene.dt_s",
    "tick_hz": "archive.scene.tick_hz",
    "sim_time_s": "archive.scene.sim_time_s",
    "frame_seq": "archive.scene.frame_seq",
    "render_mode": "archive.scene.render_mode",
    "sumo_segment": "archive.scene.sumo_segment",
    "sumo_semantics": "archive.scene.sumo_semantics",
    "sumo_traffic_light_states": "archive.scene.sumo_traffic_light_states",
    "sumo_active_incidents": "archive.scene.sumo_active_incidents",
    "pad_boundary_policy": "archive.scene.pad_boundary_policy",
    "inspect_observes_boundary": "archive.scene.inspect_observes_boundary",
    "uav_crosses_boundary": "archive.scene.uav_crosses_boundary",
    "uav_global_flow": "archive.scene.uav_global_flow",
    "uav_segment": "archive.scene.uav_segment",
    "entity_motion_state": "archive.scene.entity_motion_state",
    "roster_summary": "archive.scene.roster_summary",
}
_FRAME_HEADER_CONSUMED = ("episode_id", "schema_name", "schema_version",
                          "frame_id", "tick", "entities")


def import_truth_frames(episode: CanonicalEpisode, path: Path,
                        report: ImportReport) -> None:
    lines, n = _open_lines(path)
    report.source_records["truth_frames.jsonl"] = n
    for lineno, line in enumerate(lines, start=1):
        src = json.loads(line)
        if lineno == 1:
            report.stream_inventory("truth_frames.jsonl")["source_keys"].update(src)
            report.stream_inventory("truth_frames.jsonl")["source_keys"].add("entities[]")
        tick = src["tick"]
        # ---- v2: frame-level scene stream -> typed archive.scene families
        scene: dict[str, Any] = {}
        for key, value in src.items():
            if key in _FRAME_HEADER_CONSUMED:
                # consumed by the canonical record header (ids/ticks/entities)
                report.note_mapped("truth_frames.jsonl", [key])
                continue
            family = _FRAME_SCENE_FAMILIES.get(key)
            if family is None:
                report.note_residual("truth_frames.jsonl", [key])
                continue
            scene[family] = value
            report.note_mapped("truth_frames.jsonl", [key])
        first = True
        for ent in src["entities"]:
            tp = ent.get("truth_pose", {})
            ann = ent.get("annotations", {})
            rp = ent.get("render_presence", {})
            # native state-mapping checkpoint v5: the registered string/frame
            # family annotations.state carries the raw producer state
            # (top-level ent.state, a sibling of truth_pose/annotations);
            # absent source state keeps the missing marker.  The raw entry
            # itself stays verbatim in archive.truth_frames.
            state = ent.get("state")
            act_top = ann.get("activity_type")
            # cross-check the nested facet copy of the same producer facet:
            # conflicts surface, they are never silently overwritten; the
            # registered annotations.activity_type path is reused, no second
            # same-meaning family is invented
            act_nested = (ann.get("state_facets") or {}).get("activity") or {}
            act_nested = act_nested.get("activity_type")
            if (act_top is not None and act_nested is not None
                    and str(act_top) != str(act_nested)):
                raise SchemaError(
                    f"{path}:{lineno} entity {ent['entity_id']}: conflicting "
                    f"activity_type (annotations.activity_type={act_top!r} vs "
                    f"state_facets.activity.activity_type={act_nested!r})")
            act_value = act_top if act_top is not None else act_nested
            # raw-facet checkpoint v8: the remaining real producer facet
            # leaves, read from the exact source sub-dicts (None -> keep the
            # source shape, no guessing, no unit invention)
            act_facets = ann.get("state_facets") or {}
            activity = act_facets.get("activity") or {}
            net = act_facets.get("network") or {}
            fields: dict[str, Any] = {
                "pose.position_enu_m": _frame_vec3(tp.get("position_enu_m")),
                "pose.velocity_enu_mps": _frame_vec3(tp.get("velocity_enu_mps")),
                "motion.speed_mps": (
                    float(ann["speed_mps"]) if isinstance(ann.get("speed_mps"), (int, float))
                    else V["missing"]),
                "annotations.state": (
                    str(state) if state is not None else V["missing"]),
                "annotations.activity_type": (
                    str(act_value) if act_value is not None else V["missing"]),
                "annotations.visibility_state": (
                    str(rp.get("visibility_state"))
                    if rp.get("visibility_state") is not None else V["missing"]),
            }
            # raw-facet checkpoint v8: exact producer facet leaves.  Renderer
            # visibility is constant and stays in render_presence semantics;
            # the network leaves are renderer-declared defaults, NOT
            # measured ns3/telemetry and NOT supervision targets.
            if activity.get("animation_hint") is not None:
                fields["annotations.animation_hint"] = \
                    str(activity["animation_hint"])
            if activity.get("posture") is not None:
                fields["annotations.posture"] = str(activity["posture"])
            if activity.get("social_state") is not None:
                fields["annotations.social_state"] = \
                    str(activity["social_state"])
            if net.get("status") is not None:
                fields["annotations.network_status"] = str(net["status"])
            if net.get("latency_ms") is not None:
                fields["annotations.network_latency_ms"] = \
                    float(net["latency_ms"])
            # packet_loss checkpoint v9 (native-approved): the raw REAL
            # scalar is preserved exactly -- no ratio/percent inference from
            # the literal, no unit invention, no range clamp.  The
            # renderer-default distinction (NOT measured ns3/telemetry)
            # stays in the annotations.network_packet_loss family
            # description; a separate provenance metadata field was removed
            # (user decision): the final typed consumer only copies
            # registered fields and has no special use for it, and the
            # verbatim source entry stays in archive.truth_frames.
            if net.get("packet_loss") is not None:
                fields["annotations.network_packet_loss"] = \
                    float(net["packet_loss"])
            yaw = tp.get("rotation_deg", {}).get("yaw_deg")
            if yaw is not None:
                fields["pose.yaw_deg"] = float(yaw)
            # v2: lossless per-entity archive of the exact source entry
            fields["archive.truth_frames"] = ent
            archived = [k for k in ent if k not in
                        ("truth_pose", "annotations", "render_presence")]
            if first:
                archived.append("entities[]")
            report.note_archive("truth_frames.jsonl", records=1,
                                bytes_=len(line.encode("utf-8")) // max(1, len(src["entities"])),
                                archived_keys=archived)
            if first:
                fields.update(scene)
                first = False
            episode.add({
                "record_kind": "frame",
                "id": f"{src['frame_id']}|{ent['entity_id']}",
                "tick": tick,
                "source": {"path": str(path), "line": lineno},
                "source_family": "truth_frames",
                "entity_id": ent["entity_id"],
                "fields": fields,
            })
    report.canonical_records["frame"] = sum(1 for _ in episode.by_kind("frame"))
    report.ticks_covered = sorted({r["tick"] for r in episode.by_kind("frame")})


def import_roster(episode: CanonicalEpisode, path: Path,
                  report: ImportReport) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    entities = doc["entities"]
    report.source_records["global_entity_roster.json"] = len(entities)
    report.stream_inventory("global_entity_roster.json")["source_keys"].update(
        doc.keys())
    for ent in doc["entities"]:
        report.stream_inventory("global_entity_roster.json")["source_keys"].update(
            f"entities[].{k}" for k in ent)
    observer_entity_id = report.event_traceability.get("_observer_entity_id")
    for key in doc:
        if key != "entities":
            report.note_residual("global_entity_roster.json", [key])
    for ent in entities:
        role = "background"
        if observer_entity_id is not None and ent["entity_id"] == observer_entity_id:
            role = "observer"
        fields = {
            "roster.entity_id": ent["entity_id"],
            "roster.entity_category": ent.get("entity_category", ""),
            "roster.entity_kind": ent.get("entity_kind", ""),
            "roster.observer_role": role,
        }
        # v2: lossless archive of the exact roster entity entry
        fields["archive.global_entity_roster"] = ent
        consumed = ("entity_id", "entity_category", "entity_kind")
        report.note_mapped("global_entity_roster.json",
                           ["entities", *(f"entities[].{k}" for k in consumed if k in ent)])
        report.note_archive("global_entity_roster.json", records=1,
                            bytes_=len(json.dumps(ent, ensure_ascii=False).encode("utf-8")),
                            archived_keys=[f"entities[].{k}" for k in ent
                                           if k not in consumed])
        episode.add({
            "record_kind": "entity",
            "id": ent["entity_id"],
            "source": {"path": str(path), "line": None},
            "source_family": "global_entity_roster",
            "fields": fields,
        }, check_id=True)
    report.canonical_records["entity"] = sum(1 for _ in episode.by_kind("entity"))


def import_truth_base(episode: CanonicalEpisode, path: Path,
                      report: ImportReport) -> None:
    """Import native initial assertions as state edges from their actual source."""
    base = json.loads(path.read_text(encoding="utf-8"))
    if base["episode_id"] != episode.episode_id:
        raise SchemaError(f"{path}: base episode binding differs from imported episode")
    tick = base["tick"]
    if not isinstance(tick, int) or isinstance(tick, bool):
        raise SchemaError(f"{path}: base tick must be an explicit integer")
    assertions = base["initial_assertions"]
    report.source_records["world_truth_graph_base.json"] = len(assertions)
    report.stream_inventory("world_truth_graph_base.json")["source_keys"].update(base.keys())
    report.note_mapped("world_truth_graph_base.json", ("tick", "episode_id", "initial_assertions"))
    for index, body in enumerate(assertions):
        evidence = {key: body[key] for key in
                    ("observations", "source_refs", "binding_provenance", "candidate_authority")
                    if key in body}
        # Existing JSON evidence preserves the complete native assertion, including
        # units and unknown-source requirements, without presenting it as a delta.
        evidence["initial_assertion"] = body
        episode.add({
            "record_kind": "edge",
            "id": f"{body['assertion_id']}|initial",
            "tick": tick,
            "source": {"path": str(path), "pointer": f"/initial_assertions/{index}",
                       "assertion_id": body["assertion_id"]},
            "source_family": "world_truth_graph_base",
            "fields": {"relation.predicate_id": body["predicate_id"],
                       "relation.tuple_id": body["tuple_id"],
                       "relation.bindings": dict(body["bindings"]),
                       "relation.asserted": True,
                       "relation.value": body["value"],
                       "relation.evidence": evidence},
        })
        report.note_archive("world_truth_graph_base.json", records=1,
                            bytes_=len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
                            archived_keys=body.keys())
    report.note_residual("world_truth_graph_base.json",
                         [key for key in base if key not in ("episode_id", "tick", "initial_assertions")])
    report.canonical_records["edge"] += len(assertions)


def import_truth_deltas(episode: CanonicalEpisode, path: Path,
                        report: ImportReport) -> None:
    lines, n = _open_lines(path)
    report.source_records["world_truth_graph_deltas.jsonl"] = n
    report.stream_inventory("world_truth_graph_deltas.jsonl")["source_keys"].update(
        ("delta_id", "tick", "operations", "operations[].operation"))
    count = 0
    op_kind_counts: Counter = Counter()
    structural_mapped = False
    for lineno, line in enumerate(lines, start=1):
        delta = json.loads(line)
        tick = delta["tick"]
        if not structural_mapped:
            # structural keys consumed by the record structure (delta_id/tick/
            # operations container and the per-op operation discriminator)
            report.note_mapped("world_truth_graph_deltas.jsonl",
                               ("delta_id", "tick", "operations", "operations[].operation"))
            structural_mapped = True
        for op_idx, op in enumerate(delta["operations"]):
            op_kind = op.get("operation")
            body = op.get("assertion", op)  # add_predicate_assertion nests the body
            op_kind_counts[op_kind] += 1
            fields: dict[str, Any] = {
                "relation.predicate_id": body["predicate_id"],
                "relation.tuple_id": body["tuple_id"],
                "relation.bindings": dict(body.get("bindings", {})),
                "relation.asserted": op_kind != "remove_predicate_assertion",
            }
            evidence: dict[str, Any] = {}
            if op_kind == "remove_predicate_assertion":
                # v2 (coordinator gap 4 / review F2): a removal asserts NO
                # current truth.  from_value is evidence of what the assertion
                # carried before removal; relation.value is explicitly absent.
                fields["relation.value"] = V["absent"]
                fields["relation.removed_from_value"] = str(body["from_value"])
            else:
                to_value = body.get("to_value")
                if to_value is None:
                    to_value = body.get("truth_value")
                if to_value is None:
                    to_value = body.get("value")
                if to_value is None:
                    raise SchemaError(
                        f"{path}:{lineno} op{op_idx} ({op_kind}): no truth value in payload")
                fields["relation.value"] = str(to_value)
                for ev_key in ("observations", "source_refs", "binding_provenance",
                               "candidate_authority"):
                    if ev_key in body:
                        evidence[ev_key] = body[ev_key]
                        report.note_mapped("world_truth_graph_deltas.jsonl", [ev_key])
            if evidence:
                fields["relation.evidence"] = evidence
            # v2: lossless archive of the exact op payload
            fields["archive.world_truth_graph_deltas"] = body
            report.note_archive("world_truth_graph_deltas.jsonl", records=1,
                                bytes_=len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
                                archived_keys=[k for k in body
                                               if k not in ("predicate_id", "tuple_id", "bindings",
                                                            "from_value", "to_value", "truth_value",
                                                            "value", "observations", "source_refs",
                                                            "binding_provenance",
                                                            "candidate_authority")])
            episode.add({
                "record_kind": "edge",
                "id": f"{delta['delta_id']}|op{op_idx}",
                "tick": tick,
                "source": {"path": str(path), "line": lineno,
                           "operation": op_kind,
                           "assertion_id": body.get("assertion_id"),
                           "delta_id": delta["delta_id"]},
                "source_family": "world_truth_graph_deltas",
                "fields": fields,
            })
            count += 1
    report.canonical_records["edge"] += count
    report.event_traceability["delta_op_kinds"] = dict(op_kind_counts)


def import_events(episode: CanonicalEpisode, path: Path,
                  report: ImportReport) -> None:
    lines, n = _open_lines(path)
    report.source_records["event_occurrences.jsonl"] = n
    report.stream_inventory("event_occurrences.jsonl")["source_keys"].update(
        json.loads(lines[0]).keys() if lines else ())
    first_mid: dict[str, Any] | None = None
    for lineno, line in enumerate(lines, start=1):
        ev = json.loads(line)
        end_tick = ev.get("end_tick")
        fields: dict[str, Any] = {
            "event.event_id": ev["event_id"],
            "event.event_family_id": ev["event_family_id"],
            "event.detection_tick": int(ev["detection_tick"]),
            "event.end_tick": (int(end_tick) if end_tick is not None else V["missing"]),
            "event.bindings": dict(ev.get("bindings", {})),
        }
        # v2: lossless archive of the exact event source line
        fields["archive.event_occurrences"] = ev
        consumed = ("event_id", "event_family_id", "detection_tick", "end_tick", "bindings")
        report.note_mapped("event_occurrences.jsonl", [k for k in consumed if k in ev])
        report.note_archive("event_occurrences.jsonl", records=1,
                            bytes_=len(line.encode("utf-8")),
                            archived_keys=[k for k in ev if k not in consumed])
        episode.add({
            "record_kind": "event",
            "id": ev["event_id"],
            "tick": ev["detection_tick"],
            "source": {"path": str(path), "line": lineno},
            "source_family": "event_occurrences",
            "fields": fields,
        })
        if first_mid is None and 0 < lineno <= n:
            first_mid = {
                "event_id": ev["event_id"],
                "source_path": str(path),
                "source_line": lineno,
                "detection_tick": ev["detection_tick"],
                "end_tick": end_tick,
                "canonical_id": ev["event_id"],
            }
    report.canonical_records["event"] = sum(1 for _ in episode.by_kind("event"))
    report.event_traceability["mid_event_sample"] = first_mid




def _npz_point_count(npz_path: Path) -> int | None:
    import numpy as np
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "points_sensor_ned_m" in z:
                return int(z["points_sensor_ned_m"].shape[0])
    except FileNotFoundError:
        return None
    return None


def import_capture_path(episode: CanonicalEpisode, capture_root: Path,
                        view_dir_name: str, report: ImportReport) -> None:
    view_dir = capture_root / view_dir_name
    if not view_dir.is_dir():
        report.missing_modalities.extend(["rgb", "lidar"])
        return
    seen_modalities: set[str] = set()
    capture_inv = report.stream_inventory("capture_meta")
    for modality, ext, array_tag in (("rgb", ".json", "image"),
                                     ("lidar", ".json", "points")):
        mod_dir = view_dir / modality
        if not mod_dir.is_dir():
            report.missing_modalities.append(modality)
            continue
        metas = sorted(p for p in mod_dir.glob(f"*{ext}") if p.name.endswith(ext))
        for meta_path in metas:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tick = int(meta["tick"])
            if capture_inv["source_keys"] or not capture_inv["archived_occurrences"]:
                capture_inv["source_keys"].update(meta.keys())
            payload_path = modality_dir_payload(mod_dir, meta_path, modality)
            record_count: int | None
            if payload_path is not None and payload_path.exists():
                if modality == "lidar":
                    record_count = _npz_point_count(payload_path)
                else:
                    record_count = int(payload_path.stat().st_size)
            else:
                record_count = None
            if record_count is None:
                fields_obs = {
                    "observation.path": str(meta_path.relative_to(capture_root)),
                    "observation.modality": modality,
                    "observation.tick": tick,
                    "observation.observer_entity_id": str(meta.get("uav_entity_id") or
                                                            meta.get("source_uav_entity_id") or ""),
                    "observation.record_count": {"kind": "missing"},
                }
            else:
                fields_obs = {
                    "observation.path": str(meta_path.relative_to(capture_root)),
                    "observation.modality": modality,
                    "observation.tick": tick,
                    "observation.observer_entity_id": str(meta.get("uav_entity_id") or
                                                          meta.get("source_uav_entity_id") or ""),
                    "observation.record_count": record_count,
                }
            # v2: lossless archive of the exact capture meta object
            fields_obs["archive.capture_meta"] = meta
            capture_inv["mapped_keys"].update(
                k for k in ("tick", "uav_entity_id", "source_uav_entity_id") if k in meta)
            capture_inv["mapped_occurrences"].update(
                {k: 1 for k in ("tick", "uav_entity_id", "source_uav_entity_id") if k in meta})
            capture_inv["source_keys"].update(meta.keys())
            report.note_archive("capture_meta", records=1,
                                bytes_=meta_path.stat().st_size,
                                archived_keys=[k for k in meta
                                               if k not in ("tick", "uav_entity_id",
                                                            "source_uav_entity_id")])
            episode.add({
                "record_kind": "observation",
                "id": f"{meta.get('logical_sample_id', meta_path.stem)}|{modality}",
                "tick": tick,
                "source": {"path": str(meta_path), "line": None,
                           # v2 (native 095638 followup): the observation
                           # declares the capture run it actually belongs to,
                           # so a same-run binding can verify it (repair 2)
                           "capture": {"run_id": capture_root.name}},
                "source_family": "capture",
                "fields": fields_obs,
            })
            seen_modalities.add(modality)
            report.modalities[modality] += 1
    for m in ("rgb", "lidar"):
        if m not in seen_modalities and m not in report.missing_modalities:
            report.missing_modalities.append(m)
    report.canonical_records["observation"] = sum(1 for _ in episode.by_kind("observation"))


def modality_dir_payload(mod_dir: Path, meta_path: Path, modality: str) -> Path | None:
    if modality == "lidar":
        cand = meta_path.with_suffix(".npz")
        return cand if cand.exists() else None
    cand = meta_path.with_suffix(".png")
    return cand if cand.exists() else None


def import_episode(render_ready_root: Path, semantic_truth_root: Path,
                   capture_root: Path | None, episode_id: str,
                   view_dir_name: str | None, observer_entity_id: str | None) -> tuple[
        CanonicalEpisode, ImportReport]:
    """Import one complete episode from read-only sources."""
    index_path = render_ready_root.parent / "domain_state_supplement/source_index.json"
    source_index = json.loads(index_path.read_text())
    entries = [entry for entry in source_index["episodes"] if entry["episode_id"] == episode_id]
    if len(entries) != 1:
        raise ValueError(f"Current source index must uniquely bind episode: {episode_id}")
    ep_dir = Path(entries[0]["render_ready_input"])
    if not ep_dir.is_absolute():
        raise ValueError(f"Current render input must be an absolute source path: {ep_dir}")
    ep_dir = ep_dir.resolve()
    objective = entries[0]["objective_semantic_truth"]
    if "matches_current_execution" not in objective:
        raise ValueError(f"Current objective binding flag is missing: {episode_id}")
    matches_current = objective["matches_current_execution"]
    if matches_current is False:
        raise ValueError(
            f"Full objective has not been generated for the current execution: {episode_id}; "
            f"full-semantic import is refused. Actual native execution remains independently readable at "
            f"{entries[0]['actual_execution_source']}")
    if matches_current is not True:
        raise ValueError(f"Current objective binding flag must be an explicit boolean for {episode_id}: {matches_current!r}")
    sem_dir = semantic_truth_root / episode_id
    report = ImportReport(episode_id=episode_id)
    report.event_traceability["_observer_entity_id"] = observer_entity_id
    report.sources = {
        "truth_frames": str(ep_dir / "truth_frames.jsonl"),
        "roster": str(ep_dir / "global_entity_roster.json"),
        "world_truth_base": str(sem_dir / "world_truth_graph_base.json"),
        "world_truth_deltas": str(sem_dir / "world_truth_graph_deltas.jsonl"),
        "events": str(sem_dir / "event_occurrences.jsonl"),
        "capture": str(capture_root) if capture_root else None,
        "episode_source_index": str(index_path),
    }
    episode = CanonicalEpisode(episode_id)
    import_roster(episode, ep_dir / "global_entity_roster.json", report)
    import_truth_frames(episode, ep_dir / "truth_frames.jsonl", report)
    declared = Path(objective["path"])
    if not declared.is_absolute():
        declared = render_ready_root.parent.parent / declared
    if declared.resolve() != sem_dir.resolve():
        raise ValueError(f"Current objective source binding differs: {declared} != {sem_dir}")
    import_truth_base(episode, sem_dir / "world_truth_graph_base.json", report)
    import_truth_deltas(episode, sem_dir / "world_truth_graph_deltas.jsonl", report)
    import_events(episode, sem_dir / "event_occurrences.jsonl", report)
    if capture_root is not None and view_dir_name:
        import_capture_path(episode, capture_root, view_dir_name, report)
    else:
        report.missing_modalities.extend(["rgb", "lidar"])
    # final tick-coverage consistency: frame stream must be gapless over its span
    ticks = report.ticks_covered
    if ticks:
        expected = list(range(ticks[0], ticks[-1] + 1))
        if ticks != expected:
            report.errors.append(
                f"frame tick coverage gap: observed {len(ticks)} ticks over "
                f"{ticks[0]}..{ticks[-1]} (expected {len(expected)})")
    report.finalize_inventories()
    return episode, report


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import time

    ap = argparse.ArgumentParser(description="Read-only source episode import")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--view", required=True, help="capture view directory name")
    ap.add_argument("--observer", required=True, help="observer entity id")
    ap.add_argument("--render-ready-root", default=str(REPO / "aw_data/render_ready_episodes_capture_filtered"))
    ap.add_argument("--semantic-truth-root", default=str(REPO / "aw_data/objective_semantic_truth"))
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--out-derived", required=True, help="output canonical records jsonl")
    ap.add_argument("--out-manifest", required=True, help="output import manifest json")
    args = ap.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    t0 = time.time()
    episode, report = import_episode(
        render_ready_root=Path(args.render_ready_root),
        semantic_truth_root=Path(args.semantic_truth_root),
        capture_root=Path(args.capture_root),
        episode_id=args.episode,
        view_dir_name=args.view,
        observer_entity_id=args.observer,
    )
    out = Path(args.out_derived)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in episode.records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = report.to_dict()
    manifest["canonical_records_jsonl"] = str(out)
    manifest["canonical_record_total"] = len(episode.records)
    manifest["wall_seconds"] = round(time.time() - t0, 3)
    mp = Path(args.out_manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"episode": args.episode, "records": len(episode.records),
                      "wall_seconds": manifest["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
