"""S01.03 lossless structural text: JSONL round-trip and compact-table encoding.

Two reversible encodings over the same canonical records, plus graph/typed
dictionary views for downstream consumers:

- ``encode_jsonl`` / ``decode_jsonl``: canonical record -> one JSON object per
  line.  Lossless by construction (JSON round-trip of the exact record).
- ``encode_compact`` / ``decode_compact``: per-record-kind columnar tables with
  per-column type tags and explicit non-present kind markers.  Values are
  stored as their exact source JSON text; no rounding or float re-formatting.
  Field order inside records is normalized (sorted keys) so permuting
  entity/field order decodes to identical records; list order is preserved and
  is semantically significant (vector axes, bindings order in source).

Lossless definition here: decode(encode(x)) == x for every canonical record,
compared with exact equality on ints, strings, bools and the *binary64* float
values parsed from source text; no epsilon is introduced.  Tolerance therefore
comes from the original data's JSON precision, not from an arbitrary setting.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from p01v4.contracts.schema import (
    FIELD_REGISTRY,
    VALUE_KINDS,
    SchemaError,
    validate_value,
)

# Column layout per record kind is uniform (see _COMPACT_COLUMNS below).

# v2 cell grammar: structural markers are explicit wire shapes ({"np": kind},
# {"f", "v"}) with an escape level {"d": x} for colliding data; see
# _COMPACT_CELL_ENCODING below.  v1 used a bare '!kind' string, which collided
# with present '!'-prefixed source strings (coordinator gap 2 / review F5).


def canonicalize(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical in-memory form with sorted field keys.

    Validation is NOT re-run here; callers must import through
    p01v4.data.import_episode (which validates) or call
    p01v4.contracts.schema.validate_record themselves.

    Semantic metadata is preserved (v2, native third replay): ``provenance``
    (available_time policy, capture binding) and ``source_family`` are part
    of the record's declared identity, so canonicalize -> compact keeps them
    and downstream availability gates (window input, model-visible views,
    scaler fit) stay effective over the canonical form.
    """
    out = {
        "record_kind": record["record_kind"],
        "id": record["id"],
        "episode_id": record.get("episode_id"),
        "tick": record.get("tick"),
        "source": record.get("source"),
        "entity_id": record.get("entity_id"),
        "source_family": record.get("source_family"),
        "provenance": record.get("provenance"),
        "fields": {k: record["fields"][k] for k in sorted(record["fields"])},
    }
    return {k: v for k, v in out.items() if v is not None or k in ("tick", "episode_id")}


# ------------------------------ JSONL ------------------------------------

def encode_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for rec in records:
        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")


def decode_jsonl(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line]


# --------------------------- compact table -------------------------------

_COMPACT_COLUMNS = ["id", "episode_id", "tick", "entity_id", "source",
                    "provenance", "source_family", "field_names", "field_cells"]


#: v2 cell grammar (coordinator gap 2 / review F5): exactly reversible for
#: every present value.  Reserved wire shapes: ``{"np": kind}`` = non-present
#: kind; ``{"f": frame, "v": value}`` = frame-tagged vector; ``{"d": x}`` =
#: escape level for data that would collide with a reserved shape.  All other
#: cells are literal data - a present string like '!missing' or a plain
#: object like {'f': 'N', 'v': 'x'} round-trips exactly; nothing is rejected.
_COMPACT_CELL_ENCODING = "p01v4-compact-cells-2"


def _encode_cell(value: Any) -> Any:
    """Cell -> compact cell (v2 grammar).

    Canonical non-present markers ``{"kind": K}`` become ``{"np": K}``;
    canonical frame-tagged vectors ``{"frame": F, "value": V}`` become
    ``{"f": F, "v": V}``; any data Mapping whose key set would collide with
    a reserved wire shape gets one escape level ``{"d": ...}``.  Everything
    else - including '!'-prefixed strings - passes through literally.
    """
    if isinstance(value, Mapping):
        keys = set(value)
        if keys == {"kind"} and value["kind"] in VALUE_KINDS:
            return {"np": value["kind"]}
        if keys == {"frame", "value"}:
            return {"f": value["frame"], "v": value["value"]}
        if keys in ({"np"}, {"d"}, {"f", "v"}):
            return {"d": dict(value)}
    return value


def _decode_cell(value: Any) -> Any:
    if isinstance(value, Mapping):
        keys = set(value)
        if keys == {"np"}:
            kind = value["np"]
            if kind not in VALUE_KINDS:
                raise SchemaError(f"compact cell: unknown non-present kind '{kind}'")
            return {"kind": kind}
        if keys == {"f", "v"}:
            return {"frame": value["f"], "value": value["v"]}
        if keys == {"d"}:
            return value["d"]
    return value


def encode_compact(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Records -> one columnar table per record kind.

    Repeated strings and structures (source provenance dicts, field-name
    lists) are interned into a per-table ``pool`` of unique values; rows hold
    pool indices.  This is a standard columnar dictionary encoding and stays
    exactly lossless: decode resolves every index back to its value.
    """
    tables: dict[str, dict[str, Any]] = {}
    for rec in records:
        kind = rec["record_kind"]
        table = tables.setdefault(kind, {
            "columns": list(_COMPACT_COLUMNS), "pool": [], "_index": {}, "rows": []})
        pool = table["pool"]
        index = table["_index"]

        def ref(obj: Any) -> int:
            key = json.dumps(obj, ensure_ascii=False, sort_keys=True)
            i = index.get(key)
            if i is None:
                i = len(pool)
                index[key] = i
                pool.append(obj)
            return i

        names = sorted(rec["fields"])
        prov = rec.get("provenance")
        # tick column (native-approved exact line): an absent tick key
        # encodes the {"np": "tick"} marker; a present value (None included)
        # encodes verbatim.  The decoder is retained unchanged (native
        # decision): marker cell -> tick=None, null cell -> key absent.
        # episode_id keeps the inverse convention (line below).
        ep_cell = (ref(rec["episode_id"]) if rec.get("episode_id") is not None
                   else {"np": "episode_id"}) if "episode_id" in rec else None
        tick_cell = rec["tick"] if "tick" in rec else {"np": "tick"}
        table["rows"].append([
            rec["id"],
            ep_cell,
            tick_cell,
            ref(rec.get("entity_id")) if rec.get("entity_id") is not None else None,
            ref(rec.get("source")) if rec.get("source") is not None else None,
            ref(prov) if prov is not None else None,
            ref(rec["source_family"]) if rec.get("source_family") is not None else None,
            ref(names),
            [_encode_cell(rec["fields"][n]) for n in names],
        ])
    for table in tables.values():
        table.pop("_index")
    return tables


def decode_compact(tables: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for kind, table in tables.items():
        cols = table["columns"]
        if cols != _COMPACT_COLUMNS:
            raise SchemaError(f"compact table '{kind}': unexpected column layout {cols}")
        pool = table["pool"]
        for row in table["rows"]:
            if len(row) != len(cols):
                raise SchemaError(f"compact table '{kind}': ragged row width")
            cid, episode_ref, tick, entity_ref, source_ref, prov_ref, fam_ref, names_ref, cells = row
            if not isinstance(names_ref, int) or names_ref >= len(pool):
                raise SchemaError(f"compact table '{kind}': bad field_names pool ref")
            names = pool[names_ref]
            if len(names) != len(cells):
                raise SchemaError(f"compact table '{kind}': names/cells width mismatch")
            rec: dict[str, Any] = {
                "record_kind": kind,
                "id": cid,
                "fields": {n: _decode_cell(c) for n, c in zip(names, cells)},
            }
            # mirror the approved tick-cell convention: {"np": "tick"} =
            # tick key absent; null cell = tick present with None (the
            # approved encoder line reserves the null cell for present-None);
            # otherwise the value verbatim.  Time-indexed kinds validate
            # their tick at the schema boundary.
            if isinstance(episode_ref, dict):
                if episode_ref != {"np": "episode_id"}:
                    raise SchemaError(f"compact table '{kind}': bad episode_id cell")
                rec["episode_id"] = None
            elif episode_ref is not None:
                rec["episode_id"] = pool[episode_ref]
            if isinstance(tick, dict):
                if tick != {"np": "tick"}:
                    raise SchemaError(f"compact table '{kind}': bad tick cell")
            else:
                rec["tick"] = tick
            if source_ref is not None:
                rec["source"] = pool[source_ref]
            if prov_ref is not None:
                rec["provenance"] = pool[prov_ref]
            if fam_ref is not None:
                rec["source_family"] = pool[fam_ref]
            if entity_ref is not None:
                rec["entity_id"] = pool[entity_ref]
            records.append(rec)
    return records


# --------------------------- typed / graph views -------------------------

def to_typed_input(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Typed dictionaries required by graph/typed consumers, from the same
    canonical records: entities, per-tick entity states, edges, events,
    observations.

    Registry-driven field retention (native state-typed checkpoint v5):
    every field family the registry admits for a record kind is carried
    into the typed output verbatim — no per-name whitelist.  The declared
    output keys are preserved (entities/states/edges/events plus the
    registered observations branch); canonical and archive records are not
    modified.  Schema-legal non-present values are preserved with their
    exact tag (native non-present checkpoint v6): a family's declared
    ``{"kind": ...}`` cell stays in the typed ``fields`` map verbatim so
    missing/inapplicable/absent/source_unknown remain distinct in the final
    dictionary — no default, no fallback, no blanket UNKNOWN.  Only the old
    edge top-level ``value`` alias omits a no-current-value marker (the
    exact current semantics stay in ``fields`` as ``relation.value`` plus
    ``asserted``/``removed_from_value``); no fake truth value is created.
    """
    def _registry_fields(rec: Mapping[str, Any]) -> dict[str, Any]:
        kind = rec["record_kind"]
        out: dict[str, Any] = {}
        for key, value in rec.get("fields", {}).items():
            if key.startswith("archive."):
                # verbatim archive families are never admitted into the
                # model-visible view (model_input_projection strips them);
                # the typed input keeps that contract
                continue
            spec = FIELD_REGISTRY.get(key)
            if spec is None or kind not in spec["records"]:
                continue
            if isinstance(value, Mapping) and "kind" in value \
                    and "value" not in value:
                # singleton-kind cell: the schema boundary decides (native
                # v7).  A legal non-present tag for this family is retained
                # verbatim; every invalid cell -- unknown kind, naked
                # {"kind": "present"}, or a kind not declared for the
                # family -- raises SchemaError instead of being silently
                # dropped or passed through.
                validate_value(key, value,
                               f"{kind}#{rec.get('id', '?')}.{key}",
                               record_kind=kind)
            out[key] = value
        return out

    entities: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for rec in records:
        kind = rec["record_kind"]
        reg = _registry_fields(rec)
        if kind == "entity":
            entities[rec["id"]] = {
                "entity_id": rec["fields"]["roster.entity_id"],
                "entity_category": rec["fields"].get("roster.entity_category"),
                "entity_kind": rec["fields"].get("roster.entity_kind"),
                "observer_role": rec["fields"].get("roster.observer_role"),
            }
        elif kind == "frame":
            entry = {
                "tick": rec["tick"],
                "entity_id": rec.get("entity_id"),
                "source": rec.get("source"),
            }
            # declared typed pose/speed/visibility keys keep their names and
            # their previous always-present .get() semantics (a non-present
            # marker passes through unchanged, as before); the registry
            # field map below adds every other admitted family
            for typed_key, field in (
                    ("position_enu_m", "pose.position_enu_m"),
                    ("velocity_enu_mps", "pose.velocity_enu_mps"),
                    ("yaw_deg", "pose.yaw_deg"),
                    ("speed_mps", "motion.speed_mps"),
                    ("visibility_state", "annotations.visibility_state")):
                entry[typed_key] = rec["fields"].get(field)
            entry["fields"] = reg
            states.append(entry)
        elif kind == "edge":
            entry = {
                "tick": rec["tick"],
                "predicate_id": rec["fields"]["relation.predicate_id"],
                "tuple_id": rec["fields"]["relation.tuple_id"],
                "bindings": rec["fields"]["relation.bindings"],
                "source": rec.get("source"),
            }
            # asserted/removed semantics (v2): a removal carries no current
            # value; relation.value is absent-marker (not KeyError, never a
            # fake value), the prior value stays in removed_from_value.
            # The old top-level alias (native v6) is emitted only for
            # actual content: a schema-legal no-current-value marker stays
            # in fields only.
            if "relation.value" in reg and not (
                    isinstance(reg["relation.value"], dict)
                    and set(reg["relation.value"]) == {"kind"}
                    and reg["relation.value"]["kind"] in VALUE_KINDS):
                entry["value"] = reg["relation.value"]
            entry["asserted"] = rec["fields"].get("relation.asserted")
            if "relation.removed_from_value" in reg:
                entry["removed_from_value"] = reg["relation.removed_from_value"]
            entry["fields"] = reg
            edges.append(entry)
        elif kind == "event":
            events.append({
                "event_id": rec["fields"]["event.event_id"],
                "family": rec["fields"]["event.event_family_id"],
                "detection_tick": rec["fields"]["event.detection_tick"],
                "end_tick": rec["fields"]["event.end_tick"],
                "bindings": rec["fields"]["event.bindings"],
                "source": rec.get("source"),
                "fields": reg,
            })
        elif kind == "observation":
            observations.append({
                "tick": rec["tick"],
                "observer_entity_id":
                    rec["fields"]["observation.observer_entity_id"],
                "modality": rec["fields"]["observation.modality"],
                "path": rec["fields"]["observation.path"],
                "source": rec.get("source"),
                "fields": reg,
            })
    return {"entities": entities, "states": states, "edges": edges,
            "events": events, "observations": observations}


# ----------------------------- comparison --------------------------------

def records_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Exact equality of canonical records (no epsilon)."""
    return a == b


def semantic_equivalent(a: Iterable[Mapping[str, Any]],
                        b: Iterable[Mapping[str, Any]]) -> bool:
    """Order-independent equivalence: compare as multisets of canonical JSON."""
    def key(rec: Mapping[str, Any]) -> str:
        return json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sorted(map(key, a)) == sorted(map(key, b))
