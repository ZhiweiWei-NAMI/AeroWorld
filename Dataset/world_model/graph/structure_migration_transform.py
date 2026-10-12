"""Migrate one structural shard in memory without changing its source values."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from Dataset.world_model.graph.field_structure import STRUCTURE_RULES, StructureRule
from Dataset.world_model.graph.structure_recovery import (
    RecoveredStructureShard, recover_structure_tables,
)
from Dataset.world_model.graph.structure_reuse import (
    ReuseRule, iter_reuse_rules, resolve_reuse,
)
from Dataset.world_model.graph.structure_storage import (
    PROJECTION_SCHEMA, RECORD_SCHEMA, SOURCE_LEAF_SCHEMA,
)


_SEED_PARENT = "parameters.deterministic_sumo_traffic_template.seed_profiles"
_SEED_PREFIX = "arm_script_plan:" + _SEED_PARENT + "."
_SEED_TOKEN_PREFIX = ("parameters", "deterministic_sumo_traffic_template", "seed_profiles")
_LIST_PATHS = (
    ("arm_script_plan", "events[].log_event.target_ids"),
    ("arm_scene_setup", "validation_rules[].contract.pad_boundary_policy.inside_required_for"),
)
_LIST_RULES = {
    (rule.source_family, rule.base_path): rule for rule in STRUCTURE_RULES
    if (rule.source_family, rule.base_path) in _LIST_PATHS
}
if set(_LIST_RULES) != set(_LIST_PATHS):
    raise ValueError("structural migration lacks an exact list declaration")

_REUSE_RULES = iter_reuse_rules()
_REUSE_BY_SOURCE = {rule.source_id: rule for rule in _REUSE_RULES}
_CONDITIONAL_SOURCE_IDS = frozenset(
    rule.source_id for rule in _REUSE_RULES if rule.condition != "always")
_CONTEXT_SOURCE_IDS = {
    key: source_id for rule in _REUSE_RULES
    for key, source_id in rule.required_context_source_ids
}
_MISSING_MARKER = b"\x00"  # JSON output cannot contain an unescaped NUL byte.
_RELATION_COLUMNS = (
    "old_index", "old_id", "new_local_index", "new_id", "instance_count",
)


@dataclass(frozen=True)
class MigratedShardTables:
    records: pa.Table
    projections: pa.Table
    source_leaves: pa.Table
    recovered: RecoveredStructureShard
    source_index_relations: pa.Table
    normalized_index_relations: pa.Table
    column_change_counts: dict[str, int]


def _all_true(values: pa.Array | pa.ChunkedArray) -> bool:
    return len(values) == 0 or pc.all(pc.fill_null(values, False)).as_py() is True


def _same_values(left: pa.Array | pa.ChunkedArray,
                 right: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.or_(pc.and_(pc.is_null(left), pc.is_null(right)),
                  pc.fill_null(pc.equal(left, right), False))


def _require_same(left: pa.Array | pa.ChunkedArray,
                  right: pa.Array | pa.ChunkedArray, message: str) -> None:
    if not _all_true(_same_values(left, right)):
        raise ValueError(message)


def _require_ordered(table: pa.Table, index_name: str) -> None:
    expected = pa.array(range(table.num_rows), type=pa.int64())
    if not _all_true(pc.equal(table[index_name], expected)):
        raise ValueError(f"structure {index_name} differs from row order")


def _replace(table: pa.Table, name: str,
             values: pa.Array | pa.ChunkedArray) -> pa.Table:
    return table.set_column(table.schema.get_field_index(name), name, values)


def _patch_columns(table: pa.Table, indices: pa.Array | pa.ChunkedArray,
                   values: dict[str, pa.Array | pa.ChunkedArray]) -> pa.Table:
    if not len(indices):
        return table
    if pc.count_distinct(indices).as_py() != len(indices):
        raise ValueError("structural migration updates one row twice")
    positions = pc.index_in(
        pa.array(range(table.num_rows), type=pa.int64()), value_set=indices)
    mask = pc.is_valid(positions)
    for name, replacements in values.items():
        if len(replacements) != len(indices):
            raise ValueError(f"structural migration {name} update length differs")
        table = _replace(table, name, pc.if_else(
            mask, pc.take(replacements, positions), table[name]))
    return table


def _single_source_projections(candidates: pa.Table,
                               projections: pa.Table) -> pa.ChunkedArray:
    indices = candidates["projection_index"]
    if len(indices) and (
            pc.count_distinct(indices).as_py() != len(indices) or
            not _all_true(pc.equal(
                pc.take(projections["source_leaf_count"], indices), 1))):
        raise ValueError("affected projection has multiple source leaves")
    return indices


def _entity_index(tokens_json: pa.Array | pa.ChunkedArray) -> pa.Array:
    match = pc.extract_regex(
        pc.cast(tokens_json, pa.string()),
        r'^\["entities",(?P<entity_index>[0-9]+),')
    result = pc.struct_field(match, "entity_index")
    if not _all_true(pc.is_valid(result)):
        raise ValueError("conditional reuse has no entities[i] source location")
    return pc.cast(result, pa.int64())


def _with_entity_keys(rows: pa.Table, projections: pa.Table) -> pa.Table:
    records = pc.take(projections["record_index"], rows["projection_index"])
    return rows.append_column("record_index", records).append_column(
        "entity_index", _entity_index(rows["source_tokens_json"]))


def _conditional_selected(candidates: pa.Table, projections: pa.Table,
                          leaves: pa.Table) -> pa.Array:
    conditional = candidates.filter(pc.is_in(
        candidates["source_field_id"],
        value_set=pa.array(sorted(_CONDITIONAL_SOURCE_IDS), type=pa.string())))
    if not conditional.num_rows:
        return pa.array([], type=pa.int64())
    conditional_count = conditional.num_rows
    conditional = conditional.select((
        "source_leaf_index", "projection_index", "source_field_id",
        "source_tokens_json", "raw_value_json", "raw_value_type"))
    conditional = _with_entity_keys(conditional, projections)
    markers = leaves.filter(pc.is_in(
        leaves["source_field_id"],
        value_set=pa.array(sorted(_CONTEXT_SOURCE_IDS.values()), type=pa.string())))
    markers = markers.select((
        "projection_index", "source_field_id", "source_tokens_json",
        "raw_value_json"))
    markers = _with_entity_keys(markers, projections)
    for key, source_id in _CONTEXT_SOURCE_IDS.items():
        selected = markers.filter(pc.equal(markers["source_field_id"], source_id))
        marker = selected.select(("record_index", "entity_index", "raw_value_json"))
        marker = marker.rename_columns(("record_index", "entity_index", key + "_json"))
        if marker.num_rows:
            grouped = marker.group_by(("record_index", "entity_index"),
                                      use_threads=False).aggregate([(key + "_json", "count")])
            if pc.max(grouped[key + "_json_count"]).as_py() != 1:
                raise ValueError(f"conditional entity repeats {key}")
        conditional = conditional.join(
            marker, keys=("record_index", "entity_index"),
            join_type="left outer", use_threads=False)
        if conditional.num_rows != conditional_count:
            raise ValueError("conditional marker join changed row count")

    context_keys = tuple(key + "_json" for key in _CONTEXT_SOURCE_IDS)
    for name in context_keys:
        conditional = _replace(conditional, name, pc.fill_null(
            conditional[name], pa.scalar(_MISSING_MARKER, type=pa.binary())))
    conditional = conditional.append_column(
        "candidate_nonempty", pc.not_equal(
            conditional["raw_value_json"], pa.scalar(b'""', type=pa.binary())))
    group_keys = ("source_field_id", *context_keys, "candidate_nonempty",
                  "raw_value_type")
    combinations = conditional.select(group_keys).group_by(
        group_keys, use_threads=False).aggregate(
            [("candidate_nonempty", "count")])
    decisions: list[dict[str, Any]] = []
    for row in combinations.to_pylist():
        rule = _REUSE_BY_SOURCE[row["source_field_id"]]
        context = {
            key: json.loads(row[key + "_json"])
            for key in _CONTEXT_SOURCE_IDS
            if row[key + "_json"] != _MISSING_MARKER
        }
        candidate_value: Any = None
        if rule.condition == "sumo_nonempty_source_entity_id" and row["raw_value_type"] == "string":
            candidate_value = "present" if row["candidate_nonempty"] else ""
        target = resolve_reuse(
            source_family=rule.source_family, field_path=rule.source_path,
            current_normalized_id=rule.old_normalized_id,
            current_binding_slot=rule.existing_binding_slot,
            source_branch=rule.source_branch, source_subtype=rule.source_subtype,
            context=context, candidate_value=candidate_value)
        if target is not None and (target.normalized_id != rule.normalized_id or
                                   target.binding_slot != rule.target_binding_slot):
            raise ValueError(f"conditional reuse target differs: {rule.source_id}")
        decisions.append({**{key: row[key] for key in group_keys},
                          "selected": target is not None})
    decision_schema = conditional.select(group_keys).schema.append(pa.field("selected", pa.bool_()))
    decision_table = pa.Table.from_pylist(decisions, schema=decision_schema)
    decided = conditional.join(decision_table, keys=group_keys,
                               join_type="left outer", use_threads=False)
    if (decided.num_rows != conditional.num_rows or
            not _all_true(pc.is_valid(decided["selected"]))):
        raise ValueError("conditional reuse decisions do not cover candidates")
    return decided.filter(decided["selected"])["source_leaf_index"].combine_chunks()


def _verify_exact_copies(selected: pa.Table, rules_by_source: dict[str, ReuseRule],
                         projections: pa.Table, leaves: pa.Table) -> None:
    exact_ids = [rule.source_id for rule in rules_by_source.values()
                 if rule.value_relation == "exact_copy"]
    aliases = selected.filter(pc.is_in(
        selected["source_field_id"],
        value_set=pa.array(exact_ids, type=pa.string())))
    if not aliases.num_rows:
        return
    canonical_ids = sorted({rules_by_source[source_id].canonical_source_id
                            for source_id in aliases["source_field_id"].unique().to_pylist()})
    counterparts = leaves.filter(pc.is_in(
        leaves["source_field_id"],
        value_set=pa.array(canonical_ids, type=pa.string())))
    for rank in range(3):
        source_ids = [rule.source_id for rule in rules_by_source.values()
                      if rule.value_relation == "exact_copy" and
                      rule.source_path.count("[]") == rank]
        left = aliases.filter(pc.is_in(
            aliases["source_field_id"],
            value_set=pa.array(source_ids, type=pa.string())))
        if not left.num_rows:
            continue
        right_ids = sorted({rules_by_source[source_id].canonical_source_id
                            for source_id in left["source_field_id"].unique().to_pylist()})
        right = counterparts.filter(pc.is_in(
            counterparts["source_field_id"],
            value_set=pa.array(right_ids, type=pa.string())))
        if not _all_true(pc.equal(pc.list_value_length(
                left["element_positions"]), rank)) or not _all_true(pc.equal(
                pc.list_value_length(right["element_positions"]), rank)):
            raise ValueError("exact-copy source instance rank differs")
        lookup_ids = pa.array(source_ids, type=pa.string())
        target_ids = pc.take(pa.array([
            rules_by_source[source_id].canonical_source_id for source_id in source_ids],
            type=pa.string()), pc.index_in(left["source_field_id"], value_set=lookup_ids))
        alias_positions = left["element_positions"]
        canonical_positions = right["element_positions"]
        left = pa.table({
            "source_leaf_index": left["source_leaf_index"],
            "record_index": pc.take(projections["record_index"], left["projection_index"]),
            "canonical_source_id": target_ids,
            "alias_json": left["raw_value_json"],
            "alias_type": left["raw_value_type"],
        })
        right = pa.table({
            "record_index": pc.take(projections["record_index"], right["projection_index"]),
            "canonical_source_id": right["source_field_id"],
            "counterpart_json": right["raw_value_json"],
            "counterpart_type": right["raw_value_type"],
            "counterpart_leaf_index": right["source_leaf_index"],
        })
        for axis in range(rank):
            name = f"position{axis}"
            left = left.append_column(name, pc.list_element(alias_positions, axis))
            right = right.append_column(name, pc.list_element(canonical_positions, axis))
        keys = ("record_index", "canonical_source_id") + tuple(
            f"position{axis}" for axis in range(rank))
        joined = left.join(right, keys=keys, join_type="left outer", use_threads=False)
        if (joined.num_rows != left.num_rows or
                not _all_true(pc.is_valid(joined["counterpart_leaf_index"])) or
                not _all_true(pc.equal(joined["alias_json"], joined["counterpart_json"])) or
                not _all_true(pc.equal(joined["alias_type"], joined["counterpart_type"]))):
            raise ValueError(f"exact-copy counterpart differs for rank {rank}")


def _reuse_updates(projections: pa.Table, leaves: pa.Table
                   ) -> tuple[pa.Table, pa.Table]:
    source_ids = pa.array(sorted(_REUSE_BY_SOURCE), type=pa.string())
    candidates = leaves.filter(pc.is_in(leaves["source_field_id"], value_set=source_ids))
    if not candidates.num_rows:
        return projections, leaves
    projection_indices = _single_source_projections(candidates, projections)
    rules = [_REUSE_BY_SOURCE[source_id] for source_id in source_ids.to_pylist()]
    rule_positions = pc.index_in(candidates["source_field_id"], value_set=source_ids)
    old_ids = pc.take(pa.array([rule.old_normalized_id for rule in rules]), rule_positions)
    old_slots = pc.take(pa.array([rule.existing_binding_slot for rule in rules],
                                 type=pa.string()), rule_positions)
    _require_same(pc.take(projections["normalized_id"], projection_indices),
                  old_ids, "reuse source has a different old normalized ID")
    _require_same(pc.take(projections["binding_slot"], projection_indices),
                  old_slots, "reuse source has a different old binding slot")
    selected_conditional = _conditional_selected(candidates, projections, leaves)
    selected_mask = pc.or_(
        pc.invert(pc.is_in(candidates["source_field_id"],
            value_set=pa.array(sorted(_CONDITIONAL_SOURCE_IDS), type=pa.string()))),
        pc.is_in(candidates["source_leaf_index"], value_set=selected_conditional))
    selected = candidates.filter(selected_mask)
    if not selected.num_rows:
        return projections, leaves
    selected_positions = pc.index_in(selected["source_field_id"], value_set=source_ids)
    target_ids = pc.take(pa.array([rule.normalized_id for rule in rules]),
                         selected_positions)
    target_slots = pc.take(pa.array([rule.target_binding_slot for rule in rules]),
                           selected_positions)
    _verify_exact_copies(selected, _REUSE_BY_SOURCE, projections, leaves)
    projections = _patch_columns(projections, selected["projection_index"], {
        "normalized_id": target_ids,
        "binding_slot": target_slots,
    })
    return projections, leaves


def _seed_updates(projections: pa.Table, leaves: pa.Table
                  ) -> tuple[pa.Table, pa.Table]:
    candidates = leaves.filter(pc.starts_with(leaves["source_field_id"],
                                              _SEED_PREFIX))
    if not candidates.num_rows:
        return projections, leaves
    projection_indices = _single_source_projections(candidates, projections)
    old_sources = candidates["source_field_id"].unique().to_pylist()
    new_sources = []
    for old in old_sources:
        key, sep, suffix = old[len(_SEED_PREFIX):].partition(".")
        if key not in {"seed00", "seed01", "seed02"}:
            raise ValueError(f"unknown existing seed-profile key: {old}")
        new_sources.append(_SEED_PREFIX + "{seed_profile}" + (sep + suffix if sep else ""))
    source_positions = pc.index_in(candidates["source_field_id"],
                                   value_set=pa.array(old_sources))
    expected_paths = pc.utf8_slice_codeunits(candidates["source_field_id"],
                                             len("arm_script_plan:"))
    _require_same(candidates["field_path"], expected_paths,
                  "seed-profile old source ID and field path differ")
    for row in candidates.select(("source_field_id", "source_tokens_json")).to_pylist():
        tokens = json.loads(row["source_tokens_json"])
        key = row["source_field_id"][len(_SEED_PREFIX):].split(".", 1)[0]
        if (len(tokens) < 4 or tuple(tokens[:3]) != _SEED_TOKEN_PREFIX or
                tokens[3] != key):
            raise ValueError("seed-profile source tokens differ from instance key")
    expected_norm = pc.binary_join_element_wise(
        pa.scalar("structural:source:"), candidates["source_field_id"], "")
    _require_same(pc.take(projections["normalized_id"], projection_indices),
                  expected_norm, "seed-profile old normalized ID differs")
    if (not _all_true(pc.is_null(pc.take(projections["binding_slot"], projection_indices))) or
            not _all_true(pc.is_null(pc.take(projections["declared_shape_json"], projection_indices))) or
            not _all_true(pc.equal(pc.take(projections["structure_status"],
                                           projection_indices), "source_only"))):
        raise ValueError("seed-profile old projection is not source-only")
    new_ids = pc.take(pa.array(new_sources), source_positions)
    new_paths = pc.utf8_slice_codeunits(new_ids, len("arm_script_plan:"))
    leaves = _patch_columns(leaves, candidates["source_leaf_index"], {
        "source_field_id": new_ids,
        "field_path": new_paths,
    })
    new_norm = pc.binary_join_element_wise(pa.scalar("structural:source:"),
                                           new_ids, "")
    projections = _patch_columns(projections, projection_indices,
                                 {"normalized_id": new_norm})
    return projections, leaves


def _shape_tuple(value: bytes | None) -> tuple[int | None, ...] | None:
    if value is None:
        return None
    result = json.loads(value)
    if not isinstance(result, list) or any(
            item is not None and (type(item) is not int or item < 0)
            for item in result):
        raise ValueError(f"invalid structural shape: {value!r}")
    return tuple(result)


def _list_status(rule: StructureRule, is_base: bool,
                 value_status: str, raw_type: str) -> str:
    if not is_base:
        if value_status != "present" or raw_type not in rule.element_types:
            raise ValueError(f"list element differs from declaration: {rule.source_family}:{rule.base_path}")
        return "resolved"
    expected_types = {
        "empty_list": "empty_list", "empty_object": "empty_object",
        "null": "null", "unknown": "string",
    }
    if (value_status not in rule.base_statuses or
            raw_type != expected_types.get(value_status)):
        raise ValueError(f"list base status differs from declaration: {rule.source_family}:{rule.base_path}")
    if value_status == "empty_object":
        raise ValueError("list empty-object conflict would change its normalized value")
    return "declared_only" if value_status in {"null", "unknown"} else "resolved"


def _list_updates(projections: pa.Table, leaves: pa.Table
                  ) -> tuple[pa.Table, pa.Table]:
    for family, base in _LIST_PATHS:
        rule = _LIST_RULES[family, base]
        source_base = family + ":" + base
        source_element = source_base + "[]"
        candidates = leaves.filter(pc.is_in(
            leaves["source_field_id"],
            value_set=pa.array((source_base, source_element), type=pa.string())))
        if not candidates.num_rows:
            continue
        projection_indices = _single_source_projections(candidates, projections)
        source_ids = candidates["source_field_id"]
        old_ids = pc.binary_join_element_wise(pa.scalar("structural:source:"),
                                              source_ids, "")
        _require_same(pc.take(projections["normalized_id"], projection_indices),
                      old_ids, "list old normalized ID differs")
        if (not _all_true(pc.is_null(pc.take(projections["binding_slot"], projection_indices))) or
                not _all_true(pc.is_null(pc.take(projections["declared_shape_json"], projection_indices))) or
                not _all_true(pc.equal(pc.take(projections["structure_status"],
                                               projection_indices), "source_only"))):
            raise ValueError("list old projection is not source-only")
        _require_same(pc.take(projections["normalized_value_json"], projection_indices),
                      candidates["raw_value_json"], "list normalized value differs from source")
        _require_same(pc.take(projections["value_type"], projection_indices),
                      candidates["raw_value_type"], "list normalized value type differs from source")
        observed = pc.take(projections["observed_shape_json"], projection_indices)
        normalized = pc.take(projections["normalized_shape_json"], projection_indices)
        payload = pc.take(projections["payload_shape_json"], projection_indices)
        _require_same(observed, normalized, "list observed and normalized shapes differ")
        _require_same(observed, payload, "list payload and observed shapes differ")
        statuses = pc.take(projections["value_status"], projection_indices)
        types = candidates["raw_value_type"]
        positions = pc.take(projections["payload_positions"], projection_indices)
        status_rows = pa.table({"source_id": source_ids, "status": statuses,
                                "raw_type": types}).group_by(
            ("source_id", "status", "raw_type"), use_threads=False).aggregate(
                [("status", "count")]).to_pylist()
        status_map = {}
        for row in status_rows:
            is_base = row["source_id"] == source_base
            status_map[(row["source_id"], row["status"], row["raw_type"])] = (
                _list_status(rule, is_base, row["status"], row["raw_type"]))
        target_statuses = []
        for source_id, status, raw_type, shape, item_positions in zip(
                source_ids.to_pylist(), statuses.to_pylist(), types.to_pylist(),
                observed.to_pylist(), positions.to_pylist()):
            target_statuses.append(status_map[source_id, status, raw_type])
            actual_shape = _shape_tuple(shape)
            if (actual_shape is None or len(actual_shape) != len(rule.axes) or
                    any(fixed is not None and actual != fixed
                        for actual, fixed in zip(actual_shape, rule.axes)) or
                    (source_id == source_base and status == "empty_list" and
                     actual_shape != (0,)) or
                    (source_id == source_base and item_positions) or
                    (source_id == source_element and
                     (any(axis is None or axis == 0 for axis in actual_shape) or
                      len(item_positions) != len(rule.axes) or
                      any(position < 0 or position >= axis
                          for position, axis in zip(item_positions, actual_shape))))):
                raise ValueError(f"list shape would change under current rule: {source_id}")
        declared_shape = json.dumps(list(rule.axes), separators=(",", ":")).encode()
        projections = _patch_columns(projections, projection_indices, {
            "normalized_id": pa.array([rule.normalized_id] * candidates.num_rows),
            "binding_slot": pa.array([rule.binding_slot] * candidates.num_rows),
            "declared_shape_json": pa.array([declared_shape] * candidates.num_rows,
                                              type=pa.binary()),
            "structure_status": pa.array(target_statuses),
        })
    return projections, leaves


def _first_seen_local(table: pa.Table, id_name: str,
                      index_name: str, row_name: str) -> pa.Table:
    if not table.num_rows:
        return _replace(table, index_name, pa.array([], type=pa.int64()))
    first = table.select((id_name, row_name)).group_by(
        id_name, use_threads=False).aggregate([(row_name, "min")])
    first = first.sort_by(((row_name + "_min", "ascending"),))
    ordered = first[id_name]
    if not _all_true(pc.is_valid(ordered)):
        raise ValueError(f"migrated {id_name} is absent")
    local = pc.index_in(table[id_name], value_set=ordered)
    if not _all_true(pc.is_valid(local)):
        raise ValueError(f"migrated {id_name} has no local index")
    return _replace(table, index_name, pc.cast(local, pa.int64()))


def _relations(before: pa.Table, after: pa.Table, *,
               index_name: str, id_name: str) -> pa.Table:
    if not before.num_rows:
        return pa.table({
            "old_index": pa.array([], type=pa.int64()),
            "old_id": pa.array([], type=pa.string()),
            "new_local_index": pa.array([], type=pa.int64()),
            "new_id": pa.array([], type=pa.string()),
            "instance_count": pa.array([], type=pa.int64()),
        })
    pairs = pa.table({
        "old_index": before[index_name], "old_id": before[id_name],
        "new_local_index": after[index_name], "new_id": after[id_name],
    })
    grouped = pairs.group_by(_RELATION_COLUMNS[:-1], use_threads=False).aggregate(
        [("old_index", "count")])
    grouped = grouped.rename_columns(_RELATION_COLUMNS).sort_by((
        ("old_index", "ascending"), ("new_local_index", "ascending"),
        ("old_id", "ascending"), ("new_id", "ascending")))
    if pc.sum(grouped["instance_count"]).as_py() != before.num_rows:
        raise ValueError("migration index relations do not cover all rows")
    old_keys = grouped.select(("old_index", "old_id")).group_by(
        ("old_index", "old_id"), use_threads=False).aggregate(
            [("old_index", "count")])
    by_index = old_keys.group_by("old_index", use_threads=False).aggregate(
        [("old_id", "count")])
    if by_index.num_rows and pc.max(by_index["old_id_count"]).as_py() != 1:
        raise ValueError(f"one old {index_name} identifies multiple IDs")
    return grouped


def _change_count(left: pa.Array | pa.ChunkedArray,
                  right: pa.Array | pa.ChunkedArray) -> int:
    return int(pc.sum(pc.cast(pc.invert(_same_values(left, right)), pa.int64())).as_py() or 0)


def migrate_shard_tables(
        records: pa.Table, projections: pa.Table, source_leaves: pa.Table, *,
        shard_id: int, relative_path: str,
        ) -> MigratedShardTables:
    """Apply current structural definitions to one complete old shard."""
    if (records.schema != RECORD_SCHEMA or projections.schema != PROJECTION_SCHEMA or
            source_leaves.schema != SOURCE_LEAF_SCHEMA):
        raise ValueError("migration input table schema differs")
    _require_ordered(records, "record_index")
    _require_ordered(projections, "projection_index")
    _require_ordered(source_leaves, "source_leaf_index")
    old_projections, old_leaves = projections, source_leaves
    projections, source_leaves = _reuse_updates(projections, source_leaves)
    projections, source_leaves = _seed_updates(projections, source_leaves)
    projections, source_leaves = _list_updates(projections, source_leaves)
    projections = _first_seen_local(
        projections, "normalized_id", "normalized_index", "projection_index")
    source_leaves = _first_seen_local(
        source_leaves, "source_field_id", "source_index", "source_leaf_index")
    if (projections.num_rows != old_projections.num_rows or
            source_leaves.num_rows != old_leaves.num_rows or
            not old_projections.drop((
                "normalized_index", "normalized_id", "binding_slot",
                "declared_shape_json", "structure_status")).equals(
                projections.drop((
                    "normalized_index", "normalized_id", "binding_slot",
                    "declared_shape_json", "structure_status"))) or
            not old_leaves.drop((
                "source_index", "source_field_id", "field_path")).equals(
                source_leaves.drop((
                    "source_index", "source_field_id", "field_path")))):
        raise ValueError("migration changed a protected source or projection column")
    recovered = recover_structure_tables(
        records, projections, source_leaves, shard_id, relative_path, 0)
    if (recovered.source_global_indices != tuple(range(len(recovered.catalog_fragment.source_rows))) or
            recovered.normalized_global_indices != tuple(range(len(recovered.catalog_fragment.normalized_rows)))):
        raise ValueError("migrated fragment indexes are not first-seen local indexes")
    changes = {
        "normalized_id": _change_count(old_projections["normalized_id"], projections["normalized_id"]),
        "source_field_id": _change_count(old_leaves["source_field_id"], source_leaves["source_field_id"]),
        "field_path": _change_count(old_leaves["field_path"], source_leaves["field_path"]),
        "binding_slot": _change_count(old_projections["binding_slot"], projections["binding_slot"]),
        "declared_shape_json": _change_count(old_projections["declared_shape_json"], projections["declared_shape_json"]),
        "structure_status": _change_count(old_projections["structure_status"], projections["structure_status"]),
    }
    return MigratedShardTables(
        records, projections, source_leaves, recovered,
        _relations(old_leaves, source_leaves,
                   index_name="source_index", id_name="source_field_id"),
        _relations(old_projections, projections,
                   index_name="normalized_index", id_name="normalized_id"),
        changes,
    )
