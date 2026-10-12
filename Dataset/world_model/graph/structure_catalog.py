"""Aggregate observed source leaves and exact structural declarations.

Indexes are dense within one scan. Field IDs remain the references across scans.
The caller owns source ordering and output serialization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from Dataset.world_model.graph.field_paths import source_field_id
from Dataset.world_model.graph.field_structure import (
    COMPUTE_QUEUE_SOURCE_IDS, extra_structure_bindings, iter_structure_bindings,
    IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID,
    IMPLEMENTATION_SOURCES_GAP_SOURCE_ID,
    is_declared_implementation_sources_gap,
    is_declared_implementation_sources_gap_projection,
)
from Dataset.world_model.graph.structure_reuse import iter_reuse_rules
from Dataset.world_model.graph.structure_schema import is_publication_excluded_path
from Dataset.world_model.graph.structure_grammar_catalog import (
    GrammarCatalogPlan, GrammarSourceRule, compile_catalog_grammars,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from Dataset.world_model.graph.field_catalog import SourceDeclaration
    from Dataset.world_model.graph.source_projection import ProjectedField


_POSE_PATHS = ("pos_enu[]", "values.pos_enu[]", "vel_mps[]", "values.vel_mps[]")
_POSE_IDS = {
    source_field_id("arm_states", path):
        "structural:arm_states:observed_pose." + path.removeprefix("values.")
    for path in _POSE_PATHS
}
_QUEUE_PATHS = ("queue_depth", "resource_balance.queue_depth")


@dataclass(frozen=True)
class CatalogIndices:
    normalized_index: int
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class CatalogRows:
    source_fields: tuple[dict[str, Any], ...]
    normalized_fields: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CatalogFragment:
    """Observed rows with indexes local to one worker batch."""

    source_rows: tuple[dict[str, Any], ...]
    normalized_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CatalogIndexMaps:
    """Global indexes addressed by a fragment's local indexes."""

    source_indices: tuple[int, ...]
    normalized_indices: tuple[int, ...]


def _raw_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "empty_list"
    if type(value) is dict:
        return "empty_object"
    raise TypeError(f"source leaf has unsupported type: {type(value).__name__}")


def _digest_map_tokens(field_path: str) -> tuple[str, str] | None:
    """Return the two traversal tokens of a digest-map leaf, else ``None``.

    A publication-excluded integrity map (``code``, ``files``,
    ``implementation``) holds one producer-declared file key per digest string,
    and those keys contain dots and slashes of their own
    (``Dataset/tools/batch_generate.py``), so a published ``field_path`` is
    ``<map>.<key>`` with exactly one separator however many dots the key has.
    Splitting once therefore recovers the two tokens
    ``is_publication_excluded_path`` is written against.  Any other shape is not
    a digest-map leaf and is never tokenized here, so it keeps failing the
    record-grammar gate untouched.
    """
    if "." not in field_path:
        return None
    map_name, key = field_path.split(".", 1)
    if not map_name or not key:
        return None
    return map_name, key


def _shape(shape: tuple[int | None, ...] | None) -> list[int | None] | None:
    if shape is None:
        return None
    if (not isinstance(shape, tuple) or
            any(axis is not None and (type(axis) is not int or axis < 0)
                for axis in shape)):
        raise ValueError(f"invalid structural shape: {shape!r}")
    return list(shape)


def _ordered(values: set[Any]) -> list[Any]:
    return sorted(values, key=lambda item: json.dumps(item, ensure_ascii=False))


def _increment(counts: dict[str, int], value: str) -> None:
    counts[value] = counts.get(value, 0) + 1


def _summary_structure_status(instance_count: int,
                              counts: dict[str, int]) -> str:
    if counts.get("structure_conflict"):
        return "structure_conflict"
    if instance_count == 0:
        return "declared_only"
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed_observation"


class StructureCatalog:
    """Assign live indexes while consuming each projected source value once."""

    def __init__(self) -> None:
        self.source_index_by_id: dict[str, int] = {}
        self.normalized_index_by_id: dict[str, int] = {}
        self._source_rows: list[dict[str, Any]] = []
        self._normalized_rows: list[dict[str, Any]] = []
        self._finished: CatalogRows | None = None
        self._resumed = False
        # Published instance counts, recorded only by resume_from_published.  A
        # fresh catalog keeps this empty, so the historical integrity exception
        # below can never open there.
        self._published_instance_counts: dict[str, int] = {}
        self.grammar_declaration_statuses: tuple[str, ...] | None = None

    @classmethod
    def resume_from_published(cls, rows: CatalogRows) -> StructureCatalog:
        """Resume first-observation indexes from a validated published catalog."""
        validate_catalog_rows(rows)
        catalog = cls()
        catalog._resumed = True
        for source in rows.source_fields:
            item = deepcopy(source)
            item["observed_source_value_types"] = set(item["observed_source_value_types"])
            bindings = {}
            for binding in item["normalizations"]:
                for name in (
                    "observed_source_value_types", "observed_value_types",
                    "observed_value_statuses", "observed_structure_statuses",
                ):
                    binding[name] = set(binding[name])
                for name in ("observed_shapes", "normalized_shapes"):
                    binding[name] = {
                        tuple(shape) if shape is not None else None
                        for shape in binding[name]
                    }
                key = binding["normalized_id"], binding["binding_slot"]
                if key in bindings:
                    raise ValueError(f"duplicate published structure binding: {source['id']}:{key}")
                bindings[key] = binding
            item["normalizations"] = bindings
            catalog.source_index_by_id[item["id"]] = item["source_index"]
            catalog._source_rows.append(item)
            catalog._published_instance_counts[item["id"]] = item["instance_count"]
        for normalized in rows.normalized_fields:
            item = deepcopy(normalized)
            item["source_indices"] = {
                member["source_index"] for member in item.pop("source_members")
            }
            item.pop("structure_status")
            item["observed_value_types"] = set(item["observed_value_types"])
            for name in ("observed_shapes", "normalized_shapes"):
                item[name] = {
                    tuple(shape) if shape is not None else None
                    for shape in item[name]
                }
            catalog.normalized_index_by_id[item["id"]] = item["normalized_index"]
            catalog._normalized_rows.append(item)
        return catalog

    def reclassify_implementation_sources_gap(self, expected: int) -> None:
        """Apply the current producer's exact empty-object metadata declaration."""
        if not self._resumed or self._finished is not None:
            raise ValueError("metadata-gap reclassification requires an open published catalog")
        if type(expected) is not int or expected < 1:
            raise ValueError("metadata-gap count must be positive")
        source = self._source_rows[self.source_index_by_id[
            IMPLEMENTATION_SOURCES_GAP_SOURCE_ID]]
        normalized = self._normalized_rows[self.normalized_index_by_id[
            IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID]]
        counts = normalized["structure_status_counts"]
        if (counts.get("structure_conflict") != expected or
                counts.get("declared_only", 0) != 0 or
                any(status not in {"resolved", "structure_conflict"} for status in counts)):
            raise ValueError("published implementation_sources conflict count differs")
        bindings = [item for item in source["normalizations"].values()
                    if item["normalized_id"] == IMPLEMENTATION_SOURCES_GAP_NORMALIZED_ID
                    and item["selector"] == "base_status"]
        if len(bindings) != 1:
            raise ValueError("published implementation_sources base binding differs")
        binding = bindings[0]
        if (binding["declaration_status"] != "structure_declared" or
                binding["structure_statuses"] != ["resolved", "structure_conflict"] or
                binding["observed_structure_statuses"] != {"structure_conflict"} or
                "empty_object" not in binding["observed_value_statuses"] or
                "empty_object" not in binding["observed_source_value_types"]):
            raise ValueError("published implementation_sources source binding differs")
        del counts["structure_conflict"]
        counts["declared_only"] = expected
        binding["structure_statuses"] = ["declared_only", "resolved"]
        binding["observed_structure_statuses"] = {"declared_only"}

    def _source(self, source_id: str, family: str, path: str,
                branch: str | None, subtype: str | None) -> dict[str, Any]:
        expected = source_field_id(family, path, branch, subtype)
        if source_id != expected:
            raise ValueError(f"source field ID differs from its path: {source_id}:{expected}")
        index = self.source_index_by_id.get(source_id)
        if index is None:
            index = len(self._source_rows)
            self.source_index_by_id[source_id] = index
            self._source_rows.append({
                "kind": "source_field", "id": source_id, "source_index": index,
                "source_family": family, "field_path": path,
                "source_branch": branch, "source_subtype": subtype,
                "instance_count": 0, "observed_source_value_types": set(),
                "normalizations": {},
            })
        row = self._source_rows[index]
        if (row["source_family"], row["field_path"], row["source_branch"],
                row["source_subtype"]) != (family, path, branch, subtype):
            raise ValueError(f"one source ID has conflicting paths: {source_id}")
        return row

    def _normalized(self, normalized_id: str,
                    declared_shape: tuple[int | None, ...] | None) -> dict[str, Any]:
        if not isinstance(normalized_id, str) or not normalized_id.startswith("structural:"):
            raise ValueError(f"invalid structural normalized ID: {normalized_id!r}")
        shape = _shape(declared_shape)
        index = self.normalized_index_by_id.get(normalized_id)
        if index is None:
            index = len(self._normalized_rows)
            self.normalized_index_by_id[normalized_id] = index
            self._normalized_rows.append({
                "kind": "normalized_field", "id": normalized_id,
                "normalized_index": index, "declared_shape": shape,
                "instance_count": 0, "source_indices": set(),
                "observed_value_types": set(), "value_status_counts": {},
                "structure_status_counts": {}, "observed_shapes": set(),
                "normalized_shapes": set(),
            })
        row = self._normalized_rows[index]
        if row["declared_shape"] != shape:
            raise ValueError(f"one normalized field has conflicting declared shapes: {normalized_id}")
        return row

    @staticmethod
    def _binding(source: dict[str, Any], normalized: dict[str, Any],
                 slot: str | None) -> dict[str, Any]:
        if slot is not None and (not isinstance(slot, str) or not slot):
            raise ValueError(f"invalid binding slot: {slot!r}")
        key = normalized["id"], slot
        binding = source["normalizations"].get(key)
        if binding is None:
            binding = {
                "normalized_id": normalized["id"],
                "normalized_index": normalized["normalized_index"],
                "binding_slot": slot, "selector": "source_leaf",
                "declaration_status": "observed_only", "declaration_basis": None,
                "declared_shape": normalized["declared_shape"],
                "source_value_types": None, "value_types": None,
                "value_statuses": None, "structure_statuses": None,
                "observed_source_value_types": set(),
                "observed_value_types": set(),
                "observed_value_statuses": set(),
                "observed_structure_statuses": set(),
                "observed_shapes": set(), "normalized_shapes": set(),
                "instance_count": 0,
            }
            source["normalizations"][key] = binding
        return binding

    def observe(self, projected: ProjectedField, *, source_family: str) -> CatalogIndices:
        """Return indexes in the same order as ``projected.sources``."""
        if self._finished is not None:
            raise ValueError("structure catalog is already finished")
        if not isinstance(source_family, str) or not source_family:
            raise ValueError("source family is absent")
        if not projected.sources or projected.absent_source_field_id is not None:
            raise ValueError("catalog observation requires original source leaves")
        declared_gap = is_declared_implementation_sources_gap(
            projected.normalized_id, projected.value_type,
            projected.value_status, projected.structure_status,
        )
        if declared_gap and not is_declared_implementation_sources_gap_projection(projected):
            raise ValueError("implementation_sources metadata gap has an invalid original value")
        if (projected.value_type is not None and
                projected.value_type != _raw_type(projected.value)):
            raise ValueError(f"normalized value type differs: {projected.normalized_id}")
        if (projected.value_type is None and
                (projected.value is not None or
                 projected.structure_status != "structure_conflict" and not declared_gap)):
            raise ValueError(f"untyped normalized value lacks a declared gap: {projected.normalized_id}")
        declared = _shape(projected.declared_shape)
        observed = _shape(projected.observed_shape)
        normalized_shape = _shape(projected.normalized_shape)
        if declared is not None and projected.structure_status != "structure_conflict" and not declared_gap:
            if (observed is None or normalized_shape is None or
                    len(observed) != len(declared) or len(normalized_shape) != len(declared)):
                raise ValueError(f"declared shape lacks matching observations: {projected.normalized_id}")
            for fixed, seen, result in zip(declared, observed, normalized_shape):
                if ((fixed is not None and result != fixed) or
                        (seen is not None and result != seen)):
                    raise ValueError(f"normalized shape differs from source: {projected.normalized_id}")
        source_descriptors = []
        seen_ids: set[str] = set()
        for source in projected.sources:
            leaf = source.field_value
            source_id = source_field_id(source_family, leaf.field_path,
                                        leaf.source_branch, leaf.source_subtype)
            if source_id != source.source_field_id or source_id in seen_ids:
                raise ValueError(f"source binding is duplicated or misidentified: {source.source_field_id}")
            seen_ids.add(source_id)
            source_descriptors.append((source_id, leaf, _raw_type(leaf.value)))

        source_rows = [self._source(source_id, source_family, leaf.field_path,
                                    leaf.source_branch, leaf.source_subtype)
                       for source_id, leaf, _ in source_descriptors]
        normalized = self._normalized(projected.normalized_id, projected.declared_shape)
        _increment(normalized["value_status_counts"], projected.value_status)
        _increment(normalized["structure_status_counts"], projected.structure_status)
        normalized["instance_count"] += 1
        if projected.value_type is not None:
            normalized["observed_value_types"].add(projected.value_type)
        normalized["observed_shapes"].add(tuple(observed) if observed is not None else None)
        normalized["normalized_shapes"].add(
            tuple(normalized_shape) if normalized_shape is not None else None)
        indices = []
        for row, (_, _, raw_type) in zip(source_rows, source_descriptors):
            row["instance_count"] += 1
            row["observed_source_value_types"].add(raw_type)
            normalized["source_indices"].add(row["source_index"])
            binding = self._binding(row, normalized, projected.binding_slot)
            binding["instance_count"] += 1
            binding["observed_source_value_types"].add(raw_type)
            if projected.value_type is not None:
                binding["observed_value_types"].add(projected.value_type)
            binding["observed_value_statuses"].add(projected.value_status)
            binding["observed_structure_statuses"].add(projected.structure_status)
            binding["observed_shapes"].add(tuple(observed) if observed is not None else None)
            binding["normalized_shapes"].add(
                tuple(normalized_shape) if normalized_shape is not None else None)
            indices.append(row["source_index"])
        return CatalogIndices(normalized["normalized_index"], tuple(indices))

    def export_fragment(self) -> CatalogFragment:
        """Snapshot real observations without appending zero-instance rules."""
        if self._finished is not None:
            raise ValueError("structure catalog is already finished")
        source_rows, normalized_rows = deepcopy((self._source_rows, self._normalized_rows))
        return CatalogFragment(tuple(source_rows), tuple(normalized_rows))

    def merge_fragment(self, fragment: CatalogFragment) -> CatalogIndexMaps:
        """Merge a batch in input order and return its local-to-global indexes."""
        if self._finished is not None:
            raise ValueError("structure catalog is already finished")
        _validate_fragment(fragment)
        return self._merge_validated_fragment(fragment)

    def _merge_validated_fragment(self, fragment: CatalogFragment) -> CatalogIndexMaps:
        """Merge a fragment already validated by its producing worker."""
        if self._finished is not None:
            raise ValueError("structure catalog is already finished")
        source_indices = []
        for local in fragment.source_rows:
            global_row = self._source(
                local["id"], local["source_family"], local["field_path"],
                local["source_branch"], local["source_subtype"],
            )
            global_row["instance_count"] += local["instance_count"]
            global_row["observed_source_value_types"].update(
                local["observed_source_value_types"])
            source_indices.append(global_row["source_index"])

        normalized_indices = []
        for local in fragment.normalized_rows:
            declared_shape = local["declared_shape"]
            global_row = self._normalized(
                local["id"], tuple(declared_shape) if declared_shape is not None else None,
            )
            global_row["instance_count"] += local["instance_count"]
            global_row["source_indices"].update(
                source_indices[index] for index in local["source_indices"])
            global_row["observed_value_types"].update(local["observed_value_types"])
            global_row["observed_shapes"].update(local["observed_shapes"])
            global_row["normalized_shapes"].update(local["normalized_shapes"])
            for key in ("value_status_counts", "structure_status_counts"):
                for status, count in local[key].items():
                    global_row[key][status] = global_row[key].get(status, 0) + count
            normalized_indices.append(global_row["normalized_index"])

        observed_keys = (
            "observed_source_value_types", "observed_value_types",
            "observed_value_statuses", "observed_structure_statuses",
            "observed_shapes", "normalized_shapes",
        )
        for local in fragment.source_rows:
            global_source = self._source_rows[source_indices[local["source_index"]]]
            for local_binding in local["normalizations"].values():
                global_normalized = self._normalized_rows[
                    normalized_indices[local_binding["normalized_index"]]]
                binding = self._binding(
                    global_source, global_normalized, local_binding["binding_slot"])
                binding["instance_count"] += local_binding["instance_count"]
                for key in observed_keys:
                    binding[key].update(local_binding[key])
        return CatalogIndexMaps(tuple(source_indices), tuple(normalized_indices))

    def _declare(self, source_id: str, family: str, path: str,
                 branch: str | None, subtype: str | None,
                 rule: dict[str, Any]) -> None:
        source = self._source(source_id, family, path, branch, subtype)
        shape = (tuple(rule["declared_shape"])
                 if rule["declared_shape"] is not None else None)
        normalized = self._normalized(rule["normalized_id"], shape)
        normalized["source_indices"].add(source["source_index"])
        binding = self._binding(source, normalized, rule["binding_slot"])
        if binding["declaration_status"] != "observed_only":
            if not self._resumed:
                raise ValueError(f"duplicate exact structure declaration: {source_id}:{rule['binding_slot']}")
            expected = {
                "selector": rule["selector"],
                "declaration_status": "structure_declared",
                "declaration_basis": rule["declaration_basis"],
                "declared_shape": _shape(shape),
                "source_value_types": (sorted(set(rule["source_value_types"]))
                                       if rule["source_value_types"] is not None else None),
                "value_types": (sorted(set(rule["value_types"]))
                                if rule["value_types"] is not None else None),
                "value_statuses": (sorted(set(rule["value_statuses"]))
                                   if rule["value_statuses"] is not None else None),
                "structure_statuses": (sorted(set(rule["structure_statuses"]))
                                       if rule["structure_statuses"] is not None else None),
                "record_family": rule.get("record_family"),
                "record_schema": rule.get("record_schema"),
            }
            if any(binding.get(name) != value for name, value in expected.items()):
                raise ValueError(
                    f"published structure declaration differs: {source_id}:{rule['binding_slot']}")
            return
        if binding["declared_shape"] != _shape(shape):
            raise ValueError(f"declaration shape differs from observed shape: {source_id}")
        for allowed_name, observed_name in (
            ("source_value_types", "observed_source_value_types"),
            ("value_types", "observed_value_types"),
            ("value_statuses", "observed_value_statuses"),
            ("structure_statuses", "observed_structure_statuses"),
        ):
            declared_allowed = rule[allowed_name]
            if declared_allowed is None:
                binding[allowed_name] = None
                continue
            allowed = set(declared_allowed)
            if not binding[observed_name] <= allowed:
                raise ValueError(f"structure observation exceeds declaration: {source_id}:{allowed_name}")
            binding[allowed_name] = sorted(allowed)
        binding["selector"] = rule["selector"]
        binding["declaration_status"] = "structure_declared"
        binding["declaration_basis"] = rule["declaration_basis"]
        for name in ("record_family", "record_schema"):
            if rule.get(name) is not None:
                binding[name] = rule[name]

    @staticmethod
    def _broaden_grammar_binding(source: dict[str, Any],
                                 binding: dict[str, Any],
                                 rule: dict[str, Any]) -> None:
        if (binding["selector"] != "record_grammar_source_leaf" or
                binding["declaration_status"] != "structure_declared" or
                binding["declaration_basis"] != rule["declaration_basis"] or
                binding["declared_shape"] is not None or
                binding["structure_statuses"] != ["source_only"] or
                binding.get("record_family") != source["source_family"]):
            raise ValueError(f"published grammar binding differs: {source['id']}")
        for name in ("source_value_types", "value_types", "value_statuses"):
            old = binding[name]
            new = set(rule[name])
            if old is None or not set(old) <= new:
                raise ValueError(f"published grammar declaration is not retained: {source['id']}:{name}")
            observed = binding["observed_" + name] if name != "value_statuses" else binding["observed_value_statuses"]
            if not observed <= new:
                raise ValueError(f"published grammar observation exceeds declaration: {source['id']}:{name}")
            binding[name] = sorted(new)
        schema_ref = rule.get("record_schema")
        if schema_ref is None:
            binding.pop("record_schema", None)
        else:
            binding["record_schema"] = schema_ref

    def _bind_historical_dynamic_keys(
            self, grammar_plan: GrammarCatalogPlan,
            grammar_ids: set[str]) -> frozenset[str]:
        """Bind each published literal key back to the template that declares it.

        A runtime stores a dynamic map under its concrete keys, so a published
        source row can name ``window_status.event_family_counts.road_closure``
        where the record grammar only declares
        ``window_status.event_family_counts.{event_family}``.  Such a row is not
        a new data field: it is the historical spelling of one already declared
        template, and it is declared as that template's instance here.  Nothing
        is rewritten -- the row keeps its source index, its normalized index and
        every stored observation -- and the declaration is the ordinary
        observed-only promotion ``_declare`` already performs, so the admitted
        types and statuses stay the record grammar's.

        Five conditions make the binding exact instead of a wildcard.  The ID
        must already be published with a count that has not grown, so a resumed
        delivery can never admit a path that was never observed.  The key must
        be admitted by the dynamic map's own ``DynamicMapKey`` rule, the same
        pattern the grammar walk used to template the path.  Exactly one grammar
        rule must match the literal path under one key assignment, so the
        mapping cannot be ambiguous.  The template must share the row's family,
        branch and subtype.  And the row's observed types and statuses must fall
        inside the template's declaration, which ``_declare`` checks while it
        promotes the binding.
        """
        templates: dict[tuple[str, str | None, str | None], list[GrammarSourceRule]] = {}
        for rule in grammar_plan.rules:
            if not rule.dynamic_keys:
                continue
            templates.setdefault(
                (rule.source_family, rule.source_branch, rule.source_subtype),
                []).append(rule)
        bound: set[str] = set()
        for source in self._source_rows:
            if (not source["instance_count"] or source["id"] in grammar_ids or
                    source["id"] in bound or
                    source["source_family"] not in grammar_plan.wholly_closed_families):
                continue
            published = self._published_instance_counts.get(source["id"])
            if published is None or source["instance_count"] > published:
                continue
            bindings = source["normalizations"]
            if len(bindings) != 1:
                continue
            binding = next(iter(bindings.values()))
            if (binding["normalized_id"] != "structural:source:" + source["id"] or
                    binding["binding_slot"] is not None or
                    binding["declaration_status"] != "observed_only"):
                continue
            matches = [rule for rule in templates.get(
                (source["source_family"], source["source_branch"],
                 source["source_subtype"]), ())
                if rule.bind_dynamic_key(source["field_path"]) is not None]
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError(
                    f"historical dynamic key has ambiguous grammar bindings: "
                    f"{source['id']}")
            self._declare(source["id"], source["source_family"],
                source["field_path"], source["source_branch"], source["source_subtype"],
                matches[0].bound_key_rule(source["id"]))
            bound.add(source["id"])
        return frozenset(bound)

    def _require_historical_integrity_observation(self, source: dict[str, Any]) -> None:
        """Accept one observed path the record grammar deliberately never declares.

        The publication projection drops integrity digest subtrees from the data
        view, so the grammar catalog declares no field for them and a resumed
        delivery still holds their historical observations.  Such a row may
        stand only as it was published: the same exclusion matcher must accept
        its path, the row must already have been published, and its instance
        count must not have moved.  A new record can therefore never add an
        observation to these IDs, and every other undeclared path still fails
        the record-grammar gate with the same error as before.
        """
        published = self._published_instance_counts.get(source["id"])
        if published is None:
            raise ValueError(f"observed source path is absent from record grammar: {source['id']}")
        if source["instance_count"] != published:
            raise ValueError(
                f"published integrity observation count changed: "
                f"{source['id']}:{published}->{source['instance_count']}")
        tokens = _digest_map_tokens(source["field_path"])
        if tokens is None or not is_publication_excluded_path(tokens):
            raise ValueError(f"observed source path is absent from record grammar: {source['id']}")

    def _append_exact_declarations(self, grammar_plan: GrammarCatalogPlan) -> None:
        for item in iter_structure_bindings():
            self._declare(item.source_field_id, item.source_family,
                item.source_field_path, item.source_branch, item.source_subtype,
                {"normalized_id": item.normalized_id,
                 "binding_slot": item.binding_slot, "selector": item.selector,
                 "source_value_types": item.source_value_types,
                 "value_types": item.value_types,
                 "value_statuses": item.value_statuses,
                 "structure_statuses": item.structure_statuses,
                 "declared_shape": item.declared_shape,
                 "declaration_basis": item.declaration_basis,
                 "record_family": item.record_family,
                 "record_schema": item.record_schema})
        for rule in iter_reuse_rules():
            if rule.structure_mode != "source_only":
                continue
            self._declare(rule.source_id, rule.source_family, rule.source_path,
                rule.source_branch, rule.source_subtype, {
                    "normalized_id": rule.normalized_id,
                    "binding_slot": rule.target_binding_slot,
                    "selector": "reuse:" + rule.condition,
                    "source_value_types": rule.source_value_types,
                    "value_types": rule.value_types,
                    "value_statuses": rule.value_statuses,
                    "structure_statuses": ("source_only",),
                    "declared_shape": None,
                    "declaration_basis": rule.producer_basis,
                })
        for path, source_id in zip(_QUEUE_PATHS, COMPUTE_QUEUE_SOURCE_IDS):
            if source_field_id("compute_state", path) != source_id:
                raise ValueError(f"compute queue source declaration differs: {source_id}")
            for rule in extra_structure_bindings(source_id):
                self._declare(source_id, "compute_state", path, None, None, rule)
        for path in _POSE_PATHS:
            source_id = source_field_id("arm_states", path)
            normalized_id = _POSE_IDS[source_id]
            self._declare(source_id, "arm_states", path, None, None, {
                "normalized_id": normalized_id, "binding_slot": "observed_pose",
                "selector": "record_observed=true;finite_three_component_vector",
                "source_value_types": ("int", "float"),
                "value_types": ("int", "float"),
                "value_statuses": ("present",),
                "structure_statuses": ("resolved",), "declared_shape": (3,),
                "declaration_basis": "Dataset/world_model/graph/source_projection.py:_arm_state_observation",
            })
        for source in self._source_rows:
            for binding in source["normalizations"].values():
                if binding["binding_slot"] != "sample_identity":
                    continue
                if source["field_path"] not in {
                    "seed", "episode_id", "parent_episode_id", "arm_key",
                    "profile", "group_key", "arm_id",
                }:
                    raise ValueError(f"sample identity has an undeclared field: {source['id']}")
                self._declare(source["id"], source["source_family"],
                    source["field_path"], source["source_branch"],
                    source["source_subtype"], {
                        "normalized_id": binding["normalized_id"],
                        "binding_slot": "sample_identity",
                        "selector": "sample_identity",
                        "source_value_types": ("int", "string") if source["field_path"] == "seed" else ("string",),
                        "value_types": ("string",), "value_statuses": ("present",),
                        "structure_statuses": ("resolved",), "declared_shape": (),
                        "declaration_basis": "Dataset/world_model/graph/sample_identity.py:normalize_identity_field",
                    })
        grammar_ids = set()
        grammar_rules = {}
        for item in grammar_plan.rules:
            grammar_ids.add(item.source_field_id)
            grammar_rules[item.source_field_id] = item
            source_index = self.source_index_by_id.get(item.source_field_id)
            if source_index is not None:
                source = self._source_rows[source_index]
                default_key = ("structural:source:" + item.source_field_id, None)
                existing = source["normalizations"].get(default_key)
                if (existing is not None and
                        existing["declaration_status"] == "structure_declared"):
                    self._broaden_grammar_binding(source, existing, item.catalog_rule)
                    continue
                if (default_key not in source["normalizations"] and
                        any(binding["declaration_status"] == "structure_declared"
                            for binding in source["normalizations"].values())):
                    # A more specific producer rule owns this source path.
                    # Its branch/value selector remains the catalog authority.
                    continue
            self._declare(item.source_field_id, item.source_family,
                item.field_path, item.source_branch, item.source_subtype,
                item.catalog_rule)
        historical_keys = self._bind_historical_dynamic_keys(grammar_plan, grammar_ids)
        for source in self._source_rows:
            if (source["source_family"] in grammar_plan.wholly_closed_families and
                    source["instance_count"] and source["id"] not in grammar_ids
                    and source["id"] not in historical_keys):
                self._require_historical_integrity_observation(source)
            if (source["source_family"] in grammar_plan.wholly_closed_families and
                    source["id"] in grammar_rules and
                    not source["observed_source_value_types"] <=
                    set(grammar_rules[source["id"]].source_value_types)):
                raise ValueError(f"observed source type is absent from record grammar: {source['id']}")
        for source in self._source_rows:
            for binding in source["normalizations"].values():
                if binding["declaration_status"] == "observed_only" and binding["binding_slot"] is not None:
                    raise ValueError(f"structural slot lacks an exact declaration: {source['id']}:{binding['binding_slot']}")

    def finish(self, *, declarations: Iterable[SourceDeclaration] | None = None) -> CatalogRows:
        """Validate and serialize using every declared source role.

        A resumed catalog retains already published rules. A new catalog needs
        its declaration inventory so zero-instance grammar fields are not lost.
        """
        if self._finished is not None:
            return self._finished
        if declarations is None:
            if not self._resumed:
                raise ValueError("new structure catalog requires source declarations")
        else:
            grammar_plan = compile_catalog_grammars(declarations)
            self._append_exact_declarations(grammar_plan)
            self.grammar_declaration_statuses = grammar_plan.declaration_statuses
        result = _catalog_rows(self._source_rows, self._normalized_rows)
        validate_catalog_rows(result)
        self._finished = result
        return result


def _catalog_rows(source_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
                  normalized_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
                  ) -> CatalogRows:
    source_fields = []
    for source in source_rows:
        row = {key: value for key, value in source.items() if key != "normalizations"}
        row["observed_source_value_types"] = sorted(row["observed_source_value_types"])
        bindings = []
        for binding in source["normalizations"].values():
            item = dict(binding)
            for key in ("observed_source_value_types", "observed_value_types",
                        "observed_value_statuses", "observed_structure_statuses"):
                item[key] = sorted(item[key])
            for key in ("observed_shapes", "normalized_shapes"):
                item[key] = [list(shape) if shape is not None else None
                             for shape in _ordered(item[key])]
            bindings.append(item)
        row["normalizations"] = bindings
        source_fields.append(row)
    normalized_fields = []
    for normalized in normalized_rows:
        row = {key: value for key, value in normalized.items()
               if key != "source_indices"}
        row["source_members"] = [
            {"source_index": index, "source_field_id": source_rows[index]["id"]}
            for index in sorted(normalized["source_indices"])]
        row["observed_value_types"] = sorted(row["observed_value_types"])
        row["value_status_counts"] = dict(sorted(row["value_status_counts"].items()))
        row["structure_status_counts"] = dict(sorted(row["structure_status_counts"].items()))
        for key in ("observed_shapes", "normalized_shapes"):
            row[key] = [list(shape) if shape is not None else None
                        for shape in _ordered(row[key])]
        row["structure_status"] = _summary_structure_status(
            row["instance_count"], row["structure_status_counts"])
        normalized_fields.append(row)
    return CatalogRows(tuple(source_fields), tuple(normalized_fields))


def _validate_fragment(fragment: CatalogFragment) -> None:
    if (not isinstance(fragment, CatalogFragment) or
            not isinstance(fragment.source_rows, tuple) or
            not isinstance(fragment.normalized_rows, tuple)):
        raise TypeError("structure fragment has invalid rows")
    source_count = len(fragment.source_rows)
    for source in fragment.source_rows:
        if (source["kind"] != "source_field" or
                type(source["instance_count"]) is not int or
                source["instance_count"] <= 0 or
                not isinstance(source["normalizations"], dict)):
            raise ValueError(f"fragment has a non-observed source: {source['id']}")
        for key, binding in source["normalizations"].items():
            if (key != (binding["normalized_id"], binding["binding_slot"]) or
                    type(binding["instance_count"]) is not int or
                    binding["instance_count"] <= 0 or
                    binding["selector"] != "source_leaf" or
                    binding["declaration_status"] != "observed_only" or
                    binding["declaration_basis"] is not None or
                    any(binding[name] is not None for name in (
                        "source_value_types", "value_types", "value_statuses",
                        "structure_statuses")) or
                    "record_family" in binding or "record_schema" in binding):
                raise ValueError(f"fragment has a declared or invalid binding: {source['id']}")
    for normalized in fragment.normalized_rows:
        shape = normalized["declared_shape"]
        if (normalized["kind"] != "normalized_field" or
                not isinstance(normalized["id"], str) or
                not normalized["id"].startswith("structural:") or
                type(normalized["instance_count"]) is not int or
                normalized["instance_count"] <= 0 or
                (shape is not None and not isinstance(shape, list)) or
                not isinstance(normalized["source_indices"], set) or
                not normalized["source_indices"] or
                any(type(index) is not int or index < 0 or index >= source_count
                    for index in normalized["source_indices"])):
            raise ValueError(f"fragment has an invalid normalized field: {normalized['id']}")
        if shape is not None:
            _shape(tuple(shape))
    validate_catalog_rows(_catalog_rows(fragment.source_rows, fragment.normalized_rows))


def validate_catalog_rows(rows: CatalogRows) -> None:
    """Check indexes, exact bindings, source coverage, and shared membership."""
    source_ids = {row["id"]: index for index, row in enumerate(rows.source_fields)}
    normalized_ids = {row["id"]: index for index, row in enumerate(rows.normalized_fields)}
    if len(source_ids) != len(rows.source_fields) or len(normalized_ids) != len(rows.normalized_fields):
        raise ValueError("catalog has duplicate field IDs")
    expected_members: list[set[int]] = [set() for _ in rows.normalized_fields]
    expected_types: list[set[str]] = [set() for _ in rows.normalized_fields]
    expected_statuses: list[set[str]] = [set() for _ in rows.normalized_fields]
    expected_structures: list[set[str]] = [set() for _ in rows.normalized_fields]
    expected_observed_shapes: list[set[str]] = [set() for _ in rows.normalized_fields]
    expected_normalized_shapes: list[set[str]] = [set() for _ in rows.normalized_fields]
    binding_counts: list[list[int]] = [[] for _ in rows.normalized_fields]
    for index, source in enumerate(rows.source_fields):
        if (type(source["source_index"]) is not int or source["source_index"] != index or
                type(source["instance_count"]) is not int or source["instance_count"] < 0 or
                source_field_id(source["source_family"], source["field_path"],
                                source["source_branch"], source["source_subtype"]) != source["id"]):
            raise ValueError(f"source index or path differs: {source['id']}")
        bindings = source["normalizations"]
        if not bindings or len({(item["normalized_id"], item["binding_slot"])
                                for item in bindings}) != len(bindings):
            raise ValueError(f"source bindings are absent or duplicate: {source['id']}")
        if sum(item["instance_count"] for item in bindings) != source["instance_count"]:
            raise ValueError(f"source observation count differs: {source['id']}")
        raw_types = set()
        for binding in bindings:
            normalized_id = binding["normalized_id"]
            normalized_index = normalized_ids.get(normalized_id)
            if (normalized_index is None or type(binding["normalized_index"]) is not int or
                    binding["normalized_index"] != normalized_index or
                    type(binding["instance_count"]) is not int or binding["instance_count"] < 0):
                raise ValueError(f"binding index differs: {source['id']}:{normalized_id}")
            normalized = rows.normalized_fields[normalized_index]
            if binding["declared_shape"] != normalized["declared_shape"]:
                raise ValueError(f"binding declared shape differs: {source['id']}:{normalized_id}")
            for allowed, observed in (
                ("source_value_types", "observed_source_value_types"),
                ("value_types", "observed_value_types"),
                ("value_statuses", "observed_value_statuses"),
                ("structure_statuses", "observed_structure_statuses"),
            ):
                if (binding[allowed] is not None and
                        not set(binding[observed]) <= set(binding[allowed])):
                    raise ValueError(f"binding exceeds declaration: {source['id']}:{allowed}")
            raw_types.update(binding["observed_source_value_types"])
            expected_members[normalized_index].add(index)
            binding_counts[normalized_index].append(binding["instance_count"])
            expected_types[normalized_index].update(binding["observed_value_types"])
            expected_statuses[normalized_index].update(binding["observed_value_statuses"])
            expected_structures[normalized_index].update(binding["observed_structure_statuses"])
            expected_observed_shapes[normalized_index].update(
                json.dumps(shape) for shape in binding["observed_shapes"])
            expected_normalized_shapes[normalized_index].update(
                json.dumps(shape) for shape in binding["normalized_shapes"])
        if raw_types != set(source["observed_source_value_types"]):
            raise ValueError(f"source raw types differ from bindings: {source['id']}")
    for index, normalized in enumerate(rows.normalized_fields):
        if (type(normalized["normalized_index"]) is not int or
                normalized["normalized_index"] != index or
                type(normalized["instance_count"]) is not int or
                normalized["instance_count"] < 0):
            raise ValueError(f"normalized index differs: {normalized['id']}")
        members = [{"source_index": source_index,
                    "source_field_id": rows.source_fields[source_index]["id"]}
                   for source_index in sorted(expected_members[index])]
        if normalized["source_members"] != members:
            raise ValueError(f"normalized source members differ: {normalized['id']}")
        if (set(normalized["observed_value_types"]) != expected_types[index] or
                set(normalized["value_status_counts"]) != expected_statuses[index] or
                set(normalized["structure_status_counts"]) != expected_structures[index] or
                {json.dumps(shape) for shape in normalized["observed_shapes"]} !=
                expected_observed_shapes[index] or
                {json.dumps(shape) for shape in normalized["normalized_shapes"]} !=
                expected_normalized_shapes[index] or
                sum(normalized["value_status_counts"].values()) != normalized["instance_count"] or
                sum(normalized["structure_status_counts"].values()) != normalized["instance_count"]):
            raise ValueError(f"normalized observations differ from bindings: {normalized['id']}")
        if (normalized["structure_status"] != _summary_structure_status(
                normalized["instance_count"], normalized["structure_status_counts"]) or
                normalized["instance_count"] < max(binding_counts[index]) or
                normalized["instance_count"] > sum(binding_counts[index])):
            raise ValueError(f"normalized structure status or count differs: {normalized['id']}")
