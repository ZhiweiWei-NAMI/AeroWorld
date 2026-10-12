"""Project source leaves once for the catalog and graph value readers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
import math
import re
from typing import Any, Mapping

from Dataset.world_model.graph.arm_observation import record_observed, state_profile
from Dataset.world_model.graph.controller_scope import controller_namespaces_for_record
from Dataset.world_model.graph.field_contract import (
    EXACT_MISSING_RULES, coordinate_context_for_value, coordinate_policy, unit_policy,
)
from Dataset.world_model.graph.field_paths import (
    DYNAMIC_KEYED_OBJECTS, MIXED_KEYED_OBJECTS, FieldValue,
    catalog_source_branch, catalog_source_subtype, iter_field_values,
    source_branch_for_record, source_field_id, source_subtype_for_record,
    source_value_at_tokens, WORLD_OBSERVATION_VALUE_PATH,
    world_observation_variant_field_path,
)
from Dataset.world_model.graph.field_structure import (
    is_declared_implementation_sources_gap,
    is_declared_implementation_sources_gap_projection,
    project_structural_leaf,
)
from Dataset.world_model.graph.sample_identity import (
    arm_chain_row_family, node_source_scope, normalize_identity_field, validate_record_identity,
)
from Dataset.world_model.graph.structure_reuse import resolve_reuse, rules_for_source


_TRAILING_ARRAYS = re.compile(r"(?:\[\])+$")
_SEED_PROFILE_TEMPLATE = (
    "parameters.deterministic_sumo_traffic_template.seed_profiles.{seed_profile}"
)
_SEED_PROFILE_PARENT_TOKENS = (
    "parameters", "deterministic_sumo_traffic_template", "seed_profiles",
)
_COMPUTE_QUEUE_ID = "structural:compute_state:queue_depth"
_COMPUTE_QUEUE_SOURCES = (
    "compute_state:queue_depth",
    "compute_state:resource_balance.queue_depth",
)
_ARM_OBSERVED_POSE_IDS = {
    "arm_states:pos_enu[]": "structural:arm_states:observed_pose.pos_enu[]",
    "arm_states:values.pos_enu[]": "structural:arm_states:observed_pose.pos_enu[]",
    "arm_states:vel_mps[]": "structural:arm_states:observed_pose.vel_mps[]",
    "arm_states:values.vel_mps[]": "structural:arm_states:observed_pose.vel_mps[]",
}
_ABSENT_LOCATIONS = {
    ("formal_truth_frame", "entities[].truth_pose.position_enu_m[]"):
        ("EntityState", "entities[]."),
    ("formal_truth_frame", "entities[].truth_pose.velocity_enu_mps[]"):
        ("EntityState", "entities[]."),
}


def source_value_type(value: Any) -> str:
    """Name a source leaf type without coercing its original value."""
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
    raise TypeError(type(value))


@dataclass(frozen=True)
class ProjectedSource:
    source_field_id: str
    field_value: FieldValue


@dataclass(frozen=True)
class ProjectedField:
    normalized_id: str
    value: Any
    value_type: str | None
    envelope_tokens: tuple[str | int, ...]
    payload_tokens: tuple[str | int, ...]
    payload_positions: tuple[int, ...]
    payload_shape: tuple[int | None, ...]
    value_status: str
    sources: tuple[ProjectedSource, ...]
    record_observed: bool | None = None
    absent_source_field_id: str | None = None
    binding_slot: str | None = None
    declared_shape: tuple[int | None, ...] | None = None
    observed_shape: tuple[int | None, ...] | None = None
    normalized_shape: tuple[int | None, ...] | None = None
    structure_status: str = "source_only"


def projected_source_rows(projected: ProjectedField, *,
                          catalog: Mapping[str, Mapping[str, Any]] | None = None
                          ) -> list[dict[str, Any]]:
    """Serialize every original leaf without changing its value or location."""
    rows = []
    for source in projected.sources:
        row = {field.name: getattr(source.field_value, field.name)
               for field in fields(source.field_value)}
        # Scalar metadata is immutable; only a source container needs its own
        # value copy. Recursively copying every scalar costs a walk per leaf.
        if isinstance(row["value"], (list, dict)):
            row["value"] = deepcopy(row["value"])
        for key in ("element_positions", "source_tokens", "input_tokens"):
            row[key] = list(row[key])
        item = {"source_field_id": source.source_field_id,
                "value_type": source_value_type(source.field_value.value), **row}
        if catalog is not None:
            declared = catalog.get(source.source_field_id)
            if declared is None:
                raise ValueError(f"source field is absent from catalog: {source.source_field_id}")
            source_index = declared.get("index")
            if type(source_index) is not int or source_index < 0:
                raise ValueError(f"source field lacks a valid index: {source.source_field_id}")
            item["source_index"] = source_index
            item["catalog"] = {"id": declared["id"], "semantic_id": declared["semantic_id"],
                "meaning_status": declared["meaning_status"],
                "unit": declared.get("unit"), "unit_status": declared["unit_status"],
                "unit_basis": declared.get("unit_basis"),
                "source_unit": declared["source_unit"],
                "canonical_unit": declared["canonical_unit"],
                "coordinate_frame": declared.get("coordinate_frame"),
                "coordinate_status": declared.get("coordinate_status")}
        rows.append(item)
    return rows


def projected_catalog_binding(projected: ProjectedField,
                              catalog: Mapping[str, Mapping[str, Any]]) -> int:
    """Resolve one structural index against every original source binding."""
    declared_gap = is_declared_implementation_sources_gap(
        projected.normalized_id, projected.value_type,
        projected.value_status, projected.structure_status,
    )
    if declared_gap and not is_declared_implementation_sources_gap_projection(projected):
        raise ValueError("implementation_sources metadata gap has an invalid original value")
    sources = ([(source.source_field_id, source_value_type(source.field_value.value))
                for source in projected.sources]
                  if projected.sources else
                  [(projected.absent_source_field_id, None)] if projected.absent_source_field_id else [])
    if not sources:
        raise ValueError(f"projected field has no source binding: {projected.normalized_id}")
    normalized_index: int | None = None
    for source_id, raw_type in sources:
        declared = catalog.get(source_id)
        if declared is None:
            raise ValueError(f"source field is absent from catalog: {source_id}")
        bindings = declared.get("normalizations")
        if not isinstance(bindings, list):
            raise ValueError(f"source field lacks normalization declarations: {source_id}")
        matches = [binding for binding in bindings
                   if binding.get("normalized_id") == projected.normalized_id
                   and binding.get("binding_slot") == projected.binding_slot]
        if len(matches) != 1:
            raise ValueError(f"source field lacks one exact structural binding: "
                             f"{source_id}:{projected.normalized_id}:{projected.binding_slot}")
        binding = matches[0]
        if "source_value_types" not in binding:
            raise ValueError(f"structural binding lacks source value types: {source_id}")
        source_types = binding["source_value_types"]
        if source_types is not None and (not isinstance(source_types, list) or
                                         raw_type is not None and raw_type not in source_types):
            raise ValueError(f"original source type differs from structural binding: {source_id}:{raw_type}")
        index = binding.get("normalized_index")
        if type(index) is not int or index < 0:
            raise ValueError(f"structural binding has no index: {source_id}")
        if normalized_index is not None and index != normalized_index:
            raise ValueError(f"projected sources disagree on structural index: {projected.normalized_id}")
        normalized_index = index
        for name, value in (("value_types", projected.value_type),
                            ("value_statuses", projected.value_status),
                            ("structure_statuses", projected.structure_status)):
            if name not in binding:
                raise ValueError(f"structural binding lacks {name}: {source_id}")
            allowed = binding[name]
            if allowed is None:
                continue
            if not isinstance(allowed, list) or (value is not None and value not in allowed):
                raise ValueError(f"projected {name} differs from binding: {source_id}:{value}")
            if (name == "value_types" and value is None and allowed and
                    projected.structure_status != "structure_conflict" and
                    projected.value_status != "absent" and not declared_gap):
                raise ValueError(f"projected null structural value differs from binding: {source_id}")
        binding_shape = binding.get("declared_shape")
        if binding_shape is not None:
            if (not isinstance(binding_shape, list) or
                    tuple(binding_shape) != projected.declared_shape):
                raise ValueError(f"projected declared shape differs from binding: {source_id}")
    if (projected.declared_shape is not None and
            projected.structure_status != "structure_conflict" and not declared_gap):
        if (projected.observed_shape is None or projected.normalized_shape is None or
                len(projected.observed_shape) != len(projected.declared_shape) or
                len(projected.normalized_shape) != len(projected.declared_shape)):
            raise ValueError(f"projected structural shape is incomplete: {projected.normalized_id}")
        for declared_axis, observed_axis, normalized_axis in zip(
                projected.declared_shape, projected.observed_shape, projected.normalized_shape):
            if ((declared_axis is not None and normalized_axis != declared_axis) or
                    (observed_axis is not None and normalized_axis != observed_axis)):
                raise ValueError(f"projected structural shape differs from observations: "
                                 f"{projected.normalized_id}")
    assert normalized_index is not None
    return normalized_index


def projected_numeric_authority(
        projected: ProjectedField, catalog: Mapping[str, Mapping[str, Any]],
        source_file: str, raw_record: Mapping[str, Any],
        ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, str | None, str]:
    """Check every original source before selecting a numeric unit or meaning."""
    if not projected.sources:
        raise ValueError(f"numeric field has no original sources: {projected.normalized_id}")
    signatures: set[tuple[Any, ...]] = set()
    semantic_bindings: set[tuple[str, str]] = set()
    contexts: list[Mapping[str, Any] | None] = []
    mappings: list[Mapping[str, Any]] = []
    for source in projected.sources:
        row = catalog.get(source.source_field_id)
        if row is None:
            raise ValueError(f"numeric source is absent from catalog: {source.source_field_id}")
        expected_unit = unit_policy(row["source_family"], source.field_value.field_path,
                                    row["value_types"])
        if any(row.get(key) != value for key, value in expected_unit.items()):
            raise ValueError(f"numeric source unit declaration differs: {source.source_field_id}")
        expected_coordinate = coordinate_policy(row["source_family"], source.field_value.field_path)
        if any(row.get(key) != value for key, value in expected_coordinate.items()):
            raise ValueError(f"numeric source coordinate declaration differs: {source.source_field_id}")
        signatures.add((row["unit_status"], row["source_unit"], row["canonical_unit"],
                        row.get("quantity_role"), row.get("time_role"),
                        row.get("coordinate_status"), row.get("coordinate_frame"),
                        row.get("coordinate_contract_id"),
                        tuple(row.get("coordinate_axes") or ()), row.get("coordinate_basis")))
        semantic_bindings.add((row["semantic_id"], row["meaning_status"]))
        context = coordinate_context_for_value(
            row["source_family"], source.field_value.field_path,
            source_file, raw_record)
        if row["coordinate_status"] == "declared":
            if (context is None or
                    context.get("coordinate_frame") != row["coordinate_frame"] or
                    context.get("coordinate_contract_id") != row["coordinate_contract_id"] or
                    context.get("axis_order") != row["coordinate_axes"]):
                raise ValueError(f"numeric source coordinate context differs: {source.source_field_id}")
        elif context is not None:
            raise ValueError(f"numeric source has undeclared coordinate context: {source.source_field_id}")
        contexts.append(context)
        mappings.append(row)
    if len(signatures) != 1 or any(context != contexts[0] for context in contexts[1:]):
        raise ValueError(f"normalized numeric source mappings conflict: {projected.normalized_id}")
    if len(semantic_bindings) == 1:
        semantic_id, meaning_status = next(iter(semantic_bindings))
        if meaning_status == "declared":
            return mappings[0], contexts[0], semantic_id, "declared"
    return mappings[0], contexts[0], None, "unbound"


def _normalized_id(source_id: str, observed: bool | None = None) -> str:
    """Use a source-specific structure slot until an exact structural rule applies."""
    if observed is True and source_id in _ARM_OBSERVED_POSE_IDS:
        return _ARM_OBSERVED_POSE_IDS[source_id]
    return "structural:source:" + source_id


def _arm_state_observation(raw: Any, observed: bool | None,
                           scope: Mapping[str, Any] | None,
                           source_file: str | None, source_line: int | None) -> bool:
    if not isinstance(raw, Mapping):
        raise TypeError("ARM state source row is not an object")
    if source_file is None:
        if scope is None or scope.get("source_kind") != "arm":
            raise ValueError("ARM state projection lacks its producer source path")
        source_file = (scope.get("source_file") or
                       str(scope["source_root"]).rstrip("/") + "/raw/states.jsonl")
    profile = state_profile(source_file, scope)
    resolved = record_observed(profile, "state", raw, source_file, source_line)
    if observed is not None and (type(observed) is not bool or observed is not resolved):
        raise ValueError(f"ARM state observation differs from producer row: {source_file}:{source_line}")
    for prefix, values in (("", raw), ("values.", raw.get("values"))):
        if not isinstance(values, Mapping):
            continue
        for name in ("pos_enu", "vel_mps"):
            if name not in values:
                continue
            field = prefix + name
            if not resolved:
                raise ValueError(f"unobserved ARM state carries {field}: {source_file}:{source_line}")
            vector = values[name]
            if (type(vector) is not list or len(vector) != 3 or
                    any(type(component) not in (int, float) or not math.isfinite(component)
                        for component in vector)):
                raise ValueError(f"ARM state {field} is not a finite three-component vector: "
                                 f"{source_file}:{source_line}")
    return resolved


def _compute_queue_value(raw: Any) -> int | None:
    """Return the shared queue value only when the producer marked the row complete."""
    if not isinstance(raw, Mapping) or raw["schema_name"] != "compute_state":
        raise ValueError("compute queue structure requires a complete compute_state row")
    entity_id = raw["entity_id"]
    if (not isinstance(entity_id, str) or not entity_id or
            type(raw["tick"]) is not int or
            raw["node_id"] != f"compute_node:{entity_id}" or
            raw["queue_id"] != f"compute_queue:{entity_id}"):
        raise ValueError("compute queue row identity differs from producer")
    failure = raw["failure"]
    if not isinstance(failure, Mapping):
        raise TypeError("compute queue failure marker is not an object")
    requirement = failure["requirement_status"]
    missing = failure["missing_parameters"]
    if (type(missing) is not list or
            any(not isinstance(item, str) or not item for item in missing)):
        raise ValueError("compute queue missing parameters are malformed")
    top = raw["queue_depth"]
    balance = raw["resource_balance"]
    if not isinstance(balance, Mapping):
        raise TypeError("compute resource balance is not an object")
    nested = balance["queue_depth"]
    if (type(nested) is not float or not math.isfinite(nested) or
            nested < 0 or not nested.is_integer()):
        raise ValueError("compute resource balance queue depth is not a finite nonnegative integer value")
    if requirement == "missing_source_record":
        if not missing or top != "unknown":
            raise ValueError("compute queue missing marker differs from its raw values")
        return None
    if requirement != "not_required" or missing:
        raise ValueError("compute queue requirement marker differs from producer")
    if type(top) is not int or top < 0 or top != int(nested):
        raise ValueError("compute queue source values are not equal nonnegative integers")
    return top


def _value_status(value: Any, source_family: str, field_path: str,
                  declared: Mapping[str, Any] | None) -> str:
    if value is None:
        return "null"
    if value == [] and type(value) is list:
        return "empty_list"
    if value == {} and type(value) is dict:
        return "empty_object"
    missing_rule = EXACT_MISSING_RULES.get((source_family, field_path))
    status_literals = {"unknown", "missing_source_record", "inactive", "not_applicable"}
    if (type(value) is str and value in status_literals
            and isinstance(missing_rule, Mapping) and value in missing_rule
            and (declared is None or (declared.get("missing_rule") or {}).get(value) == missing_rule[value])):
        return value
    return "present"


def _payload_location(raw: Any, leaf: FieldValue,
                      source_prefix_tokens: tuple[str | int, ...]
                      ) -> tuple[tuple[str | int, ...], tuple[str | int, ...],
                                 tuple[int, ...], tuple[int, ...]]:
    """Keep record arrays in the envelope and measure payload axes from data."""
    trailing = _TRAILING_ARRAYS.search(leaf.field_path)
    payload_rank = len(trailing.group()) // 2 if trailing else 0
    template_shell_rank = leaf.field_path.count("[]") - payload_rank
    prefix_rank = sum(type(token) is int for token in source_prefix_tokens)
    local_shell_rank = max(0, template_shell_rank - prefix_rank)
    local_indices = [index for index, token in enumerate(leaf.input_tokens) if type(token) is int]
    if local_shell_rank > len(local_indices):
        raise ValueError(f"source field array location differs: {leaf.field_path}:{leaf.source_tokens}")
    split = local_indices[local_shell_rank - 1] + 1 if local_shell_rank else 0
    envelope = source_prefix_tokens + leaf.input_tokens[:split]
    payload = leaf.input_tokens[split:]
    if len([token for token in payload if type(token) is int]) != payload_rank:
        raise ValueError(f"payload array rank differs: {leaf.field_path}:{leaf.source_tokens}")
    positions = tuple(token for token in payload if type(token) is int)
    current = source_value_at_tokens(raw, leaf.input_tokens[:split])
    shape: list[int] = []
    for token in payload:
        if type(token) is int:
            if not isinstance(current, list):
                raise TypeError(f"payload axis is not an array: {leaf.raw_path}")
            shape.append(len(current))
        current = current[token]
    if type(current) is list and not current:
        shape.append(0)
    return envelope, payload, positions, tuple(shape)


def _seed_profile_instance(source_family: str, leaf: FieldValue) -> str | None:
    """Separate instances of the one declared seed-profile template."""
    if (source_family != "arm_script_plan" or
            not (leaf.field_path == _SEED_PROFILE_TEMPLATE or
                 leaf.field_path.startswith(_SEED_PROFILE_TEMPLATE + "."))):
        return None
    tokens = leaf.source_tokens
    if (len(tokens) < 4 or tokens[:3] != _SEED_PROFILE_PARENT_TOKENS or
            type(tokens[3]) is not str):
        raise ValueError(f"seed profile source location differs: {tokens}")
    return tokens[3]


_TEMPLATE_SEGMENT = re.compile(r"\{([^{}]*)\}")
# Templates that already occupy their own slot in the grouping key below.
_RESERVED_TEMPLATES = frozenset({"tick", "entity"})


def _declared_key_dimensions() -> frozenset[str]:
    """Dimension names bound by a declared keyed-object template."""
    names = {dimension for table in DYNAMIC_KEYED_OBJECTS.values()
             for dimension in table.values()}
    names.update(dimension for table in MIXED_KEYED_OBJECTS.values()
                 for dimension, _ in table.values())
    return frozenset(names)


_DECLARED_KEY_DIMENSIONS = _declared_key_dimensions()


def _dynamic_instance(leaf: FieldValue,
                      field_prefix: str = "") -> tuple[tuple[str, str | int], ...]:
    """Return ``(dimension, key)`` for every declared dynamic key on this leaf.

    A declared keyed object templates its open keys into one field path, so every
    entry of the map shares one source field id while carrying its own key.  The
    key is therefore the only thing that separates two entries, and the envelope
    and payload positions stay empty for a map of scalars.  Recording the bound
    key per template position gives each entry its own instance identity.

    Leaves that meet again through structural reuse or the compute-queue binding
    do so under literal paths with no template, so they all record ``()`` and
    still group together; two entries of one map can never group together,
    because the catalog rejects a single projection holding two source leaves
    with the same source field id.

    Alignment runs over the walked value exactly as ``_payload_location`` does:
    ``field_path`` carries ``field_prefix`` while ``input_tokens`` are relative to
    the walked node, so the prefix is dropped from the path and the prefix's own
    keys stay in ``source_tokens``.  A source key may itself contain a dot
    (``communication.backup_link_active`` is one key), so one name spans
    ``key.count(".") + 1`` path parts.  Splitting on ``.`` leaves no dot inside a
    part, which makes that span exact.  A template occupies a single part
    whatever its key carries, so keys stay whole.  Keys are read from the tokens
    and never from the path.
    """
    if "{" not in leaf.field_path:
        return ()
    if field_prefix:
        if not leaf.field_path.startswith(field_prefix):
            raise ValueError(
                f"source field path does not start at its prefix: "
                f"{field_prefix}:{leaf.field_path}")
        path = leaf.field_path[len(field_prefix):]
    else:
        path = leaf.field_path
    parts: list[tuple[str, int]] = []
    for segment in path.split("."):
        arrays = 0
        while segment.endswith("[]"):
            segment, arrays = segment[:-2], arrays + 1
        parts.append((segment, arrays))

    result: list[tuple[str, str | int]] = []
    index = 0
    declared_arrays = 0
    name_arrays = 0
    seen_indices = 0
    for token in leaf.input_tokens:
        if type(token) is int:
            seen_indices += 1
            continue
        if index >= len(parts):
            raise ValueError(
                f"source field path ends before its keys: {leaf.field_path}:{leaf.source_tokens}")
        name, _ = parts[index]
        matched = _TEMPLATE_SEGMENT.fullmatch(name)
        if matched is not None:
            # One template stands for one open key, dots included.
            consumed = 1
            dimension = matched.group(1)
            if dimension in _RESERVED_TEMPLATES:
                pass
            elif dimension not in _DECLARED_KEY_DIMENSIONS:
                raise ValueError(
                    f"source field has an undeclared key dimension: {leaf.field_path}")
            else:
                result.append((dimension, token))
        else:
            consumed = token.count(".") + 1
            spelled = ".".join(part[0] for part in parts[index:index + consumed])
            if spelled != token:
                raise ValueError(
                    f"source field path does not match its keys: {leaf.field_path}:{leaf.source_tokens}")
        name_arrays = sum(part[1] for part in parts[index:index + consumed])
        declared_arrays += name_arrays
        index += consumed
    if index != len(parts):
        raise ValueError(
            f"source field path does not cover its keys: {leaf.field_path}:{leaf.source_tokens}")
    # Array indices are consumed as they are walked; only an empty container at
    # the end of the path may leave its [] unmatched.
    if not 0 <= declared_arrays - seen_indices <= name_arrays:
        raise ValueError(
            f"source field path does not match its indices: {leaf.field_path}:{leaf.source_tokens}")
    return tuple(result)


def _reuse_entity_context(raw: Any, leaf: FieldValue,
                          source_prefix_tokens: tuple[str | int, ...]
                          ) -> Mapping[str, Any]:
    """Read conditional reuse markers from this leaf's own formal entity."""
    if not leaf.field_path.startswith("entities[]."):
        raise ValueError(f"conditional structural reuse has no entity path: {leaf.field_path}")
    entity_tokens = leaf.source_tokens[:2]
    if (len(entity_tokens) != 2 or entity_tokens[0] != "entities" or
            type(entity_tokens[1]) is not int):
        raise ValueError(f"conditional structural reuse has no entity location: {leaf.source_tokens}")
    if source_prefix_tokens:
        if source_prefix_tokens != entity_tokens:
            raise ValueError(f"conditional structural reuse entity location differs: {leaf.source_tokens}")
        entity = raw
    else:
        entity = source_value_at_tokens(raw, entity_tokens)
    if not isinstance(entity, Mapping):
        raise TypeError(f"conditional structural reuse entity is not an object: {entity_tokens}")
    return entity


def project_values(raw: Any, *, source_family: str, field_prefix: str = "",
                   source_prefix_tokens: tuple[str | int, ...] = (),
                   entity_key_namespaces: Mapping[str, str] | None = None,
                   source_branch: str | None = None,
                   source_subtype: str | None = None,
                   catalog: Mapping[str, Mapping[str, Any]] | None = None,
                   observed: bool | None = None,
                   scope: Mapping[str, Any] | None = None,
                   source_file: str | None = None,
                   source_line: int | None = None,
                   include_absent: bool = False,
                   owner: str | None = None,
                   numeric_only: bool = False) -> tuple[ProjectedField, ...]:
    """Walk and group values within one complete source record or node payload.

    ``scope`` verifies sample identity fields while retaining every source leaf.
    Missing source identity fields remain absent from the source projection.
    """
    if scope is not None and isinstance(raw, Mapping):
        validate_record_identity(raw, scope, source_family=source_family,
                                 source_file=source_file or source_family,
                                 source_line=source_line)
    if source_family == "arm_states":
        if field_prefix or source_prefix_tokens:
            raise ValueError("ARM state structure projection requires a complete source row")
        observed = _arm_state_observation(raw, observed, scope, source_file, source_line)
    queue_value = None
    if source_family == "compute_state":
        if field_prefix or source_prefix_tokens:
            raise ValueError("compute queue structure projection requires a complete source row")
        queue_value = _compute_queue_value(raw)
    grouped: dict[tuple[Any, ...], ProjectedField] = {}
    mapping_signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    present_ids: set[str] = set()
    present_paths: set[str] = set()
    for leaf in iter_field_values(raw, source_family=source_family,
                                  field_prefix=field_prefix,
                                  source_prefix_tokens=source_prefix_tokens,
                                  entity_key_namespaces=entity_key_namespaces,
                                  source_branch=source_branch,
                                  source_subtype=source_subtype):
        if numeric_only and type(leaf.value) not in {int, float}:
            continue
        source_value = source_value_at_tokens(raw, leaf.input_tokens)
        if type(source_value) is not type(leaf.value) or source_value != leaf.value:
            raise ValueError(f"source value location differs: {source_family}:{leaf.raw_path}")
        structural_source_id = source_field_id(source_family, leaf.field_path,
                                               leaf.source_branch, leaf.source_subtype)
        variant_normalized_id: str | None = None
        if (source_family == "arm_predicate_truth" and
                leaf.field_path == WORLD_OBSERVATION_VALUE_PATH and
                type(leaf.value) in {int, float}):
            if leaf.source_path_key is None:
                raise ValueError(f"numeric world observation lacks its sibling path: {leaf.raw_path}")
            semantic_path = world_observation_variant_field_path(
                source_family, leaf.field_path, leaf.source_path_key)
            semantic_source_id = source_field_id(
                source_family, semantic_path, leaf.source_branch, leaf.source_subtype)
            variant_normalized_id = _normalized_id(semantic_source_id)
            if catalog is not None:
                if semantic_source_id not in catalog:
                    raise ValueError(
                        f"numeric world-observation branch is absent from catalog: "
                        f"{semantic_source_id}")
                declared_variant = catalog[semantic_source_id]
                catalog_source_branch(declared_variant)
                if (declared_variant.get("source_path_key") != leaf.source_path_key or
                        leaf.predicate_id not in declared_variant.get("predicate_ids", ())):
                    raise ValueError(
                        f"numeric world-observation branch authority differs: "
                        f"{leaf.predicate_id}:{leaf.source_path_key}")
                leaf = replace(leaf, field_path=semantic_path)
                source_id = semantic_source_id
            else:
                # The structure catalog retains the grammar's raw carrier ID,
                # while its normalized index records the tagged union branch.
                source_id = structural_source_id
        else:
            source_id = structural_source_id
        present_ids.add(source_id)
        present_paths.add(leaf.field_path)
        declared = catalog.get(source_id) if catalog is not None else None
        identity = (normalize_identity_field(scope, source_family, leaf.field_path,
                    leaf.value, source_file=source_file or source_family,
                    source_line=source_line) if scope is not None else None)
        normalized_id, normalized_value = (identity if identity is not None else
                                            (_normalized_id(source_id, observed), leaf.value))
        structure = (project_structural_leaf(
            source_family=source_family, field_path=leaf.field_path,
            branch=leaf.source_branch, subtype=leaf.source_subtype,
            source_tokens=leaf.source_tokens, input_tokens=leaf.input_tokens,
            value=leaf.value, record=raw) if identity is None else None)
        if structure is not None:
            normalized_id, normalized_value = structure.normalized_id, structure.value
        if variant_normalized_id is not None:
            if identity is not None or structure is not None:
                raise ValueError(
                    f"numeric world-observation branch unexpectedly has a structural rewrite: "
                    f"{leaf.field_path}")
            normalized_id = variant_normalized_id
        if source_family == "compute_state" and source_id in _COMPUTE_QUEUE_SOURCES:
            normalized_id, normalized_value = ((_COMPUTE_QUEUE_ID, queue_value)
                if queue_value is not None else (_normalized_id(source_id), leaf.value))
        if structure is not None:
            envelope = structure.envelope_tokens
            payload = structure.payload_tokens
            positions = structure.payload_positions
            shape = structure.normalized_shape or ()
            status = structure.value_status
            value_type = structure.value_type
            binding_slot = structure.binding_slot
            declared_shape = structure.declared_shape
            observed_shape = structure.observed_shape
            normalized_shape = structure.normalized_shape
            structure_status = structure.structure_status
        else:
            envelope, payload, positions, shape = _payload_location(
                raw, leaf, source_prefix_tokens)
            status = _value_status(leaf.value, source_family, leaf.field_path, declared)
            value_type = source_value_type(normalized_value)
            binding_slot = None
            declared_shape = None
            observed_shape = shape
            normalized_shape = shape
            structure_status = "source_only"
            if identity is not None:
                binding_slot = "sample_identity"
                declared_shape = observed_shape = normalized_shape = ()
                structure_status = "resolved"
            elif observed is True and source_id in _ARM_OBSERVED_POSE_IDS:
                if shape != (3,):
                    raise ValueError(f"observed ARM pose shape differs: {source_id}:{shape}")
                binding_slot = "observed_pose"
                declared_shape = observed_shape = normalized_shape = (3,)
                structure_status = "resolved"
            elif source_family == "compute_state" and source_id in _COMPUTE_QUEUE_SOURCES:
                binding_slot = "queue_complete" if queue_value is not None else "queue_missing"
                declared_shape = observed_shape = normalized_shape = ()
                structure_status = "resolved" if queue_value is not None else "declared_only"
        if source_family == "compute_state" and source_id in _COMPUTE_QUEUE_SOURCES and queue_value is None:
            status = "unknown"
        reuse_rules = rules_for_source(source_id)
        if reuse_rules:
            context = (_reuse_entity_context(raw, leaf, source_prefix_tokens)
                       if reuse_rules[0].condition != "always" else None)
            reuse = resolve_reuse(
                source_family=source_family, field_path=leaf.field_path,
                current_normalized_id=normalized_id, current_binding_slot=binding_slot,
                source_branch=leaf.source_branch, source_subtype=leaf.source_subtype,
                context=context, candidate_value=leaf.value)
            if reuse is not None:
                normalized_id, binding_slot = reuse.normalized_id, reuse.binding_slot
        source = ProjectedSource(source_id, leaf)
        key = (normalized_id, binding_slot, envelope, positions,
               leaf.entity_namespace, leaf.entity_id,
               leaf.snapshot_tick, leaf.value_tick, leaf.time_role, leaf.endpoint_role,
               leaf.target_scope, leaf.predicate_id, leaf.source_path_key,
               _seed_profile_instance(source_family, leaf),
               _dynamic_instance(leaf, field_prefix),
               leaf.source_branch, leaf.source_subtype)
        mapping_signature = ((declared["source_unit"], declared["canonical_unit"],
                              declared.get("coordinate_frame")) if declared is not None else
                             (None, None, None))
        existing = grouped.get(key)
        if existing is None:
            mapping_signatures[key] = mapping_signature
            grouped[key] = ProjectedField(normalized_id, normalized_value,
                value_type, envelope, payload, positions, shape,
                status, (source,), record_observed=observed,
                binding_slot=binding_slot, declared_shape=declared_shape,
                observed_shape=observed_shape, normalized_shape=normalized_shape,
                structure_status=structure_status)
        else:
            if mapping_signatures[key] != mapping_signature:
                raise ValueError(f"normalized source unit mapping conflicts: {normalized_id}:{envelope}")
            if (type(existing.value) is not type(normalized_value) or existing.value != normalized_value
                    or existing.payload_shape != shape or existing.value_status != status
                    or existing.value_type != value_type
                    or existing.declared_shape != declared_shape
                    or existing.observed_shape != observed_shape
                    or existing.normalized_shape != normalized_shape
                    or existing.structure_status != structure_status):
                raise ValueError(f"normalized source values conflict: {normalized_id}:{envelope}")
            grouped[key] = ProjectedField(existing.normalized_id, existing.value,
                existing.value_type, existing.envelope_tokens, existing.payload_tokens,
                existing.payload_positions, existing.payload_shape,
                existing.value_status, existing.sources + (source,),
                record_observed=observed, binding_slot=existing.binding_slot,
                declared_shape=existing.declared_shape,
                observed_shape=existing.observed_shape,
                normalized_shape=existing.normalized_shape,
                structure_status=existing.structure_status)
    if source_family == "compute_state" and queue_value is not None:
        queue_keys = [key for key, projected in grouped.items()
                      if projected.normalized_id == _COMPUTE_QUEUE_ID]
        if len(queue_keys) != 1:
            raise ValueError("compute queue sources did not form one structural field")
        key = queue_keys[0]
        projected = grouped[key]
        by_id = {source.source_field_id: source for source in projected.sources}
        if len(projected.sources) != 2 or set(by_id) != set(_COMPUTE_QUEUE_SOURCES):
            raise ValueError("compute queue structural field lacks its exact two source leaves")
        top_source, nested_source = (by_id[source_id] for source_id in _COMPUTE_QUEUE_SOURCES)
        envelope, payload, positions, shape = _payload_location(
            raw, top_source.field_value, source_prefix_tokens)
        grouped[key] = ProjectedField(_COMPUTE_QUEUE_ID, queue_value, "int",
            envelope, payload, positions, shape, "present",
            (top_source, nested_source), record_observed=observed,
            binding_slot="queue_complete", declared_shape=(), observed_shape=(),
            normalized_shape=(), structure_status="resolved")
    if include_absent:
        if catalog is None or owner is None:
            raise ValueError("absent projection requires catalog and owner authority")
        branch = source_branch
        subtype = source_subtype
        if source_family == "domain_state" and not field_prefix:
            branch = source_branch_for_record(source_family, raw)
            subtype = source_subtype_for_record(source_family, raw)
        for (family, field_path), location in _ABSENT_LOCATIONS.items():
            if family != source_family or location != (owner, field_prefix):
                continue
            source_id = source_field_id(source_family, field_path, branch, subtype)
            declared = catalog.get(source_id)
            if declared is None:
                continue
            exact_rule = EXACT_MISSING_RULES.get((source_family, field_path))
            base_path = _TRAILING_ARRAYS.sub("", field_path)
            if (declared["source_family"] != source_family or declared["owner"] != owner
                    or source_id in present_ids
                    or base_path in present_paths
                    or not field_path.startswith(field_prefix) or "{" in field_path
                    or location != (owner, field_prefix)
                    or len(source_prefix_tokens) != 2
                    or source_prefix_tokens[0] != "entities"
                    or type(source_prefix_tokens[1]) is not int
                    or exact_rule is None or "absent" not in exact_rule
                    or (declared.get("missing_rule") or {}).get("absent") != exact_rule["absent"]
                    or catalog_source_branch(declared) != branch
                    or catalog_source_subtype(declared) != subtype):
                continue
            normalized_id = _normalized_id(source_id)
            key = ("absent", source_id, source_prefix_tokens)
            grouped[key] = ProjectedField(normalized_id, None, None,
                source_prefix_tokens, (), (), (), "absent", (),
                record_observed=observed, absent_source_field_id=source_id)
    return tuple(grouped.values())


def project_node_values(node: Mapping[str, Any], *,
                        catalog: Mapping[str, Mapping[str, Any]] | None = None,
                        scope: Mapping[str, Any] | None = None,
                        owner_scope: Mapping[str, Any] | None = None,
                        require_stored_scope: bool = False,
                        include_absent: bool = False,
                        numeric_only: bool = False) -> tuple[ProjectedField, ...]:
    """Project a graph node's raw payload using its source record location."""
    family, prefix = _source_family(node)
    raw = node["attrs"].get("raw")
    actual_scope = node_source_scope(node, family, owner_scope,
                                    require_stored=require_stored_scope)
    if scope is not None and scope != actual_scope:
        raise ValueError(f"node source projection scope differs: {node['node_id']}")
    if family is None or raw is None:
        if raw is None and node["attrs"].get("fields") is not None:
            raise ValueError(f"structured fields lack source raw record: {node['node_id']}")
        return ()
    if not isinstance(raw, dict):
        raise TypeError(f"source raw record is not an object: {node['node_id']}")
    observed = None
    if family == "arm_states" and node["node_type"] == "EntityState":
        observed = node["attrs"].get("observed")
        if type(observed) is not bool:
            raise ValueError(f"ARM state lacks resolved observation status: {node['node_id']}")
    namespaces = (controller_namespaces_for_record(node["source_file"], raw)
                  if family == "domain_state" else None)
    return project_values(raw, source_family=family, field_prefix=prefix,
                          source_prefix_tokens=_source_location(node, family, prefix),
                          entity_key_namespaces=namespaces, catalog=catalog,
                          observed=observed, scope=actual_scope,
                          source_file=node["source_file"], source_line=node["source_line"],
                          include_absent=include_absent, owner=node["node_type"],
                          numeric_only=numeric_only)


def _source_family(node: dict) -> tuple[str | None, str]:
    source = node["source_file"]
    if node["node_type"] == "EventPhase" and source.endswith("/event_occurrences.jsonl"):
        return ("arm_event_occurrences" if source.startswith("arms/")
                else "formal_event_occurrences"), "event_phases[]."
    if node["attrs"].get("source_record_family") == "arm_semantic_history":
        if node["node_type"] == "PredicateAssertion":
            return "arm_semantic_history", "objective.predicate_truth[]."
        if node["node_type"] == "PredicateTransition":
            return "arm_semantic_history", "objective.predicate_transitions[]."
    if source.startswith("aw_data/render_ready_episodes_capture_filtered/") and source.endswith("/event_realization.jsonl"):
        return "formal_event_realization", ""
    if source.startswith("aw_data/render_ready_episodes_capture_filtered/") and source.endswith("/weather_meta.jsonl"):
        return "formal_weather_meta", ""
    if source.endswith("/truth_frames.jsonl"):
        return "formal_truth_frame", "entities[]." if node["node_type"] == "EntityState" else ""
    if source.endswith("/global_entity_roster.json"):
        return ("arm_branch_roster" if source.startswith("arms/") else "formal_roster"), ""
    if node["node_type"] == "EntityIdentity" and node["attrs"].get("source_completeness") == "scene":
        return "arm_scene_setup", ""
    if "/objective_semantic_truth/" in source:
        name = Path(source).stem
        if name in {"compute_predicate_truth", "compute_predicate_matrix", "compute_events"}:
            return {"compute_predicate_truth": "predicate_truth",
                    "compute_predicate_matrix": "predicate_truth_matrix", "compute_events": "events"}[name], ""
        if name in {"domain_state_observations", "compute_state", "communication_state"}:
            return name.replace("_observations", ""), ""
        if (name == "world_truth_graph_base" and node["node_type"] == "EntityIdentity"
                and node["attrs"].get("source_completeness") == "world_scope"):
            return "formal_world_truth_graph_base", "scope_entities[]."
        return "formal_" + name, ""
    if source.startswith("arms/") and source.endswith("/ue/event_script.json"):
        prefix = ("events[].actions[]." if node["node_type"] == "ActionPlan" else
                  "events[]." if node["node_type"] == "ScriptEvent" else "")
        return "arm_script_plan", prefix
    if node["node_type"] in {"ScriptEvent", "ActionPlan", "ContextAnchor"} and not source.startswith("aw_data/") and not source.startswith("arms/"):
        prefix = ("events[].actions[]." if node["node_type"] == "ActionPlan" else
                  "events[]." if node["node_type"] == "ScriptEvent" else "")
        return "formal_source_script", prefix
    if source.startswith("arms/") and source.endswith(("/chain/nodes.jsonl", "/chain/events.jsonl")):
        family = "arm_chain_" + Path(source).stem
        return arm_chain_row_family(family, node["attrs"]["raw"]), ""
    if source.startswith("arms/") and source.endswith("/chain/relations.jsonl"):
        return "arm_chain_relations", ""
    if (source.startswith("arms/") and source.endswith("/objective/predicate_truth_ticks.jsonl")
            and node["node_type"] == "ContextAnchor"
            and node["attrs"].get("source_layer") == "domain_state_observation"):
        return "arm_domain_state_ticks", ""
    if source.startswith("arms/"):
        return "arm_" + source.rsplit("/", 1)[-1].removesuffix(".jsonl"), ""
    if "/domain_state_supplement/" in source:
        return "domain_state", ""
    if "/compute_comm_supplement/" in source:
        return source.rsplit("/", 1)[-1].removesuffix(".jsonl"), ""
    return None, ""


def _requires_source_projection(node: dict, family: str, field_path: str) -> bool:
    """Require a catalog-backed value for source payloads represented by this node."""
    if family == "formal_weather_meta":
        return node["node_type"] == "ContextAnchor" and field_path != "tick"
    if family == "formal_truth_frame":
        return (node["node_type"] == "ContextAnchor"
                and node["category_id"] == "sumo_controller_state"
                and field_path.startswith("sumo_traffic_light_states.{entity}."))
    if family == "arm_actions":
        return node["node_type"] == "ContextAnchor" and node["category_id"] == "arm_action_audit"
    return family == "formal_event_realization" and node["node_type"] == "ScriptEvent"


def _source_location(node: dict, family: str, field_prefix: str) -> tuple[str | int, ...]:
    """Locate attrs.raw inside its complete source-file record."""
    attrs = node["attrs"]

    def index(name: str) -> int:
        value = attrs.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"source array index missing: {node['node_id']}:{name}")
        return value

    if family in {"formal_roster", "arm_branch_roster", "arm_scene_setup"}:
        entity_index = index("source_entity_index")
        return "entities", entity_index
    if not field_prefix:
        return ()
    if family in {"arm_event_occurrences", "formal_event_occurrences"} and field_prefix == "event_phases[].":
        return "event_phases", index("phase_index")
    if family == "arm_semantic_history":
        name = ("predicate_truth" if field_prefix == "objective.predicate_truth[]."
                else "predicate_transitions" if field_prefix == "objective.predicate_transitions[]." else None)
        if name is None:
            raise ValueError(f"history projection lacks an exact source prefix: {field_prefix}")
        return "objective", name, index("source_history_index")
    if family == "formal_event_realization" and field_prefix == "action_realizations[].":
        action_index = index("source_action_index")
        return "action_realizations", action_index
    if family == "formal_truth_frame" and field_prefix == "entities[].":
        entity_index = index("source_entity_index")
        return "entities", entity_index
    if family == "formal_world_truth_graph_base" and field_prefix == "scope_entities[].":
        entity_index = index("source_scope_entity_index")
        return "scope_entities", entity_index
    if family in {"arm_script_plan", "formal_source_script"} and field_prefix == "events[].":
        event_index = index("source_event_index")
        return "events", event_index
    if family in {"arm_script_plan", "formal_source_script"} and field_prefix == "events[].actions[].":
        event_index = index("source_event_index")
        action_index = index("source_action_index")
        return "events", event_index, "actions", action_index
    raise ValueError(f"source array location has no rule: {node['node_id']}:{family}:{field_prefix}")
