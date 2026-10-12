"""Read-only projection from objective AERO truth to compact model tensors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import orjson

from Dataset.semantic_truth.provenance import open_artifact_binary
from Dataset.semantic_truth.world_truth import replay_world_truth

from .contract import (
    EPISODE_SCHEMA_NAME,
    EPISODE_SCHEMA_VERSION,
    FORMAL_TICK_COUNT,
    FORMAL_TICKS,
    INACTIVE_PREDICATE_CODE,
    PREDICATE_TRUTH_CODES,
    TICK_TO_INDEX,
    UNASSIGNED_SPLIT,
    ModelDataError,
    canonical_json,
    field_unit,
    is_missing_state_value,
    pack_bits,
    parse_episode_id,
    sha256_bytes,
    stable_identifier,
    stable_int63,
    validate_state_value,
)


_UNASSIGNED = object()
_DIRECT_MOBILE_PAIR_DISTANCE_FAMILIES = frozenset(
    {
        "predicate_contract_aircraft_vehicle_geometry",
        "predicate_contract_aircraft_pedestrian_geometry",
        "predicate_contract_vehicle_pedestrian_geometry",
    }
)


@dataclass
class EpisodeProjection:
    episode: dict[str, Any]
    entities: list[dict[str, Any]]
    state_slots: list[dict[str, Any]]
    predicate_slots: list[dict[str, Any]]
    tensor_payload: dict[str, Any]
    ragged_state: list[dict[str, Any]]
    categories: dict[int, tuple[str, set[str]]]
    audit: dict[str, Any]


@dataclass
class _EntityAccumulator:
    entity_id: str
    scope_types: set[str] = field(default_factory=set)
    scope_subtypes: set[str] = field(default_factory=set)
    ontology_class_ids: set[str] = field(default_factory=set)
    subject_categories: set[str] = field(default_factory=set)
    subject_kinds: set[str] = field(default_factory=set)
    source_kinds: set[str] = field(default_factory=set)
    observer_capable: bool = False

    def update(
        self,
        *,
        scope_type: Any = None,
        scope_subtype: Any = None,
        ontology_class_id: Any = None,
        subject_category: Any = None,
        subject_kind: Any = None,
        source_kind: str,
    ) -> None:
        _add_string(self.scope_types, scope_type)
        _add_string(self.scope_subtypes, scope_subtype)
        _add_string(self.ontology_class_ids, ontology_class_id)
        _add_string(self.subject_categories, subject_category)
        _add_string(self.subject_kinds, subject_kind)
        self.source_kinds.add(source_kind)
        if scope_type == "uav" or ontology_class_id == "world:UnmannedAircraft":
            self.observer_capable = True


@dataclass
class _StateSlotAccumulator:
    episode_id: str
    subject_id: str
    observation_family: str
    field_name: str
    value_kind: str
    width: int
    values: list[Any] = field(
        default_factory=lambda: [_UNASSIGNED for _ in range(FORMAL_TICK_COUNT)]
    )
    source_classes: set[str] = field(default_factory=set)
    model_ids: set[str] = field(default_factory=set)
    model_versions: set[str] = field(default_factory=set)
    rule_ids: set[str] = field(default_factory=set)
    rule_versions: set[str] = field(default_factory=set)
    input_digests: set[str] = field(default_factory=set)
    parameter_digests: set[str] = field(default_factory=set)

    @property
    def slot_id(self) -> str:
        return stable_identifier(
            "model_state_slot",
            self.episode_id,
            self.subject_id,
            self.observation_family,
            self.field_name,
        )

    def put(self, tick: int, value: Any, row: Mapping[str, Any]) -> None:
        index = TICK_TO_INDEX.get(tick)
        if index is None:
            raise ModelDataError(f"L0 row uses a non-formal tick: {tick}")
        if self.values[index] is not _UNASSIGNED:
            raise ModelDataError(
                f"duplicate state assignment: {self.subject_id}/"
                f"{self.observation_family}/{self.field_name}@{tick}"
            )
        stored_value = value
        if self.value_kind in {
            "multi_reference",
            "ragged_continuous",
        } and not is_missing_state_value(value):
            stored_value = canonical_json(value)
        self.values[index] = stored_value
        _add_string(self.source_classes, row.get("source_class"))
        _add_string(self.model_ids, row.get("model_id"))
        _add_string(self.model_versions, row.get("model_version"))
        _add_string(self.rule_ids, row.get("rule_id"))
        _add_string(self.rule_versions, row.get("rule_version"))
        _add_string(self.input_digests, row.get("input_digest"))
        _add_string(self.parameter_digests, row.get("parameter_digest"))


def project_episode(
    episode_root: Path,
    *,
    source_root: Path,
) -> EpisodeProjection:
    """Project one objective semantic-truth episode without mutating its source."""

    episode_root = episode_root.resolve()
    source_root = source_root.resolve()
    try:
        episode_root.relative_to(source_root)
    except ValueError as exc:
        raise ModelDataError(
            f"episode escapes objective source root: {episode_root}"
        ) from exc
    episode_id = episode_root.name
    profile_id, seed_label = parse_episode_id(episode_id)
    manifest_path = episode_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_object_bytes(manifest_bytes, manifest_path)
    _validate_objective_manifest(manifest, episode_id)

    render_root = Path(
        _required_string(manifest, "episode_root", str(manifest_path))
    ).resolve()
    if render_root.name != episode_id:
        raise ModelDataError(f"{episode_id}: render episode root has the wrong name")
    render_manifest_path = render_root / "episode_manifest.json"
    render_manifest_bytes = render_manifest_path.read_bytes()
    render_manifest = _load_object_bytes(render_manifest_bytes, render_manifest_path)
    _validate_time_contract(render_manifest, episode_id)
    capture_boundary_id = _required_string(
        render_manifest,
        "capture_boundary_id",
        str(render_manifest_path),
    )
    trajectory_path = render_root / "trajectories.jsonl"
    if not trajectory_path.is_file():
        raise ModelDataError(f"{episode_id}: capture-filtered trajectories are absent")

    base_path = episode_root / "world_truth_graph_base.json"
    delta_path = episode_root / "world_truth_graph_deltas.jsonl"
    l0_path = episode_root / "l0_predicate_state.jsonl"
    base_bytes = base_path.read_bytes()
    base = _load_object_bytes(base_bytes, base_path)
    delta_bytes, deltas = _load_jsonl_with_bytes(delta_path)

    entities: dict[str, _EntityAccumulator] = {}
    for item in base.get("scope_entities", ()):
        if not isinstance(item, Mapping):
            raise ModelDataError(f"{base_path}: scope entity must be an object")
        entity_id = _required_string(item, "entity_id", str(base_path))
        _entity(entities, entity_id).update(
            scope_type=item.get("scope_type"),
            scope_subtype=item.get("scope_subtype"),
            ontology_class_id=item.get("ontology_class_id"),
            source_kind="world_scope",
        )

    predicate_slots, predicate_tensor = _project_predicates(
        base,
        deltas,
        entities,
        context=str(episode_root),
    )

    (
        state_slots,
        state_tensors,
        ragged_rows,
        categories,
        l0_audit,
        l0_digest,
        trajectory_digest,
    ) = _project_l0_state(
        l0_path,
        episode_id=episode_id,
        entities=entities,
        trajectory_path=trajectory_path,
    )
    if l0_audit["l0_row_count"] != manifest["record_counts"]["l0_predicate_state"]:
        raise ModelDataError(f"{episode_id}: L0 row count disagrees with manifest")

    _entity(entities, capture_boundary_id).update(
        ontology_class_id="world:OperationalRegion",
        source_kind="episode_manifest",
    )
    entity_rows, entity_index = _finalize_entities(episode_id, entities)
    for row in state_slots:
        row["subject_entity_index"] = entity_index[row["subject_id"]]
    for row in predicate_slots:
        binding_ids = row.pop("binding_entity_ids")
        row["binding_entity_indices"] = [entity_index[value] for value in binding_ids]

    tensor_payload = {
        "episode_id": episode_id,
        **state_tensors,
        "predicate_truth_codes": predicate_tensor.astype("<u1", copy=False).tobytes(),
    }
    predicate_counts = Counter(predicate_tensor.reshape(-1).tolist())
    episode_row = {
        "schema_name": EPISODE_SCHEMA_NAME,
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "profile_id": profile_id,
        "seed_label": seed_label,
        "split": UNASSIGNED_SPLIT,
        "capture_boundary_id": capture_boundary_id,
        "formal_tick_count": FORMAL_TICK_COUNT,
        "simulation_tick_seconds": 0.1,
        "formal_frame_seconds": 0.5,
        "entity_count": len(entity_rows),
        "state_slot_count": len(state_slots),
        "continuous_width": state_tensors["continuous_width"],
        "boolean_width": state_tensors["boolean_width"],
        "categorical_width": state_tensors["categorical_width"],
        "reference_width": state_tensors["reference_width"],
        "ragged_slot_count": sum(
            row["value_kind"] in {"multi_reference", "ragged_continuous"}
            for row in state_slots
        ),
        "predicate_slot_count": len(predicate_slots),
        "source_manifest_path": f"{episode_id}/manifest.json",
        "source_manifest_sha256": sha256_bytes(manifest_bytes),
        "source_l0_path": f"{episode_id}/l0_predicate_state.jsonl",
        "source_l0_sha256": l0_digest,
        "source_render_manifest_path": str(render_manifest_path),
        "source_render_manifest_sha256": sha256_bytes(render_manifest_bytes),
        "source_trajectory_path": str(trajectory_path),
        "source_trajectory_sha256": trajectory_digest,
        "source_world_base_path": f"{episode_id}/world_truth_graph_base.json",
        "source_world_base_sha256": sha256_bytes(base_bytes),
        "source_world_delta_path": f"{episode_id}/world_truth_graph_deltas.jsonl",
        "source_world_delta_sha256": sha256_bytes(delta_bytes),
        "source_context_input_digest": str(manifest["context_input_digest"]),
        "source_domain_input_digest": str(manifest["domain_input_digest"]),
        "source_l0_state_digest": str(
            manifest["l0_state_materialization"]["state_digest"]
        ),
        "source_world_base_digest": str(base["base_graph_digest"]),
    }
    audit = {
        "episode_id": episode_id,
        **l0_audit,
        "entity_count": len(entity_rows),
        "predicate_slot_count": len(predicate_slots),
        "predicate_false_count": predicate_counts[0],
        "predicate_true_count": predicate_counts[1],
        "predicate_unknown_count": predicate_counts[2],
        "predicate_out_of_scope_count": predicate_counts[3],
        "predicate_inactive_count": predicate_counts[4],
    }
    projection = EpisodeProjection(
        episode=episode_row,
        entities=entity_rows,
        state_slots=state_slots,
        predicate_slots=predicate_slots,
        tensor_payload=tensor_payload,
        ragged_state=ragged_rows,
        categories=categories,
        audit=audit,
    )
    finalize_reference_tensor(projection)
    return projection


def _project_l0_state(
    path: Path,
    *,
    episode_id: str,
    entities: dict[str, _EntityAccumulator],
    trajectory_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[int, tuple[str, set[str]]],
    dict[str, Any],
    str,
    str,
]:
    slots: dict[tuple[str, str, str], _StateSlotAccumulator] = {}
    ticks: set[int] = set()
    row_count = 0
    assignment_count = 0
    valid_assignment_count = 0
    mobile_pair_distances: list[tuple[int, str, str, float]] = []
    mobile_pair_entity_ids: set[str] = set()
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            hasher.update(line)
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ModelDataError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ModelDataError(f"{path}:{line_number}: row must be an object")
            if row.get("episode_id") != episode_id:
                raise ModelDataError(f"{path}:{line_number}: episode mismatch")
            tick = row.get("tick")
            if not isinstance(tick, int) or tick not in TICK_TO_INDEX:
                raise ModelDataError(f"{path}:{line_number}: invalid formal tick")
            ticks.add(tick)
            family = _required_string(
                row, "observation_family", f"{path}:{line_number}"
            )
            subject_id = _required_string(row, "subject_id", f"{path}:{line_number}")
            values = row.get("values")
            if not isinstance(values, dict):
                raise ModelDataError(f"{path}:{line_number}: values must be an object")
            if family in _DIRECT_MOBILE_PAIR_DISTANCE_FAMILIES:
                distance = values.get("distance_m")
                if isinstance(distance, (int, float)) and not isinstance(
                    distance, bool
                ):
                    bindings = subject_id.split("|")
                    if len(bindings) != 2 or not all(bindings):
                        raise ModelDataError(
                            f"{path}:{line_number}: malformed mobile-pair subject"
                        )
                    mobile_pair_distances.append(
                        (tick, family, subject_id, float(distance))
                    )
                    mobile_pair_entity_ids.update(bindings)
            assigned, valid = _assign_state_values(
                slots,
                episode_id=episode_id,
                tick=tick,
                family=family,
                subject_id=subject_id,
                values=values,
                provenance=row,
                entities=entities,
                subject_category=row.get("subject_category"),
                subject_kind=row.get("subject_kind"),
                source_kind="l0_subject",
                context=f"{path}:{line_number}",
            )
            assignment_count += assigned
            valid_assignment_count += valid
            row_count += 1
    if tuple(sorted(ticks)) != FORMAL_TICKS:
        raise ModelDataError(f"{path}: incomplete formal tick coverage")

    trajectory_hasher = hashlib.sha256()
    trajectory_rows = Counter()
    trajectory_assignment_count = 0
    trajectory_valid_assignment_count = 0
    seen_mobile_frames: set[tuple[int, str]] = set()
    pair_trajectory_positions: dict[tuple[int, str], tuple[float, float, float]] = {}
    with trajectory_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            trajectory_hasher.update(line)
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ModelDataError(
                    f"{trajectory_path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ModelDataError(
                    f"{trajectory_path}:{line_number}: row must be an object"
                )
            tick = row.get("tick")
            if not isinstance(tick, int):
                raise ModelDataError(
                    f"{trajectory_path}:{line_number}: tick must be an integer"
                )
            entity_id = row.get("entity_id")
            if (
                tick in TICK_TO_INDEX
                and isinstance(entity_id, str)
                and entity_id in mobile_pair_entity_ids
            ):
                position = row.get("pos_enu")
                if (
                    not isinstance(position, list)
                    or len(position) != 3
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not np.isfinite(value)
                        for value in position
                    )
                ):
                    raise ModelDataError(
                        f"{trajectory_path}:{line_number}: mobile-pair entity lacks "
                        "a finite pos_enu"
                    )
                pair_trajectory_positions[(tick, entity_id)] = tuple(
                    float(value) for value in position
                )
            if tick not in TICK_TO_INDEX:
                continue
            category = row.get("entity_category") or row.get("category")
            if category not in {"vehicle", "pedestrian"}:
                continue
            entity_id = _required_string(
                row, "entity_id", f"{trajectory_path}:{line_number}"
            )
            frame_key = (tick, entity_id)
            if frame_key in seen_mobile_frames:
                raise ModelDataError(
                    f"{trajectory_path}:{line_number}: duplicate mobile frame {frame_key}"
                )
            seen_mobile_frames.add(frame_key)
            provenance = {
                "source_class": "oracle_object_observation",
                "rule_id": "capture_filtered_trajectory.same_tick_projection",
                "rule_version": EPISODE_SCHEMA_VERSION,
            }
            assigned, valid = _assign_state_values(
                slots,
                episode_id=episode_id,
                tick=tick,
                family="oracle_object_state",
                subject_id=entity_id,
                values={
                    "position_enu_m": row.get("pos_enu"),
                    "velocity_enu_mps": row.get("vel_mps"),
                    "yaw_deg": row.get("yaw_deg"),
                },
                provenance=provenance,
                entities=entities,
                scope_type=str(category),
                ontology_class_id=(
                    "world:GroundVehicle"
                    if category == "vehicle"
                    else "world:Pedestrian"
                ),
                subject_category=category,
                subject_kind=row.get("entity_kind"),
                source_kind="oracle_object_observation",
                context=f"{trajectory_path}:{line_number}",
            )
            trajectory_assignment_count += assigned
            trajectory_valid_assignment_count += valid
            trajectory_rows[str(category)] += 1
            if category != "vehicle":
                continue
            vehicle = row.get("sumo_vehicle")
            if vehicle is None:
                trajectory_rows["vehicle_without_runtime_context"] += 1
                continue
            if not isinstance(vehicle, Mapping):
                raise ModelDataError(
                    f"{trajectory_path}:{line_number}: malformed SUMO runtime state"
                )
            trajectory_rows["vehicle_runtime_context"] += 1
            runtime_values = _vehicle_runtime_values(vehicle)
            assigned, valid = _assign_state_values(
                slots,
                episode_id=episode_id,
                tick=tick,
                family="vehicle_runtime_context",
                subject_id=entity_id,
                values=runtime_values,
                provenance=provenance,
                entities=entities,
                scope_type="vehicle",
                ontology_class_id="world:GroundVehicle",
                subject_category="vehicle",
                subject_kind=row.get("entity_kind"),
                source_kind="oracle_object_observation",
                context=f"{trajectory_path}:{line_number}",
            )
            trajectory_assignment_count += assigned
            trajectory_valid_assignment_count += valid

    checked_mobile_pair_geometry_count = _validate_mobile_pair_geometry(
        mobile_pair_distances,
        pair_trajectory_positions,
        context=path,
    )
    (
        unobservable_mobile_pair_row_count,
        unobservable_mobile_pair_assignment_count,
    ) = _remove_unobservable_mobile_pair_state(
        slots,
        mobile_pair_distances,
        pair_trajectory_positions,
    )

    if trajectory_rows["vehicle"] != (
        trajectory_rows["vehicle_runtime_context"]
        + trajectory_rows["vehicle_without_runtime_context"]
    ):
        raise ModelDataError(f"{trajectory_path}: vehicle runtime accounting diverged")

    ordered = sorted(slots.values(), key=lambda item: item.slot_id)
    widths = Counter()
    for slot in ordered:
        if slot.value_kind == "continuous":
            widths["continuous"] += slot.width
        elif slot.value_kind in {"boolean", "categorical", "reference"}:
            widths[slot.value_kind] += 1

    continuous = np.full((FORMAL_TICK_COUNT, widths["continuous"]), np.nan, dtype="<f8")
    continuous_scope = np.zeros_like(continuous, dtype=np.bool_)
    continuous_valid = np.zeros_like(continuous, dtype=np.bool_)
    boolean = np.zeros((FORMAL_TICK_COUNT, widths["boolean"]), dtype=np.bool_)
    boolean_scope = np.zeros_like(boolean)
    boolean_valid = np.zeros_like(boolean)
    categorical = np.zeros((FORMAL_TICK_COUNT, widths["categorical"]), dtype="<i8")
    categorical_scope = np.zeros_like(categorical, dtype=np.bool_)
    categorical_valid = np.zeros_like(categorical, dtype=np.bool_)
    reference_values: list[tuple[_StateSlotAccumulator, int, list[Any]]] = []
    reference_scope = np.zeros((FORMAL_TICK_COUNT, widths["reference"]), dtype=np.bool_)
    reference_valid = np.zeros_like(reference_scope)
    offsets = Counter()
    rows: list[dict[str, Any]] = []
    ragged_rows: list[dict[str, Any]] = []
    categories: dict[int, tuple[str, set[str]]] = {}
    kind_counts = Counter()
    for slot_index, slot in enumerate(ordered):
        storage_index = -1
        if slot.value_kind in {"continuous", "boolean", "categorical", "reference"}:
            storage_index = offsets[slot.value_kind]
            offsets[slot.value_kind] += slot.width
        assigned_count = 0
        valid_count = 0
        distinct_values: set[Any] = set()
        if slot.value_kind == "reference":
            reference_values.append((slot, storage_index, slot.values))
        for tick_index, raw_value in enumerate(slot.values):
            if raw_value is _UNASSIGNED:
                continue
            assigned_count += 1
            if slot.value_kind in {"multi_reference", "ragged_continuous"}:
                valid = raw_value is not None and raw_value != "unknown"
                ragged_rows.append(
                    {
                        "episode_id": episode_id,
                        "tick": FORMAL_TICKS[tick_index],
                        "slot_id": slot.slot_id,
                        "valid": valid,
                        "value_json": raw_value if valid else None,
                    }
                )
                if valid:
                    valid_count += 1
                    distinct_values.add(str(raw_value))
                continue
            if slot.value_kind == "continuous":
                sl = slice(storage_index, storage_index + slot.width)
                continuous_scope[tick_index, sl] = True
            elif slot.value_kind == "boolean":
                boolean_scope[tick_index, storage_index] = True
            elif slot.value_kind == "categorical":
                categorical_scope[tick_index, storage_index] = True
            elif slot.value_kind == "reference":
                reference_scope[tick_index, storage_index] = True
            if is_missing_state_value(raw_value):
                continue
            valid_count += 1
            distinct_values.add(
                tuple(raw_value) if isinstance(raw_value, list) else raw_value
            )
            if slot.value_kind == "continuous":
                sl = slice(storage_index, storage_index + slot.width)
                value = (
                    np.asarray(raw_value, dtype="<f8")
                    if slot.width > 1
                    else np.asarray([raw_value], dtype="<f8")
                )
                continuous[tick_index, sl] = value
                continuous_valid[tick_index, sl] = True
            elif slot.value_kind == "boolean":
                boolean[tick_index, storage_index] = bool(raw_value)
                boolean_valid[tick_index, storage_index] = True
            elif slot.value_kind == "categorical":
                code = stable_int63("state_category", str(raw_value))
                _register_category(
                    categories,
                    code,
                    str(raw_value),
                    f"{slot.observation_family}.{slot.field_name}",
                )
                categorical[tick_index, storage_index] = code
                categorical_valid[tick_index, storage_index] = True
            elif slot.value_kind == "reference":
                reference_valid[tick_index, storage_index] = True
        value_temporal_mode = (
            "all_missing"
            if valid_count == 0
            else "static"
            if len(distinct_values) == 1
            else "dynamic"
        )
        rows.append(
            {
                "episode_id": episode_id,
                "slot_id": slot.slot_id,
                "slot_index": slot_index,
                "subject_id": slot.subject_id,
                "observation_family": slot.observation_family,
                "field_name": slot.field_name,
                "value_kind": slot.value_kind,
                "storage_index": storage_index,
                "width": slot.width,
                "unit": field_unit(slot.field_name, slot.value_kind),
                "assigned_count": assigned_count,
                "valid_count": valid_count,
                "distinct_value_count": len(distinct_values),
                "value_temporal_mode": value_temporal_mode,
                "scope_temporal_mode": (
                    "static" if assigned_count == FORMAL_TICK_COUNT else "dynamic"
                ),
                "source_classes_json": canonical_json(sorted(slot.source_classes)),
                "model_ids_json": canonical_json(sorted(slot.model_ids)),
                "model_versions_json": canonical_json(sorted(slot.model_versions)),
                "rule_ids_json": canonical_json(sorted(slot.rule_ids)),
                "rule_versions_json": canonical_json(sorted(slot.rule_versions)),
                "input_digests_json": canonical_json(sorted(slot.input_digests)),
                "parameter_digests_json": canonical_json(
                    sorted(slot.parameter_digests)
                ),
            }
        )
        kind_counts[slot.value_kind] += 1

    tensor_values = {
        "continuous_width": widths["continuous"],
        "continuous_values": continuous.tobytes(order="C"),
        "continuous_scope": pack_bits(continuous_scope),
        "continuous_valid": pack_bits(continuous_valid),
        "boolean_width": widths["boolean"],
        "boolean_values": pack_bits(boolean),
        "boolean_scope": pack_bits(boolean_scope),
        "boolean_valid": pack_bits(boolean_valid),
        "categorical_width": widths["categorical"],
        "categorical_values": categorical.tobytes(order="C"),
        "categorical_scope": pack_bits(categorical_scope),
        "categorical_valid": pack_bits(categorical_valid),
        "reference_width": widths["reference"],
        "reference_values": reference_values,
        "reference_scope": pack_bits(reference_scope),
        "reference_valid": pack_bits(reference_valid),
    }
    audit = {
        "l0_row_count": row_count,
        "l0_assignment_count": assignment_count,
        "l0_valid_assignment_count": valid_assignment_count,
        "trajectory_formal_mobile_row_count": (
            trajectory_rows["vehicle"] + trajectory_rows["pedestrian"]
        ),
        "trajectory_formal_vehicle_row_count": trajectory_rows["vehicle"],
        "trajectory_formal_pedestrian_row_count": trajectory_rows["pedestrian"],
        "trajectory_vehicle_runtime_row_count": trajectory_rows[
            "vehicle_runtime_context"
        ],
        "trajectory_vehicle_without_runtime_context_count": trajectory_rows[
            "vehicle_without_runtime_context"
        ],
        "trajectory_assignment_count": trajectory_assignment_count,
        "trajectory_valid_assignment_count": trajectory_valid_assignment_count,
        "checked_mobile_pair_geometry_count": checked_mobile_pair_geometry_count,
        "unobservable_mobile_pair_row_count": unobservable_mobile_pair_row_count,
        "unobservable_mobile_pair_assignment_count": (
            unobservable_mobile_pair_assignment_count
        ),
        "state_slot_count": len(rows),
        "state_slot_counts_by_kind": dict(sorted(kind_counts.items())),
        "ragged_record_count": len(ragged_rows),
    }
    return (
        rows,
        tensor_values,
        ragged_rows,
        categories,
        audit,
        "sha256:" + hasher.hexdigest(),
        "sha256:" + trajectory_hasher.hexdigest(),
    )


def _validate_mobile_pair_geometry(
    pair_rows: Sequence[tuple[int, str, str, float]],
    positions: Mapping[tuple[int, str], tuple[float, float, float]],
    *,
    context: Path,
) -> int:
    checked = 0
    for tick, family, subject_id, stored_distance in pair_rows:
        first_id, second_id = subject_id.split("|")
        first = positions.get((tick, first_id))
        second = positions.get((tick, second_id))
        if first is None or second is None:
            continue
        recomputed = float(np.linalg.norm(np.subtract(first, second)))
        if not np.isclose(stored_distance, recomputed, rtol=0.0, atol=1.0e-6):
            raise ModelDataError(
                f"{context}: {family}:{subject_id}@{tick} distance_m "
                f"{stored_distance} disagrees with same-tick trajectory geometry "
                f"{recomputed}"
            )
        checked += 1
    return checked


def _remove_unobservable_mobile_pair_state(
    slots: Mapping[tuple[str, str, str], _StateSlotAccumulator],
    pair_rows: Sequence[tuple[int, str, str, float]],
    positions: Mapping[tuple[int, str], tuple[float, float, float]],
) -> tuple[int, int]:
    slots_by_subject: dict[tuple[str, str], list[_StateSlotAccumulator]] = {}
    for (subject_id, family, _field_name), slot in slots.items():
        if family in _DIRECT_MOBILE_PAIR_DISTANCE_FAMILIES:
            slots_by_subject.setdefault((family, subject_id), []).append(slot)
    removed_rows = 0
    removed_assignments = 0
    for tick, family, subject_id, _distance in pair_rows:
        first_id, second_id = subject_id.split("|")
        required_external_ids = (
            (first_id, second_id)
            if family == "predicate_contract_vehicle_pedestrian_geometry"
            else (second_id,)
        )
        if all((tick, entity_id) in positions for entity_id in required_external_ids):
            continue
        tick_index = TICK_TO_INDEX[tick]
        removed_rows += 1
        for slot in slots_by_subject[(family, subject_id)]:
            if slot.values[tick_index] is _UNASSIGNED:
                raise ModelDataError(
                    "mobile-pair L0 row has inconsistent field assignments"
                )
            slot.values[tick_index] = _UNASSIGNED
            removed_assignments += 1
    return removed_rows, removed_assignments


def _assign_state_values(
    slots: dict[tuple[str, str, str], _StateSlotAccumulator],
    *,
    episode_id: str,
    tick: int,
    family: str,
    subject_id: str,
    values: Mapping[str, Any],
    provenance: Mapping[str, Any],
    entities: dict[str, _EntityAccumulator],
    source_kind: str,
    context: str,
    scope_type: Any = None,
    ontology_class_id: Any = None,
    subject_category: Any = None,
    subject_kind: Any = None,
) -> tuple[int, int]:
    _entity(entities, subject_id).update(
        scope_type=scope_type,
        ontology_class_id=ontology_class_id,
        subject_category=subject_category,
        subject_kind=subject_kind,
        source_kind=source_kind,
    )
    valid_count = 0
    for field_name, value in values.items():
        if not isinstance(field_name, str) or not field_name:
            raise ModelDataError(f"{context}: invalid state field name")
        kind, width = validate_state_value(field_name, value)
        _register_value_reference(
            field_name,
            value,
            value_kind=kind,
            all_values=values,
            entities=entities,
        )
        key = (subject_id, family, field_name)
        slot = slots.get(key)
        if slot is None:
            slot = _StateSlotAccumulator(
                episode_id=episode_id,
                subject_id=subject_id,
                observation_family=family,
                field_name=field_name,
                value_kind=kind,
                width=width,
            )
            slots[key] = slot
        elif slot.value_kind != kind or slot.width != width:
            raise ModelDataError(f"{context}: state slot type changed for {key}")
        slot.put(tick, value, provenance)
        valid_count += not is_missing_state_value(value)
    return len(values), valid_count


def _vehicle_runtime_values(vehicle: Mapping[str, Any]) -> dict[str, Any]:
    lane_id = _optional_reference(vehicle.get("lane_id"))
    speed_regulation_id = _optional_reference(vehicle.get("speed_limit_regulation_id"))
    following_rule_id = _optional_reference(vehicle.get("following_distance_rule_id"))
    signal_id = _optional_reference(vehicle.get("controlling_signal_id"))
    stop_line_id = _optional_reference(vehicle.get("stop_line_id"))
    return {
        "lane_id": lane_id,
        "lane_ontology_class_id": (
            vehicle.get("lane_ontology_class_id") if lane_id is not None else None
        ),
        "speed_mps": vehicle.get("speed_mps"),
        "posted_speed_limit_mps": vehicle.get("allowed_speed_mps"),
        "speed_limit_regulation_id": speed_regulation_id,
        "speed_limit_regulation_ontology_class_id": (
            vehicle.get("speed_limit_regulation_ontology_class_id")
            if speed_regulation_id is not None
            else None
        ),
        "following_distance_rule_id": following_rule_id,
        "following_distance_rule_ontology_class_id": (
            vehicle.get("following_distance_rule_ontology_class_id")
            if following_rule_id is not None
            else None
        ),
        "controlling_signal_id": signal_id,
        "controlling_signal_ontology_class_id": (
            vehicle.get("controlling_signal_ontology_class_id")
            if signal_id is not None
            else None
        ),
        "controlling_signal_state": (
            _optional_reference(vehicle.get("controlling_signal_state"))
            if signal_id is not None
            else None
        ),
        "stop_line_id": stop_line_id,
        "stop_line_ontology_class_id": (
            vehicle.get("stop_line_ontology_class_id")
            if stop_line_id is not None
            else None
        ),
        "crossed_stop_line": vehicle.get("crossed_stop_line"),
    }


def _optional_reference(value: Any) -> str | None:
    if value in {None, "", "none", "unknown"}:
        return None
    if not isinstance(value, str):
        raise ModelDataError(f"reference value must be a string, found {value!r}")
    return value


def _project_predicates(
    base: Mapping[str, Any],
    deltas: list[dict[str, Any]],
    entities: dict[str, _EntityAccumulator],
    *,
    context: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    final_replay = replay_world_truth(base, deltas, FORMAL_TICKS[-1])
    current: dict[tuple[str, str], dict[str, Any]] = {}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    values: dict[tuple[str, str], np.ndarray] = {}

    def register(assertion: Mapping[str, Any]) -> tuple[str, str]:
        predicate_id = _required_string(assertion, "predicate_id", context)
        tuple_id = _required_string(assertion, "tuple_id", context)
        key = (predicate_id, tuple_id)
        roles = assertion.get("ontology_argument_roles")
        bindings = assertion.get("bindings")
        classes = assertion.get("binding_ontology_classes")
        if not isinstance(roles, list) or not all(
            isinstance(item, str) for item in roles
        ):
            raise ModelDataError(f"{context}: predicate roles must be strings")
        if not isinstance(bindings, dict) or not isinstance(classes, dict):
            raise ModelDataError(f"{context}: predicate bindings must be objects")
        if set(roles) != set(bindings) or set(roles) != set(classes):
            raise ModelDataError(f"{context}: predicate binding roles disagree")
        normalized = {
            "assertion_id": _required_string(assertion, "assertion_id", context),
            "predicate_id": predicate_id,
            "tuple_id": tuple_id,
            "roles": tuple(roles),
            "bindings": tuple(str(bindings[role]) for role in roles),
            "classes": tuple(str(classes[role]) for role in roles),
        }
        previous = metadata.get(key)
        if previous is not None and previous != normalized:
            raise ModelDataError(f"{context}: predicate tuple metadata drift: {key}")
        metadata[key] = normalized
        values.setdefault(
            key,
            np.full(FORMAL_TICK_COUNT, INACTIVE_PREDICATE_CODE, dtype=np.uint8),
        )
        for role, entity_id, ontology_class_id in zip(
            normalized["roles"], normalized["bindings"], normalized["classes"]
        ):
            _entity(entities, entity_id).update(
                ontology_class_id=ontology_class_id,
                subject_kind=role,
                source_kind="predicate_binding",
            )
        return key

    for assertion in base.get("initial_assertions", ()):
        if not isinstance(assertion, Mapping):
            raise ModelDataError(f"{context}: base assertion must be an object")
        key = register(assertion)
        if key in current:
            raise ModelDataError(f"{context}: duplicate base predicate tuple: {key}")
        current[key] = dict(assertion)
    _write_predicate_tick(current, values, 0)

    delta_ticks: list[int] = []
    for delta in sorted(deltas, key=lambda row: int(row["tick"])):
        tick = int(delta["tick"])
        delta_ticks.append(tick)
        for operation in delta.get("operations", ()):
            if not isinstance(operation, Mapping):
                raise ModelDataError(f"{context}: delta operation must be an object")
            name = operation.get("operation")
            if name == "add_predicate_assertion":
                assertion = operation.get("assertion")
                if not isinstance(assertion, Mapping):
                    raise ModelDataError(f"{context}: add lacks assertion")
                key = register(assertion)
                if key in current:
                    raise ModelDataError(f"{context}: duplicate predicate add: {key}")
                current[key] = dict(assertion)
            elif name == "remove_predicate_assertion":
                key = (
                    _required_string(operation, "predicate_id", context),
                    _required_string(operation, "tuple_id", context),
                )
                if key not in current:
                    raise ModelDataError(
                        f"{context}: predicate remove is absent: {key}"
                    )
                del current[key]
            elif name == "set_predicate_value":
                key = (
                    _required_string(operation, "predicate_id", context),
                    _required_string(operation, "tuple_id", context),
                )
                assertion = current.get(key)
                if assertion is None:
                    raise ModelDataError(f"{context}: predicate set is absent: {key}")
                assertion["value"] = _required_string(operation, "to_value", context)
            elif name == "refresh_predicate_evidence":
                key = (
                    _required_string(operation, "predicate_id", context),
                    _required_string(operation, "tuple_id", context),
                )
                assertion = current.get(key)
                if assertion is None or assertion["value"] != operation.get("value"):
                    raise ModelDataError(
                        f"{context}: predicate evidence refresh mismatch: {key}"
                    )
            else:
                raise ModelDataError(f"{context}: unknown predicate operation: {name}")
        _write_predicate_tick(current, values, tick)
    if tuple(delta_ticks) != FORMAL_TICKS[1:]:
        raise ModelDataError(f"{context}: predicate delta ticks are incomplete")

    replay_state = {
        (str(row["predicate_id"]), str(row["tuple_id"])): str(row["value"])
        for row in final_replay["assertions"]
    }
    streamed_state = {key: str(row["value"]) for key, row in current.items()}
    if streamed_state != replay_state:
        raise ModelDataError(
            f"{context}: streamed predicate replay diverges from authority"
        )

    ordered_keys = sorted(values, key=lambda key: (key[0], key[1]))
    tensor = np.stack([values[key] for key in ordered_keys], axis=1)
    rows: list[dict[str, Any]] = []
    for slot_index, key in enumerate(ordered_keys):
        item = metadata[key]
        column = values[key]
        active = np.flatnonzero(column != INACTIVE_PREDICATE_CODE)
        counts = Counter(column.tolist())
        rows.append(
            {
                "episode_id": str(base["episode_id"]),
                "slot_id": item["tuple_id"],
                "slot_index": slot_index,
                "assertion_id": item["assertion_id"],
                "predicate_id": item["predicate_id"],
                "binding_roles": list(item["roles"]),
                "binding_entity_ids": list(item["bindings"]),
                "binding_ontology_classes": list(item["classes"]),
                "first_active_tick": FORMAL_TICKS[int(active[0])],
                "last_active_tick": FORMAL_TICKS[int(active[-1])],
                "candidate_active_count": int(active.size),
                "false_count": counts[PREDICATE_TRUTH_CODES["false"]],
                "true_count": counts[PREDICATE_TRUTH_CODES["true"]],
                "unknown_count": counts[PREDICATE_TRUTH_CODES["unknown"]],
                "out_of_scope_count": counts[PREDICATE_TRUTH_CODES["out_of_scope"]],
            }
        )
    return rows, tensor


def _write_predicate_tick(
    current: Mapping[tuple[str, str], Mapping[str, Any]],
    values: Mapping[tuple[str, str], np.ndarray],
    tick: int,
) -> None:
    tick_index = TICK_TO_INDEX[tick]
    for key, assertion in current.items():
        truth = assertion.get("value")
        code = PREDICATE_TRUTH_CODES.get(str(truth))
        if code is None:
            raise ModelDataError(f"predicate {key} has invalid truth value: {truth}")
        values[key][tick_index] = code


def _finalize_entities(
    episode_id: str,
    entities: Mapping[str, _EntityAccumulator],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ordered = sorted(entities.values(), key=lambda item: item.entity_id)
    rows: list[dict[str, Any]] = []
    codes: dict[int, str] = {}
    index: dict[str, int] = {}
    for entity_index, item in enumerate(ordered):
        code = stable_int63("graph_entity", item.entity_id)
        previous = codes.get(code)
        if previous is not None and previous != item.entity_id:
            raise ModelDataError(
                f"entity code collision between {previous!r} and {item.entity_id!r}"
            )
        codes[code] = item.entity_id
        index[item.entity_id] = entity_index
        rows.append(
            {
                "episode_id": episode_id,
                "entity_index": entity_index,
                "entity_code": code,
                "entity_id": item.entity_id,
                "scope_types": sorted(item.scope_types),
                "scope_subtypes": sorted(item.scope_subtypes),
                "ontology_class_ids": sorted(item.ontology_class_ids),
                "subject_categories": sorted(item.subject_categories),
                "subject_kinds": sorted(item.subject_kinds),
                "source_kinds": sorted(item.source_kinds),
                "observer_capable": item.observer_capable,
            }
        )

    # Reference values are encoded only after the complete entity registry exists.
    # The caller replaces the temporary raw reference payload before serialization.
    return rows, index


def finalize_reference_tensor(
    projection: EpisodeProjection,
) -> None:
    """Replace the private raw-reference accumulator with exact entity indices."""

    raw = projection.tensor_payload.get("reference_values")
    if not isinstance(raw, list):
        raise ModelDataError("projection lacks raw reference values")
    entity_index = {
        row["entity_id"]: row["entity_index"] for row in projection.entities
    }
    width = int(projection.tensor_payload["reference_width"])
    values = np.full((FORMAL_TICK_COUNT, width), -1, dtype="<i4")
    for slot, storage_index, raw_values in raw:
        for tick_index, value in enumerate(raw_values):
            if value is _UNASSIGNED or is_missing_state_value(value):
                continue
            values[tick_index, storage_index] = entity_index[str(value)]
    projection.tensor_payload["reference_values"] = values.tobytes(order="C")


def _register_value_reference(
    field_name: str,
    value: Any,
    *,
    value_kind: str,
    all_values: Mapping[str, Any],
    entities: dict[str, _EntityAccumulator],
) -> None:
    if is_missing_state_value(value):
        return
    if value_kind == "reference":
        ontology_class = None
        if field_name.endswith("_id") and not field_name.endswith("_ontology_class_id"):
            ontology_class = all_values.get(
                field_name[: -len("_id")] + "_ontology_class_id"
            )
            if is_missing_state_value(ontology_class):
                ontology_class = None
        _entity(entities, str(value)).update(
            ontology_class_id=ontology_class,
            source_kind=(
                "ontology_reference"
                if field_name.endswith("_ontology_class_id")
                else "state_reference"
            ),
        )
    elif value_kind == "multi_reference":
        for entity_id in value:
            _entity(entities, str(entity_id)).update(source_kind="state_reference")


def _register_category(
    categories: dict[int, tuple[str, set[str]]],
    code: int,
    value: str,
    domain: str,
) -> None:
    existing = categories.get(code)
    if existing is None:
        categories[code] = (value, {domain})
        return
    if existing[0] != value:
        raise ModelDataError(
            f"category code collision between {existing[0]!r} and {value!r}"
        )
    existing[1].add(domain)


def _entity(
    entities: dict[str, _EntityAccumulator], entity_id: str
) -> _EntityAccumulator:
    if not isinstance(entity_id, str) or not entity_id:
        raise ModelDataError("graph entity id must be a non-empty string")
    item = entities.get(entity_id)
    if item is None:
        item = _EntityAccumulator(entity_id=entity_id)
        entities[entity_id] = item
    return item


def _add_string(target: set[str], value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ModelDataError(f"expected a non-empty string, found {value!r}")
    target.add(value)


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ModelDataError(f"{context}: {key} must be a non-empty string")
    return item


def _load_object_bytes(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        value = orjson.loads(payload)
    except orjson.JSONDecodeError as exc:
        raise ModelDataError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ModelDataError(f"{path}: root must be an object")
    return value


def _load_jsonl_with_bytes(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    with open_artifact_binary(path) as stream:
        payload = stream.read()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        try:
            value = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise ModelDataError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ModelDataError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return payload, rows


def _validate_objective_manifest(manifest: Mapping[str, Any], episode_id: str) -> None:
    expected = {
        "schema_name": "objective_semantic_episode_manifest",
        "schema_version": "4.0.0",
        "episode_id": episode_id,
        "closure_status": "PASS",
        "forbidden_inputs_used": [],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ModelDataError(
                f"{episode_id}: objective manifest {key} is {manifest.get(key)!r}, "
                f"expected {value!r}"
            )
    counts = manifest.get("record_counts")
    if not isinstance(counts, Mapping):
        raise ModelDataError(f"{episode_id}: objective manifest lacks record counts")
    if counts.get("world_truth_graph_bases") != 1:
        raise ModelDataError(f"{episode_id}: objective manifest lacks one world base")
    if counts.get("world_truth_graph_deltas") != FORMAL_TICK_COUNT - 1:
        raise ModelDataError(f"{episode_id}: objective manifest delta count is invalid")


def _validate_time_contract(manifest: Mapping[str, Any], episode_id: str) -> None:
    if (
        manifest.get("episode_id") != episode_id
        or manifest.get("duration_ticks") != 900
    ):
        raise ModelDataError(f"{episode_id}: render manifest time contract mismatch")
    time_range = manifest.get("time_range")
    if time_range != {
        "tick_start": 0,
        "tick_end": 900,
        "sim_time_start": 0.0,
        "sim_time_end": 90.0,
    }:
        raise ModelDataError(f"{episode_id}: unexpected render time range")
