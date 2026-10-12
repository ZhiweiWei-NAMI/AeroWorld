"""Atomic Parquet builder for the AERO unified graph dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from .contract import (
    ALLOWED_HISTORY_LENGTHS,
    ALLOWED_HORIZONS,
    DATASET_SCHEMA_NAME,
    DATASET_SCHEMA_VERSION,
    FORMAL_FRAME_SECONDS,
    FORMAL_TICK_COUNT,
    FORMAL_TICKS,
    MATERIALIZED_WINDOWS,
    SIMULATION_TICK_SECONDS,
    UNASSIGNED_SPLIT,
    ModelDataError,
    canonical_json,
    parse_episode_id,
    sha256_file,
    stable_identifier,
)
from .projection import EpisodeProjection, project_episode


DATASET_FILES = (
    "episodes.parquet",
    "entities.parquet",
    "state_slots.parquet",
    "predicate_slots.parquet",
    "ragged_state.parquet",
)


EPISODE_SCHEMA = pa.schema(
    [
        ("schema_name", pa.string()),
        ("schema_version", pa.string()),
        ("episode_id", pa.string()),
        ("profile_id", pa.string()),
        ("seed_label", pa.string()),
        ("split", pa.string()),
        ("capture_boundary_id", pa.string()),
        ("row_group_index", pa.int32()),
        ("formal_tick_count", pa.int16()),
        ("simulation_tick_seconds", pa.float64()),
        ("formal_frame_seconds", pa.float64()),
        ("entity_count", pa.int32()),
        ("state_slot_count", pa.int32()),
        ("continuous_width", pa.int32()),
        ("boolean_width", pa.int32()),
        ("categorical_width", pa.int32()),
        ("reference_width", pa.int32()),
        ("ragged_slot_count", pa.int32()),
        ("predicate_slot_count", pa.int32()),
        ("l0_row_count", pa.int64()),
        ("l0_assignment_count", pa.int64()),
        ("l0_valid_assignment_count", pa.int64()),
        ("trajectory_formal_mobile_row_count", pa.int64()),
        ("trajectory_formal_vehicle_row_count", pa.int64()),
        ("trajectory_formal_pedestrian_row_count", pa.int64()),
        ("trajectory_vehicle_runtime_row_count", pa.int64()),
        ("trajectory_vehicle_without_runtime_context_count", pa.int64()),
        ("trajectory_assignment_count", pa.int64()),
        ("trajectory_valid_assignment_count", pa.int64()),
        ("ragged_record_count", pa.int64()),
        ("predicate_false_count", pa.int64()),
        ("predicate_true_count", pa.int64()),
        ("predicate_unknown_count", pa.int64()),
        ("predicate_out_of_scope_count", pa.int64()),
        ("predicate_inactive_count", pa.int64()),
        ("state_slot_counts_by_kind_json", pa.string()),
        ("source_manifest_path", pa.string()),
        ("source_manifest_sha256", pa.string()),
        ("source_l0_path", pa.string()),
        ("source_l0_sha256", pa.string()),
        ("source_render_manifest_path", pa.string()),
        ("source_render_manifest_sha256", pa.string()),
        ("source_trajectory_path", pa.string()),
        ("source_trajectory_sha256", pa.string()),
        ("source_world_base_path", pa.string()),
        ("source_world_base_sha256", pa.string()),
        ("source_world_delta_path", pa.string()),
        ("source_world_delta_sha256", pa.string()),
        ("source_context_input_digest", pa.string()),
        ("source_domain_input_digest", pa.string()),
        ("source_l0_state_digest", pa.string()),
        ("source_world_base_digest", pa.string()),
        ("continuous_values", pa.binary()),
        ("continuous_scope", pa.binary()),
        ("continuous_valid", pa.binary()),
        ("boolean_values", pa.binary()),
        ("boolean_scope", pa.binary()),
        ("boolean_valid", pa.binary()),
        ("categorical_values", pa.binary()),
        ("categorical_scope", pa.binary()),
        ("categorical_valid", pa.binary()),
        ("reference_values", pa.binary()),
        ("reference_scope", pa.binary()),
        ("reference_valid", pa.binary()),
        ("predicate_truth_codes", pa.binary()),
    ],
    metadata={
        b"schema_name": DATASET_SCHEMA_NAME.encode(),
        b"schema_version": DATASET_SCHEMA_VERSION.encode(),
    },
)

ENTITY_SCHEMA = pa.schema(
    [
        ("episode_id", pa.string()),
        ("entity_index", pa.int32()),
        ("entity_code", pa.int64()),
        ("entity_id", pa.string()),
        ("scope_types", pa.list_(pa.string())),
        ("scope_subtypes", pa.list_(pa.string())),
        ("ontology_class_ids", pa.list_(pa.string())),
        ("subject_categories", pa.list_(pa.string())),
        ("subject_kinds", pa.list_(pa.string())),
        ("source_kinds", pa.list_(pa.string())),
        ("observer_capable", pa.bool_()),
    ]
)

STATE_SLOT_SCHEMA = pa.schema(
    [
        ("episode_id", pa.string()),
        ("slot_id", pa.string()),
        ("slot_index", pa.int32()),
        ("subject_id", pa.string()),
        ("subject_entity_index", pa.int32()),
        ("observation_family", pa.string()),
        ("field_name", pa.string()),
        ("value_kind", pa.string()),
        ("storage_index", pa.int32()),
        ("width", pa.int16()),
        ("unit", pa.string()),
        ("assigned_count", pa.int16()),
        ("valid_count", pa.int16()),
        ("distinct_value_count", pa.int16()),
        ("value_temporal_mode", pa.string()),
        ("scope_temporal_mode", pa.string()),
        ("source_classes_json", pa.string()),
        ("model_ids_json", pa.string()),
        ("model_versions_json", pa.string()),
        ("rule_ids_json", pa.string()),
        ("rule_versions_json", pa.string()),
        ("input_digests_json", pa.string()),
        ("parameter_digests_json", pa.string()),
    ]
)

PREDICATE_SLOT_SCHEMA = pa.schema(
    [
        ("episode_id", pa.string()),
        ("slot_id", pa.string()),
        ("slot_index", pa.int32()),
        ("assertion_id", pa.string()),
        ("predicate_id", pa.string()),
        ("binding_roles", pa.list_(pa.string())),
        ("binding_entity_indices", pa.list_(pa.int32())),
        ("binding_ontology_classes", pa.list_(pa.string())),
        ("first_active_tick", pa.int16()),
        ("last_active_tick", pa.int16()),
        ("candidate_active_count", pa.int16()),
        ("false_count", pa.int16()),
        ("true_count", pa.int16()),
        ("unknown_count", pa.int16()),
        ("out_of_scope_count", pa.int16()),
    ]
)

RAGGED_SCHEMA = pa.schema(
    [
        ("episode_id", pa.string()),
        ("tick", pa.int16()),
        ("slot_id", pa.string()),
        ("valid", pa.bool_()),
        ("value_json", pa.string()),
    ]
)


@dataclass(frozen=True)
class BuildResult:
    output_root: Path
    manifest: Mapping[str, Any]


class _ParquetWriters:
    def __init__(self, root: Path) -> None:
        options = {
            "compression": "zstd",
            "compression_level": 6,
            "use_dictionary": True,
            "write_statistics": True,
            "data_page_version": "2.0",
        }
        self.episodes = pq.ParquetWriter(
            root / "episodes.parquet", EPISODE_SCHEMA, **options
        )
        self.entities = pq.ParquetWriter(
            root / "entities.parquet", ENTITY_SCHEMA, **options
        )
        self.state_slots = pq.ParquetWriter(
            root / "state_slots.parquet", STATE_SLOT_SCHEMA, **options
        )
        self.predicate_slots = pq.ParquetWriter(
            root / "predicate_slots.parquet", PREDICATE_SLOT_SCHEMA, **options
        )
        self.ragged = pq.ParquetWriter(
            root / "ragged_state.parquet", RAGGED_SCHEMA, **options
        )

    def write(self, projection: EpisodeProjection, row_group_index: int) -> None:
        episode = _episode_parquet_row(projection, row_group_index)
        self.episodes.write_table(
            pa.Table.from_pylist([episode], schema=EPISODE_SCHEMA)
        )
        self.entities.write_table(
            pa.Table.from_pylist(projection.entities, schema=ENTITY_SCHEMA)
        )
        self.state_slots.write_table(
            pa.Table.from_pylist(projection.state_slots, schema=STATE_SLOT_SCHEMA)
        )
        self.predicate_slots.write_table(
            pa.Table.from_pylist(
                projection.predicate_slots, schema=PREDICATE_SLOT_SCHEMA
            )
        )
        self.ragged.write_table(
            pa.Table.from_pylist(projection.ragged_state, schema=RAGGED_SCHEMA)
        )

    def close(self) -> None:
        self.episodes.close()
        self.entities.close()
        self.state_slots.close()
        self.predicate_slots.close()
        self.ragged.close()


def _episode_parquet_row(
    projection: EpisodeProjection,
    row_group_index: int,
) -> dict[str, Any]:
    return {
        **projection.episode,
        "row_group_index": row_group_index,
        **{
            key: projection.audit[key]
            for key in (
                "l0_row_count",
                "l0_assignment_count",
                "l0_valid_assignment_count",
                "trajectory_formal_mobile_row_count",
                "trajectory_formal_vehicle_row_count",
                "trajectory_formal_pedestrian_row_count",
                "trajectory_vehicle_runtime_row_count",
                "trajectory_vehicle_without_runtime_context_count",
                "trajectory_assignment_count",
                "trajectory_valid_assignment_count",
                "ragged_record_count",
                "predicate_false_count",
                "predicate_true_count",
                "predicate_unknown_count",
                "predicate_out_of_scope_count",
                "predicate_inactive_count",
            )
        },
        "state_slot_counts_by_kind_json": canonical_json(
            projection.audit["state_slot_counts_by_kind"]
        ),
        **{
            key: value
            for key, value in projection.tensor_payload.items()
            if key
            not in {
                "episode_id",
                "continuous_width",
                "boolean_width",
                "categorical_width",
                "reference_width",
            }
        },
    }


def discover_formal_episodes(source_root: Path) -> tuple[list[Path], dict[str, str]]:
    source_root = source_root.resolve()
    roots = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir()
        and (path / "manifest.json").is_file()
        and (path / "l0_predicate_state.jsonl").is_file()
        and (path / "world_truth_graph_base.json").is_file()
        and (path / "world_truth_graph_deltas.jsonl").is_file()
    )
    profiles: dict[str, set[str]] = defaultdict(set)
    for root in roots:
        profile_id, seed = parse_episode_id(root.name)
        profiles[profile_id].add(seed)
    if len(roots) != 210 or len(profiles) != 70:
        raise ModelDataError(
            f"formal source requires 210 episodes and 70 profiles; found "
            f"{len(roots)} episodes and {len(profiles)} profiles"
        )
    invalid = {
        profile: sorted(seeds)
        for profile, seeds in profiles.items()
        if seeds != {"00", "01", "02"}
    }
    if invalid:
        raise ModelDataError(f"formal profiles lack the exact three seeds: {invalid}")
    splits = {profile_id: UNASSIGNED_SPLIT for profile_id in profiles}
    return roots, splits


def build_dataset(
    source_root: Path,
    output_root: Path,
    *,
    episode_ids: Sequence[str] | None = None,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> BuildResult:
    """Build an atomic, deterministic projection.

    The objective source is read-only. Existing output directories are rejected.
    """

    if workers < 1:
        raise ModelDataError("workers must be positive")
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    all_roots, profile_splits = discover_formal_episodes(source_root)
    by_id = {path.name: path for path in all_roots}
    selected_ids = sorted(episode_ids) if episode_ids else sorted(by_id)
    if not selected_ids:
        raise ModelDataError("at least one episode must be selected")
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ModelDataError(f"unknown episode ids: {unknown}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ModelDataError("episode selection contains duplicates")
    selected = [by_id[episode_id] for episode_id in selected_ids]

    if output_root.exists():
        raise ModelDataError(f"output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging.exists():
        raise ModelDataError(f"staging path already exists: {staging}")
    staging.mkdir()

    writers: _ParquetWriters | None = None
    try:
        writers = _ParquetWriters(staging)
        aggregate = Counter()
        state_kinds = Counter()
        categories: dict[int, tuple[str, set[str]]] = {}
        jobs = [
            (
                path,
                profile_splits[parse_episode_id(path.name)[0]],
                source_root,
            )
            for path in selected
        ]
        for index, projection in enumerate(
            _ordered_projections(jobs, workers=workers), start=0
        ):
            writers.write(projection, index)
            _merge_categories(categories, projection.categories)
            for key, value in projection.audit.items():
                if isinstance(value, int):
                    aggregate[key] += value
            state_kinds.update(projection.audit["state_slot_counts_by_kind"])
            if progress is not None:
                progress(index + 1, len(selected), projection.episode["episode_id"])
        writers.close()
        writers = None

        category_document = [
            {
                "category_code": code,
                "value": value,
                "domains": sorted(domains),
            }
            for code, (value, domains) in sorted(categories.items())
        ]
        artifact_manifest = {
            name: {
                "bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name),
            }
            for name in DATASET_FILES
        }
        split_profile_counts = Counter(profile_splits.values())
        split_episode_counts = Counter(
            profile_splits[parse_episode_id(episode_id)[0]]
            for episode_id in selected_ids
        )
        canonical_window_count = sum(
            FORMAL_TICK_COUNT - history - horizon + 1
            for history, horizon in MATERIALIZED_WINDOWS
        ) * len(selected_ids)
        manifest = {
            "schema_name": DATASET_SCHEMA_NAME,
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": stable_identifier(
                "aero_model_dataset",
                DATASET_SCHEMA_VERSION,
                selected_ids,
                [artifact_manifest[name]["sha256"] for name in DATASET_FILES],
            ),
            "build_mode": (
                "formal_full"
                if len(selected_ids) == len(all_roots)
                else "selected_canary"
            ),
            "source_root": str(source_root),
            "episode_count": len(selected_ids),
            "profile_count": len(
                {parse_episode_id(value)[0] for value in selected_ids}
            ),
            "formal_tick_contract": {
                "tick_start": FORMAL_TICKS[0],
                "tick_end": FORMAL_TICKS[-1],
                "tick_stride": FORMAL_TICKS[1] - FORMAL_TICKS[0],
                "frame_count_per_episode": FORMAL_TICK_COUNT,
                "simulation_tick_seconds": SIMULATION_TICK_SECONDS,
                "formal_frame_seconds": FORMAL_FRAME_SECONDS,
            },
            "split_contract": {
                "status": "unassigned",
                "embedded_split": False,
                "group_key": "profile_id",
                "seed_labels_per_profile": ["00", "01", "02"],
                "profile_counts": dict(sorted(split_profile_counts.items())),
                "episode_counts": dict(sorted(split_episode_counts.items())),
                "rule": (
                    "projection bytes carry only the literal 'unassigned'; an external "
                    "profile-grouped manifest is required before training"
                ),
            },
            "window_contract": {
                "allowed_history_lengths": sorted(ALLOWED_HISTORY_LENGTHS),
                "allowed_horizons": sorted(ALLOWED_HORIZONS),
                "canonical_report_pairs": [
                    list(value) for value in MATERIALIZED_WINDOWS
                ],
                "canonical_window_count": canonical_window_count,
                "storage": "derived from episode frame count; no redundant window file",
            },
            "observer_bundle_contract": {
                "status": "not_materialized_in_server_handoff",
                "supported_modalities": ["rgb", "lidar", "text"],
                "current_materialized_modalities": [],
                "structured_l0_is_supervision_not_sensor_tokens": True,
                "authored_event_trace_allowed_as_input": False,
                "attachment_key": [
                    "dataset_id",
                    "episode_id",
                    "tick",
                    "observer_uav_id",
                    "sensor_id",
                    "modality",
                ],
            },
            "oracle_object_observation_contract": {
                "status": "materialized",
                "semantics": "perfect upstream detection, localization, and tracking; not object-mounted GPS",
                "entity_scope_types": ["vehicle", "pedestrian"],
                "observation_family": "oracle_object_state",
                "fields": [
                    "entity_id",
                    "class",
                    "position_enu_m",
                    "velocity_enu_mps",
                    "yaw_deg",
                    "confidence",
                    "formal_tick_timestamp",
                ],
                "confidence": 1.0,
                "confidence_semantics": "constant by the perfect-upstream-tracker Oracle assumption; not a learned score",
                "history_role": "structured Oracle input at the matching tick",
                "target_role": "future state supervision at the matching tick",
                "presence_rule": "same-tick trajectory row exists",
                "absence_rule": "scope=false, valid=false, continuous value=NaN",
                "derived_mobile_pair_scope_rule": (
                    "every referenced external vehicle or pedestrian has a "
                    "same-tick structured position observation; UAV pose authority "
                    "comes from central telemetry"
                ),
                "future_fill_forbidden": True,
                "terminal_coordinate_fill_forbidden": True,
                "uav_coordinate_duplication_forbidden": True,
            },
            "projection_counts": {
                "formal_frame_count": len(selected_ids) * FORMAL_TICK_COUNT,
                "one_step_transition_count": len(selected_ids)
                * (FORMAL_TICK_COUNT - 1),
                "l0_row_count": aggregate["l0_row_count"],
                "l0_assignment_count": aggregate["l0_assignment_count"],
                "l0_valid_assignment_count": aggregate["l0_valid_assignment_count"],
                "trajectory_formal_mobile_row_count": aggregate[
                    "trajectory_formal_mobile_row_count"
                ],
                "trajectory_formal_vehicle_row_count": aggregate[
                    "trajectory_formal_vehicle_row_count"
                ],
                "trajectory_formal_pedestrian_row_count": aggregate[
                    "trajectory_formal_pedestrian_row_count"
                ],
                "trajectory_vehicle_runtime_row_count": aggregate[
                    "trajectory_vehicle_runtime_row_count"
                ],
                "trajectory_vehicle_without_runtime_context_count": aggregate[
                    "trajectory_vehicle_without_runtime_context_count"
                ],
                "trajectory_assignment_count": aggregate["trajectory_assignment_count"],
                "trajectory_valid_assignment_count": aggregate[
                    "trajectory_valid_assignment_count"
                ],
                "state_slot_count": aggregate["state_slot_count"],
                "state_slot_counts_by_kind": dict(sorted(state_kinds.items())),
                "ragged_record_count": aggregate["ragged_record_count"],
                "entity_count": aggregate["entity_count"],
                "predicate_slot_count": aggregate["predicate_slot_count"],
                "predicate_truth_counts": {
                    "false": aggregate["predicate_false_count"],
                    "true": aggregate["predicate_true_count"],
                    "unknown": aggregate["predicate_unknown_count"],
                    "out_of_scope": aggregate["predicate_out_of_scope_count"],
                    "inactive": aggregate["predicate_inactive_count"],
                },
                "categorical_vocabulary_size": len(category_document),
            },
            "representation_upper_bound": {
                "boolean_categorical_reference_predicate_fidelity": 1.0,
                "continuous_fidelity_to_parsed_float64_authority": 1.0,
                "ragged_value_fidelity": 1.0,
                "float32_training_cast_unit_roundoff_upper_bound": 2**-24,
                "formula": "abs(cast_error) <= 2^-24 * abs(x) for normal finite float32 x",
                "sensor_to_truth_upper_bound": "not_defined_until_real ObserverBundle features are attached",
            },
            "logical_schema": _schema_document(category_document),
            "artifacts": artifact_manifest,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_root)
        return BuildResult(output_root=output_root, manifest=manifest)
    except BaseException:
        if writers is not None:
            writers.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_dataset(
    dataset_root: Path,
    *,
    source_root: Path | None = None,
    deep: bool = False,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Validate physical files, graph contracts, and optional source round-trips."""

    from .dataset import AeroGraphStore

    if workers < 1:
        raise ModelDataError("workers must be positive")
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ModelDataError("dataset manifest must be an object")
    expected_contract = (DATASET_SCHEMA_NAME, DATASET_SCHEMA_VERSION)
    actual_contract = (manifest.get("schema_name"), manifest.get("schema_version"))
    if actual_contract != expected_contract:
        raise ModelDataError(
            f"dataset contract mismatch: {actual_contract} != {expected_contract}"
        )
    expected_files = {"manifest.json", *DATASET_FILES}
    actual_files = {path.name for path in dataset_root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ModelDataError(
            f"dataset file set mismatch: expected {sorted(expected_files)}, "
            f"found {sorted(actual_files)}"
        )
    for name in DATASET_FILES:
        artifact = manifest["artifacts"].get(name)
        if not isinstance(artifact, Mapping):
            raise ModelDataError(f"manifest lacks artifact metadata for {name}")
        path = dataset_root / name
        if path.stat().st_size != int(artifact["bytes"]):
            raise ModelDataError(f"artifact byte count mismatch: {name}")

    episode_columns = [
        "episode_id",
        "profile_id",
        "seed_label",
        "split",
        "row_group_index",
        "entity_count",
        "state_slot_count",
        "continuous_width",
        "boolean_width",
        "categorical_width",
        "reference_width",
        "predicate_slot_count",
        "l0_row_count",
        "l0_assignment_count",
        "l0_valid_assignment_count",
        "trajectory_formal_mobile_row_count",
        "trajectory_formal_vehicle_row_count",
        "trajectory_formal_pedestrian_row_count",
        "trajectory_vehicle_runtime_row_count",
        "trajectory_vehicle_without_runtime_context_count",
        "trajectory_assignment_count",
        "trajectory_valid_assignment_count",
        "ragged_record_count",
        "predicate_false_count",
        "predicate_true_count",
        "predicate_unknown_count",
        "predicate_out_of_scope_count",
        "predicate_inactive_count",
    ]
    episode_rows = pq.read_table(
        dataset_root / "episodes.parquet", columns=episode_columns
    ).to_pylist()
    episode_count = int(manifest["episode_count"])
    if len(episode_rows) != episode_count:
        raise ModelDataError("episode Parquet count disagrees with manifest")
    if [row["row_group_index"] for row in episode_rows] != list(range(episode_count)):
        raise ModelDataError("episode row-group indices are not contiguous")
    profiles: dict[str, set[str]] = defaultdict(set)
    profile_splits: dict[str, set[str]] = defaultdict(set)
    for row in episode_rows:
        profiles[str(row["profile_id"])].add(str(row["seed_label"]))
        profile_splits[str(row["profile_id"])].add(str(row["split"]))
    if any(len(values) != 1 for values in profile_splits.values()):
        raise ModelDataError("a profile leaks across dataset splits")
    if manifest["build_mode"] == "formal_full" and any(
        seeds != {"00", "01", "02"} for seeds in profiles.values()
    ):
        raise ModelDataError("a formal profile does not contain all three seeds")
    actual_profile_splits = Counter(
        next(iter(values)) for values in profile_splits.values()
    )
    split_contract = manifest.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise ModelDataError("dataset manifest lacks a split contract")
    if split_contract.get("status") != "unassigned" or split_contract.get(
        "embedded_split"
    ) is not False:
        raise ModelDataError("projection must not embed a train/validation/test split")
    if set(actual_profile_splits) != {UNASSIGNED_SPLIT}:
        raise ModelDataError("projection contains an assigned dataset split")
    expected_profile_splits = split_contract["profile_counts"]
    if (
        manifest["build_mode"] == "formal_full"
        and dict(sorted(actual_profile_splits.items())) != expected_profile_splits
    ):
        raise ModelDataError("profile split counts disagree with manifest")
    actual_episode_splits = Counter(str(row["split"]) for row in episode_rows)
    if set(actual_episode_splits) != {UNASSIGNED_SPLIT}:
        raise ModelDataError("projection contains an assigned episode split")
    expected_episode_splits = split_contract["episode_counts"]
    if dict(sorted(actual_episode_splits.items())) != expected_episode_splits:
        raise ModelDataError("episode split counts disagree with manifest")

    counts = manifest["projection_counts"]
    summed_fields = {
        "l0_row_count": "l0_row_count",
        "l0_assignment_count": "l0_assignment_count",
        "l0_valid_assignment_count": "l0_valid_assignment_count",
        "trajectory_formal_mobile_row_count": "trajectory_formal_mobile_row_count",
        "trajectory_formal_vehicle_row_count": "trajectory_formal_vehicle_row_count",
        "trajectory_formal_pedestrian_row_count": "trajectory_formal_pedestrian_row_count",
        "trajectory_vehicle_runtime_row_count": "trajectory_vehicle_runtime_row_count",
        "trajectory_vehicle_without_runtime_context_count": "trajectory_vehicle_without_runtime_context_count",
        "trajectory_assignment_count": "trajectory_assignment_count",
        "trajectory_valid_assignment_count": "trajectory_valid_assignment_count",
        "state_slot_count": "state_slot_count",
        "ragged_record_count": "ragged_record_count",
        "entity_count": "entity_count",
        "predicate_slot_count": "predicate_slot_count",
    }
    for manifest_key, column in summed_fields.items():
        if sum(int(row[column]) for row in episode_rows) != int(counts[manifest_key]):
            raise ModelDataError(f"aggregate count mismatch: {manifest_key}")
    predicate_columns = {
        "false": "predicate_false_count",
        "true": "predicate_true_count",
        "unknown": "predicate_unknown_count",
        "out_of_scope": "predicate_out_of_scope_count",
        "inactive": "predicate_inactive_count",
    }
    for truth_name, column in predicate_columns.items():
        actual = sum(int(row[column]) for row in episode_rows)
        if actual != int(counts["predicate_truth_counts"][truth_name]):
            raise ModelDataError(f"predicate count mismatch: {truth_name}")
    physical_rows = {
        "entities.parquet": sum(int(row["entity_count"]) for row in episode_rows),
        "state_slots.parquet": sum(
            int(row["state_slot_count"]) for row in episode_rows
        ),
        "predicate_slots.parquet": sum(
            int(row["predicate_slot_count"]) for row in episode_rows
        ),
        "ragged_state.parquet": sum(
            int(row["ragged_record_count"]) for row in episode_rows
        ),
    }
    for name, expected_rows in physical_rows.items():
        actual_rows = pq.ParquetFile(dataset_root / name).metadata.num_rows
        if actual_rows != expected_rows:
            raise ModelDataError(f"physical row count mismatch: {name}")

    store = AeroGraphStore(dataset_root, cache_size=1)
    vocabulary = {
        int(row["category_code"])
        for row in manifest["logical_schema"]["categorical_vocabulary"]
    }
    source_categories: dict[int, tuple[str, set[str]]] = {}
    resolved_source = (
        source_root.resolve()
        if source_root is not None
        else Path(str(manifest["source_root"])).resolve()
    )
    source_projections: Iterable[EpisodeProjection | None]
    if deep:
        jobs = [
            (
                resolved_source / str(row["episode_id"]),
                str(row["split"]),
                resolved_source,
            )
            for row in episode_rows
        ]
        source_projections = _ordered_projections(jobs, workers=workers)
    else:
        source_projections = (None for _row in episode_rows)
    for index, (row, source_projection) in enumerate(
        zip(episode_rows, source_projections), start=1
    ):
        episode_id = str(row["episode_id"])
        sequence = store.get_episode(episode_id)
        _validate_sequence(sequence, vocabulary)
        if deep:
            if source_projection is None:
                raise ModelDataError("deep validation projection is missing")
            _compare_projection(dataset_root, row, source_projection)
            _merge_categories(source_categories, source_projection.categories)
        if progress is not None:
            progress(index, episode_count, episode_id)
    if deep:
        expected_categories = [
            {
                "category_code": code,
                "value": value,
                "domains": sorted(domains),
            }
            for code, (value, domains) in sorted(source_categories.items())
        ]
        if expected_categories != manifest["logical_schema"]["categorical_vocabulary"]:
            raise ModelDataError("categorical vocabulary does not round-trip to source")
    return {
        "status": "PASS",
        "dataset_id": manifest["dataset_id"],
        "episode_count": episode_count,
        "profile_count": len(profiles),
        "formal_frame_count": int(counts["formal_frame_count"]),
        "one_step_transition_count": int(counts["one_step_transition_count"]),
        "deep_source_roundtrip": deep,
    }


def _validate_sequence(sequence: Any, vocabulary: set[int]) -> None:
    metadata = sequence.metadata
    episode_id = sequence.episode_id
    if [row["entity_index"] for row in sequence.entities] != list(
        range(len(sequence.entities))
    ):
        raise ModelDataError(f"{episode_id}: entity indices are not contiguous")
    if [row["slot_index"] for row in sequence.state_slots] != list(
        range(len(sequence.state_slots))
    ):
        raise ModelDataError(f"{episode_id}: state slot indices are not contiguous")
    if [row["slot_index"] for row in sequence.predicate_slots] != list(
        range(len(sequence.predicate_slots))
    ):
        raise ModelDataError(f"{episode_id}: predicate slot indices are not contiguous")
    arrays = {
        "continuous": (
            sequence.continuous,
            sequence.continuous_scope,
            sequence.continuous_valid,
        ),
        "boolean": (sequence.boolean, sequence.boolean_scope, sequence.boolean_valid),
        "categorical": (
            sequence.categorical,
            sequence.categorical_scope,
            sequence.categorical_valid,
        ),
        "reference": (
            sequence.reference,
            sequence.reference_scope,
            sequence.reference_valid,
        ),
    }
    for kind, (values, scope, valid) in arrays.items():
        if values.shape != scope.shape or values.shape != valid.shape:
            raise ModelDataError(f"{episode_id}: {kind} tensor shapes disagree")
        if np.any(valid & ~scope):
            raise ModelDataError(f"{episode_id}: {kind} valid mask escapes scope")
    if np.any(~np.isfinite(sequence.continuous[sequence.continuous_valid])):
        raise ModelDataError(f"{episode_id}: valid continuous state is non-finite")
    if np.any(~np.isnan(sequence.continuous[~sequence.continuous_valid])):
        raise ModelDataError(f"{episode_id}: invalid continuous state is not NaN")
    if np.any(sequence.categorical[~sequence.categorical_valid] != 0):
        raise ModelDataError(f"{episode_id}: invalid categorical state is not zero")
    used_categories = set(
        map(int, sequence.categorical[sequence.categorical_valid].tolist())
    )
    if not used_categories <= vocabulary:
        raise ModelDataError(f"{episode_id}: categorical tensor escapes vocabulary")
    if np.any(sequence.reference[~sequence.reference_valid] != -1):
        raise ModelDataError(f"{episode_id}: invalid reference state is not -1")
    references = sequence.reference[sequence.reference_valid]
    if references.size and (
        int(references.min()) < 0 or int(references.max()) >= len(sequence.entities)
    ):
        raise ModelDataError(f"{episode_id}: reference tensor escapes entity registry")
    _validate_slot_storage(sequence.state_slots, metadata)
    topology = sequence.static_topology()
    if topology["edge_index"].shape[0] != 2:
        raise ModelDataError(f"{episode_id}: topology edge index is malformed")


def _validate_slot_storage(
    state_slots: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> None:
    occupied: dict[str, list[int]] = defaultdict(list)
    for row in state_slots:
        kind = str(row["value_kind"])
        storage_index = int(row["storage_index"])
        width = int(row["width"])
        if kind in {"multi_reference", "ragged_continuous"}:
            if storage_index != -1:
                raise ModelDataError("ragged slot unexpectedly occupies dense storage")
            continue
        occupied[kind].extend(range(storage_index, storage_index + width))
    for kind in ("continuous", "boolean", "categorical", "reference"):
        expected_width = int(metadata[f"{kind}_width"])
        if sorted(occupied[kind]) != list(range(expected_width)):
            raise ModelDataError(f"{kind} storage offsets are not a complete partition")


def _compare_projection(
    dataset_root: Path,
    episode_row: Mapping[str, Any],
    source: EpisodeProjection,
) -> None:
    row_group = int(episode_row["row_group_index"])
    actual_episode_rows = (
        pq.ParquetFile(dataset_root / "episodes.parquet")
        .read_row_group(row_group)
        .to_pylist()
    )
    if actual_episode_rows != [_episode_parquet_row(source, row_group)]:
        raise ModelDataError(
            f"{source.episode['episode_id']}: source round-trip differs in episodes.parquet"
        )
    comparisons = (
        ("entities.parquet", source.entities),
        ("state_slots.parquet", source.state_slots),
        ("predicate_slots.parquet", source.predicate_slots),
    )
    for filename, expected in comparisons:
        actual = (
            pq.ParquetFile(dataset_root / filename)
            .read_row_group(row_group)
            .to_pylist()
        )
        if actual != expected:
            raise ModelDataError(
                f"{source.episode['episode_id']}: source round-trip differs in {filename}"
            )
    actual_ragged = pq.read_table(
        dataset_root / "ragged_state.parquet",
        filters=[("episode_id", "=", source.episode["episode_id"])],
    ).to_pylist()
    if actual_ragged != source.ragged_state:
        raise ModelDataError(
            f"{source.episode['episode_id']}: source round-trip differs in ragged state"
        )
    tensor_columns = list(source.tensor_payload)
    actual_tensor_table = pq.ParquetFile(
        dataset_root / "episodes.parquet"
    ).read_row_group(row_group, columns=tensor_columns)
    actual_tensor = {
        name: actual_tensor_table[name][0].as_py()
        for name in actual_tensor_table.column_names
    }
    if actual_tensor != source.tensor_payload:
        raise ModelDataError(
            f"{source.episode['episode_id']}: source round-trip differs in tensors"
        )


def _ordered_projections(
    jobs: Sequence[tuple[Path, str, Path]],
    *,
    workers: int,
) -> Iterable[EpisodeProjection]:
    if workers == 1:
        for job in jobs:
            yield _project_job(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: dict[int, Future[EpisodeProjection]] = {}
        next_submit = 0
        for next_yield in range(len(jobs)):
            while next_submit < len(jobs) and next_submit < next_yield + workers:
                futures[next_submit] = executor.submit(_project_job, jobs[next_submit])
                next_submit += 1
            yield futures.pop(next_yield).result()


def _project_job(job: tuple[Path, str, Path]) -> EpisodeProjection:
    path, split, source_root = job
    if split != UNASSIGNED_SPLIT:
        raise ModelDataError("projection jobs must remain split-unassigned")
    return project_episode(path, source_root=source_root)


def _merge_categories(
    target: dict[int, tuple[str, set[str]]],
    source: Mapping[int, tuple[str, set[str]]],
) -> None:
    for code, (value, domains) in source.items():
        current = target.get(code)
        if current is None:
            target[code] = (value, set(domains))
            continue
        if current[0] != value:
            raise ModelDataError(
                f"category code collision between {current[0]!r} and {value!r}"
            )
        current[1].update(domains)


def _schema_document(categories: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_name": DATASET_SCHEMA_NAME,
        "schema_version": DATASET_SCHEMA_VERSION,
        "graph_node_partitions": [
            "entity",
            "state_slot",
            "predicate_slot",
            "observer_bundle",
        ],
        "implicit_typed_edges": {
            "has_state": "state_slots.subject_entity_index -> entities.entity_index",
            "predicate_argument": "predicate_slots.binding_entity_indices with binding_roles",
            "state_reference": "reference tensor value -> entities.entity_index",
            "produces": "observer_uav_id -> ObserverBundle",
            "observes_or_supports": "ObserverBundle associations -> entity/predicate slots",
        },
        "state_value_kinds": {
            "continuous": "float64 values plus independent scope and valid bit masks",
            "boolean": "bit-packed values plus independent scope and valid bit masks",
            "categorical": "verified collision-free int63 codes plus vocabulary",
            "reference": "episode-local entity_index plus scope and valid masks",
            "multi_reference": "canonical JSON in ragged_state.parquet",
            "ragged_continuous": "canonical JSON in ragged_state.parquet",
        },
        "predicate_truth_codes": {
            "false": 0,
            "true": 1,
            "unknown": 2,
            "out_of_scope": 3,
            "inactive_union_slot": 4,
        },
        "predicate_training_masks": {
            "true": {"activation": 1, "supervision": 1, "scope": 1},
            "false": {"activation": 0, "supervision": 1, "scope": 1},
            "unknown": {"activation": 0, "supervision": 0, "scope": 1},
            "out_of_scope": {"activation": 0, "supervision": 0, "scope": 0},
            "inactive_union_slot": {
                "activation": 0,
                "supervision": 0,
                "scope": 0,
            },
        },
        "edit_codes": {
            "retain": 0,
            "birth": 1,
            "death": 2,
            "value_change": 3,
        },
        "observer_bundle_required_fields": [
            "dataset_id",
            "episode_id",
            "tick",
            "observer_uav_id",
            "sensor_id",
            "modality",
            "feature_tensor",
            "token_or_point_coordinates",
            "sensor_intrinsics",
            "sensor_extrinsics",
            "presence_mask",
            "encoder_name",
            "encoder_weight_hash",
            "preprocess_hash",
            "source_provenance",
        ],
        "categorical_vocabulary": categories,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    )
