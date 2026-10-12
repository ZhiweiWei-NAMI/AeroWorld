"""Executable ROI contracts for targeted truth regeneration.

The formal capture-filtered corpus remains the authority for every preserved
ROI.  A profile may diverge only when it appears in the explicit minimal
rebuild contract and its recorded formal baseline still matches all three
release seeds exactly.  No spatial fallback or automatic grid reassignment is
allowed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


RELEASE_SEEDS = (0, 1, 2)
CAPTURE_STEP_TICKS = 5
EXPECTED_TICK_START = 0
EXPECTED_TICK_END = 900
EXPECTED_TRUTH_FRAME_COUNT = EXPECTED_TICK_END - EXPECTED_TICK_START + 1
EXPECTED_CAPTURE_FRAME_COUNT = (
    (EXPECTED_TICK_END - EXPECTED_TICK_START) // CAPTURE_STEP_TICKS + 1
)
DEFAULT_SCENARIOS_ROOT = Path(__file__).resolve().parents[1] / "scenarios"
DEFAULT_REBUILD_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "roi_rebuild_contracts.json"
)
TARGET_RECAPTURE_SCENARIO_IDS: tuple[str, ...] = (
    "L2-1_v2",
    "L2-3_v1",
    "L2-3_v2",
    "L2-4_v1",
    "L2-5_v1",
    "L3-1_v1",
    "L4-1_v2",
    "L4-3_v1",
    "L4-3_v2",
    "L4-3_v3",
    "L4-4_v1",
    "L4-4_v2",
    "L4-5_v1",
    "L4-5_v2",
    "L4-5_v3",
    "L4-6_v1",
    "L4-6_v2",
    "L4-7_v1",
    "L4-7_v2",
    "L4-8_v1",
    "L4-8_v2",
    "L4-9_v1",
    "L4-9_v2",
    "L4-10_v1",
    "L4-10_v2",
    "L4-11_v1",
    "L4-11_v2",
    "L5-1_v1",
    "L5-1_v2",
    "L5-1_v3",
    "X1_rain_to_c2loss_to_forced_landing",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _float_list(
    value: Any,
    *,
    length: int,
    context: str,
) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context}: expected a sequence")
    if len(value) != length:
        raise ValueError(f"{context}: expected {length} values, got {len(value)}")
    result = [float(item) for item in value]
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise ValueError(f"{context}: values must be finite")
    return result


def polygon_from_runtime_crop(
    bbox_enu_m: Sequence[float],
    padding_m: float,
) -> tuple[tuple[float, float], ...]:
    """Recover the exact axis-aligned release polygon from its persisted crop."""

    bbox = _float_list(
        bbox_enu_m,
        length=4,
        context="runtime_spatial_crop.bbox_enu_m",
    )
    padding = float(padding_m)
    if padding <= 0.0:
        raise ValueError("runtime_spatial_crop.expanded_boundary_padding_m must be positive")
    min_x = round(bbox[0] + padding, 3)
    min_y = round(bbox[1] + padding, 3)
    max_x = round(bbox[2] - padding, 3)
    max_y = round(bbox[3] - padding, 3)
    if min_x >= max_x or min_y >= max_y:
        raise ValueError(
            "runtime crop cannot encode a positive-area capture polygon: "
            f"bbox={bbox}, padding={padding}"
        )
    reconstructed_bbox = [
        round(min_x - padding, 3),
        round(min_y - padding, 3),
        round(max_x + padding, 3),
        round(max_y + padding, 3),
    ]
    if reconstructed_bbox != bbox:
        raise ValueError(
            "runtime crop is not an exact three-decimal axis-aligned polygon expansion: "
            f"expected {bbox}, reconstructed {reconstructed_bbox}"
        )
    return (
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    )


@dataclass(frozen=True)
class RoiContract:
    scenario_id: str
    map_id: str
    active_site_id: str
    active_roi_id: str
    capture_boundary_id: str
    capture_polygon_enu_m: tuple[tuple[float, float], ...]
    runtime_crop_bbox_enu_m: tuple[float, float, float, float]
    expanded_boundary_padding_m: float
    runtime_crop_policy: str
    tick_start: int
    tick_end: int
    capture_step_ticks: int
    capture_frames_per_episode: int
    authority_root: Path
    baseline_capture_polygon_enu_m: tuple[tuple[float, float], ...]
    geometry_authority: str
    full_chain_rebuild_required: bool
    minimal_change_evidence: Mapping[str, Any] | None = None

    @property
    def center_enu_m(self) -> list[float]:
        return [
            round(
                sum(point[0] for point in self.capture_polygon_enu_m)
                / len(self.capture_polygon_enu_m),
                3,
            ),
            round(
                sum(point[1] for point in self.capture_polygon_enu_m)
                / len(self.capture_polygon_enu_m),
                3,
            ),
            0.0,
        ]

    def polygon_lists(self) -> list[list[float]]:
        return [list(point) for point in self.capture_polygon_enu_m]

    def contract_payload(self, *, authority_root: Path | None = None) -> dict[str, Any]:
        source_root = self.authority_root
        if authority_root is not None:
            source_root = source_root.relative_to(authority_root.resolve())
        return {
            "policy": self.geometry_authority,
            "source_root": source_root.as_posix(),
            "scenario_id": self.scenario_id,
            "map_id": self.map_id,
            "active_site_id": self.active_site_id,
            "active_roi_id": self.active_roi_id,
            "capture_boundary_id": self.capture_boundary_id,
            "capture_polygon_enu_m": self.polygon_lists(),
            "baseline_capture_polygon_enu_m": [
                list(point) for point in self.baseline_capture_polygon_enu_m
            ],
            "runtime_crop_bbox_enu_m": list(self.runtime_crop_bbox_enu_m),
            "expanded_boundary_padding_m": self.expanded_boundary_padding_m,
            "runtime_crop_policy": self.runtime_crop_policy,
            "tick_start": self.tick_start,
            "tick_end": self.tick_end,
            "capture_step_ticks": self.capture_step_ticks,
            "capture_frames_per_episode": self.capture_frames_per_episode,
            "seed_consistency": "exact_across_seed00_seed01_seed02",
            "full_chain_rebuild_required": self.full_chain_rebuild_required,
            "minimal_change_evidence": (
                dict(self.minimal_change_evidence)
                if self.minimal_change_evidence is not None
                else None
            ),
        }


def load_episode_roi_contract_payload(
    authority_root: Path,
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
    episode_id = f"{scenario_id}__seed{seed:02d}"
    episode_dir = authority_root / episode_id
    manifest = _read_json(episode_dir / "episode_manifest.json")
    plan = _read_json(episode_dir / "scenario_plan.json")
    roster = _read_json(episode_dir / "global_entity_roster.json")

    if str(manifest.get("episode_id") or "") != episode_id:
        raise ValueError(f"{episode_dir}: manifest episode_id does not match directory")
    if str(manifest.get("scenario_id") or "") != scenario_id:
        raise ValueError(f"{episode_dir}: manifest scenario_id does not match request")
    if int(manifest.get("seed")) != seed:
        raise ValueError(f"{episode_dir}: manifest seed does not match directory")

    site_contracts = dict(
        (plan.get("compiled_plan_summary") or {}).get("site_contracts") or {}
    )
    if len(site_contracts) != 1:
        raise ValueError(
            f"{episode_dir}: expected exactly one active site contract, got "
            f"{sorted(site_contracts)}"
        )
    site_key, site_contract_value = next(iter(site_contracts.items()))
    site_contract = dict(site_contract_value or {})
    site_id = str(site_contract.get("site_id") or "")
    roi_id = str(site_contract.get("roi_id") or "")
    if not site_id or site_id != str(site_key):
        raise ValueError(f"{episode_dir}: active site id is missing or inconsistent")
    if not roi_id:
        raise ValueError(f"{episode_dir}: active ROI id is missing")

    roster_entities = roster.get("entities")
    if not isinstance(roster_entities, list) or not roster_entities:
        raise ValueError(f"{episode_dir}: global roster is empty")
    roster_site_ids = {
        str(entity.get("site_id") or "")
        for entity in roster_entities
        if isinstance(entity, Mapping)
    }
    roster_roi_ids = {
        str(entity.get("roi_id") or "")
        for entity in roster_entities
        if isinstance(entity, Mapping)
    }
    if roster_site_ids != {site_id} or roster_roi_ids != {roi_id}:
        raise ValueError(
            f"{episode_dir}: roster site/ROI diverges from active contract: "
            f"sites={sorted(roster_site_ids)}, rois={sorted(roster_roi_ids)}"
        )

    map_id = str(manifest.get("map_id") or "")
    if not map_id or str(plan.get("map_id") or "") != map_id:
        raise ValueError(f"{episode_dir}: map_id is missing or inconsistent")
    sumo_map_id = str(
        ((manifest.get("sumo_traffic") or {}).get("source") or {}).get("map_id")
        or ""
    )
    uav_map_id = str(
        ((manifest.get("uav_global_flow") or {}).get("source") or {}).get("map_id")
        or ""
    )
    if sumo_map_id != map_id or uav_map_id != map_id:
        raise ValueError(
            f"{episode_dir}: external authority map ids diverge from {map_id!r}"
        )

    boundary_id = str(manifest.get("capture_boundary_id") or "")
    if not boundary_id or str(site_contract.get("capture_boundary_id") or "") != boundary_id:
        raise ValueError(f"{episode_dir}: capture boundary id is missing or inconsistent")

    crop = dict(manifest.get("runtime_spatial_crop") or {})
    if crop.get("enabled") is not True:
        raise ValueError(f"{episode_dir}: runtime spatial crop is not enabled")
    if str(crop.get("capture_boundary_id") or "") != boundary_id:
        raise ValueError(f"{episode_dir}: runtime crop boundary id diverges")
    bbox = _float_list(
        crop.get("bbox_enu_m"),
        length=4,
        context=f"{episode_dir}: runtime_spatial_crop.bbox_enu_m",
    )
    padding = float(crop.get("expanded_boundary_padding_m"))
    polygon = polygon_from_runtime_crop(bbox, padding)
    polygon_point_count = int(
        ((manifest.get("sumo_traffic") or {}).get("visibility_geometry") or {}).get(
            "capture_polygon_points"
        )
        or 0
    )
    if polygon_point_count != 4:
        raise ValueError(
            f"{episode_dir}: release polygon cannot be recovered fail-closed; "
            f"persisted point count is {polygon_point_count}"
        )
    formal_roi_bbox = list(
        (manifest.get("capture_truth_sync") or {}).get("formal_roi_bbox_enu_m")
        or []
    )
    if len(formal_roi_bbox) < 4 or [float(value) for value in formal_roi_bbox[:4]] != bbox:
        raise ValueError(f"{episode_dir}: capture truth bbox diverges from runtime crop")

    time_range = dict(manifest.get("time_range") or {})
    tick_start = int(time_range.get("tick_start"))
    tick_end = int(time_range.get("tick_end"))
    duration_ticks = int(manifest.get("duration_ticks"))
    truth_frame_count = int((manifest.get("record_counts") or {}).get("truth_frames"))
    if (
        tick_start != EXPECTED_TICK_START
        or tick_end != EXPECTED_TICK_END
        or duration_ticks != EXPECTED_TICK_END
        or truth_frame_count != EXPECTED_TRUTH_FRAME_COUNT
    ):
        raise ValueError(
            f"{episode_dir}: formal time contract diverges: "
            f"start={tick_start}, end={tick_end}, duration={duration_ticks}, "
            f"truth_frames={truth_frame_count}"
        )

    return {
        "scenario_id": scenario_id,
        "map_id": map_id,
        "active_site_id": site_id,
        "active_roi_id": roi_id,
        "capture_boundary_id": boundary_id,
        "capture_polygon_enu_m": polygon,
        "runtime_crop_bbox_enu_m": tuple(bbox),
        "expanded_boundary_padding_m": padding,
        "runtime_crop_policy": str(crop.get("policy") or ""),
        "tick_start": tick_start,
        "tick_end": tick_end,
        "capture_step_ticks": CAPTURE_STEP_TICKS,
        "capture_frames_per_episode": EXPECTED_CAPTURE_FRAME_COUNT,
    }


def load_formal_roi_contract(
    authority_root: Path,
    scenario_id: str,
) -> RoiContract:
    resolved_root = authority_root.resolve()
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    seed_payloads = [
        load_episode_roi_contract_payload(resolved_root, scenario_id, seed)
        for seed in RELEASE_SEEDS
    ]
    reference = seed_payloads[0]
    divergences = [
        seed
        for seed, payload in zip(RELEASE_SEEDS[1:], seed_payloads[1:])
        if payload != reference
    ]
    if divergences:
        raise RuntimeError(
            f"{scenario_id}: release ROI contract diverges in seeds {divergences}"
        )
    formal_polygon = tuple(reference["capture_polygon_enu_m"])
    return RoiContract(
        authority_root=resolved_root,
        baseline_capture_polygon_enu_m=formal_polygon,
        geometry_authority="formal_release_roi_exact_v1",
        full_chain_rebuild_required=False,
        **reference,
    )


def load_rebuild_contracts(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("schema_name") != "aero_minimal_roi_rebuild_contract":
        raise ValueError(f"{path}: unexpected ROI rebuild schema_name")
    if payload.get("schema_version") != "1.0.0":
        raise ValueError(f"{path}: unsupported ROI rebuild schema_version")
    replacements = payload.get("replacements")
    if not isinstance(replacements, list):
        raise TypeError(f"{path}: replacements must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(replacements):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{path}: replacements[{index}] must be an object")
        replacement = dict(raw)
        scenario_id = str(replacement.get("scenario_id") or "")
        if not scenario_id:
            raise ValueError(f"{path}: replacements[{index}] lacks scenario_id")
        if scenario_id in indexed:
            raise ValueError(f"{path}: duplicate replacement for {scenario_id}")
        indexed[scenario_id] = replacement
    return indexed


def _apply_rebuild_contract(
    formal: RoiContract,
    replacement: Mapping[str, Any],
    *,
    contract_path: Path,
) -> RoiContract:
    identity_fields = (
        "scenario_id",
        "map_id",
        "active_site_id",
        "active_roi_id",
        "capture_boundary_id",
    )
    for field in identity_fields:
        actual = str(replacement.get(field) or "")
        expected = str(getattr(formal, field))
        if actual != expected:
            raise ValueError(
                f"{contract_path}: {formal.scenario_id} {field}={actual!r}, "
                f"formal baseline requires {expected!r}"
            )
    baseline_polygon = tuple(
        tuple(point)
        for point in (
            _float_list(
                raw_point,
                length=2,
                context=(
                    f"{contract_path}:{formal.scenario_id}:"
                    "baseline_capture_polygon_enu_m"
                ),
            )
            for raw_point in replacement.get("baseline_capture_polygon_enu_m") or []
        )
    )
    replacement_polygon = tuple(
        tuple(point)
        for point in (
            _float_list(
                raw_point,
                length=2,
                context=(
                    f"{contract_path}:{formal.scenario_id}:"
                    "replacement_capture_polygon_enu_m"
                ),
            )
            for raw_point in replacement.get("replacement_capture_polygon_enu_m") or []
        )
    )
    if len(replacement_polygon) != 4:
        raise ValueError(
            f"{contract_path}: {formal.scenario_id} replacement polygon must have four points"
        )
    if len(baseline_polygon) != 4:
        raise ValueError(
            f"{contract_path}: {formal.scenario_id} baseline polygon must have four points"
        )
    # ``baseline_capture_polygon_enu_m`` is the immutable geometry recorded
    # before the explicitly authorised rebuild.  Once the three-seed parent
    # corpus has been rebuilt, the formal authority naturally equals the
    # replacement polygon rather than that historical baseline.  Accept both
    # lifecycle states, but never accept a third, undocumented geometry.
    if formal.capture_polygon_enu_m not in (baseline_polygon, replacement_polygon):
        raise ValueError(
            f"{contract_path}: {formal.scenario_id} formal authority matches neither "
            "the recorded baseline nor the authorised replacement polygon"
        )
    replacement_bbox = tuple(
        _float_list(
            replacement.get("replacement_runtime_crop_bbox_enu_m"),
            length=4,
            context=(
                f"{contract_path}:{formal.scenario_id}:"
                "replacement_runtime_crop_bbox_enu_m"
            ),
        )
    )
    padding_m = float(replacement.get("expanded_boundary_padding_m"))
    if padding_m != formal.expanded_boundary_padding_m:
        raise ValueError(
            f"{contract_path}: {formal.scenario_id} replacement padding must "
            f"remain {formal.expanded_boundary_padding_m}m"
        )
    if polygon_from_runtime_crop(replacement_bbox, padding_m) != replacement_polygon:
        raise ValueError(
            f"{contract_path}: {formal.scenario_id} replacement crop and polygon diverge"
        )
    baseline_width = baseline_polygon[1][0] - baseline_polygon[0][0]
    baseline_height = baseline_polygon[2][1] - baseline_polygon[1][1]
    replacement_width = replacement_polygon[1][0] - replacement_polygon[0][0]
    replacement_height = replacement_polygon[2][1] - replacement_polygon[1][1]
    if (
        round(replacement_width, 3),
        round(replacement_height, 3),
    ) != (
        round(baseline_width, 3),
        round(baseline_height, 3),
    ):
        raise ValueError(
            f"{contract_path}: {formal.scenario_id} replacement must preserve exact dimensions"
        )
    if replacement.get("full_chain_rebuild_required") is not True:
        raise ValueError(
            f"{contract_path}: {formal.scenario_id} replacement must require full-chain rebuild"
        )
    evidence = replacement.get("minimal_change_evidence")
    if not isinstance(evidence, Mapping):
        raise TypeError(
            f"{contract_path}: {formal.scenario_id} lacks minimal_change_evidence"
        )
    return RoiContract(
        scenario_id=formal.scenario_id,
        map_id=formal.map_id,
        active_site_id=formal.active_site_id,
        active_roi_id=formal.active_roi_id,
        capture_boundary_id=formal.capture_boundary_id,
        capture_polygon_enu_m=replacement_polygon,
        runtime_crop_bbox_enu_m=replacement_bbox,
        expanded_boundary_padding_m=padding_m,
        runtime_crop_policy=formal.runtime_crop_policy,
        tick_start=formal.tick_start,
        tick_end=formal.tick_end,
        capture_step_ticks=formal.capture_step_ticks,
        capture_frames_per_episode=formal.capture_frames_per_episode,
        authority_root=formal.authority_root,
        baseline_capture_polygon_enu_m=baseline_polygon,
        geometry_authority="minimal_rebuilt_roi_contract_v1",
        full_chain_rebuild_required=True,
        minimal_change_evidence=dict(evidence),
    )


def load_roi_contract(
    authority_root: Path,
    scenario_id: str,
    *,
    rebuild_contract_path: Path = DEFAULT_REBUILD_CONTRACT_PATH,
) -> RoiContract:
    formal = load_formal_roi_contract(authority_root, scenario_id)
    replacements = load_rebuild_contracts(rebuild_contract_path)
    replacement = replacements.get(scenario_id)
    if replacement is None:
        return formal
    return _apply_rebuild_contract(
        formal,
        replacement,
        contract_path=rebuild_contract_path,
    )


def load_roi_contracts(
    authority_root: Path,
    scenario_ids: Sequence[str],
    *,
    rebuild_contract_path: Path = DEFAULT_REBUILD_CONTRACT_PATH,
) -> dict[str, RoiContract]:
    unique_ids = sorted({str(value) for value in scenario_ids if str(value)})
    if not unique_ids:
        raise ValueError("At least one scenario id is required for ROI contracts")
    return {
        scenario_id: load_roi_contract(
            authority_root,
            scenario_id,
            rebuild_contract_path=rebuild_contract_path,
        )
        for scenario_id in unique_ids
    }


def source_capture_boundary(event_script: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(event_script.get("parameters") or {})
    semantic_contract = dict(parameters.get("semantic_event_contract") or {})
    return {
        **dict(semantic_contract.get("capture_boundary") or {}),
        **dict(parameters.get("capture_boundary") or {}),
    }


def source_capture_boundary_id(event_script: Mapping[str, Any]) -> str:
    return str(source_capture_boundary(event_script).get("boundary_id") or "")


def source_capture_polygon(event_script: Mapping[str, Any]) -> list[list[float]]:
    raw_polygon = source_capture_boundary(event_script).get("polygon_enu_m")
    if not isinstance(raw_polygon, list) or len(raw_polygon) != 4:
        raise ValueError("source capture boundary must contain exactly four polygon points")
    polygon: list[list[float]] = []
    for index, point in enumerate(raw_polygon):
        polygon.append(
            _float_list(
                point,
                length=2,
                context=f"capture_boundary.polygon_enu_m[{index}]",
            )
        )
    return polygon


def index_event_scripts(scenarios_root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(scenarios_root.rglob("event_script.json")):
        payload = _read_json(path)
        scenario_id = str(payload.get("scenario_id") or "")
        if not scenario_id:
            raise ValueError(f"{path}: event_script is missing scenario_id")
        if scenario_id in indexed:
            duplicates.append(scenario_id)
            continue
        indexed[scenario_id] = path
    if duplicates:
        raise RuntimeError(f"duplicate event_script scenario ids: {sorted(set(duplicates))}")
    return indexed


def _mismatch_fields(
    *,
    boundary_id: str,
    polygon: list[list[float]],
    roi_contract: RoiContract,
) -> list[str]:
    mismatches: list[str] = []
    if boundary_id != roi_contract.capture_boundary_id:
        mismatches.append(
            f"capture_boundary_id={boundary_id!r} expected "
            f"{roi_contract.capture_boundary_id!r}"
        )
    if polygon != roi_contract.polygon_lists():
        mismatches.append(
            f"capture_polygon={polygon!r} expected "
            f"{roi_contract.polygon_lists()!r}"
        )
    return mismatches


def assert_source_contract_matches_roi(
    *,
    source_contract: Mapping[str, Any],
    roi_contract: RoiContract,
    context: str,
) -> None:
    boundary_id = str(source_contract.get("capture_boundary_id") or "")
    polygon = [
        [float(point[0]), float(point[1])]
        for point in source_contract.get("capture_boundary_polygon_enu_m") or []
    ]
    mismatches = _mismatch_fields(
        boundary_id=boundary_id,
        polygon=polygon,
        roi_contract=roi_contract,
    )
    if mismatches:
        raise RuntimeError(f"{context}: ROI contract mismatch: {mismatches}")


def assert_event_script_matches_roi(
    *,
    event_script: Mapping[str, Any],
    roi_contract: RoiContract,
    context: str,
) -> None:
    mismatches = _mismatch_fields(
        boundary_id=source_capture_boundary_id(event_script),
        polygon=source_capture_polygon(event_script),
        roi_contract=roi_contract,
    )
    if mismatches:
        raise RuntimeError(f"{context}: ROI contract mismatch: {mismatches}")


def validate_source_roi_contracts(
    *,
    authority_root: Path,
    scenarios_root: Path,
    scenario_ids: Sequence[str],
    rebuild_contract_path: Path = DEFAULT_REBUILD_CONTRACT_PATH,
) -> dict[str, Any]:
    unique_ids = [str(value) for value in scenario_ids if str(value)]
    if not unique_ids:
        raise ValueError("At least one scenario id is required")
    roi_contracts = load_roi_contracts(
        authority_root,
        unique_ids,
        rebuild_contract_path=rebuild_contract_path,
    )
    scripts = index_event_scripts(scenarios_root)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scenario_id in unique_ids:
        roi_contract = roi_contracts[scenario_id]
        script_path = scripts.get(scenario_id)
        if script_path is None:
            failure = {
                "scenario_id": scenario_id,
                "ok": False,
                "errors": [f"{scenario_id}: missing Dataset event_script.json"],
            }
            results.append(failure)
            failures.append(failure)
            continue
        event_script = _read_json(script_path)
        source_polygon: list[list[float]] | None = None
        try:
            source_polygon = source_capture_polygon(event_script)
            assert_event_script_matches_roi(
                event_script=event_script,
                roi_contract=roi_contract,
                context=str(script_path),
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            failure = {
                "scenario_id": scenario_id,
                "ok": False,
                "script_path": script_path.as_posix(),
                "errors": [str(exc)],
                "source_polygon_enu_m": source_polygon,
                "expected_polygon_enu_m": roi_contract.polygon_lists(),
                "expected_center_enu_m": roi_contract.center_enu_m,
                "geometry_authority": roi_contract.geometry_authority,
            }
            results.append(failure)
            failures.append(failure)
            continue
        results.append(
            {
                "scenario_id": scenario_id,
                "ok": True,
                "script_path": script_path.as_posix(),
                "capture_boundary_id": roi_contract.capture_boundary_id,
                "expected_center_enu_m": roi_contract.center_enu_m,
                "geometry_authority": roi_contract.geometry_authority,
                "full_chain_rebuild_required": (
                    roi_contract.full_chain_rebuild_required
                ),
            }
        )
    return {
        "schema_name": "target_roi_contract_validation",
        "schema_version": "1.0.0",
        "policy": "preserve_formal_roi_except_proven_infeasible_v1",
        "authority_root": str(authority_root.resolve()),
        "scenarios_root": str(scenarios_root.resolve()),
        "scenario_count": len(unique_ids),
        "failed_scenario_count": len(failures),
        "full_chain_rebuild_scenarios": sorted(
            scenario_id
            for scenario_id, contract in roi_contracts.items()
            if contract.full_chain_rebuild_required
        ),
        "ok": not failures,
        "scenarios": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority-root",
        type=Path,
        required=True,
        help="Read-only formal capture-filtered root used as ROI polygon authority.",
    )
    parser.add_argument(
        "--scenarios-root",
        type=Path,
        default=DEFAULT_SCENARIOS_ROOT,
    )
    parser.add_argument(
        "--rebuild-contract",
        type=Path,
        default=DEFAULT_REBUILD_CONTRACT_PATH,
    )
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    scenario_ids = list(args.scenario) or list(TARGET_RECAPTURE_SCENARIO_IDS)
    payload = validate_source_roi_contracts(
        authority_root=args.authority_root,
        scenarios_root=args.scenarios_root,
        scenario_ids=scenario_ids,
        rebuild_contract_path=args.rebuild_contract,
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
