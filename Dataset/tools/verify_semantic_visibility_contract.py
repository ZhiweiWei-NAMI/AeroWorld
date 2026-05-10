from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "Dataset" / "render_ready_episodes"
DEFAULT_SUMMARY = PROJECT_ROOT / "Saved" / "AirSim" / "semantic_70events_rgb_depth_seg_tick100_summary.csv"
DEFAULT_RULES = PROJECT_ROOT / "Config" / "LowAltitude" / "semantic_stencil_rules.json"
DEFAULT_CAPTURE_PRESETS = PROJECT_ROOT / "Plugins" / "SumoImporter" / "Scripts" / "episode_capture_presets.json"

DEFAULT_REQUIRED_TRUTH_CLASSES = ("drone", "hazard_trigger", "uav_corridor", "pedestrian", "vehicle")
DEFAULT_REQUIRED_SEG_CLASSES = (
    "city_base_background",
    "drone",
    "hazard_trigger",
    "uav_corridor",
    "pedestrian",
    "vehicle",
)
DEFAULT_LOGICAL_CLASSES = ("hazard_trigger", "uav_corridor")


@dataclass(frozen=True)
class TruthCandidate:
    episode: str
    tick: int
    capture_entity_id: str
    altitude_m: float
    fov_degrees: float
    footprint_radius_m: float
    semantic_classes_in_footprint: tuple[str, ...]
    missing_semantic_classes: tuple[str, ...]
    other_uav_entity_ids: tuple[str, ...]
    entities_in_footprint: tuple[str, ...]

    @property
    def satisfies_requirements(self) -> bool:
        return not self.missing_semantic_classes


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def resolve_path(value: Any, project_root: Path = PROJECT_ROOT) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = project_root / path
    return path


def load_class_name_to_id(rules_path: Path) -> dict[str, int]:
    rules = read_json(rules_path)
    return {str(name): int(value) for name, value in dict(rules.get("classes") or {}).items()}


def load_default_uav_camera_preset(capture_presets_path: Path) -> dict[str, Any]:
    presets = read_json(capture_presets_path)
    cameras = list((presets.get("uav_cameras") or {}).get("default") or [])
    if not cameras:
        raise RuntimeError(f"No default UAV camera preset found in {capture_presets_path}")
    return dict(cameras[0])


def validate_nadir_preset(preset: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rotation = dict(preset.get("fixed_rotation_offset_deg") or {})
    pitch = float(rotation.get("pitch_deg", 0.0) or 0.0)
    yaw = float(rotation.get("yaw_deg", 0.0) or 0.0)
    roll = float(rotation.get("roll_deg", 0.0) or 0.0)
    if abs(pitch - -90.0) > 0.001 or abs(yaw) > 0.001 or abs(roll) > 0.001:
        errors.append(f"default UAV camera preset is not the existing nadir view: pitch={pitch} yaw={yaw} roll={roll}")
    if str(preset.get("camera_name") or "") != "bottom_center":
        errors.append(f"default UAV camera_name changed from bottom_center: {preset.get('camera_name')!r}")
    if not bool(preset.get("set_runtime_camera_pose", False)):
        errors.append("default UAV camera preset no longer sets the runtime camera pose")
    return errors


def entity_position_enu_m(entity: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = dict(entity.get("truth_pose") or {})
    position = pose.get("position_enu_m")
    if not isinstance(position, list) or len(position) < 3:
        return None
    return (float(position[0]), float(position[1]), float(position[2]))


def is_render_visible(entity: dict[str, Any]) -> bool:
    presence = dict(entity.get("render_presence") or {})
    submission_state = str(presence.get("submission_state") or "").strip()
    visibility_state = str(presence.get("visibility_state") or "").strip()
    return submission_state in {"", "submit_to_ue"} and visibility_state in {"", "visible"}


def semantic_classes_for_entity(entity: dict[str, Any]) -> set[str]:
    values = {
        str(entity.get("label_class") or ""),
        str(entity.get("entity_category") or ""),
        str(entity.get("entity_kind") or ""),
        str(entity.get("entity_type") or ""),
        str(entity.get("logical_asset_id") or ""),
        str(entity.get("proxy_template_id") or ""),
    }
    values.update(str(tag or "") for tag in (entity.get("tags") or []))
    haystack = " ".join(value.lower() for value in values if value)
    classes: set[str] = set()
    if "pedestrian" in haystack or "walker" in haystack or "ped_" in haystack:
        classes.add("pedestrian")
    if "vehicle" in haystack or "ambulance" in haystack or "police" in haystack or "suv" in haystack:
        classes.add("vehicle")
    if "uav" in haystack or "drone" in haystack or "quadrotor" in haystack or "flyingpawn" in haystack:
        classes.add("drone")
    if "uav_corridor" in haystack or "highaltitudecorridor" in haystack:
        classes.add("uav_corridor")
    if "no_fly" in haystack or "hazard" in haystack or "trigger.no_fly" in haystack or "geofence" in haystack:
        classes.add("hazard_trigger")
    if "landing_pad" in haystack or "facility" in haystack or "charger" in haystack:
        classes.add("facility")
    return classes


def footprint_radius_m(altitude_m: float, fov_degrees: float) -> float:
    return math.tan(math.radians(float(fov_degrees) / 2.0)) * max(0.0, float(altitude_m))


def iter_truth_frames(episode_dir: Path) -> Iterable[dict[str, Any]]:
    truth_path = episode_dir / "truth_frames.jsonl"
    with truth_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_truth_candidates(
    episodes_root: Path,
    *,
    required_classes: Iterable[str],
    altitude_min_m: float,
    altitude_max_m: float,
    fov_degrees: float,
    require_other_uav: bool = True,
) -> list[TruthCandidate]:
    required = {str(value).strip() for value in required_classes if str(value).strip()}
    candidates: list[TruthCandidate] = []
    for episode_dir in sorted(path for path in episodes_root.iterdir() if path.is_dir()):
        truth_path = episode_dir / "truth_frames.jsonl"
        if not truth_path.exists():
            continue
        for frame in iter_truth_frames(episode_dir):
            active_entities = [dict(entity) for entity in frame.get("entities") or [] if is_render_visible(dict(entity))]
            active_positions = [
                (entity, entity_position_enu_m(entity))
                for entity in active_entities
                if entity_position_enu_m(entity) is not None
            ]
            for entity, position in active_positions:
                assert position is not None
                if str(entity.get("entity_category") or "").lower() != "uav":
                    continue
                altitude = float(position[2])
                if altitude < altitude_min_m or altitude > altitude_max_m:
                    continue
                radius = footprint_radius_m(altitude, fov_degrees)
                capture_id = str(entity.get("entity_id") or "")
                classes: set[str] = set()
                other_uavs: set[str] = set()
                entity_ids: set[str] = set()
                for other, other_position in active_positions:
                    assert other_position is not None
                    distance_xy = math.hypot(float(other_position[0]) - position[0], float(other_position[1]) - position[1])
                    if distance_xy > radius:
                        continue
                    other_id = str(other.get("entity_id") or "")
                    entity_ids.add(other_id)
                    classes.update(semantic_classes_for_entity(other))
                    if str(other.get("entity_category") or "").lower() == "uav" and other_id != capture_id:
                        other_uavs.add(other_id)
                if require_other_uav and not other_uavs:
                    missing = set(required)
                    missing.add("other_uav")
                else:
                    missing = set(required) - classes
                candidates.append(
                    TruthCandidate(
                        episode=episode_dir.name,
                        tick=int(frame.get("tick") or 0),
                        capture_entity_id=capture_id,
                        altitude_m=round(altitude, 3),
                        fov_degrees=float(fov_degrees),
                        footprint_radius_m=round(radius, 3),
                        semantic_classes_in_footprint=tuple(sorted(classes)),
                        missing_semantic_classes=tuple(sorted(missing)),
                        other_uav_entity_ids=tuple(sorted(other_uavs)),
                        entities_in_footprint=tuple(sorted(entity_ids)),
                    )
                )
    return candidates


def candidate_key(candidate: TruthCandidate) -> tuple[str, int, str]:
    return (candidate.episode, int(candidate.tick), candidate.capture_entity_id)


def best_candidate_diagnostics(candidates: list[TruthCandidate], limit: int = 10) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            len(item.missing_semantic_classes),
            -len(item.semantic_classes_in_footprint),
            item.episode,
            item.tick,
            item.capture_entity_id,
        ),
    )
    return [asdict(item) for item in ordered[: max(0, limit)]]


def modality_payload(row: dict[str, str], modality: str) -> dict[str, Any]:
    outputs = json.loads(row.get("modality_outputs") or "{}")
    payload = dict(outputs.get(modality) or {})
    if modality == "rgb" and not payload.get("path"):
        payload["path"] = row.get("rgb_path") or ""
    if modality == "depth" and not payload.get("path"):
        payload["path"] = row.get("depth_path") or ""
    if modality == "seg":
        if not payload.get("png"):
            payload["png"] = row.get("seg_path") or ""
        if not payload.get("palette"):
            payload["palette"] = row.get("seg_palette_path") or ""
    return payload


def sidecar_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def semantic_histogram_from_png(path: Path) -> dict[str, int]:
    from PIL import Image

    image = Image.open(path).convert("L")
    histogram = image.histogram()
    return {str(index): int(count) for index, count in enumerate(histogram) if count}


def histogram_from_payload(payload: dict[str, Any], seg_path: Path) -> dict[str, int]:
    histogram = payload.get("histogram")
    if not histogram:
        histogram = semantic_histogram_from_png(seg_path)
    return {str(key): int(value) for key, value in dict(histogram).items()}


def class_count(histogram: dict[str, int], class_name: str, class_name_to_id: dict[str, int]) -> int:
    class_id = class_name_to_id.get(class_name)
    if class_id is None:
        return 0
    return int(histogram.get(str(class_id), 0))


def custom_stencil_targets(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in sidecar.get("event_semantic_objects") or []:
        entry = dict(item)
        logical = " ".join(
            str(entry.get(key) or "")
            for key in ("logical_asset_id", "spawn_logical_asset_id", "spawn_asset_id", "entity_id")
        ).lower()
        if "trigger.no_fly" in logical or "hazard" in logical or "semantic.trigger_box" in logical or "semantic.uav_corridor" in logical:
            targets.append(entry)
    return targets


def validate_rgb_invisibility_contract(rgb_sidecar: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(rgb_sidecar.get("capture_backend") or "") != "airsim_native_uav_camera":
        errors.append("rgb sidecar capture_backend is not airsim_native_uav_camera")
    targets = custom_stencil_targets(rgb_sidecar)
    if not targets:
        errors.append("rgb sidecar has no custom-stencil-only logical proxy targets to validate")
        return errors
    sanitizer = rgb_sidecar.get("event_semantic_proxy_sanitizer")
    if not isinstance(sanitizer, dict):
        errors.append("rgb sidecar is missing event_semantic_proxy_sanitizer")
    else:
        if str(sanitizer.get("status") or "").lower() != "ok":
            errors.append(f"event_semantic_proxy_sanitizer status is not ok: {sanitizer.get('status')!r}")
        target_count = int(sanitizer.get("target_count") or 0)
        sanitized_count = int(sanitizer.get("sanitized_actor_count") or 0)
        missing_count = int(sanitizer.get("missing_actor_count") or 0)
        if target_count <= 0 or sanitized_count != target_count or missing_count != 0:
            errors.append(
                "event_semantic_proxy_sanitizer did not sanitize every logical proxy "
                f"(target={target_count}, sanitized={sanitized_count}, missing={missing_count})"
            )
    exclusion = rgb_sidecar.get("airsim_proxy_capture_exclusion")
    if not isinstance(exclusion, dict):
        errors.append("rgb sidecar is missing airsim_proxy_capture_exclusion")
    else:
        if str(exclusion.get("method") or "") != "proxy_primitive_render_flags":
            errors.append(f"unexpected RGB proxy exclusion method: {exclusion.get('method')!r}")
        if bool(exclusion.get("pipcamera_hidden_lists_mutated", False)):
            errors.append("RGB proxy exclusion mutated PIPCamera hidden lists")
        if str(exclusion.get("status") or "").lower() != "ok":
            errors.append(f"RGB proxy exclusion status is not ok: {exclusion.get('status')!r}")
    return errors


def validate_sample_row(
    row: dict[str, str],
    *,
    class_name_to_id: dict[str, int],
    required_seg_classes: Iterable[str],
    logical_classes: Iterable[str],
    min_pixels_per_class: int,
    project_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    details: dict[str, Any] = {
        "episode": row.get("episode", ""),
        "tick": int(row.get("tick") or 0),
        "capture_entity_id": row.get("capture_entity_id", ""),
        "capture_view_id": row.get("capture_view_id", ""),
    }
    rgb_payload = modality_payload(row, "rgb")
    seg_payload = modality_payload(row, "seg")
    rgb_path = resolve_path(rgb_payload.get("path"), project_root)
    seg_path = resolve_path(seg_payload.get("png") or seg_payload.get("path"), project_root)
    if rgb_path is None or not rgb_path.exists():
        errors.append(f"missing RGB output: {rgb_path}")
        return errors, details
    if seg_path is None or not seg_path.exists():
        errors.append(f"missing seg output: {seg_path}")
        return errors, details
    rgb_sidecar_path = resolve_path(rgb_payload.get("sidecar"), project_root) or sidecar_path_for(rgb_path)
    seg_sidecar_path = resolve_path(seg_payload.get("sidecar"), project_root) or sidecar_path_for(seg_path)
    if not rgb_sidecar_path.exists():
        errors.append(f"missing RGB sidecar: {rgb_sidecar_path}")
        return errors, details
    if not seg_sidecar_path.exists():
        errors.append(f"missing seg sidecar: {seg_sidecar_path}")
        return errors, details
    rgb_sidecar = read_json(rgb_sidecar_path)
    seg_sidecar = read_json(seg_sidecar_path)
    histogram = histogram_from_payload(seg_payload, seg_path)
    details["seg_histogram"] = histogram
    if str(seg_sidecar.get("capture_backend") or "") != "ue_custom_stencil_fixed_world_camera":
        errors.append("seg sidecar capture_backend is not ue_custom_stencil_fixed_world_camera")
    if str(seg_sidecar.get("segmentation_kind") or "") != "ue_custom_stencil_class_id_u8":
        errors.append("seg sidecar segmentation_kind is not ue_custom_stencil_class_id_u8")
    if str(seg_sidecar.get("capture_backend") or "") == "airsim_native_uav_camera":
        errors.append("seg output used AirSim native capture backend")
    requested_rotation = dict(seg_sidecar.get("requested_camera_rotation_body_deg") or {})
    if abs(float(requested_rotation.get("pitch_deg", 0.0) or 0.0) - -90.0) > 0.001:
        errors.append("seg sidecar does not preserve the existing nadir camera pitch")
    if str(seg_sidecar.get("camera_name") or "") != "bottom_center":
        errors.append("seg sidecar camera_name is not bottom_center")
    for class_name in required_seg_classes:
        class_name = str(class_name).strip()
        if not class_name:
            continue
        count = class_count(histogram, class_name, class_name_to_id)
        if count < min_pixels_per_class:
            errors.append(f"seg class {class_name!r} has {count} pixels, expected >= {min_pixels_per_class}")
    for class_name in logical_classes:
        count = class_count(histogram, str(class_name), class_name_to_id)
        if count < min_pixels_per_class:
            errors.append(f"logical seg class {class_name!r} is not visible")
    errors.extend(validate_rgb_invisibility_contract(rgb_sidecar))
    if bool((rgb_sidecar.get("airsim_proxy_capture_exclusion") or {}).get("pipcamera_hidden_lists_mutated", False)):
        errors.append("RGB sidecar says PIPCamera hidden lists were mutated")
    return errors, details


def matching_candidate(row: dict[str, str], candidates: list[TruthCandidate]) -> TruthCandidate | None:
    key = (str(row.get("episode") or ""), int(row.get("tick") or 0), str(row.get("capture_entity_id") or ""))
    for candidate in candidates:
        if candidate_key(candidate) == key:
            return candidate
    return None


def verify_summary_rows(
    rows: list[dict[str, str]],
    *,
    candidates: list[TruthCandidate],
    class_name_to_id: dict[str, int],
    required_seg_classes: Iterable[str],
    logical_classes: Iterable[str],
    min_pixels_per_class: int,
    altitude_min_m: float,
    altitude_max_m: float,
    enforce_truth_candidate: bool,
    project_root: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    verified: list[dict[str, Any]] = []
    ok_rows = [
        row
        for row in rows
        if str(row.get("view") or "") == "uav_tick100"
        and str(row.get("status") or "") == "ok"
        and altitude_min_m <= float(row.get("altitude_m") or 0.0) <= altitude_max_m
    ]
    if not ok_rows:
        errors.append(f"no ok UAV summary rows in altitude band {altitude_min_m:g}..{altitude_max_m:g}m")
        return errors, verified
    for row in ok_rows:
        label = f"{row.get('episode')} tick={row.get('tick')} entity={row.get('capture_entity_id')}"
        candidate = matching_candidate(row, candidates)
        if enforce_truth_candidate:
            if candidate is None:
                errors.append(f"{label}: no matching 70event truth candidate")
                continue
            if candidate.missing_semantic_classes:
                errors.append(
                    f"{label}: 70event truth candidate is missing classes "
                    f"{list(candidate.missing_semantic_classes)} in the existing nadir footprint"
                )
                continue
        row_errors, details = validate_sample_row(
            row,
            class_name_to_id=class_name_to_id,
            required_seg_classes=required_seg_classes,
            logical_classes=logical_classes,
            min_pixels_per_class=min_pixels_per_class,
            project_root=project_root,
        )
        if row_errors:
            errors.extend(f"{label}: {error}" for error in row_errors)
        else:
            if candidate is not None:
                details["truth_candidate"] = asdict(candidate)
            verified.append(details)
    return errors, verified


def parse_csv_list(values: list[str] | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return defaults
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(parsed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that existing 70event UAV samples expose semantic-only logical regions in seg, not RGB."
    )
    parser.add_argument("--mode", choices=["search", "verify-summary"], default="verify-summary")
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--capture-presets", type=Path, default=DEFAULT_CAPTURE_PRESETS)
    parser.add_argument("--altitude-min-m", type=float, default=75.0)
    parser.add_argument("--altitude-max-m", type=float, default=85.0)
    parser.add_argument("--required-truth-classes", nargs="*", default=None)
    parser.add_argument("--required-seg-classes", nargs="*", default=None)
    parser.add_argument("--logical-classes", nargs="*", default=None)
    parser.add_argument("--min-pixels-per-class", type=int, default=1)
    parser.add_argument("--require-other-uav", action="store_true", default=True)
    parser.add_argument("--no-require-other-uav", dest="require_other_uav", action="store_false")
    parser.add_argument("--no-enforce-truth-candidate", dest="enforce_truth_candidate", action="store_false")
    parser.add_argument("--diagnostic-limit", type=int, default=10)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    required_truth_classes = parse_csv_list(args.required_truth_classes, DEFAULT_REQUIRED_TRUTH_CLASSES)
    required_seg_classes = parse_csv_list(args.required_seg_classes, DEFAULT_REQUIRED_SEG_CLASSES)
    logical_classes = parse_csv_list(args.logical_classes, DEFAULT_LOGICAL_CLASSES)
    preset = load_default_uav_camera_preset(args.capture_presets)
    preset_errors = validate_nadir_preset(preset)
    fov_degrees = float(preset.get("fov_degrees") or 85.0)
    class_name_to_id = load_class_name_to_id(args.rules)
    candidates = find_truth_candidates(
        args.episodes_root,
        required_classes=required_truth_classes,
        altitude_min_m=args.altitude_min_m,
        altitude_max_m=args.altitude_max_m,
        fov_degrees=fov_degrees,
        require_other_uav=bool(args.require_other_uav),
    )
    matching = [candidate for candidate in candidates if candidate.satisfies_requirements]
    result: dict[str, Any] = {
        "mode": args.mode,
        "status": "ok",
        "nadir_preset_errors": preset_errors,
        "required_truth_classes": list(required_truth_classes),
        "required_seg_classes": list(required_seg_classes),
        "logical_classes": list(logical_classes),
        "candidate_count": len(candidates),
        "matching_candidate_count": len(matching),
        "best_candidates": best_candidate_diagnostics(candidates, args.diagnostic_limit),
    }
    errors: list[str] = list(preset_errors)
    if args.mode == "search":
        if not matching:
            errors.append("no existing 70event tick satisfies the requested truth-frame footprint constraints")
        result["matching_candidates"] = [asdict(candidate) for candidate in matching[: args.diagnostic_limit]]
    else:
        if not args.summary.exists():
            errors.append(f"summary is missing: {args.summary}")
            verified: list[dict[str, Any]] = []
        else:
            rows = load_rows(args.summary)
            row_errors, verified = verify_summary_rows(
                rows,
                candidates=candidates,
                class_name_to_id=class_name_to_id,
                required_seg_classes=required_seg_classes,
                logical_classes=logical_classes,
                min_pixels_per_class=max(1, int(args.min_pixels_per_class)),
                altitude_min_m=args.altitude_min_m,
                altitude_max_m=args.altitude_max_m,
                enforce_truth_candidate=bool(args.enforce_truth_candidate),
                project_root=args.project_root,
            )
            errors.extend(row_errors)
        result["verified_samples"] = verified
        result["verified_sample_count"] = len(verified)
    if errors:
        result["status"] = "failed"
        result["errors"] = errors
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
