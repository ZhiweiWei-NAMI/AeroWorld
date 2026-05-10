from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_MODALITIES = ("rgb", "depth", "seg")
LOGICAL_PRIMARY_SEG_CLASS_IDS = {"11": "hazard_trigger", "12": "uav_corridor"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sidecar_for_output(path: Path) -> Path:
    if path.suffix.lower() == ".npy":
        return path.with_suffix(".json")
    return path.with_suffix(".json")


def parse_int(value: Any, default: int = -1) -> int:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def semantic_histogram_from_png(path: Path) -> dict[str, int]:
    from PIL import Image

    image = Image.open(path).convert("L")
    histogram = image.histogram()
    return {str(index): int(count) for index, count in enumerate(histogram) if count}


def normalize_histogram(histogram: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in dict(histogram or {}).items() if int(value) > 0}


def histogram_consistency_errors(reference: dict[str, int], label: str, histogram: Any) -> list[str]:
    if not histogram:
        return []
    candidate = normalize_histogram(histogram)
    if candidate == reference:
        return []
    return [f"seg: {label} histogram does not match primary seg PNG histogram: {candidate} != {reference}"]


def verify_ok_uav_row(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    episode = str(row.get("episode") or "")
    tick = parse_int(row.get("tick"), -1)
    capture_view_id = str(row.get("capture_view_id") or "")
    capture_entity_id = str(row.get("capture_entity_id") or "")
    logical_sample_id = str(row.get("logical_sample_id") or "")
    alignment_key = str(row.get("alignment_key") or "")
    alignment_source = str(row.get("alignment_source") or "")
    batch_id = str(row.get("batch_id") or "")
    frame_id = str(row.get("frame_id") or "")
    frame_seq = parse_int(row.get("frame_seq"), tick)
    if not episode:
        errors.append("missing episode")
    if tick < 0:
        errors.append("missing tick")
    if not capture_view_id:
        errors.append("missing capture_view_id")
    if not capture_entity_id:
        errors.append("missing capture_entity_id")
    expected_logical_sample_id = f"{episode}:tick{tick:06d}:{capture_view_id}"
    if logical_sample_id != expected_logical_sample_id:
        errors.append(f"logical_sample_id mismatch: {logical_sample_id!r} != {expected_logical_sample_id!r}")
    if not alignment_key:
        errors.append("missing alignment_key")
    if not alignment_source:
        errors.append("missing alignment_source")
    if not batch_id:
        errors.append("missing batch_id")
    if not frame_id:
        errors.append("missing frame_id")
    if frame_seq < 0:
        errors.append("missing frame_seq")

    outputs = json.loads(row.get("modality_outputs") or "{}")
    for modality in REQUIRED_MODALITIES:
        payload = dict(outputs.get(modality) or {})
        output_value = payload.get("path") or payload.get("png")
        if not output_value:
            errors.append(f"{modality}: missing output path")
            continue
        output_path = Path(str(output_value))
        if not output_path.exists():
            errors.append(f"{modality}: output missing: {output_path}")
            continue
        sidecar_value = payload.get("sidecar")
        sidecar_path = Path(str(sidecar_value)) if sidecar_value else sidecar_for_output(output_path)
        if not sidecar_path.exists():
            errors.append(f"{modality}: sidecar missing: {sidecar_path}")
            continue
        sidecar = read_json(sidecar_path)
        checks = {
            "episode_id": episode,
            "tick": tick,
            "capture_view_id": capture_view_id,
            "source_uav_entity_id": capture_entity_id,
            "logical_sample_id": logical_sample_id,
            "capture_alignment_key": alignment_key,
            "capture_alignment_source": alignment_source,
            "batch_id": batch_id,
            "frame_id": frame_id,
            "frame_seq": frame_seq,
        }
        for key, expected in checks.items():
            actual = sidecar.get(key)
            if key in {"tick", "frame_seq"}:
                actual = parse_int(actual, -1)
            else:
                actual = str(actual or "")
            if actual != expected:
                errors.append(f"{modality}: sidecar {key} mismatch: {actual!r} != {expected!r}")
        payload_checks = {
            "alignment_key": alignment_key,
            "alignment_source": alignment_source,
            "batch_id": batch_id,
            "frame_id": frame_id,
            "frame_seq": frame_seq,
            "logical_sample_id": logical_sample_id,
        }
        for key, expected in payload_checks.items():
            actual = payload.get(key)
            if key == "frame_seq":
                actual = parse_int(actual, -1)
            else:
                actual = str(actual or "")
            if actual != expected:
                errors.append(f"{modality}: summary payload {key} mismatch: {actual!r} != {expected!r}")
        if modality == "seg":
            palette = payload.get("palette") or row.get("seg_palette_path")
            if not palette or not Path(str(palette)).exists():
                errors.append(f"seg: palette preview missing: {palette}")
            histogram = semantic_histogram_from_png(output_path)
            errors.extend(histogram_consistency_errors(histogram, "summary payload", payload.get("histogram")))
            if row.get("histogram"):
                try:
                    errors.extend(
                        histogram_consistency_errors(histogram, "summary row", json.loads(row.get("histogram") or "{}"))
                    )
                except json.JSONDecodeError as exc:
                    errors.append(f"seg: summary row histogram is not valid JSON: {exc}")
            sidecar_histogram = sidecar.get("class_histogram")
            errors.extend(histogram_consistency_errors(histogram, "sidecar", sidecar_histogram))
            nonzero = {str(k): int(v) for k, v in dict(histogram or {}).items() if int(k) != 0 and int(v) > 0}
            if not nonzero:
                errors.append("seg: class histogram has no nonzero class pixels")
            policy = dict(sidecar.get("event_semantic_logical_region_policy") or {})
            if not bool(policy.get("logical_region_primary_segmentation_enabled", False)):
                for class_id, class_name in LOGICAL_PRIMARY_SEG_CLASS_IDS.items():
                    count = int(dict(histogram or {}).get(class_id) or 0)
                    if count > 0:
                        errors.append(
                            f"seg: sidecar-only logical class {class_name!r} appears in primary seg histogram "
                            f"with {count} pixels"
                        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--require-ok-uav", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.summary)
    ok_uav_rows = [
        row
        for row in rows
        if str(row.get("view") or "") == "uav_tick100" and str(row.get("status") or "") == "ok"
    ]
    if args.require_ok_uav and not ok_uav_rows:
        raise SystemExit(f"No ok UAV rows found in {args.summary}")

    failures: list[str] = []
    for index, row in enumerate(ok_uav_rows, start=1):
        row_errors = verify_ok_uav_row(row)
        failures.extend([f"row#{index} {row.get('episode')} {row.get('capture_view_id')}: {error}" for error in row_errors])

    if failures:
        print(json.dumps({"status": "failed", "errors": failures}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"status": "ok", "verified_ok_uav_rows": len(ok_uav_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
