"""Collect the current, validated ARM packages for UE capture."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from Dataset.tools.arm_ue_export import ROOT, read_json, write_json
from Dataset.tools.validate_arm_ue import validate_arm


def collect(*, full: bool = False) -> dict:
    expected = read_json(ROOT / "design/ue_arm_delivery/expected_coverage.json")
    families = {
        row["profile"]: family
        for family, profiles in expected["families"].items()
        for row in profiles
    }
    required = {(profile, f"seed{seed:02d}") for profile in families for seed in range(3)}
    covered = set()
    packages = []
    groups = set()
    arm_keys = set()
    for manifest_path in sorted((ROOT / "arms").glob("*/seed*/*/*/manifest.json")):
        arm = manifest_path.parent
        profile, seed, group, arm_id = arm.relative_to(ROOT / "arms").parts
        if profile not in families:
            raise ValueError(f"Unexpected ARM profile: {profile}")
        manifest = read_json(manifest_path)
        if manifest["mode"] not in ("factual_observation", "intervention"):
            raise ValueError(f"Unsupported ARM mode: {manifest['mode']}")
        ue = arm / "ue"
        saved = read_json(ue / "validation.json")
        if saved["ok"] is not True:
            raise ValueError(f"Failed UE validation: {arm}")
        window = {k: manifest["window"][k] for k in ("start_tick", "end_tick")}
        if saved["window"] != window:
            raise ValueError(f"Stale validation window: {arm}")
        if full and validate_arm(arm) != saved:
            raise ValueError(f"Saved validation differs from current files: {arm}")
        package = read_json(ue / "scenario_package.json")
        for key, value in package.items():
            if key in {"scenario_id", "episode_id"}:
                continue
            path = ROOT / value
            if Path(value).is_absolute() or not path.resolve().is_relative_to(ue.resolve()):
                raise ValueError(f"Nonlocal UE binding: {arm}/{key}")
            if not path.exists():
                raise FileNotFoundError(path)
        capture = read_json(ue / "capture_plan.json")
        ticks = list(range(window["start_tick"], window["end_tick"] + 1))
        if capture["capture_ticks"] != [tick for tick in ticks if tick % 5 == 0]:
            raise ValueError(f"Invalid capture grid: {arm}")
        if saved["frame_count"] != len(ticks) or saved["capture_frame_count"] != len(capture["capture_ticks"]):
            raise ValueError(f"Invalid saved counts: {arm}")
        episode = read_json(ue / "episode_manifest.json")
        background = Path(episode["background_source"])
        if background.is_absolute() or not (ROOT / background).is_file():
            raise ValueError(f"Missing or nonportable background provenance: {arm}")
        covered.add((profile, seed))
        groups.add((profile, seed, group))
        key = arm.relative_to(ROOT / "arms").as_posix()
        arm_keys.add(key)
        config_suffix = ".json" if families[profile] == "L6" else ".yaml"
        configuration = f"interventions/{key}{config_suffix}"
        entrypoint = f"scripts/arms/{key}.py"
        for path in (configuration, entrypoint):
            if not (ROOT / path).is_file():
                raise FileNotFoundError(ROOT / path)
        packages.append({
            "key": key, "path": arm.relative_to(ROOT).as_posix(),
            "family": families[profile], "profile": profile, "seed": seed,
            "group": group, "arm_id": arm_id, "window": window,
            "configuration": configuration, "entrypoint": entrypoint,
            "scenario_package": (ue / "scenario_package.json").relative_to(ROOT).as_posix(),
            "render_host_config": (ue / "render_host_config.json").relative_to(ROOT).as_posix(),
            "validation": (ue / "validation.json").relative_to(ROOT).as_posix(),
            "frame_count": saved["frame_count"],
            "capture_frame_count": saved["capture_frame_count"],
            "ue_bytes": sum(path.stat().st_size for path in ue.iterdir() if path.is_file()),
        })
    if covered != required:
        raise ValueError(f"Episode coverage mismatch: missing={sorted(required-covered)}, extra={sorted(covered-required)}")
    actual_arm_dirs = {p.parent.relative_to(ROOT / "arms").as_posix()
                       for p in (ROOT / "arms").rglob("raw") if p.is_dir()}
    actual_arm_dirs.update(p.parent.relative_to(ROOT / "arms").as_posix()
                           for p in (ROOT / "arms").rglob("manifest.json"))
    if actual_arm_dirs != arm_keys:
        raise ValueError(f"ARM directories lack current manifests: {sorted(actual_arm_dirs-arm_keys)}")
    for directory in ("interventions", "scripts/arms"):
        base = ROOT / directory
        keys = set()
        extensions = (".py",) if directory == "scripts/arms" else (".yaml", ".json")
        for path in base.glob("*/seed*/*/*"):
            if not path.is_file() or path.suffix not in extensions:
                continue
            profile = path.relative_to(base).parts[0]
            if profile not in families:
                raise ValueError(f"Unexpected configuration/entry profile: {path}")
            # Non-L6 families store YAML configs alongside JSON source scripts.
            if directory == "interventions" and families[profile] != "L6" and path.suffix == ".json":
                continue
            expected_suffix = ".py" if directory == "scripts/arms" else ".json" if families[profile] == "L6" else ".yaml"
            if path.suffix != expected_suffix:
                raise ValueError(f"Wrong current configuration format: {path}")
            keys.add(path.relative_to(base).with_suffix("").as_posix())
        if keys != arm_keys:
            raise ValueError(f"{directory} differs from current ARM data: missing={sorted(arm_keys-keys)}, extra={sorted(keys-arm_keys)}")
    old_align = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "arms").glob("*/seed*/*/*/ue_align"))
    old_align += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "arms").glob("*/seed*/*/ue_align"))
    if old_align:
        raise ValueError(f"Obsolete ue_align packages remain: {old_align}")
    code_files = ["arm_ue_export.py", "validate_arm_ue.py", "collect_arm_ue_delivery.py"]
    return {
        "scope": "Current L1–L6 and X ARM inputs for UE RGB/depth/segmentation capture",
        "capture_executed": False,
        "validation": "full content revalidation" if full else "all export validation reports, package references and coverage",
        "profile_count": len(families), "episode_count": len(covered),
        "group_count": len(groups), "arm_count": len(packages),
        "family_arm_counts": dict(sorted(Counter(row["family"] for row in packages).items())),
        "frame_count": sum(row["frame_count"] for row in packages),
        "capture_frame_count": sum(row["capture_frame_count"] for row in packages),
        "ue_bytes": sum(row["ue_bytes"] for row in packages),
        "packages": packages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Recheck all measured rows as well as package references")
    parser.add_argument("--output", type=Path, default=ROOT / "design/ue_arm_delivery/delivery_manifest.json")
    args = parser.parse_args()
    result = collect(full=args.full)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "packages"}, ensure_ascii=False, indent=2))
