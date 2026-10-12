"""Shared IO, identity and time helpers for the L3 v2 ARM workflow."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "Dataset/tools", ROOT / "Plugins/SumoImporter/Scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Dataset.semantic_truth.provenance import open_artifact_text

TICK_HZ = 10
FORMAL_STEP_TICKS = 5
SCENARIO_ROOT = ROOT / "Dataset/scenarios/L3_dynamic_constraints/interaction"
CAPTURE = ROOT / "aw_data/render_ready_episodes_capture_filtered"
SEMANTIC = ROOT / "aw_data/objective_semantic_truth"
RUNTIME_STATE_FAMILIES = (
    "control_state", "mission_state", "navigation_state", "security_state",
    "sensor_state", "facility_state", "communication_state", "incident_state",
    "vehicle_state", "pedestrian_state",
)


def load(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path: Path | str) -> list[dict]:
    with open_artifact_text(Path(path), encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path | str, values: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def project_path(ref: str | Path) -> Path:
    path = (ROOT / ref).resolve()
    path.relative_to(ROOT)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def file_digest(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                   separators=(",", ":")).encode()
    ).hexdigest()


def grid_up(value: int) -> int:
    return (int(value) + FORMAL_STEP_TICKS - 1) // FORMAL_STEP_TICKS * FORMAL_STEP_TICKS


def scene_path(profile: str) -> str:
    return f"Dataset/scenarios/L3_dynamic_constraints/interaction/{profile}/scene_setup.json"


def script_path(profile: str) -> str:
    return f"Dataset/scenarios/L3_dynamic_constraints/interaction/{profile}/event_script.json"
