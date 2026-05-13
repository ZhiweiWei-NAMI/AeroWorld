"""Validate city-grid spatial assignments in regenerated scenario scripts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

try:
    from .incident_plan import GROUND_TRAFFIC_EVENT_RULES
    from .spatial_event_grid import SpatialEventGridPlanner
except ImportError:  # pragma: no cover - direct script execution.
    from incident_plan import GROUND_TRAFFIC_EVENT_RULES  # type: ignore
    from spatial_event_grid import SpatialEventGridPlanner  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENARIOS_ROOT = ROOT / "Dataset" / "scenarios"
DEFAULT_INCIDENT_PLAN = ROOT / "Dataset" / "sumo_outputs" / "donghu_traffic_270s" / "sumo_incident_plan.json"
TRAFFIC_MAX_TARGET_ERROR_M = 80.0
NONTRAFFIC_MAX_TARGET_ERROR_M = 140.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def distance_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def scenario_scripts(root: Path) -> list[Path]:
    return sorted(root.rglob("event_script.json"))


def traffic_scenario_ids(incident_plan_path: Path) -> set[str]:
    ids: set[str] = set()
    if incident_plan_path.exists():
        plan = read_json(incident_plan_path)
        ids.update(str(item.get("episode_scenario_id") or "") for item in plan.get("incidents") or [])
    prefixes = tuple(rule[0] for rule in GROUND_TRAFFIC_EVENT_RULES)
    for script_path in scenario_scripts(DEFAULT_SCENARIOS_ROOT):
        scenario_id = script_path.parent.name
        if any(scenario_id.startswith(prefix) for prefix in prefixes):
            ids.add(scenario_id)
    return {item for item in ids if item}


def validate(root: Path, incident_plan_path: Path) -> dict[str, Any]:
    scripts = scenario_scripts(root)
    traffic_ids = traffic_scenario_ids(incident_plan_path)
    errors: list[str] = []
    warnings: list[str] = []
    cells_by_class: defaultdict[str, set[str]] = defaultdict(set)
    assignments: list[dict[str, Any]] = []
    for script_path in scripts:
        script = read_json(script_path)
        scenario_id = str(script.get("scenario_id") or script_path.parent.name)
        params = dict(script.get("parameters") or {})
        assignment = dict(params.get("spatial_grid_assignment") or {})
        if not assignment:
            errors.append(f"{scenario_id}: missing parameters.spatial_grid_assignment")
            continue
        event_space_class = str(assignment.get("event_space_class") or "")
        target = list(assignment.get("target_center_enu_m") or [])
        actual = list(assignment.get("actual_capture_center_enu_m") or [])
        target_error = float(assignment.get("target_error_m") or distance_xy(target, actual))
        grid_cell = dict(assignment.get("grid_cell") or {})
        cell_id = str(grid_cell.get("cell_id") or "")
        if not cell_id:
            errors.append(f"{scenario_id}: missing spatial grid cell id")
        cells_by_class[event_space_class].add(cell_id)
        should_be_traffic = scenario_id in traffic_ids
        is_traffic = event_space_class == "traffic_incident_grid"
        if should_be_traffic and not is_traffic:
            errors.append(f"{scenario_id}: traffic scenario not assigned to traffic_incident_grid")
        if not should_be_traffic and is_traffic:
            errors.append(f"{scenario_id}: nontraffic scenario assigned to traffic_incident_grid")
        if is_traffic and not assignment.get("traffic_incident_id"):
            errors.append(f"{scenario_id}: traffic assignment missing traffic_incident_id")
        if is_traffic and target_error > TRAFFIC_MAX_TARGET_ERROR_M:
            errors.append(f"{scenario_id}: traffic target error {target_error:.3f}m exceeds {TRAFFIC_MAX_TARGET_ERROR_M:.1f}m")
        if not is_traffic and target_error > NONTRAFFIC_MAX_TARGET_ERROR_M:
            errors.append(f"{scenario_id}: nontraffic target error {target_error:.3f}m exceeds {NONTRAFFIC_MAX_TARGET_ERROR_M:.1f}m")
        if float(grid_cell.get("main_edge_count") or 0.0) <= 0:
            errors.append(f"{scenario_id}: assigned grid cell has no main road edge")
        assignments.append(
            {
                "scenario_id": scenario_id,
                "event_space_class": event_space_class,
                "cell_id": cell_id,
                "target_error_m": round(target_error, 6),
                "traffic_incident_id": assignment.get("traffic_incident_id") or "",
            }
        )
    class_counts = Counter(item["event_space_class"] for item in assignments)
    try:
        planner = SpatialEventGridPlanner.default()
        available_nonincident = len(planner.nonincident_cells())
    except Exception as exc:
        planner = None
        available_nonincident = 0
        warnings.append(f"unable to compute available nonincident grid cells: {exc}")
    nontraffic_distinct = len(cells_by_class.get("nontraffic_main_road_grid", set()))
    traffic_distinct = len(cells_by_class.get("traffic_incident_grid", set()))
    expected_nontraffic_distinct = min(class_counts.get("nontraffic_main_road_grid", 0), available_nonincident)
    if expected_nontraffic_distinct and nontraffic_distinct < expected_nontraffic_distinct:
        errors.append(
            "nontraffic scenarios do not exhaust available non-incident main-road grid cells before reuse: "
            f"{nontraffic_distinct}/{expected_nontraffic_distinct}"
        )
    if traffic_distinct < class_counts.get("traffic_incident_grid", 0):
        errors.append(
            "traffic incidents must use distinct traffic grid cells before reuse: "
            f"{traffic_distinct}/{class_counts.get('traffic_incident_grid', 0)}"
        )
    return {
        "ok": not errors,
        "scenario_count": len(scripts),
        "assignment_count": len(assignments),
        "class_counts": dict(sorted(class_counts.items())),
        "distinct_cells_by_class": {key: len(value) for key, value in sorted(cells_by_class.items())},
        "errors": errors,
        "warnings": warnings,
        "assignments": assignments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate scenario spatial grid assignments.")
    parser.add_argument("--scenarios-root", type=Path, default=DEFAULT_SCENARIOS_ROOT)
    parser.add_argument("--incident-plan", type=Path, default=DEFAULT_INCIDENT_PLAN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.scenarios_root.resolve(), args.incident_plan.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
