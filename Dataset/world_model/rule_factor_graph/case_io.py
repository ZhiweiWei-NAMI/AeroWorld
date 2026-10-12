"""Load the bounded real corridor oracle case with exact IDs and line provenance."""
from __future__ import annotations

from Dataset.semantic_truth.provenance import open_artifact_text
import json
from pathlib import Path

EPISODE = "L4-2_v2__seed01"
ACTOR = "uav_facade_l4_2_v2"
CORRIDOR = "corridor_l4_2_v2_uav_facade_l4_2_v2_02"
PREDICATES = (
    "airspace.aircraft_occupies_airspace_corridor",
    "airspace.aircraft_moving_in_corridor",
    "airspace.aircraft_stationary_in_corridor",
)


def load_oracle_case(repo_root: Path) -> dict:
    """Read 19–24 seconds as oracle evidence, never as prediction inputs."""
    directory = Path("aw_data/objective_semantic_truth") / EPISODE
    summary_path = directory / "world_truth_summary.json"
    summary = json.loads((repo_root / summary_path).read_text())
    step = summary["governed_parameters"]["formal_step_ticks"]
    if step != 5:
        raise ValueError(f"Expected capture step 5 for this declared case, got {step}")
    ticks = tuple(range(190, 241, step))
    records = {}
    record_path = directory / "domain_state_observations.jsonl"
    with (repo_root / record_path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["observation_family"] != "predicate_contract_aircraft_corridor_geometry":
                continue
            values = row["values"]
            if row["tick"] not in ticks or values["aircraft_id"] != ACTOR or values["corridor_id"] != CORRIDOR:
                continue
            if row["tick"] in records:
                raise ValueError(f"Duplicate oracle record at {row['tick']}")
            records[row["tick"]] = {"record": row, "source_path": str(record_path), "source_line": line_number}
    if set(records) != set(ticks):
        raise ValueError(f"Missing oracle ticks: {set(ticks) - set(records)}")
    labels = {tick: {} for tick in ticks}
    delta_path = directory / "world_truth_graph_deltas.jsonl"
    with open_artifact_text(repo_root / delta_path) as stream:
        for line_number, line in enumerate(stream, 1):
            delta = json.loads(line)
            if delta["tick"] not in ticks:
                continue
            for operation in delta["operations"]:
                if operation.get("predicate_id") not in PREDICATES or operation.get("bindings") != {"actor": ACTOR, "corridor": CORRIDOR}:
                    continue
                if operation["operation"] == "set_predicate_value":
                    value = operation["to_value"]
                elif operation["operation"] == "refresh_predicate_evidence":
                    value = operation["value"]
                else:
                    raise ValueError(f"Unsupported oracle delta operation: {operation['operation']}")
                predicate = operation["predicate_id"]
                if predicate in labels[delta["tick"]]:
                    raise ValueError("Duplicate saved predicate evidence")
                labels[delta["tick"]][predicate] = {"value": value, "tuple_id": operation["tuple_id"], "source_path": str(delta_path), "source_line": line_number}
    for tick in ticks:
        if set(labels[tick]) != set(PREDICATES):
            raise ValueError(f"Incomplete saved predicate evidence at {tick}")
    return {"episode_id": EPISODE, "actor_id": ACTOR, "corridor_id": CORRIDOR,
            "ticks": ticks, "summary": summary, "summary_path": str(summary_path),
            "records": records, "saved_labels": labels, "purpose": "oracle_rule_replay_only"}
