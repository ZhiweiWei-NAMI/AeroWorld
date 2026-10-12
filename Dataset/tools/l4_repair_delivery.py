"""Reproduce L4 business semantics and materialize current-engine UE inputs.

Runs serially. The old business arms are retained until the generated set passes
the semantic and UE checks; all source episodes remain read-only.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path

from Dataset.tools import l4_business_semantics as business
from Dataset.tools import l4_window as window

ROOT = window.ROOT
REPORT = ROOT / "design/L4_arm_workflow_v2/repair_20260919"


def plan_for(config: dict) -> dict:
    tree = ast.parse((ROOT / config["script"]).read_text())
    plans = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "PLAN" for t in n.targets)]
    if len(plans) != 1:
        raise ValueError(f"expected one explicit PLAN in {config['script']}")
    window.require_plan_matches_config(plans[0], config)
    return plans[0]


def write_manifest_inputs(path: Path, config: dict) -> None:
    manifest = window.read_json(path / "manifest.json")
    manifest["ue_execution"] = {
        "method": "current_engine_execution_scoped_to_declared_window",
        "scope": "all authored engine entities, actual weather, interpreter events and executed actions",
        "source_scene": config["sources"]["scene"], "source_script": config["sources"]["script"],
    }
    window.write_json(path / "manifest.json", manifest)


def process(config_path: Path) -> dict:
    from Dataset.tools.arm_ue_export import write_engine_inputs, export_arm

    config = window.load_config(config_path)
    plan = plan_for(config)
    out = ROOT / config["output"]
    local = business.profile_rule(config["episode_id"].split("__")[0])
    existing = window.read_json(out / "manifest.json")
    already_generated = local is not None
    started = time.monotonic()
    if local is not None and not already_generated:
        window.execute_config(config_path, plan)
    else:
        engine, rows, audit, _ = window.run_engine(config, plan)
        w = config["window"]
        selected = [r for r in rows if w["start_tick"] <= r["tick"] <= w["end_tick"]
                    and r["entity_id"] in config["entity_ids"]]
        saved = list(window.read_rows(out / "raw/states.jsonl"))
        if selected != saved:
            raise ValueError(f"current engine no longer reproduces {out}/raw/states.jsonl")
        actions = [r for r in audit if w["start_tick"] <= r["tick"] <= w["end_tick"]]
        if actions != list(window.read_rows(out / "raw/actions.jsonl")):
            raise ValueError(f"current actions differ from {out}/raw/actions.jsonl")
        write_engine_inputs(out, engine, engine.arm_interpreter, w, script=engine.arm_script)
    write_manifest_inputs(out, config)
    exported = export_arm(out, ROOT / config["sources"]["trajectories"].rsplit("/", 1)[0],
                          ROOT / config["sources"]["scene"], ROOT / config["sources"]["script"])
    result = {"arm": config["output"], "episode_id": config["episode_id"], "window": config["window"],
              "business_semantics_regenerated": local is not None, "seconds": time.monotonic() - started,
              "source_scene": config["sources"]["scene"], "source_script": config["sources"]["script"],
              "canonical_dir": config["sources"]["trajectories"].rsplit("/", 1)[0],
              "ue_result": exported}
    # A per-arm completion record enables an interrupted batch to resume without
    # treating a partial directory or an old semantic manifest as completed.
    record = REPORT / "completed" / config["group_key"] / (config["arm_id"] + ".json")
    window.write_json(record, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--seed")
    parser.add_argument("--arm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supersede-to", required=True,
                        help="Explicit repository-relative archive for existing arms")
    args = parser.parse_args()
    os.environ["L4_SUPERSEDE_TO"] = args.supersede_to
    paths = sorted((ROOT / "interventions").glob("L4-*/seed*/*/arm_*.yaml"))
    selected = []
    for path in paths:
        relative = path.relative_to(ROOT / "interventions")
        if args.profile and relative.parts[0] not in args.profile:
            continue
        if args.seed and relative.parts[1] != args.seed:
            continue
        if args.arm and path.stem != args.arm:
            continue
        selected.append(path)
    if not selected:
        raise ValueError("empty L4 repair selection")
    for number, path in enumerate(selected, 1):
        relative = path.relative_to(ROOT / "interventions").with_suffix(".json")
        if args.resume and (REPORT / "completed" / relative).is_file():
            continue
        result = process(path)
        print(json.dumps({"completed": number, "selected": len(selected), "arm": result["arm"],
                          "seconds": result["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
