"""
Auto-generated P0 scenario spec for: L6-1_v1
Event type: L6-1 — C2 Link Complete Loss
Category: failure
CAAC ref: CAAC-9 (C2 link loss exceeding 30 seconds)
SORA SAIL: V
Severity: major
"""

import json, sys
from pathlib import Path

# Ensure Dataset/tools is on the path
_TOOLS = Path(__file__).resolve().parent.parent.parent.parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

from spec_compiler import ScenarioSpec, EventStepSpec, ActionSpec, SpecCompiler, WaypointSpec
from action_templates import ActionTemplates as AT


def build_spec():
    """Build and return the ScenarioSpec. Edit this function to customize."""
    # This spec is rebuilt from the archetype in generate_p0_scenarios.py.
    # Load the compiled event_script.json for reference, or customize below.
    script_path = Path(__file__).resolve().parent / "event_script.json"
    if script_path.exists():
        print(f"Loading compiled spec from {script_path}")
        print("To customize: edit build_spec() above, or modify the archetype and re-run generate_p0_scenarios.py")
        return None  # Signal that event_script.json is the authoritative source

    # Fallback: define spec manually here (copy from archetype output)
    return ScenarioSpec(
        scenario_id="L6-1_v1",
        category="failure.l6-1",
        description="Link loss, successful RTH and landing",
        duration_ticks=900,
    )


if __name__ == "__main__":
    spec = build_spec()
    if spec is not None:
        compiler = SpecCompiler()
        compiled = compiler.compile(spec)
        out_path = Path(__file__).resolve().parent / "event_script.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(compiled, f, indent=2, ensure_ascii=False)
        print(f"Compiled spec -> {out_path}")
    else:
        print("event_script.json is the authoritative source. No recompilation needed.")
