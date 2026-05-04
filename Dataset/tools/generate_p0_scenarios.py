"""
P0 场景批量生成器。

从 taxonomy.json 读取 33 个事件类型定义，为每个类型生成:
  - spec.py (ScenarioSpec Python 定义)
  - event_script.json (编译产物)

使用事件原型 (archetype) 模式减少重复: 每个事件类型映射到一个原型，
原型提供默认的 entity 布局、trigger 模式、action 序列。

使用方式:
    python generate_p0_scenarios.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_compiler import (
    ScenarioSpec, EntitySpec, EventStepSpec, ActionSpec, SpecCompiler,
    make_tick_trigger, make_weather_trigger, make_proximity_trigger,
    make_event_fired_trigger, make_composite_trigger, WaypointSpec,
)
from action_templates import ActionTemplates as AT


# ============================================================================
# Event Archetypes — 每个原型定义一个可参数化的场景模板
# ============================================================================

class Archetype:
    """场景原型基类"""

    def build_spec(self, et: dict, variant: dict, layer: str, scenario_dir: str) -> ScenarioSpec:
        raise NotImplementedError


class WeatherTriggeredArchetype(Archetype):
    """
    天气变化触发多 agent 响应链。
    适用于: L5-1 (rain), L5-2 (fog), L5-3 (wind)
    """

    def build_spec(self, et, variant, layer, scenario_dir):
        eid = variant["id"]
        weather_profile = {"L5-1": "rain", "L5-2": "fog", "L5-3": "wind"}.get(et["event_type_id"], "rain")
        severity = variant.get("severity", "minor")

        intensity = {"minor": 0.4, "major": 0.7, "critical": 0.95}.get(severity, 0.7)
        vis_reduction = {"minor": 0.3, "major": 0.7, "critical": 0.95}.get(severity, 0.5)

        return ScenarioSpec(
            scenario_id=eid,
            category=f"environment.{weather_profile}",
            description=variant["name"],
            duration_ticks=900,
            parameters={
                "weather_onset_tick": 300,
            },
            entities=[
                EntitySpec("uav_patrol_01", "uav.inspect.quad.v1",
                           [50.0, 20.0, 30.0], [0, 0, 0],
                           movement_waypoints=[WaypointSpec([55.0, 25.0, 30.0])]),
                EntitySpec("car_traffic_01", "vehicle.ground.boxcar.v1",
                           [45.0, 22.0, 0.0], [0, 0, 0]),
                EntitySpec("ped_commuter_01", "pedestrian.cityops.basic.v1",
                           [52.0, 20.0, 0.0], [0, 0, 0]),
            ],
            event_chain=[
                EventStepSpec(
                    event_id="weather_onset",
                    trigger=make_tick_trigger(300),
                    priority=1,
                    actions=[ActionSpec("set_weather", AT.environment_cascade(weather_profile, intensity, vis_reduction))],
                    on_fire_emit=["uav_reaction"],
                    log_topic=f"evt_{eid}_onset", log_category="weather",
                    log_title=f"{variant['name']} — 天气变化开始",
                    log_severity="warning", log_overlay="weather",
                    log_target_ids=["weather_global"],
                ),
                EventStepSpec(
                    event_id="uav_reaction",
                    trigger=make_event_fired_trigger("weather_onset"),
                    priority=2,
                    actions=[ActionSpec("move_entity", AT.move_entity("uav_patrol_01", [[50, 20, 30], [40, 15, 30]], 3.0))],
                    log_topic=f"evt_{eid}_uav_react", log_category="uav_mission",
                    log_title="UAV 因天气变化调整航线",
                    log_severity="info" if severity == "minor" else "warning",
                    log_overlay="uav_mission", log_target_ids=["uav_patrol_01"],
                ),
            ],
        )


class CollisionNearMissArchetype(Archetype):
    """
    两个 agent 接近 → 冲突 → 避让/碰撞。
    适用于: L4-1 (UAV-UAV), L4-5 (UAV-pedestrian), L4-6 (pedestrian-vehicle), L4-9 (vehicle-vehicle)
    """

    def build_spec(self, et, variant, layer, scenario_dir):
        eid = variant["id"]
        severity = variant.get("severity", "major")
        etype = et["event_type_id"]

        # Determine agent types based on event type
        agent_configs = {
            "L4-1": ("uav.inspect.quad.v1", "uav.airsim.flying_pawn.v1", "uav_a", "uav_b", 20.0, 15.0),
            "L4-4": ("uav.inspect.quad.v1", "vehicle.ground.boxcar.v1", "uav_01", "car_01", 15.0, 3.0),
            "L4-5": ("uav.inspect.quad.v1", "pedestrian.cityops.basic.v1", "uav_01", "ped_01", 10.0, 3.0),
            "L4-6": ("pedestrian.cityops.basic.v1", "vehicle.ground.boxcar.v1", "ped_01", "car_01", 8.0, 2.0),
            "L4-9": ("vehicle.ground.boxcar.v1", "vehicle.emergency.suv.v1", "car_a", "car_b", 10.0, 2.0),
        }
        cfg = agent_configs.get(etype, agent_configs["L4-5"])
        asset_a, asset_b, id_a, id_b, approach_dist, collision_dist = cfg

        collision_action = AT.ped_stagger(id_b) if "pedestrian" in asset_b else AT.move_entity(id_b, [[50, 20, 0], [52, 22, 0]], 8.0)
        evasive_action = AT.move_entity(id_a, [[55, 18, 30], [58, 16, 30]], 10.0) if "uav" in asset_a else AT.move_entity(id_a, [[50, 20, 0], [48, 18, 0]], 5.0)

        # Determine z-levels based on agent type
        z_a = 30.0 if "uav" in asset_a else 0.0
        z_b = 25.0 if "uav" in asset_b else 0.0

        return ScenarioSpec(
            scenario_id=eid,
            category=f"collision.{etype.lower()}",
            description=variant["name"],
            duration_ticks=900,
            parameters={
                "approach_tick": 400,
                "conflict_tick": 450,
                "resolution_tick": 480,
            },
            entities=[
                EntitySpec(id_a, asset_a, [55.0, 18.0, z_a], [0, 0, 0],
                           movement_waypoints=[WaypointSpec([50.0, 20.5, z_a])]),
                EntitySpec(id_b, asset_b, [48.0, 22.0, z_b], [0, 0, 0],
                           movement_waypoints=[WaypointSpec([50.0, 20.0, z_b])]),
            ],
            event_chain=[
                EventStepSpec(
                    event_id="approach_phase",
                    trigger=make_tick_trigger(400),
                    priority=1,
                    actions=[ActionSpec("move_entity", AT.move_entity(id_a, [[55, 18, z_a], [50, 20, z_a]], 8.0)),
                             ActionSpec("move_entity", AT.move_entity(id_b, [[48, 22, z_b], [50, 20, z_b]], 7.0))],
                    on_fire_emit=["proximity_alert"],
                    log_topic=f"evt_{eid}_approach", log_category="uav_mission",
                    log_title="两实体正在接近",
                    log_severity="info", log_overlay="uav_mission",
                    log_target_ids=[id_a, id_b],
                ),
                EventStepSpec(
                    event_id="proximity_alert",
                    trigger=make_tick_trigger(450),
                    priority=2,
                    actions=[ActionSpec("move_entity", evasive_action),
                             ActionSpec("capture_screenshot", AT.capture_screenshot("demo_overview"))],
                    log_topic=f"evt_{eid}_conflict", log_category="uav_mission",
                    log_title="冲突检测 — 避让动作触发",
                    log_severity="warning" if severity != "critical" else "critical",
                    log_overlay="uav_mission",
                    log_target_ids=[id_a, id_b],
                ),
            ],
        )

class SystemFailureArchetype(Archetype):
    """
    UAV 系统故障 → 紧急响应。
    适用于: L4-3 (forced landing), L4-11 (vehicle breakdown), L6-1 (C2 loss), L6-3 (GNSS spoof)
    """

    def build_spec(self, et, variant, layer, scenario_dir):
        eid = variant["id"]
        severity = variant.get("severity", "major")
        etype = et["event_type_id"]

        return ScenarioSpec(
            scenario_id=eid,
            category=f"failure.{etype.lower()}",
            description=variant["name"],
            duration_ticks=900,
            parameters={
                "failure_tick": 400,
            },
            entities=[
                EntitySpec("uav_ops_01", "uav.inspect.quad.v1", [50.0, 20.0, 35.0], [0, 0, 0]),
                EntitySpec("pad_emergency_01", "facility.landing_pad.visible.v1", [45.0, 15.0, 0.0], [0, 0, 0]),
                EntitySpec("ped_bystander_01", "pedestrian.cityops.basic.v1", [47.0, 16.0, 0.0], [0, 0, 0]),
            ],
            event_chain=[
                EventStepSpec(
                    event_id="failure_detected",
                    trigger=make_tick_trigger(400),
                    priority=1,
                    actions=[
                        ActionSpec("move_entity", AT.move_entity("uav_ops_01", [[50, 20, 35], [46, 16, 15]], 4.0)),
                    ],
                    on_fire_emit=["emergency_landing"],
                    log_topic=f"evt_{eid}_failure", log_category="uav_mission",
                    log_title=f"{variant['name']} — 系统失效检测",
                    log_severity="critical", log_overlay="uav_mission",
                    log_target_ids=["uav_ops_01"],
                ),
                EventStepSpec(
                    event_id="emergency_landing",
                    trigger=make_event_fired_trigger("failure_detected"),
                    priority=2,
                    actions=[
                        ActionSpec("move_entity", AT.move_entity("uav_ops_01", [[46, 16, 15], [45, 15, 0.5]], 2.0)),
                        ActionSpec("capture_screenshot", AT.capture_screenshot("demo_overview")),
                    ],
                    log_topic=f"evt_{eid}_landing", log_category="uav_mission",
                    log_title="紧急迫降执行",
                    log_severity="critical", log_overlay="uav_mission",
                    log_target_ids=["uav_ops_01", "pad_emergency_01"],
                ),
            ],
        )


class ZoneViolationArchetype(Archetype):
    """
    Agent 进入禁区 → 检测 → 响应。
    适用于: L1-1 (geofence), L1-3 (intrusion), L3-2 (TFR)
    """

    def build_spec(self, et, variant, layer, scenario_dir):
        eid = variant["id"]
        severity = variant.get("severity", "major")

        return ScenarioSpec(
            scenario_id=eid,
            category=f"violation.{et['event_type_id'].lower()}",
            description=variant["name"],
            duration_ticks=900,
            parameters={
                "violation_tick": 400,
            },
            entities=[
                EntitySpec("uav_delivery_01", "uav.airsim.flying_pawn.v1", [60.0, 25.0, 25.0], [0, 0, 0],
                           movement_waypoints=[WaypointSpec([52.0, 20.0, 25.0])]),
                EntitySpec("nfz_critical_01", "trigger.no_fly.box.v1", [50.0, 20.0, 15.0], [0, 0, 0]),
            ],
            event_chain=[
                EventStepSpec(
                    event_id="zone_approach",
                    trigger=make_tick_trigger(400),
                    priority=1,
                    actions=[ActionSpec("move_entity", AT.move_entity("uav_delivery_01", [[60, 25, 25], [52, 20, 25]], 6.0))],
                    on_fire_emit=["zone_breach"],
                    log_topic=f"evt_{eid}_approach", log_category="uav_mission",
                    log_title="UAV 接近限制空域",
                    log_severity="warning", log_overlay="uav_mission",
                    log_target_ids=["uav_delivery_01", "nfz_critical_01"],
                ),
                EventStepSpec(
                    event_id="zone_breach",
                    trigger=make_event_fired_trigger("zone_approach"),
                    priority=2,
                    actions=[ActionSpec("move_entity", AT.move_entity("uav_delivery_01", [[52, 20, 25], [58, 25, 30]], 8.0))],
                    log_topic=f"evt_{eid}_breach", log_category="uav_mission",
                    log_title="空域违规 — 强制返航",
                    log_severity="critical", log_overlay="uav_mission",
                    log_target_ids=["uav_delivery_01", "nfz_critical_01"],
                ),
            ],
        )


class OperationalResponseArchetype(Archetype):
    """
    事件检测 → 应急调度 → 到达确认。
    适用于: L4-7 (pedestrian fall), L4-8 (crowd), L4-10 (emergency vehicle)
    """

    def build_spec(self, et, variant, layer, scenario_dir):
        eid = variant["id"]
        severity = variant.get("severity", "major")
        etype = et["event_type_id"]

        is_crowd = "crowd" in variant["name"].lower() or etype == "L4-8"
        is_ev = "emergency" in variant["name"].lower() or "ambulance" in variant["name"].lower() or etype == "L4-10"

        return ScenarioSpec(
            scenario_id=eid,
            category=f"operational.{etype.lower()}",
            description=variant["name"],
            duration_ticks=900,
            parameters={
                "incident_tick": 400,
            },
            entities=[
                EntitySpec("uav_patrol_01", "uav.inspect.quad.v1", [60.0, 25.0, 30.0], [0, 0, 0],
                           movement_waypoints=[WaypointSpec([52.0, 20.0, 15.0])]),
                EntitySpec("ped_victim_01", "pedestrian.cityops.basic.v1", [52.0, 19.5, 0.0], [0, 0, 0]),
                EntitySpec("ambulance_response_01", "vehicle.emergency.ambulance.v1", [80.0, 40.0, 0.0], [0, 0, 0]),
            ],
            event_chain=[
                EventStepSpec(
                    event_id="incident_occurred",
                    trigger=make_tick_trigger(400),
                    priority=1,
                    actions=[
                        ActionSpec("play_animation", AT.ped_fall_flat("ped_victim_01")),
                        ActionSpec("move_entity", AT.move_entity("uav_patrol_01", [[60, 25, 30], [52, 20, 15]], 5.0)),
                    ],
                    on_fire_emit=["dispatch_responder"],
                    log_topic=f"evt_{eid}_incident", log_category="pedestrian",
                    log_title=f"{variant['name']} — 事件检测",
                    log_severity="critical" if severity == "critical" else "warning",
                    log_overlay="pedestrian", log_target_ids=["ped_victim_01"],
                ),
                EventStepSpec(
                    event_id="dispatch_responder",
                    trigger=make_event_fired_trigger("incident_occurred"),
                    priority=2,
                    actions=[
                        ActionSpec("spawn_entity", AT.spawn_vehicle("ambulance_response_01", "ambulance", [80, 40, 0],
                                     visual_state={"mode": "response", "lights_on": True})),
                        ActionSpec("move_entity", AT.move_entity("ambulance_response_01", [[80, 40, 0], [52, 19, 0]], 12.0)),
                    ],
                    on_fire_emit=["responder_arrived"],
                    log_topic=f"evt_{eid}_dispatch", log_category="vehicle",
                    log_title="应急响应派遣",
                    log_severity="warning", log_overlay="vehicle",
                    log_target_ids=["ambulance_response_01"],
                ),
                EventStepSpec(
                    event_id="responder_arrived",
                    trigger=make_event_fired_trigger("dispatch_responder"),
                    priority=3,
                    actions=[ActionSpec("capture_screenshot", AT.capture_screenshot("demo_overview"))],
                    log_topic=f"evt_{eid}_arrival", log_category="uav_mission",
                    log_title="应急响应到达确认",
                    log_severity="info", log_overlay="uav_mission",
                    log_target_ids=["uav_patrol_01", "ambulance_response_01", "ped_victim_01"],
                ),
            ],
        )


class InfrastructureFailureArchetype(Archetype):
    """
    基础设施故障 → 服务降级 → agent 响应。
    适用于: L2-1 (station failure), L2-2 (GNSS), L2-3 (charger), L2-4 (pad contention), L2-5 (traffic signal)
    """

    def build_spec(self, et, variant, layer, scenario_dir):
        eid = variant["id"]
        severity = variant.get("severity", "minor")
        etype = et["event_type_id"]

        return ScenarioSpec(
            scenario_id=eid,
            category=f"failure.{etype.lower()}",
            description=variant["name"],
            duration_ticks=900,
            parameters={
                "failure_tick": 400,
            },
            entities=[
                EntitySpec("uav_ops_01", "uav.inspect.quad.v1", [50.0, 20.0, 30.0], [0, 0, 0]),
                EntitySpec("infra_station_01", "facility.radio.base_tower.v1", [48.0, 22.0, 0.0], [0, 0, 0]),
            ],
            event_chain=[
                EventStepSpec(
                    event_id="infra_failure",
                    trigger=make_tick_trigger(400),
                    priority=1,
                    actions=[
                        ActionSpec("set_visual_state", AT.set_visual_state("infra_station_01", mode="disabled")),
                    ],
                    on_fire_emit=["service_degraded"],
                    log_topic=f"evt_{eid}_failure", log_category="infrastructure",
                    log_title=f"{variant['name']} — 基础设施故障",
                    log_severity="warning", log_overlay="infrastructure",
                    log_target_ids=["infra_station_01"],
                ),
                EventStepSpec(
                    event_id="service_degraded",
                    trigger=make_event_fired_trigger("infra_failure"),
                    priority=2,
                    actions=[ActionSpec("move_entity", AT.move_entity("uav_ops_01", [[50, 20, 30], [48, 22, 20]], 4.0))],
                    log_topic=f"evt_{eid}_degraded", log_category="uav_mission",
                    log_title="服务降级 — UAV 调整运行模式",
                    log_severity="info" if severity == "minor" else "warning",
                    log_overlay="uav_mission", log_target_ids=["uav_ops_01"],
                ),
            ],
        )


# ============================================================================
# Event type → Archetype mapping
# ============================================================================

ARCHETYPE_MAP = {
    "L1-1": ZoneViolationArchetype,
    "L1-2": ZoneViolationArchetype,
    "L1-3": ZoneViolationArchetype,
    "L1-4": ZoneViolationArchetype,
    "L2-1": InfrastructureFailureArchetype,
    "L2-2": InfrastructureFailureArchetype,
    "L2-3": InfrastructureFailureArchetype,
    "L2-4": InfrastructureFailureArchetype,
    "L2-5": InfrastructureFailureArchetype,
    "L3-1": OperationalResponseArchetype,
    "L3-2": ZoneViolationArchetype,
    "L3-3": OperationalResponseArchetype,
    "L4-1": CollisionNearMissArchetype,
    "L4-2": CollisionNearMissArchetype,
    "L4-3": SystemFailureArchetype,
    "L4-4": CollisionNearMissArchetype,
    "L4-5": CollisionNearMissArchetype,
    "L4-6": CollisionNearMissArchetype,
    "L4-7": OperationalResponseArchetype,
    "L4-8": OperationalResponseArchetype,
    "L4-9": CollisionNearMissArchetype,
    "L4-10": OperationalResponseArchetype,
    "L4-11": SystemFailureArchetype,
    "L5-1": WeatherTriggeredArchetype,
    "L5-2": WeatherTriggeredArchetype,
    "L5-3": WeatherTriggeredArchetype,
    "L5-4": WeatherTriggeredArchetype,
    "L5-5": WeatherTriggeredArchetype,
    "L6-1": SystemFailureArchetype,
    "L6-2": SystemFailureArchetype,
    "L6-3": SystemFailureArchetype,
    "L6-4": SystemFailureArchetype,
    "L6-5": SystemFailureArchetype,
}


# ============================================================================
# Main: generate all P0 scenarios
# ============================================================================

LAYER_DIR_MAP = {
    "L1": "L1_airspace",
    "L2": "L2_infrastructure",
    "L3": "L3_dynamic_constraints",
    "L4": "L4_agents",
    "L5": "L5_environment",
    "L6": "L6_digital_layer",
}


def get_scenario_subdir(et: dict) -> str:
    """Determine the subdirectory within the layer for this event type."""
    etype = et["event_type_id"]
    mechanism = et.get("mechanism", "")
    if mechanism == "collision":
        return "collision"
    elif mechanism == "failure":
        return "failure"
    elif mechanism in ("operational", "violation", "environmental"):
        return "interaction"
    return ""


def main() -> None:
    dataset_root = Path(__file__).resolve().parent.parent
    taxonomy_path = dataset_root / "taxonomy.json"
    scenarios_root = dataset_root / "scenarios"

    if not taxonomy_path.exists():
        print(f"ERROR: taxonomy.json not found at {taxonomy_path}", file=sys.stderr)
        sys.exit(1)

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    compiler = SpecCompiler()
    generated = 0

    for et in taxonomy["event_types"]:
        etype_id = et["event_type_id"]
        layer = etype_id.split("-")[0]
        layer_dir = LAYER_DIR_MAP.get(layer, f"L{layer}_unknown")
        subdir = get_scenario_subdir(et)

        archetype_cls = ARCHETYPE_MAP.get(etype_id)
        if archetype_cls is None:
            print(f"  [SKIP] {etype_id}: no archetype mapping")
            continue

        archetype = archetype_cls()

        for variant in et.get("variants", []):
            vid = variant["id"]
            # Build scenario directory path
            if subdir:
                scenario_dir = scenarios_root / layer_dir / subdir / vid
            else:
                scenario_dir = scenarios_root / layer_dir / vid

            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Build spec and compile
            try:
                spec = archetype.build_spec(et, variant, layer, str(scenario_dir))
                compiled = compiler.compile(spec)

                # Write event_script.json
                script_path = scenario_dir / "event_script.json"
                with open(script_path, "w", encoding="utf-8") as f:
                    json.dump(compiled, f, indent=2, ensure_ascii=False)

                # Write spec.py (self-contained, recompilable)
                spec_py_path = scenario_dir / "spec.py"
                with open(spec_py_path, "w", encoding="utf-8") as f:
                    f.write(f'''"""
Auto-generated P0 scenario spec for: {vid}
Event type: {etype_id} — {et["name"]}
Category: {et.get("mechanism", "unknown")}
CAAC ref: {et.get("caac_ref", "N/A")}
SORA SAIL: {et.get("sora_sail", "N/A")}
Severity: {variant.get("severity", "unknown")}
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
        print(f"Loading compiled spec from {{script_path}}")
        print("To customize: edit build_spec() above, or modify the archetype and re-run generate_p0_scenarios.py")
        return None  # Signal that event_script.json is the authoritative source

    # Fallback: define spec manually here (copy from archetype output)
    return ScenarioSpec(
        scenario_id="{vid}",
        category="{et.get('mechanism', 'unknown')}.{etype_id.lower()}",
        description="{variant['name']}",
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
        print(f"Compiled spec -> {{out_path}}")
    else:
        print("event_script.json is the authoritative source. No recompilation needed.")
''')

                # Write README.md
                readme_path = scenario_dir / "README.md"
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(f"""# {vid}: {variant['name']}

- **Event Type**: {etype_id} — {et['name']}
- **ODD Layer**: {layer} ({et.get('odd_layer', 'N/A')})
- **Mechanism**: {et.get('mechanism', 'unknown')}
- **SORA SAIL**: {et.get('sora_sail', 'N/A')}
- **CAAC Reference**: {et.get('caac_ref', 'N/A')}
- **Severity**: {variant.get('severity', 'unknown')}
- **Belcastro Domain**: {et.get('belcastro_ref', 'N/A')}

## Causal Chain
{et.get('causal_chain_sketch', 'N/A')}

## Entities
{', '.join(et.get('entities_required', []))}

## Files
- `event_script.json` — Compiled event script (loadable by EventScriptInterpreter)
- `spec.py` — ScenarioSpec definition (auto-generated, customize for manual tuning)

## Usage
```python
from donghu_core.event_script_interpreter import EventScriptInterpreter
from pathlib import Path
interpreter = EventScriptInterpreter(Path('event_script.json'))
```
""")

                generated += 1
                print(f"  [OK] {vid} → {scenario_dir}/event_script.json")

            except Exception as e:
                print(f"  [FAIL] {vid}: {e}")

    print(f"\nGenerated {generated} P0 scenario(s)")


if __name__ == "__main__":
    main()
