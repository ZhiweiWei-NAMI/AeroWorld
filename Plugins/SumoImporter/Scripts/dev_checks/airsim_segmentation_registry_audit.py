#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SUMO_SCRIPTS_DIR = SCRIPT_DIR.parent

if str(SUMO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SUMO_SCRIPTS_DIR))

from donghu_core.discovery import project_root_from  # noqa: E402
from editor_hook_client import FixedWorldCaptureEditorHook  # noqa: E402


LEGACY_AIRSIM_SEGMENTATION_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "class_id": 1,
        "class_name": "city_base_background",
        "actor_regex": r".*(BP_CityBaseGenerator0|BP_CityBaseGenerator_C.*|BP_CityBaseGenerator.*).*",
        "component_regex": r".*(BP_CityBaseGenerator0|BP_CityBaseGenerator_C.*|BP_CityBaseGenerator.*).*",
        "canonical_actor_label": "BP_CityBaseGenerator0",
        "required_for_static_audit": True,
        "category": "static_city_base",
    },
    {
        "class_id": 2,
        "class_name": "building_style1",
        "actor_regex": r".*BP_Archi_Style1_C.*",
        "component_regex": r".*BP_Archi_Style1_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 3,
        "class_name": "building_style3",
        "actor_regex": r".*BP_Archi_Style3_C.*",
        "component_regex": r".*BP_Archi_Style3_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 4,
        "class_name": "building_style4",
        "actor_regex": r".*BP_Archi_Style4_C.*",
        "component_regex": r".*BP_Archi_Style4_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 5,
        "class_name": "building_style05",
        "actor_regex": r".*BP_Archi_Style05_C.*",
        "component_regex": r".*BP_Archi_Style05_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 6,
        "class_name": "building_roof",
        "actor_regex": r".*BP_Archi_Roof_C.*",
        "component_regex": r".*BP_Archi_Roof_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 7,
        "class_name": "building_pitched_roof",
        "actor_regex": r".*BP_Archi_PitchedRoof_C.*",
        "component_regex": r".*BP_Archi_PitchedRoof_C.*",
        "required_for_static_audit": False,
        "category": "building",
    },
    {
        "class_id": 20,
        "class_name": "uav",
    "actor_regex": r".*(CaptureUAV_0|Quadrotor|uav).*",
    "component_regex": r".*(CaptureUAV_0|Quadrotor|uav).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 21,
        "class_name": "vehicle",
        "actor_regex": r".*(Vehicle|BoxCar|SUV|Ambulance|Police).*",
        "component_regex": r".*(Vehicle|BoxCar|SUV|Ambulance|Police).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 22,
        "class_name": "pedestrian",
        "actor_regex": r".*(Pedestrian|ped_).*",
        "component_regex": r".*(Pedestrian|ped_).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 23,
        "class_name": "roadwork_prop",
        "actor_regex": r".*(Roadwork|ConstructionFence|TrafficCone|Barrier).*",
        "component_regex": r".*(Roadwork|ConstructionFence|TrafficCone|Barrier).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 24,
        "class_name": "traffic_control",
        "actor_regex": r".*(TrafficControl|Signal|PoliceSign|PoliceTape).*",
        "component_regex": r".*(TrafficControl|Signal|PoliceTape).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 25,
        "class_name": "facility",
        "actor_regex": r".*(LandingPad|Charger|BaseTower|Facility).*",
        "component_regex": r".*(LandingPad|Charger|BaseTower|Facility).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
    {
        "class_id": 26,
        "class_name": "hazard_trigger",
        "actor_regex": r".*(NoFly|Hazard|Trigger).*",
        "component_regex": r".*(NoFly|Hazard|Trigger).*",
        "required_for_static_audit": False,
        "category": "optional_rendered_trigger",
    },
    {
        "class_id": 27,
        "class_name": "service_misc_prop",
        "actor_regex": r".*(DeliveryBag|Backpack|Phone|Umbrella|Service).*",
        "component_regex": r".*(DeliveryBag|Backpack|Phone|Umbrella|Service).*",
        "required_for_static_audit": False,
        "category": "dynamic_actor",
    },
)


def parse_args() -> argparse.Namespace:
    project_root = project_root_from(Path(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Read-only legacy AirSim segmentation registry audit. This never calls "
            "add_new_actor_to_instance_segmentation or simSetSegmentationObjectID."
        )
    )
    parser.add_argument("--map-id", default="donghu_road_topo")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "Saved" / "AirSim" / "airsim_segmentation_registry_audit.json",
    )
    parser.add_argument("--discovery-timeout-s", type=float, default=15.0)
    parser.add_argument("--capture-timeout-s", type=float, default=15.0)
    return parser.parse_args()


def _remote_audit_script(classes: tuple[dict[str, Any], ...]) -> str:
    classes_for_remote = [
        {
            "class_id": int(item["class_id"]),
            "class_name": str(item["class_name"]),
            "actor_regex": str(item["actor_regex"]),
            "component_regex": str(item["component_regex"]),
            "canonical_actor_label": str(item.get("canonical_actor_label") or ""),
            "required_for_static_audit": bool(item.get("required_for_static_audit", False)),
            "category": str(item.get("category") or ""),
        }
        for item in classes
    ]
    return f"""
import json
import re
import unreal

classes = json.loads({json.dumps(json.dumps(classes_for_remote, ensure_ascii=True))})
payload = {{
    "audit_version": "legacy_airsim_segmentation_registry_audit.v1",
    "formal_output_allowed": False,
    "mutation_enabled": False,
    "mutation_policy": "read_only_contract_no_pie_or_airsim_segmentation_mutation",
    "world": "",
    "actor_count": 0,
    "sim_mode_found": False,
    "sim_mode_name": "",
    "classes": {{}},
    "errors": [],
}}

for item in classes:
    payload["classes"][item["class_name"]] = {{
        "class_id": item["class_id"],
        "class_name": item["class_name"],
        "category": item["category"],
        "required_for_static_audit": item["required_for_static_audit"],
        "actor_regex": item["actor_regex"],
        "component_regex": item["component_regex"],
        "canonical_actor_label": item["canonical_actor_label"],
        "actor_match_count": 0,
        "actor_sample": [],
        "registered_actor_count": 0,
    }}

try:
    world = unreal.EditorLevelLibrary.get_game_world()
    payload["world"] = "game_world"
except Exception as exc:
    world = None
    payload["errors"].append("get_game_world: " + str(exc))

actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor) if world else []
payload["actor_count"] = len(actors)
sim_mode = None
for actor in actors:
    cls = actor.get_class().get_name()
    name = actor.get_name()
    label = actor.get_actor_label()
    if "SimMode" in cls or "SimMode" in name or "SimMode" in label:
        sim_mode = actor
        payload["sim_mode_found"] = True
        payload["sim_mode_name"] = name + "|" + label + "|" + cls
        break

compiled = [(item["class_name"], re.compile(item["actor_regex"], re.IGNORECASE)) for item in classes]
for actor in actors:
    name = actor.get_name()
    label = actor.get_actor_label()
    cls = actor.get_class().get_name()
    values = [name, label, cls, name + "|" + label + "|" + cls]
    for class_name, pattern in compiled:
        if any(pattern.fullmatch(value) for value in values):
            row = payload["classes"][class_name]
            row["actor_match_count"] += 1
            if len(row["actor_sample"]) < 20:
                row["actor_sample"].append(name + "|" + label + "|" + cls)
            break

print("AEROWORLD_AIRSIM_SEGMENTATION_AUDIT_JSON_BEGIN" + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "AEROWORLD_AIRSIM_SEGMENTATION_AUDIT_JSON_END")
"""


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    project_root = project_root_from(Path(__file__))
    hook = FixedWorldCaptureEditorHook(
        project_root=project_root,
        discovery_timeout_s=float(args.discovery_timeout_s),
        capture_timeout_s=float(args.capture_timeout_s),
    )
    try:
        result = hook.remote.run_python(
            _remote_audit_script(LEGACY_AIRSIM_SEGMENTATION_CLASSES),
            unattended=False,
            raise_on_failure=True,
        )
    finally:
        hook.close()

    output_text = "".join(str(item.get("output", "")) for item in result.get("output") or [] if isinstance(item, dict))
    match = re.search(
        r"AEROWORLD_AIRSIM_SEGMENTATION_AUDIT_JSON_BEGIN(.*?)AEROWORLD_AIRSIM_SEGMENTATION_AUDIT_JSON_END",
        output_text,
        flags=re.S,
    )
    if not match:
        raise RuntimeError(f"Unable to parse AirSim segmentation audit output: {output_text[:2000]}")
    payload = json.loads(match.group(1))
    payload["map_id"] = str(args.map_id)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_audit(args)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "mutation_enabled": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
