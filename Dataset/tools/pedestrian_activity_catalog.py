from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PedestrianActivity:
    activity_type: str
    animation_hint: str
    posture: str = "standing"
    social_state: str = "solo"
    moving: bool = False
    render_action: dict[str, Any] | None = None


WALKING_ASSET = "/Game/MixamoAssets/Animations/Walking.Walking"
TEXTING_ASSET = "/Game/MixamoAssets/Animations/Walking_While_Texting.Walking_While_Texting"
PHONE_ASSET = "/Game/MixamoAssets/Animations/Talking_Phone_Pacing.Talking_Phone_Pacing"
TALKING_ASSET = "/Game/MixamoAssets/Animations/Talking.Talking"
YELLING_ASSET = "/Game/MixamoAssets/Animations/Yelling.Yelling"
FALL_ASSET = "/Game/MixamoAssets/Animations/AM_Fall_Flat_2Stage.AM_Fall_Flat_2Stage"


def _play(asset: str, *, loop_count: int = 999, reapply: bool = False, freeze: bool = False) -> dict[str, Any]:
    action: dict[str, Any] = {
        "op": "ped_play_animation",
        "animation_asset_path": asset,
        "play_rate": 1.0,
        "loop_count": loop_count,
    }
    if reapply:
        action["reapply_after_pose_sync"] = True
    if freeze:
        action["freeze_pose_while_active"] = True
    return action


ACTIVITIES: dict[str, PedestrianActivity] = {
    "idle": PedestrianActivity("idle", "pedestrian_idle", render_action={"op": "ped_stop"}),
    "standing": PedestrianActivity("standing", "pedestrian_idle", render_action={"op": "ped_stop"}),
    "waiting": PedestrianActivity("waiting", "pedestrian_idle", render_action={"op": "ped_stop"}),
    "stopped": PedestrianActivity("stopped", "pedestrian_idle", render_action={"op": "ped_stop"}),
    "observing": PedestrianActivity("observing", "pedestrian_observe", social_state="observer", render_action={"op": "ped_observe"}),
    "walking": PedestrianActivity("walking", "pedestrian_walk", moving=True, render_action=_play(WALKING_ASSET, reapply=True)),
    "crossing": PedestrianActivity("crossing", "pedestrian_crossing", moving=True, render_action=_play(WALKING_ASSET, reapply=True)),
    "evacuating": PedestrianActivity("evacuating", "pedestrian_evacuation_walk", social_state="group", moving=True, render_action=_play(WALKING_ASSET, reapply=True)),
    "texting_walk": PedestrianActivity("texting_walk", "pedestrian_texting_walk", moving=True, render_action=_play(TEXTING_ASSET, reapply=True)),
    "phone_call": PedestrianActivity("phone_call", "pedestrian_phone_call", render_action=_play(PHONE_ASSET, loop_count=999, freeze=True)),
    "chatting": PedestrianActivity("chatting", "pedestrian_talking", social_state="group", render_action=_play(TALKING_ASSET, loop_count=999, freeze=True)),
    "quarrel": PedestrianActivity("quarrel", "pedestrian_yelling", social_state="group", render_action=_play(YELLING_ASSET, loop_count=3, freeze=True)),
    "medical_incident": PedestrianActivity(
        "medical_incident",
        "pedestrian_fall",
        posture="fallen",
        render_action={
            "freeze_pose_while_active": True,
            "ground_lift_m": 0.9,
            "ops": [
                {"op": "ped_stop"},
                {
                    "op": "ped_play_animation",
                    "animation_asset_path": FALL_ASSET,
                    "start_section": "Start",
                    "play_rate": 1.0,
                    "loop_count": 1,
                },
            ],
        },
    ),
}


ALIASES = {
    "": "waiting",
    "observe": "observing",
    "watching": "observing",
    "fall_flat": "medical_incident",
    "fallen_hold": "medical_incident",
    "fall_start": "medical_incident",
    "moving": "walking",
}


def normalize_activity_type(value: str | None, *, moving: bool = False) -> str:
    raw = str(value or "").strip().lower()
    activity = ALIASES.get(raw, raw)
    if not activity:
        activity = "walking" if moving else "waiting"
    if moving and activity in {"idle", "standing", "waiting", "stopped"}:
        activity = "walking"
    if activity not in ACTIVITIES:
        raise ValueError(f"Unknown pedestrian activity_type: {value!r}")
    return activity


def get_activity(value: str | None, *, moving: bool = False) -> PedestrianActivity:
    return ACTIVITIES[normalize_activity_type(value, moving=moving)]


def activity_annotations(activity_type: str, *, speed_mps: float = 0.0) -> dict[str, Any]:
    activity = get_activity(activity_type)
    return {
        "activity_type": activity.activity_type,
        "speed_mps": round(float(speed_mps), 4),
        "state_facets": {
            "activity": {
                "activity_type": activity.activity_type,
                "animation_hint": activity.animation_hint,
                "posture": activity.posture,
                "social_state": activity.social_state,
            }
        },
    }


def activity_actions_for_template() -> dict[str, dict[str, Any]]:
    return {
        activity_type: dict(activity.render_action or {"op": "none"})
        for activity_type, activity in ACTIVITIES.items()
    }


def ue_asset_to_content_path(asset_path: str, repo_root: Path) -> Path | None:
    if not asset_path.startswith("/Game/") or "." not in asset_path:
        return None
    package = asset_path.split(".", 1)[0]
    return repo_root / "Content" / (package[len("/Game/") :] + ".uasset")


def iter_animation_asset_paths() -> list[str]:
    paths: list[str] = []
    for activity in ACTIVITIES.values():
        actions = [activity.render_action or {}]
        if activity.render_action and activity.render_action.get("ops"):
            actions = list(activity.render_action["ops"])
        for action in actions:
            asset_path = action.get("animation_asset_path")
            if asset_path:
                paths.append(str(asset_path))
    return sorted(set(paths))


def validate_local_animation_assets(repo_root: Path) -> list[str]:
    errors: list[str] = []
    for asset_path in iter_animation_asset_paths():
        content_path = ue_asset_to_content_path(asset_path, repo_root)
        if content_path is not None and not content_path.exists():
            errors.append(f"{asset_path} does not resolve to local asset {content_path}")
    return errors
