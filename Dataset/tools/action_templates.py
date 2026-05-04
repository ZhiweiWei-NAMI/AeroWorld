"""
共享动作模板库。

提供常用动作的工厂方法，避免在每个 scenario spec 中手工构建 ActionSpec。
模板返回 dict（可直接放入 ActionSpec.params）或完整 ActionSpec。

使用方式:
    from action_templates import ActionTemplates as AT
    action = AT.set_weather("rain", {"rain": 0.7, "visibility_m": 2000.0})
    chain = AT.emergency_responder_chain("ambulance_01", "vehicle.emergency.ambulance.v1",
                                          [80, 40, 0], [52, 19, 0])
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Action result — 可选的 action_id 生成
# ---------------------------------------------------------------------------

_action_counter: dict[str, int] = {}

def _next_action_id(prefix: str = "act") -> str:
    n = _action_counter.get(prefix, 0) + 1
    _action_counter[prefix] = n
    return f"{prefix}_{n:03d}"


# ---------------------------------------------------------------------------
# ActionTemplates
# ---------------------------------------------------------------------------

class ActionTemplates:
    """静态方法工厂，每个返回 dict（可直接用于 ActionSpec.params 或完整 ActionSpec）。"""

    # ---- 天气 ----

    @staticmethod
    def set_weather(profile: str, overrides: Optional[dict] = None,
                    action_id: Optional[str] = None) -> dict:
        """
        天气变更动作。
        profile: "clear" | "rain" | "fog" | "wind" | "snow"
        overrides: 如 {"rain": 0.7, "visibility_m": 2000.0, "wind_speed": 5.0}
        """
        result = {
            "action_id": action_id or _next_action_id("set_weather"),
            "type": "set_weather",
            "profile": profile,
        }
        if overrides:
            result["overrides"] = overrides
        return result

    @staticmethod
    def weather_clear(action_id: Optional[str] = None) -> dict:
        """恢复晴天的便捷方法。"""
        return ActionTemplates.set_weather("clear", {
            "rain": 0.0, "fog": 0.0, "wind_speed": 2.0,
            "visibility_m": 20000.0, "wetness": 0.0,
        }, action_id)

    @staticmethod
    def weather_rain(intensity: float = 0.7, visibility_m: float = 2000.0,
                     wind_speed: float = 5.0, action_id: Optional[str] = None) -> dict:
        return ActionTemplates.set_weather("rain", {
            "rain": intensity, "visibility_m": visibility_m,
            "wind_speed": wind_speed, "wetness": 0.8,
        }, action_id)

    @staticmethod
    def weather_fog(density: float = 0.8, visibility_m: float = 500.0,
                    action_id: Optional[str] = None) -> dict:
        return ActionTemplates.set_weather("fog", {
            "fog": density, "visibility_m": visibility_m,
        }, action_id)

    @staticmethod
    def weather_wind(wind_speed: float = 15.0, gust_factor: float = 1.5,
                     action_id: Optional[str] = None) -> dict:
        return ActionTemplates.set_weather("wind", {
            "wind_speed": wind_speed, "gust_factor": gust_factor,
        }, action_id)

    # ---- 实体生成/移动/移除 ----

    @staticmethod
    def spawn_entity(entity_id: str, asset_id: str, position_enu_m: list[float],
                     rotation_deg: Optional[list[float]] = None,
                     visual_state: Optional[dict] = None,
                     action_id: Optional[str] = None) -> dict:
        result = {
            "action_id": action_id or _next_action_id("spawn"),
            "type": "spawn_entity",
            "asset_id": asset_id,
            "entity_id": entity_id,
            "position_enu_m": position_enu_m,
        }
        if rotation_deg:
            result["rotation_deg"] = rotation_deg
        if visual_state:
            result["visual_state"] = visual_state
        return result

    @staticmethod
    def spawn_uav(entity_id: str, uav_type: str = "inspect",
                  position_enu_m: Optional[list[float]] = None,
                  action_id: Optional[str] = None) -> dict:
        """生成 UAV。uav_type: 'inspect' | 'delivery' | 'airsim' | 'cv'"""
        asset_map = {
            "inspect": "uav.inspect.quad.v1",
            "delivery": "uav.airsim.flying_pawn.v1",
            "airsim": "uav.airsim.flying_pawn.v1",
            "cv": "uav.airsim.cv_pawn.v1",
        }
        pos = position_enu_m or [50.0, 20.0, 30.0]
        return ActionTemplates.spawn_entity(
            entity_id, asset_map.get(uav_type, asset_map["airsim"]),
            pos, visual_state={"mode": "idle"}, action_id=action_id,
        )

    @staticmethod
    def spawn_vehicle(entity_id: str, vehicle_type: str = "car",
                      position_enu_m: Optional[list[float]] = None,
                      visual_state: Optional[dict] = None,
                      action_id: Optional[str] = None) -> dict:
        """生成地面车辆。vehicle_type: 'car' | 'emergency' | 'police' | 'ambulance' | 'box'"""
        asset_map = {
            "car": "vehicle.ground.boxcar.v1",
            "emergency": "vehicle.emergency.suv.v1",
            "police": "vehicle.emergency.police_suv.v1",
            "ambulance": "vehicle.emergency.ambulance.v1",
            "box": "vehicle.service.box.v1",
            "husky": "vehicle.ground.husky.v1",
        }
        pos = position_enu_m or [48.0, 22.0, 0.0]
        vs = visual_state or {"mode": "idle"}
        return ActionTemplates.spawn_entity(
            entity_id, asset_map.get(vehicle_type, asset_map["car"]),
            pos, visual_state=vs, action_id=action_id,
        )

    @staticmethod
    def spawn_pedestrian(entity_id: str,
                         position_enu_m: Optional[list[float]] = None,
                         action_id: Optional[str] = None) -> dict:
        """生成单个行人。"""
        pos = position_enu_m or [52.0, 20.0, 0.0]
        return ActionTemplates.spawn_entity(
            entity_id, "pedestrian.cityops.basic.v1",
            pos, visual_state={"mode": "idle"}, action_id=action_id,
        )

    @staticmethod
    def move_entity(entity_id: str, waypoints: list[list[float]],
                    velocity_mps: float = 5.0, action_id: Optional[str] = None) -> dict:
        return {
            "action_id": action_id or _next_action_id("move"),
            "type": "move_entity",
            "entity_id": entity_id,
            "waypoints_enu_m": waypoints,
            "velocity_mps": velocity_mps,
        }

    @staticmethod
    def remove_entity(entity_id: str, action_id: Optional[str] = None) -> dict:
        return {
            "action_id": action_id or _next_action_id("remove"),
            "type": "remove_entity",
            "entity_id": entity_id,
        }

    # ---- 行人 ----

    @staticmethod
    def play_animation(ped_id: str, animation_path: str,
                       start_section: str = "Start", play_rate: float = 1.0,
                       loop_count: int = 1, action_id: Optional[str] = None) -> dict:
        return {
            "action_id": action_id or _next_action_id("anim"),
            "type": "play_animation",
            "ped_id": ped_id,
            "animation_path": animation_path,
            "start_section": start_section,
            "play_rate": play_rate,
            "loop_count": loop_count,
        }

    @staticmethod
    def ped_fall_flat(ped_id: str, action_id: Optional[str] = None) -> dict:
        """行人跌倒动画。"""
        return ActionTemplates.play_animation(
            ped_id,
            "/Game/MixamoAssets/Animations/AM_Fall_Flat_2Stage.AM_Fall_Flat_2Stage",
            action_id=action_id,
        )

    @staticmethod
    def ped_stagger(ped_id: str, action_id: Optional[str] = None) -> dict:
        """行人踉跄动画（被撞击反应）。"""
        return ActionTemplates.play_animation(
            ped_id,
            "/Game/MixamoAssets/Animations/Hit_Reaction.Hit_Reaction",
            action_id=action_id,
        )

    @staticmethod
    def spawn_crowd(group_id: str, count: int, spawn_origin_enu_m: list[float],
                    spawn_box_extent_cm: Optional[list[float]] = None,
                    seed: int = 42, action_id: Optional[str] = None) -> dict:
        result = {
            "action_id": action_id or _next_action_id("crowd"),
            "type": "spawn_crowd",
            "group_id": group_id,
            "count": count,
            "spawn_origin_enu_m": spawn_origin_enu_m,
            "seed": seed,
        }
        if spawn_box_extent_cm:
            result["spawn_box_extent_cm"] = spawn_box_extent_cm
        return result

    @staticmethod
    def clear_crowd(group_id: str, action_id: Optional[str] = None) -> dict:
        return {
            "action_id": action_id or _next_action_id("clear_crowd"),
            "type": "clear_crowd",
            "group_id": group_id,
        }

    # ---- 视觉状态 ----

    @staticmethod
    def set_visual_state(entity_id: str, mode: Optional[str] = None,
                         lights_on: Optional[bool] = None,
                         action_id: Optional[str] = None) -> dict:
        vs: dict = {}
        if mode is not None:
            vs["mode"] = mode
        if lights_on is not None:
            vs["lights_on"] = lights_on
        return {
            "action_id": action_id or _next_action_id("vis"),
            "type": "set_visual_state",
            "entity_id": entity_id,
            "visual_state": vs,
        }

    @staticmethod
    def emergency_lights_on(entity_id: str, action_id: Optional[str] = None) -> dict:
        """开启应急灯光。"""
        return ActionTemplates.set_visual_state(
            entity_id, mode="response", lights_on=True, action_id=action_id,
        )

    # ---- 截图 ----

    @staticmethod
    def capture_screenshot(camera_id: str, action_id: Optional[str] = None) -> dict:
        return {
            "action_id": action_id or _next_action_id("cap"),
            "type": "capture_screenshot",
            "camera_id": camera_id,
        }

    # ---- 基础设施 ----

    @staticmethod
    def spawn_barrier(entity_id: str, position_enu_m: list[float],
                      action_id: Optional[str] = None) -> dict:
        return ActionTemplates.spawn_entity(
            entity_id, "prop.roadwork.barrier.v1",
            position_enu_m, action_id=action_id,
        )

    @staticmethod
    def spawn_traffic_cone(entity_id: str, position_enu_m: list[float],
                           action_id: Optional[str] = None) -> dict:
        return ActionTemplates.spawn_entity(
            entity_id, "prop.roadwork.traffic_cone.v1",
            position_enu_m, action_id=action_id,
        )

    @staticmethod
    def spawn_no_fly_zone(entity_id: str, position_enu_m: list[float],
                          scale_xyz: Optional[list[float]] = None,
                          action_id: Optional[str] = None) -> dict:
        return ActionTemplates.spawn_entity(
            entity_id, "trigger.no_fly.box.v1",
            position_enu_m, action_id=action_id,
        )

    @staticmethod
    def spawn_hazard_zone(entity_id: str, position_enu_m: list[float],
                          zone_kind: str = "generic",
                          action_id: Optional[str] = None) -> dict:
        asset = "trigger.hazard.construction.box.v1" if zone_kind == "construction" else "trigger.hazard.generic.box.v1"
        return ActionTemplates.spawn_entity(
            entity_id, asset, position_enu_m, action_id=action_id,
        )

    # ---- 复合模式 ----

    @staticmethod
    def emergency_responder_chain(entity_id: str, vehicle_type: str,
                                  start_pos: list[float], target_pos: list[float],
                                  anim_actions: Optional[list[dict]] = None
                                  ) -> list[dict]:
        """
        应急响应链：生成应急车辆 → 开启灯光 → 移动到目标。
        返回 action list，可直接作为 EventStepSpec.actions。
        """
        actions = [
            ActionTemplates.spawn_vehicle(entity_id, vehicle_type, start_pos,
                                          visual_state={"mode": "response", "lights_on": False}),
            ActionTemplates.emergency_lights_on(entity_id),
            ActionTemplates.move_entity(entity_id, [start_pos, target_pos], velocity_mps=12.0),
        ]
        if anim_actions:
            actions.extend(anim_actions)
        return actions

    @staticmethod
    def uav_detect_and_report(uav_id: str, target_id: str,
                              patrol_waypoints: list[list[float]],
                              camera_id: str = "demo_overview") -> list[dict]:
        """
        UAV 检测事件并报告：移动到目标 → 悬停 → 截图。
        """
        return [
            ActionTemplates.move_entity(uav_id, patrol_waypoints),
            ActionTemplates.capture_screenshot(camera_id),
        ]

    @staticmethod
    def environment_cascade(weather_profile: str, intensity: float,
                            visibility_reduction: float,
                            action_id: Optional[str] = None) -> dict:
        """
        环境渐变：多参数同时变化。
        用于构建复合天气事件。
        """
        if weather_profile == "rain":
            return ActionTemplates.set_weather("rain", {
                "rain": intensity,
                "visibility_m": 20000.0 * (1 - visibility_reduction),
                "wetness": intensity * 0.9,
                "wind_speed": 5.0 + intensity * 10.0,
            }, action_id)
        elif weather_profile == "fog":
            return ActionTemplates.set_weather("fog", {
                "fog": intensity,
                "visibility_m": 20000.0 * (1 - visibility_reduction),
            }, action_id)
        elif weather_profile == "wind":
            return ActionTemplates.set_weather("wind", {
                "wind_speed": 5.0 + intensity * 20.0,
                "gust_factor": 1.0 + intensity * 1.0,
            }, action_id)
        return ActionTemplates.set_weather(weather_profile, {}, action_id)
