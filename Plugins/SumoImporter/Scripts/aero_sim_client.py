#!/usr/bin/env python3
from __future__ import annotations

import json
import uuid
from typing import Any, Iterable, Optional, Sequence


class AeroSimError(RuntimeError):
    def __init__(self, operation: str, message: str, response: Optional[dict[str, Any]] = None) -> None:
        super().__init__(f"{operation} failed: {message}")
        self.operation = operation
        self.response = response or {}


def _vector_payload(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        raise ValueError("vector payload requires at least 2 components")
    return [float(values[0]), float(values[1]), float(values[2] if len(values) > 2 else 0.0)]


def _rotation_payload(
    rotation_deg: Sequence[float] | dict[str, Any] | None = None,
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> dict[str, float]:
    if rotation_deg is None:
        return {
            "pitch_deg": float(pitch_deg),
            "yaw_deg": float(yaw_deg),
            "roll_deg": float(roll_deg),
        }

    if isinstance(rotation_deg, dict):
        return {
            "pitch_deg": float(rotation_deg.get("pitch_deg", rotation_deg.get("pitch", pitch_deg))),
            "yaw_deg": float(rotation_deg.get("yaw_deg", rotation_deg.get("yaw", yaw_deg))),
            "roll_deg": float(rotation_deg.get("roll_deg", rotation_deg.get("roll", roll_deg))),
        }

    if len(rotation_deg) < 3:
        raise ValueError("rotation payload sequence requires 3 components: pitch, yaw, roll")
    return {
        "pitch_deg": float(rotation_deg[0]),
        "yaw_deg": float(rotation_deg[1]),
        "roll_deg": float(rotation_deg[2]),
    }


class AeroSimClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 41451,
        timeout_value: float = 60.0,
        airsim_client: Any | None = None,
        auto_connect: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_value = timeout_value
        self._airsim_module = None
        self._client = airsim_client
        self._direct_rpc_client = None

        if self._client is None:
            try:
                import cosysairsim as airsim  # type: ignore
            except Exception as exc:  # pragma: no cover - depends on local env
                raise RuntimeError(
                    "cosysairsim import failed. Install it in the active Python environment."
                ) from exc

            self._airsim_module = airsim
            self._client = airsim.VehicleClient(ip=host, port=port, timeout_value=timeout_value)

        if auto_connect and hasattr(self._client, "confirmConnection"):
            self._client.confirmConnection()

    @property
    def client(self) -> Any:
        return self._client

    @staticmethod
    def enu_m_to_world_cm(position_enu_m: Sequence[float], world_origin_cm: Sequence[float]) -> list[float]:
        enu = _vector_payload(position_enu_m)
        origin = _vector_payload(world_origin_cm)
        return [origin[0] + enu[0] * 100.0, origin[1] + enu[1] * 100.0, origin[2] + enu[2] * 100.0]

    @staticmethod
    def world_cm_to_enu_m(position_world_cm: Sequence[float], world_origin_cm: Sequence[float]) -> list[float]:
        world = _vector_payload(position_world_cm)
        origin = _vector_payload(world_origin_cm)
        return [(world[0] - origin[0]) / 100.0, (world[1] - origin[1]) / 100.0, (world[2] - origin[2]) / 100.0]

    def _make_request_json(self, payload: Optional[dict[str, Any]] = None, map_id: str = "") -> str:
        root: dict[str, Any] = {
            "api_version": "1.0",
            "request_id": uuid.uuid4().hex,
            "payload": payload or {},
        }
        if map_id:
            root["map_id"] = map_id
        return json.dumps(root, ensure_ascii=True)

    def _parse_response(self, operation: str, response_text: str) -> dict[str, Any]:
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise AeroSimError(operation, f"invalid JSON response: {response_text}") from exc

        response_op = str(response.get("op") or "").strip()
        if response_op and response_op != operation:
            raise AeroSimError(
                operation,
                f"mismatched response op '{response_op}' (expected '{operation}')",
                response=response,
            )

        if response.get("status") != "ok":
            error_obj = response.get("error") or {}
            message = error_obj.get("message") or "unknown error"
            raise AeroSimError(operation, str(message), response=response)

        return response

    def _get_direct_rpc_client(self) -> Any:
        if self._direct_rpc_client is not None:
            return self._direct_rpc_client

        try:
            import msgpackrpc  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "No direct RPC transport found on the AirSim client and msgpackrpc is unavailable."
            ) from exc

        self._direct_rpc_client = msgpackrpc.Client(
            msgpackrpc.Address(self.host, self.port),
            timeout=int(self.timeout_value * 1000),
        )
        return self._direct_rpc_client

    def _call_rpc(self, method_name: str, request_json: str) -> str:
        bound_method = getattr(self._client, method_name, None)
        if callable(bound_method):
            response = bound_method(request_json)
            if isinstance(response, bytes):
                return response.decode("utf-8")
            return str(response)

        direct_client = self._get_direct_rpc_client()
        response = direct_client.call(method_name, request_json)
        if isinstance(response, bytes):
            return response.decode("utf-8")
        return str(response)

    def _call_json_method(
        self,
        method_name: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        map_id: str = "",
    ) -> dict[str, Any]:
        request_json = self._make_request_json(payload=payload, map_id=map_id)
        response_text = self._call_rpc(method_name, request_json)
        return self._parse_response(method_name, response_text)

    def describe_capabilities(self) -> dict[str, Any]:
        return self._call_json_method("simAeroDescribeCapabilities")

    def load_context(self, map_id: str) -> dict[str, Any]:
        return self._call_json_method("simAeroLoadContext", payload={"map_id": map_id}, map_id=map_id)

    def reload_config(self, kind: str, path: str | None = None, map_id: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": kind}
        if path:
            payload["path"] = path
        return self._call_json_method("simAeroReloadConfig", payload=payload, map_id=map_id)

    def query_nearest(
        self,
        tag: str,
        *,
        pose_enu_m: Sequence[float] | None = None,
        radius_m: float | None = None,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tag": tag}
        if pose_enu_m is not None:
            payload["pose_enu_m"] = _vector_payload(pose_enu_m)
        if radius_m is not None:
            payload["radius_m"] = float(radius_m)
        return self._call_json_method("simAeroQueryNearest", payload=payload, map_id=map_id)

    def project_ground(
        self,
        *,
        point_enu_m: Sequence[float] | None = None,
        position_enu_m: Sequence[float] | None = None,
        position_world_cm: Sequence[float] | None = None,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if point_enu_m is None and position_enu_m is not None:
            point_enu_m = position_enu_m
        if point_enu_m is not None:
            payload["point_enu_m"] = _vector_payload(point_enu_m)
        elif position_world_cm is not None:
            raise ValueError(
                "simAeroProjectGround accepts point_enu_m only; convert world cm using load_context payload['world_origin_cm'] first."
            )
        else:
            raise ValueError("simAeroProjectGround requires point_enu_m (or position_enu_m alias)")
        return self._call_json_method("simAeroProjectGround", payload=payload, map_id=map_id)

    def query_ped_path(self, payload: dict[str, Any], map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroQueryPedPath", payload=payload, map_id=map_id)

    def query_ped_anchor(
        self,
        payload: dict[str, Any],
        map_id: str = "",
    ) -> dict[str, Any]:
        return self._call_json_method("simAeroQueryPedAnchor", payload=payload, map_id=map_id)

    def poll_feedback(
        self,
        *,
        since_tick: int | None = None,
        since_frame_id: int | None = None,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if since_tick is not None:
            payload["since_tick"] = int(since_tick)
        if since_frame_id is not None:
            payload["since_frame_id"] = int(since_frame_id)
        return self._call_json_method("simAeroPollFeedback", payload=payload, map_id=map_id)

    def apply_frame(self, payload: dict[str, Any], map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroApplyFrame", payload=payload, map_id=map_id)

    def spawn_asset(self, payload: dict[str, Any], map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroSpawnAsset", payload=payload, map_id=map_id)

    def move_asset(self, payload: dict[str, Any], map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroMoveAsset", payload=payload, map_id=map_id)

    def remove_asset(self, asset_id: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroRemoveAsset", payload={"asset_id": asset_id}, map_id=map_id)

    def capture_world_camera(self, payload: dict[str, Any], map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroCaptureWorldCamera", payload=payload, map_id=map_id)

    def reserve_occupancy(self, asset_id: str, entity_id: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method(
            "simAeroReserveOccupancy",
            payload={"asset_id": asset_id, "entity_id": entity_id},
            map_id=map_id,
        )

    def release_occupancy(self, asset_id: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroReleaseOccupancy", payload={"asset_id": asset_id}, map_id=map_id)

    def apply_weather(self, payload: dict[str, Any], map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroApplyWeather", payload=payload, map_id=map_id)

    def ped_spawn(
        self,
        ped_id: str,
        position_world_cm: Sequence[float] | None = None,
        *,
        position_enu_m: Sequence[float] | None = None,
        yaw_deg: float = 0.0,
        variant_id: str = "",
        snap_to_ground: bool = True,
        preserve_xy: bool = True,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ped_id": ped_id,
            "yaw_deg": float(yaw_deg),
            "snap_to_ground": bool(snap_to_ground),
            "preserve_xy": bool(preserve_xy),
        }
        if position_enu_m is not None:
            payload["position_enu_m"] = _vector_payload(position_enu_m)
        elif position_world_cm is not None:
            payload["position_world_cm"] = _vector_payload(position_world_cm)
        else:
            raise ValueError("ped_spawn requires position_enu_m or position_world_cm")
        if variant_id.strip():
            payload["variant_id"] = variant_id.strip()
        return self._call_json_method("simAeroPedSpawn", payload=payload, map_id=map_id)

    def ped_reset(
        self,
        ped_id: str,
        position_world_cm: Sequence[float] | None = None,
        *,
        position_enu_m: Sequence[float] | None = None,
        yaw_deg: float = 0.0,
        snap_to_ground: bool = True,
        preserve_xy: bool = True,
        frame_pose: bool = False,
        walking: bool = False,
        speed_cm_per_sec: float = 0.0,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload = {
            "ped_id": ped_id,
            "yaw_deg": float(yaw_deg),
            "snap_to_ground": bool(snap_to_ground),
            "preserve_xy": bool(preserve_xy),
        }
        if frame_pose:
            payload["frame_pose"] = True
            payload["walking"] = bool(walking)
            payload["speed_cm_per_sec"] = float(speed_cm_per_sec)
        if position_enu_m is not None:
            payload["position_enu_m"] = _vector_payload(position_enu_m)
        elif position_world_cm is not None:
            payload["position_world_cm"] = _vector_payload(position_world_cm)
        else:
            raise ValueError("ped_reset requires position_enu_m or position_world_cm")
        return self._call_json_method("simAeroPedReset", payload=payload, map_id=map_id)

    def ped_set_target(
        self,
        ped_id: str,
        target_world_cm: Sequence[float] | None = None,
        *,
        target_enu_m: Sequence[float] | None = None,
        speed_cm_per_sec: float,
        snap_to_ground: bool = True,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload = {
            "ped_id": ped_id,
            "speed_cm_per_sec": float(speed_cm_per_sec),
            "snap_to_ground": bool(snap_to_ground),
        }
        if target_enu_m is not None:
            payload["target_enu_m"] = _vector_payload(target_enu_m)
        elif target_world_cm is not None:
            payload["target_world_cm"] = _vector_payload(target_world_cm)
        else:
            raise ValueError("ped_set_target requires target_enu_m or target_world_cm")
        return self._call_json_method("simAeroPedSetTarget", payload=payload, map_id=map_id)

    def ped_observe(self, ped_id: str, *, start_section: str = "", map_id: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"ped_id": ped_id}
        if start_section.strip():
            payload["start_section"] = start_section.strip()
        return self._call_json_method("simAeroPedObserve", payload=payload, map_id=map_id)

    def ped_play_animation(
        self,
        ped_id: str,
        animation_asset_path: str,
        *,
        start_section: str = "",
        play_rate: float = 1.0,
        loop_count: int = 1,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ped_id": ped_id,
            "animation_asset_path": animation_asset_path,
            "play_rate": float(play_rate),
            "loop_count": max(1, int(loop_count)),
        }
        if start_section.strip():
            payload["start_section"] = start_section.strip()
        return self._call_json_method("simAeroPedPlayAnimation", payload=payload, map_id=map_id)

    def ped_commit_cross(
        self,
        ped_id: str,
        target_world_cm: Sequence[float] | None = None,
        *,
        target_enu_m: Sequence[float] | None = None,
        speed_cm_per_sec: float,
        snap_to_ground: bool = True,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload = {
            "ped_id": ped_id,
            "speed_cm_per_sec": float(speed_cm_per_sec),
            "snap_to_ground": bool(snap_to_ground),
        }
        if target_enu_m is not None:
            payload["target_enu_m"] = _vector_payload(target_enu_m)
        elif target_world_cm is not None:
            payload["target_world_cm"] = _vector_payload(target_world_cm)
        else:
            raise ValueError("ped_commit_cross requires target_enu_m or target_world_cm")
        return self._call_json_method("simAeroPedCommitCross", payload=payload, map_id=map_id)

    def ped_stop(self, ped_id: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroPedStop", payload={"ped_id": ped_id}, map_id=map_id)

    def ped_set_variant(self, ped_id: str, variant_id: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method(
            "simAeroPedSetVariant",
            payload={"ped_id": ped_id, "variant_id": variant_id},
            map_id=map_id,
        )

    def ped_release(self, ped_id: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroPedRelease", payload={"ped_id": ped_id}, map_id=map_id)

    def ped_spawn_crowd(
        self,
        group_id: str,
        count: int,
        spawn_origin_world_cm: Sequence[float] | None = None,
        *,
        spawn_origin_enu_m: Sequence[float] | None = None,
        seed: int = 0,
        spawn_box_extent_cm: Sequence[float] | None = None,
        yaw_policy: str = "random",
        fixed_yaw_deg: float = 0.0,
        snap_to_ground: bool = True,
        appearance_pool_path: str = "",
        role_profile_path: str = "",
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "group_id": group_id,
            "count": int(count),
            "seed": int(seed),
            "yaw_policy": yaw_policy,
            "fixed_yaw_deg": float(fixed_yaw_deg),
            "snap_to_ground": bool(snap_to_ground),
        }
        if spawn_origin_enu_m is not None:
            payload["spawn_origin_enu_m"] = _vector_payload(spawn_origin_enu_m)
        elif spawn_origin_world_cm is not None:
            payload["spawn_origin_world_cm"] = _vector_payload(spawn_origin_world_cm)
        else:
            raise ValueError("ped_spawn_crowd requires spawn_origin_enu_m or spawn_origin_world_cm")
        if spawn_box_extent_cm is not None:
            payload["spawn_box_extent_cm"] = _vector_payload(spawn_box_extent_cm)
        if appearance_pool_path.strip():
            payload["appearance_pool_path"] = appearance_pool_path.strip()
        if role_profile_path.strip():
            payload["role_profile_path"] = role_profile_path.strip()
        return self._call_json_method("simAeroPedSpawnCrowd", payload=payload, map_id=map_id)

    def ped_clear_crowd(self, group_id: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method("simAeroPedClearCrowd", payload={"group_id": group_id}, map_id=map_id)

    def ped_respawn_crowd(self, group_id: str, *, seed: int = 0, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method(
            "simAeroPedRespawnCrowd",
            payload={"group_id": group_id, "seed": int(seed)},
            map_id=map_id,
        )

    def create_runtime_multirotor(
        self,
        vehicle_name: str,
        position_world_cm: Sequence[float] | None = None,
        *,
        position_enu_m: Sequence[float] | None = None,
        rotation_deg: Sequence[float] | dict[str, Any] | None = None,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vehicle_name": vehicle_name,
            "rotation_deg": _rotation_payload(
                rotation_deg,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
                roll_deg=roll_deg,
            ),
        }
        if position_enu_m is not None:
            payload["position_enu_m"] = _vector_payload(position_enu_m)
        elif position_world_cm is not None:
            payload["position_world_cm"] = _vector_payload(position_world_cm)
        else:
            raise ValueError("create_runtime_multirotor requires position_enu_m or position_world_cm")
        return self._call_json_method("simAeroCreateRuntimeMultirotor", payload=payload, map_id=map_id)

    def move_runtime_multirotor(
        self,
        vehicle_name: str,
        target_world_cm: Sequence[float] | None = None,
        *,
        target_enu_m: Sequence[float] | None = None,
        velocity_mps: float = 5.0,
        map_id: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vehicle_name": vehicle_name,
            "velocity_mps": float(velocity_mps),
        }
        if target_enu_m is not None:
            resolved_target = _vector_payload(target_enu_m)
            payload["target_enu_m"] = resolved_target
            payload["position_enu_m"] = list(resolved_target)
        elif target_world_cm is not None:
            resolved_target = _vector_payload(target_world_cm)
            payload["target_world_cm"] = resolved_target
            payload["position_world_cm"] = list(resolved_target)
        else:
            raise ValueError("move_runtime_multirotor requires target_enu_m or target_world_cm")
        return self._call_json_method("simAeroMoveRuntimeMultirotor", payload=payload, map_id=map_id)

    def get_runtime_multirotor_status(self, vehicle_name: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method(
            "simAeroGetRuntimeMultirotorStatus",
            payload={"vehicle_name": vehicle_name},
            map_id=map_id,
        )

    def remove_runtime_vehicle(self, vehicle_name: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method(
            "simAeroRemoveRuntimeVehicle",
            payload={"vehicle_name": vehicle_name},
            map_id=map_id,
        )

    def get_runtime_vehicle_pose(self, vehicle_name: str, map_id: str = "") -> dict[str, Any]:
        return self._call_json_method(
            "simAeroGetRuntimeVehiclePose",
            payload={"vehicle_name": vehicle_name},
            map_id=map_id,
        )
