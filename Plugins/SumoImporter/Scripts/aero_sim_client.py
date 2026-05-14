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


def _rotation_radians(rotation_deg: Sequence[float] | dict[str, Any] | None = None) -> tuple[float, float, float]:
    rotation = _rotation_payload(rotation_deg)
    return (
        float(rotation["pitch_deg"]) * 3.141592653589793 / 180.0,
        float(rotation["roll_deg"]) * 3.141592653589793 / 180.0,
        float(rotation["yaw_deg"]) * 3.141592653589793 / 180.0,
    )


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
            self._client = airsim.MultirotorClient(ip=host, port=port, timeout_value=timeout_value)

        if auto_connect and hasattr(self._client, "confirmConnection"):
            self._client.confirmConnection()

    @property
    def client(self) -> Any:
        return self._client

    def _airsim(self) -> Any:
        if self._airsim_module is not None:
            return self._airsim_module
        try:
            import cosysairsim as airsim  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("cosysairsim import failed. Install it in the active Python environment.") from exc
        self._airsim_module = airsim
        return airsim

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

    def _rotation_quaternion(self, rotation_deg: Sequence[float] | dict[str, Any] | None = None) -> Any:
        airsim = self._airsim()
        pitch_rad, roll_rad, yaw_rad = _rotation_radians(rotation_deg)
        if hasattr(airsim, "to_quaternion"):
            return airsim.to_quaternion(pitch_rad, roll_rad, yaw_rad)
        if hasattr(airsim, "utils") and hasattr(airsim.utils, "euler_to_quaternion"):
            return airsim.utils.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
        raise RuntimeError("cosysairsim does not expose a quaternion helper.")

    def build_airsim_pose(self, position_enu_m: Sequence[float], rotation_deg: Sequence[float] | dict[str, Any] | None = None) -> Any:
        airsim = self._airsim()
        position = _vector_payload(position_enu_m)
        return airsim.Pose(
            airsim.Vector3r(float(position[0]), float(position[1]), float(-position[2])),
            self._rotation_quaternion(rotation_deg),
        )

    def build_airsim_pose_ned(self, position_ned_m: Sequence[float], rotation_deg: Sequence[float] | dict[str, Any] | None = None) -> Any:
        airsim = self._airsim()
        position = _vector_payload(position_ned_m)
        return airsim.Pose(
            airsim.Vector3r(float(position[0]), float(position[1]), float(position[2])),
            self._rotation_quaternion(rotation_deg),
        )

    @staticmethod
    def airsim_pose_to_enu_payload(pose: Any) -> dict[str, Any]:
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)
        position_enu_m = [
            float(getattr(position, "x_val", 0.0)),
            float(getattr(position, "y_val", 0.0)),
            float(-getattr(position, "z_val", 0.0)),
        ]
        orientation_payload = {
            "x_val": float(getattr(orientation, "x_val", 0.0)),
            "y_val": float(getattr(orientation, "y_val", 0.0)),
            "z_val": float(getattr(orientation, "z_val", 0.0)),
            "w_val": float(getattr(orientation, "w_val", 1.0)),
        }
        return {"position_enu_m": position_enu_m, "orientation": orientation_payload}

    def list_vehicles(self) -> list[str]:
        return [str(value) for value in self._client.listVehicles()]

    def add_vehicle(
        self,
        vehicle_name: str,
        *,
        vehicle_type: str = "SimpleFlight",
        position_enu_m: Sequence[float] = (0.0, 0.0, 20.0),
        rotation_deg: Sequence[float] | dict[str, Any] | None = None,
        pawn_path: str = "",
    ) -> bool:
        pose = self.build_airsim_pose(position_enu_m, rotation_deg)
        return bool(self._client.simAddVehicle(str(vehicle_name), str(vehicle_type), pose, str(pawn_path or "")))

    def enable_api_control(self, enabled: bool, vehicle_name: str) -> None:
        self._client.enableApiControl(bool(enabled), str(vehicle_name))

    def arm_disarm(self, arm: bool, vehicle_name: str) -> bool:
        return bool(self._client.armDisarm(bool(arm), str(vehicle_name)))

    def set_vehicle_pose_enu(
        self,
        vehicle_name: str,
        *,
        position_enu_m: Sequence[float],
        rotation_deg: Sequence[float] | dict[str, Any] | None = None,
        ignore_collision: bool = True,
    ) -> None:
        airsim = self._airsim()
        pose = self.build_airsim_pose(position_enu_m, rotation_deg)
        if hasattr(airsim, "KinematicsState") and hasattr(self._client, "simSetKinematics"):
            state = airsim.KinematicsState()
            state.position = pose.position
            state.orientation = pose.orientation
            state.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
            state.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
            state.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
            state.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
            self._client.simSetKinematics(state, bool(ignore_collision), str(vehicle_name))
            return
        self._client.simSetVehiclePose(pose, bool(ignore_collision), str(vehicle_name))

    def set_vehicle_pose_ned(
        self,
        vehicle_name: str,
        *,
        position_ned_m: Sequence[float],
        rotation_deg: Sequence[float] | dict[str, Any] | None = None,
        ignore_collision: bool = True,
    ) -> None:
        airsim = self._airsim()
        pose = self.build_airsim_pose_ned(position_ned_m, rotation_deg)
        if hasattr(airsim, "KinematicsState") and hasattr(self._client, "simSetKinematics"):
            state = airsim.KinematicsState()
            state.position = pose.position
            state.orientation = pose.orientation
            state.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
            state.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
            state.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
            state.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
            self._client.simSetKinematics(state, bool(ignore_collision), str(vehicle_name))
            return
        self._client.simSetVehiclePose(pose, bool(ignore_collision), str(vehicle_name))

    def get_vehicle_pose_ned(self, vehicle_name: str) -> dict[str, Any]:
        if hasattr(self._client, "simGetGroundTruthKinematics"):
            state = self._client.simGetGroundTruthKinematics(str(vehicle_name))
            position = state.position
            orientation = state.orientation
        else:
            pose = self._client.simGetVehiclePose(str(vehicle_name))
            position = pose.position
            orientation = pose.orientation
        return {
            "position_ned_m": [
                float(getattr(position, "x_val", 0.0)),
                float(getattr(position, "y_val", 0.0)),
                float(getattr(position, "z_val", 0.0)),
            ],
            "orientation": {
                "x_val": float(getattr(orientation, "x_val", 0.0)),
                "y_val": float(getattr(orientation, "y_val", 0.0)),
                "z_val": float(getattr(orientation, "z_val", 0.0)),
                "w_val": float(getattr(orientation, "w_val", 1.0)),
            },
        }

    def get_vehicle_pose_enu(self, vehicle_name: str) -> dict[str, Any]:
        if hasattr(self._client, "simGetGroundTruthKinematics"):
            state = self._client.simGetGroundTruthKinematics(str(vehicle_name))
            return self.airsim_pose_to_enu_payload(type("_Pose", (), {"position": state.position, "orientation": state.orientation})())
        return self.airsim_pose_to_enu_payload(self._client.simGetVehiclePose(str(vehicle_name)))

    def set_camera_pose(
        self,
        vehicle_name: str,
        camera_name: str,
        *,
        position_enu_m: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_deg: Sequence[float] | dict[str, Any] | None = None,
    ) -> None:
        pose = self.build_airsim_pose(position_enu_m, rotation_deg)
        self._client.simSetCameraPose(str(camera_name), pose, str(vehicle_name))

    def set_camera_pose_ned(
        self,
        vehicle_name: str,
        camera_name: str,
        *,
        position_ned_m: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_deg: Sequence[float] | dict[str, Any] | None = None,
    ) -> None:
        pose = self.build_airsim_pose_ned(position_ned_m, rotation_deg)
        self._client.simSetCameraPose(str(camera_name), pose, str(vehicle_name))

    def get_camera_info(self, vehicle_name: str, camera_name: str) -> dict[str, Any]:
        info = self._client.simGetCameraInfo(str(camera_name), str(vehicle_name))
        pose = getattr(info, "pose", None)
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)
        return {
            "camera_name": str(camera_name),
            "vehicle_name": str(vehicle_name),
            "fov_degrees": float(getattr(info, "fov", 0.0)),
            "position_ned_m": [
                float(getattr(position, "x_val", 0.0)),
                float(getattr(position, "y_val", 0.0)),
                float(getattr(position, "z_val", 0.0)),
            ],
            "orientation": {
                "x_val": float(getattr(orientation, "x_val", 0.0)),
                "y_val": float(getattr(orientation, "y_val", 0.0)),
                "z_val": float(getattr(orientation, "z_val", 0.0)),
                "w_val": float(getattr(orientation, "w_val", 1.0)),
            },
        }

    def _image_type_value(self, image_type: str | int) -> Any:
        airsim = self._airsim()
        if isinstance(image_type, int):
            return int(image_type)
        normalized = str(image_type or "Scene").strip()
        if not normalized:
            normalized = "Scene"
        return getattr(airsim.ImageType, normalized)

    def capture_vehicle_image(
        self,
        vehicle_name: str,
        *,
        camera_name: str,
        image_type: str | int,
        pixels_as_float: bool,
        compress: bool,
        annotation_name: str = "",
    ) -> Any:
        airsim = self._airsim()
        request = airsim.ImageRequest(
            str(camera_name),
            self._image_type_value(image_type),
            bool(pixels_as_float),
            bool(compress),
            str(annotation_name or ""),
        )
        responses = self._client.simGetImages([request], str(vehicle_name))
        if not responses:
            raise RuntimeError(f"simGetImages returned no responses for {vehicle_name}/{camera_name}/{image_type}")
        return responses[0]

    def set_segmentation_object_id(self, mesh_name: str, object_id: int, *, is_name_regex: bool = True) -> bool:
        return bool(self._client.simSetSegmentationObjectID(str(mesh_name), int(object_id), bool(is_name_regex)))

    def list_instance_segmentation_objects(self) -> list[str]:
        return [str(value) for value in self._client.simListInstanceSegmentationObjects()]

    def get_segmentation_color_map(self) -> list[list[int]]:
        values = self._airsim().MultirotorClient.simGetSegmentationColorMap()
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [[int(channel) for channel in row[:3]] for row in values]

    def get_instance_segmentation_color_map(self) -> list[list[float]]:
        if hasattr(self._client, "simGetInstanceSegmentationColorMap"):
            values = self._client.simGetInstanceSegmentationColorMap()
        elif hasattr(self._client, "client"):
            values = self._client.client.call("simGetInstanceSegmentationColorMap")
        else:
            values = []
        return [
            [
                float(getattr(value, "x_val", 0.0)),
                float(getattr(value, "y_val", 0.0)),
                float(getattr(value, "z_val", 0.0)),
            ]
            for value in values
        ]

    def get_settings_string(self) -> str:
        if hasattr(self._client, "getSettingsString"):
            return str(self._client.getSettingsString())
        direct_client = self._get_direct_rpc_client()
        response = direct_client.call("getSettingsString")
        if isinstance(response, bytes):
            return response.decode("utf-8")
        return str(response)

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
