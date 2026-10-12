"""Calibrated capture geometry, without conflating source AGL with world Z.

The capture rig stores forward/right/down sensor coordinates. Its AirSim
world NED axes map to map-world x/y/-z (the runtime's Unreal-axis convention,
not a geodetic N/E axis swap). Camera optical coordinates are right/down/forward.
Transforms are available independently of image arrays for predicted poses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quaternion_matrix(xyzw) -> np.ndarray:
    """Rotation from an explicitly stored x,y,z,w quaternion."""
    q = np.asarray(xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("invalid orientation quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def rotation_ned(rotation_deg: Mapping[str, float]) -> np.ndarray:
    """AirSim Rz(yaw) Ry(pitch) Rx(roll), with angles in degrees."""
    pitch, yaw, roll = np.deg2rad([rotation_deg[k] for k in
                                 ("pitch_deg", "yaw_deg", "roll_deg")])
    cp, sp, cy, sy, cr, sr = (np.cos(pitch), np.sin(pitch), np.cos(yaw),
                             np.sin(yaw), np.cos(roll), np.sin(roll))
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]], dtype=np.float64)


def rigid_transform(rotation, position) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64)
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def transform_points(points, transform) -> np.ndarray:
    p = np.asarray(points)
    t = np.asarray(transform, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or t.shape != (4, 4):
        raise ValueError("expected points[N,3] and transform[4,4]")
    return p @ t[:3, :3].T + t[:3, 3]


WORLD_FROM_NED = np.diag([1.0, 1.0, -1.0])
SENSOR_FROM_OPTICAL = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def source_agl_to_capture_world(source_position_agl_m, *, ground_world_z_m: float):
    """Convert source x/y/AGL to capture-world x/y/Z with explicit ground data.

    ``ground_world_z_m`` must come from a legitimate ground source available
    to the caller. Target capture metadata is supervision, not a rollout input.
    """
    source = np.asarray(source_position_agl_m, dtype=np.float64)
    if source.shape != (3,) or not np.isfinite(source).all() or not np.isfinite(ground_world_z_m):
        raise ValueError("source AGL conversion requires a finite position and ground height")
    capture = source.copy()
    capture[2] += ground_world_z_m
    return capture


def tangent_ground_height(xy_world_m, *, reference_ground_world_m, surface_normal_world):
    """Cutoff-known local tangent model, explicitly an approximation away from reference.

    Uses only a supplied measured reference point/normal; it does not read
    target capture ground. A map height lookup is preferable for long travel.
    """
    xy = np.asarray(xy_world_m, dtype=np.float64)
    point = np.asarray(reference_ground_world_m, dtype=np.float64)
    normal = np.asarray(surface_normal_world, dtype=np.float64)
    if xy.shape != (2,) or point.shape != (3,) or normal.shape != (3,) or \
            not np.isfinite(np.r_[xy, point, normal]).all() or abs(normal[2]) < 1e-9:
        raise ValueError("invalid local ground tangent reference")
    return float(point[2] - np.dot(normal[:2], xy-point[:2])/normal[2])


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    sensor_to_world: np.ndarray
    body_to_world: np.ndarray
    sensor_to_body: np.ndarray
    camera_to_body: np.ndarray

    def __post_init__(self):
        for array in (self.intrinsic, self.camera_to_world, self.sensor_to_world,
                      self.body_to_world, self.sensor_to_body, self.camera_to_body):
            array.setflags(write=False)

    @property
    def world_to_camera(self):
        return np.linalg.inv(self.camera_to_world)

    @property
    def world_to_sensor(self):
        return np.linalg.inv(self.sensor_to_world)

    def as_dict(self):
        return {"width": self.width, "height": self.height,
                **{k: getattr(self, k).tolist() for k in
                   ("intrinsic", "camera_to_world", "sensor_to_world", "body_to_world",
                    "sensor_to_body", "camera_to_body")},
                "camera_axes": ["right", "down", "forward"],
                "sensor_axes": ["forward", "right", "down"],
                "world_axes": ["map_x", "map_y", "up"],
                "pixel_origin": "top_left; principal point width/2,height/2"}


def calibration_from_capture(camera: Mapping, lidar: Mapping) -> Calibration:
    """Use capture camera extrinsic and actual reported LiDAR body mount."""
    w, h, fov = camera["width"], camera["height"], camera["fov_degrees"]
    f = w / (2 * np.tan(np.deg2rad(fov) / 2))
    intrinsic = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    actual = lidar["capture_vehicle"]["pre_scan_pose"]
    if "pose" in actual:
        pose = actual["pose"]
        orientation = pose["orientation"]
        body_rotation = quaternion_matrix([orientation[k+"_val"] for k in ("x", "y", "z", "w")])
        position_world = np.asarray(actual["requested_position_enu_m"]) + WORLD_FROM_NED @ (
            np.asarray(pose["position_ned_m"]) - np.asarray(actual["requested_position_ned_m"]))
    else:
        # Some v14 recaptures omit the read-back kinematic pose.  The requested simSetKinematics pose is
        # used instead; over every read-back pose of the earlier captures it differs by <= 6.1e-5 m
        # and <= 7e-8 in rotation-matrix entries (float32 read-back rounding).
        body_rotation = rotation_ned(actual["requested_rotation_deg"])
        position_world = np.asarray(actual["requested_position_enu_m"], dtype=np.float64)
    body = rigid_transform(WORLD_FROM_NED @ body_rotation, position_world)
    reported = lidar["lidar"]["reported_sensor_pose_body_ned"]
    sensor_body = rigid_transform(quaternion_matrix(reported["orientation_body_xyzw"]),
                                 reported["position_body_ned_m"])
    camera_world = rigid_transform(
        WORLD_FROM_NED @ rotation_ned(camera["fixed_world_camera_rotation_deg"]) @ SENSOR_FROM_OPTICAL,
        camera["fixed_world_camera_position_enu_m"])
    return Calibration(w, h, intrinsic, camera_world, body @ sensor_body,
                       body, sensor_body, np.linalg.inv(body) @ camera_world)


def project_world(points_world, calibration: Calibration):
    """Return floating UV, optical XYZ, and geometric in-frustum mask."""
    xyz = transform_points(points_world, calibration.world_to_camera)
    uv = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    front = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    uv[front] = (xyz[front, :2] / xyz[front, 2, None]) * np.diag(calibration.intrinsic)[:2]
    uv[front] += calibration.intrinsic[:2, 2]
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < calibration.width) \
        & (uv[:, 1] >= 0) & (uv[:, 1] < calibration.height)
    return uv, xyz, inside
