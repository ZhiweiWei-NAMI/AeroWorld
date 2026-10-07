"""Actual multimodal archive frames and small reusable, causal materializations.

Archive scans happen once per explicitly selected tick set. Prepared frames
contain only whitelisted binding/calibration metadata, not privileged raw
sidecars. No fitted state distributions or donor indices are constructed here.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, replace
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Mapping, Iterable

import numpy as np
from PIL import Image
import zstandard

from .geometry import (Calibration, calibration_from_capture, project_world,
                       lidar_depth_association, association_metrics)

MODALITIES = ("rgb", "depth", "seg", "lidar")
PAYLOAD_SUFFIX = {"rgb": ".png", "depth": ".npy", "seg": ".png", "lidar": ".npz"}
ARRAY_NAMES = ("rgb", "depth_m", "segmentation_class_ids", "points_sensor_ned_m", "hit_mask",
               "scan_timestamp_ns", "sensor_pose_position_body_ned_m", "sensor_pose_orientation_body_xyzw")
# One explicit class-ID palette for every frame; labels come from the sidecar.
CLASS_COLORS = ("#20252b", "#62717f", "#e45756", "#54a24b", "#4c78a8", "#f58518",
                "#b279a2", "#ff9da6", "#9d755d", "#edc949", "#76b7b2", "#af2e40",
                "#9467bd", "#737373", "#eeeeee", "#ffdc36", "#ba9e72", "#c7c7c7",
                "#397a45", "#97bb47", "#b9d978")
MEMBER_RE = re.compile(r"^(?P<profile>[^/]+)/(?P<seed>seed\d+)/(?P<view>[^/]+)/"
                       r"(?P<modality>rgb|depth|seg|lidar)/tick_(?P<tick>\d{6})(?P<suffix>\.[^/]+)$")


@dataclass(frozen=True)
class ObservationFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation_class_ids: np.ndarray
    points_sensor_ned_m: np.ndarray
    hit_mask: np.ndarray
    scan_timestamp_ns: np.ndarray
    sensor_pose_position_body_ned_m: np.ndarray
    sensor_pose_orientation_body_xyzw: np.ndarray
    calibration: Calibration
    binding: dict
    rgb_source_alpha: np.ndarray | None = None

    def __post_init__(self):
        for name in ARRAY_NAMES:
            getattr(self, name).setflags(write=False)
        if self.rgb_source_alpha is not None:
            self.rgb_source_alpha.setflags(write=False)


def _branch(tick: int, cutoff: int, branch: str):
    if branch not in ("historical", "target"):
        raise ValueError("branch must explicitly be historical or target")
    if branch == "historical" and tick > cutoff:
        raise ValueError(f"future observation tick {tick} cannot enter cutoff {cutoff}")
    if branch == "target" and tick <= cutoff:
        raise ValueError("target observation must be strictly after cutoff")


class CaptureArchive:
    """One-pass prepared archive source, with complete selected-observer inventory.

    Instantiate with all required ticks. Reading a tick not selected at creation
    is an error; it never silently rescans the whole compressed episode.
    """
    def __init__(self, path, *, episode_id: str, observer: str, ticks: Iterable[int]):
        self.path = Path(path)
        self.episode_id, self.observer = episode_id, observer
        self.ticks = tuple(sorted(set(int(t) for t in ticks)))
        if not self.ticks:
            raise ValueError("archive requires explicit selected ticks")
        selected = set(self.ticks)
        self.payloads: dict[tuple[int, str, str], tuple[str, bytes]] = {}
        self.inventory = {m: {"payload_ticks": [], "sidecar_ticks": [], "payload_members": [],
                              "sidecar_members": []} for m in MODALITIES}
        self.view_id = None
        with self.path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream, \
                tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                match = MEMBER_RE.match(member.name)
                if match is None:
                    continue
                info = match.groupdict()
                if info["profile"] + "__" + info["seed"] != episode_id:
                    continue
                if not info["view"].endswith("__" + observer):
                    continue
                if self.view_id is None:
                    self.view_id = info["view"]
                elif self.view_id != info["view"]:
                    raise ValueError("observer resolves to multiple capture views")
                modality, tick, suffix = info["modality"], int(info["tick"]), info["suffix"]
                if suffix not in (".json", PAYLOAD_SUFFIX[modality]):
                    continue
                kind = "sidecar" if suffix == ".json" else "payload"
                self.inventory[modality][kind+"_ticks"].append(tick)
                self.inventory[modality][kind+"_members"].append(member.name)
                if tick in selected:
                    key = (tick, modality, kind)
                    if key in self.payloads:
                        raise ValueError(f"duplicate capture member {key}")
                    if not member.isfile():
                        raise ValueError("capture member is not a regular file")
                    self.payloads[key] = (member.name, archive.extractfile(member).read())
        for modality, entry in self.inventory.items():
            for kind in ("payload", "sidecar"):
                if len(entry[kind+"_ticks"]) != len(set(entry[kind+"_ticks"])):
                    raise ValueError(f"duplicate inventory {modality}/{kind}")
                paired = sorted(zip(entry[kind+"_ticks"], entry[kind+"_members"]))
                entry[kind+"_ticks"] = [x[0] for x in paired]
                entry[kind+"_members"] = [x[1] for x in paired]
                entry[kind+"_count"] = len(paired)
                entry[kind+"_missing_ticks_5_to_900"] = sorted(set(range(5, 901, 5))-set(entry[kind+"_ticks"]))
        required = {(t, m, k) for t in selected for m in MODALITIES for k in ("payload", "sidecar")}
        if missing := required - self.payloads.keys():
            raise ValueError(f"missing selected archive members: {sorted(missing)}")
        self.frames: dict[int, ObservationFrame] = {}

    def load(self, tick: int) -> ObservationFrame:
        if tick not in self.ticks:
            raise ValueError(f"tick {tick} was not prepared; instantiate archive with complete desired tick set")
        if tick not in self.frames:
            self.frames[tick] = self._decode(tick)
        return self.frames[tick]

    def _decode(self, tick: int) -> ObservationFrame:
        metadata = {m: json.loads(self.payloads[tick, m, "sidecar"][1]) for m in MODALITIES}
        rgb_meta, depth_meta, seg_meta, lidar_meta = (metadata[m] for m in MODALITIES)
        keys = ("episode_id", "tick", "capture_view_id", "camera_id", "source_uav_entity_id", "capture_alignment_key")
        for modality, meta in metadata.items():
            if any(meta[k] != rgb_meta[k] for k in keys):
                raise ValueError(f"cross-modal binding disagreement for {modality}")
            if meta["episode_id"] != self.episode_id or meta["tick"] != tick or meta["source_uav_entity_id"] != self.observer:
                raise ValueError("archive path and sidecar binding disagree")
        if depth_meta["depth_unit_m"] is not True or depth_meta["output_format"] != "npy_float32_m":
            raise ValueError("depth is not actual float32 metres")
        if seg_meta["segmentation_kind"] != "ue_custom_stencil_class_id_u8":
            raise ValueError("segmentation is not declared class IDs")
        if lidar_meta["lidar"]["point_coordinate_frame"] != "sensor_local_ned_m":
            raise ValueError("unsupported LiDAR coordinate frame")
        for m in ("depth", "seg"):
            if any(metadata[m][k] != rgb_meta[k] for k in ("width", "height", "fov_degrees")):
                raise ValueError(f"{m} camera intrinsics differ from RGB")
            position_key = "ue_stencil_camera_position_enu_m" if m == "seg" else "fixed_world_camera_position_enu_m"
            rotation_key = "ue_stencil_camera_rotation_deg" if m == "seg" else "fixed_world_camera_rotation_deg"
            if metadata[m][position_key] != rgb_meta["fixed_world_camera_position_enu_m"] or \
                    metadata[m][rotation_key] != rgb_meta["fixed_world_camera_rotation_deg"]:
                raise ValueError(f"{m} camera extrinsic differs from RGB")
        with Image.open(io.BytesIO(self.payloads[tick, "rgb", "payload"][1])) as image:
            if image.mode not in ("RGB", "RGBA"):
                raise ValueError(f"unsupported actual RGB PNG mode {image.mode}")
            rgb_mode = image.mode
            image_array = np.array(image)
            rgb = image_array[..., :3].copy()
            alpha = image_array[..., 3].copy() if image.mode == "RGBA" else None
        with Image.open(io.BytesIO(self.payloads[tick, "seg", "payload"][1])) as image:
            segmentation = np.array(image)
        depth = np.load(io.BytesIO(self.payloads[tick, "depth", "payload"][1]), allow_pickle=False)
        with np.load(io.BytesIO(self.payloads[tick, "lidar", "payload"][1]), allow_pickle=False) as scan:
            lidar_arrays = {name: scan[name].copy() for name in ARRAY_NAMES[3:]}
        shape = (rgb_meta["height"], rgb_meta["width"])
        if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3) or depth.dtype != np.float32 or depth.shape != shape:
            raise ValueError("RGB/depth array contract mismatch")
        if segmentation.dtype != np.uint8 or segmentation.shape != shape:
            raise ValueError("segmentation class-ID array contract mismatch")
        classes = seg_meta["semantic_class_by_id"]
        if any(str(int(c)) not in classes for c in np.unique(segmentation)):
            raise ValueError("unregistered actual segmentation class ID")
        points, mask = lidar_arrays["points_sensor_ned_m"], lidar_arrays["hit_mask"]
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3 or mask.shape != (len(points),):
            raise ValueError("LiDAR scan contract mismatch")
        if not np.isin(mask, [0, 1]).all() or not np.isfinite(points[mask.astype(bool)]).all():
            raise ValueError("invalid real LiDAR hit mask or points")
        if len(points) != lidar_meta["point_count"] or int(mask.sum()) != lidar_meta["hit_count"]:
            raise ValueError("actual scan counts differ from sidecar")
        reported = lidar_meta["lidar"]["reported_sensor_pose_body_ned"]
        if not np.array_equal(lidar_arrays["sensor_pose_position_body_ned_m"], reported["position_body_ned_m"]) or \
                not np.array_equal(lidar_arrays["sensor_pose_orientation_body_xyzw"], reported["orientation_body_xyzw"]):
            raise ValueError("stored scan mount differs from sidecar calibration")
        calibration = calibration_from_capture(rgb_meta, lidar_meta)
        binding = {k: rgb_meta[k] for k in keys}
        binding.update({"observer": self.observer, "sim_time_s": rgb_meta["sim_time_s"],
                        "source_archive": str(self.path), "modality": {
                            m: {"payload_member": self.payloads[tick, m, "payload"][0],
                                "sidecar_member": self.payloads[tick, m, "sidecar"][0],
                                "sensor_id": lidar_meta["lidar"]["sensor_name"] if m == "lidar" else rgb_meta["camera_id"]}
                            for m in MODALITIES},
                        "capture_backend": rgb_meta["capture_backend"],
                        "rgb_source_png_mode": rgb_mode,
                        "rgb_source_alpha_present": alpha is not None,
                        "depth": {"unit": "m", "producer_image_type": depth_meta["image_type"],
                                  "semantics": "producer_source_unavailable; axial/radial measured separately",
                                  "producer_source_confirmed": False},
                        "segmentation": {"kind": seg_meta["segmentation_kind"], "class_by_id": classes,
                                         "instance_ids_available": False},
                        "source_pose_agl_m": rgb_meta["source_uav_pose_enu_m"],
                        "capture_position_world_m": rgb_meta["fixed_world_camera_position_enu_m"],
                        "capture_rotation_deg": rgb_meta["fixed_world_camera_rotation_deg"],
                        "body_rotation_deg": lidar_meta["lidar_sensor_mount_truth_pose"]["vehicle_rotation_deg"],
                        "capture_pose_adjustment": {k: rgb_meta["capture_pose_adjustment"][k] for k in
                            ("policy", "kind", "source_truth_altitude_agl_m", "projected_ground_z_enu_m",
                             "resolved_capture_position_enu_m", "resolved_capture_altitude_agl_m")},
                        "ground_reference": {k: rgb_meta["capture_ground_reference"]["projection"][k] for k in
                            ("projected_enu_m", "surface_normal_enu", "ground_resolved")},
                        "scan": {"point_count": len(points), "hit_count": int(mask.sum()),
                                 "timestamp_ns": int(lidar_arrays["scan_timestamp_ns"][0]),
                                 "point_frame": "sensor_local_ned_m", "scope": "complete full-azimuth scan",
                                 "pre_move_timestamp_ns": lidar_meta["scan"]["pre_move_timestamp_ns"],
                                 "discarded_post_move_timestamps_ns": lidar_meta["scan"]["discarded_post_move_timestamps_ns"],
                                 "accepted_timestamp_ns": lidar_meta["scan"]["accepted_timestamp_ns"],
                                 "image_acquisition_timestamp": "not_recorded; same simulation tick binding only"},
                        "lidar_capture_pose_measured": {
                            "actual_position_ned_m": lidar_meta["capture_vehicle"]["pre_scan_pose"]["pose"]["position_ned_m"],
                            "requested_position_ned_m": lidar_meta["capture_vehicle"]["pre_scan_pose"]["requested_position_ned_m"],
                            "actual_orientation_xyzw": [lidar_meta["capture_vehicle"]["pre_scan_pose"]["pose"]["orientation"][k+"_val"]
                                                        for k in ("x", "y", "z", "w")],
                            "pose_error_m": lidar_meta["capture_vehicle"]["pre_scan_pose"]["pose_error_m"],
                            "post_scan_position_ned_m": lidar_meta["capture_vehicle"]["post_scan_pose"]["pose"]["position_ned_m"],
                            "post_scan_position_error_m": lidar_meta["capture_vehicle"]["post_scan_pose"]["position_error_m"]},
                        "source_entity_geometry": [
                            {k: e[k] for k in ("entity_id", "entity_category", "entity_kind", "position_enu_m")}
                            for e in rgb_meta["entity_records"]]})
        return ObservationFrame(rgb, depth, segmentation, **lidar_arrays, calibration=calibration,
                                binding=binding, rgb_source_alpha=alpha)


def load_observation_frame(source, episode_id: str, tick: int, observer: str, *, branch: str,
                           cutoff: int) -> ObservationFrame:
    """Load prepared arrays/calibration/binding; branch/cutoff is mandatory.

    ``source`` is a prepared CaptureArchive or its materialized directory.
    Target frames are labels only; callers keep them off the model input path.
    """
    _branch(tick, cutoff, branch)
    if isinstance(source, CaptureArchive):
        frame = source.load(tick)
    else:
        root = Path(source)
        binding = json.loads((root/f"tick_{tick:06d}.json").read_text())
        geometry = binding.pop("calibration")
        calibration = Calibration(geometry["width"], geometry["height"],
                                  *(np.asarray(geometry[k], dtype=np.float64) for k in
                                    ("intrinsic", "camera_to_world", "sensor_to_world", "body_to_world",
                                     "sensor_to_body", "camera_to_body")))
        with np.load(root/f"tick_{tick:06d}.npz", allow_pickle=False) as arrays:
            frame = ObservationFrame(**{k: arrays[k].copy() for k in ARRAY_NAMES},
                                     calibration=calibration, binding=binding,
                                     rgb_source_alpha=arrays["rgb_source_alpha"].copy()
                                     if binding["rgb_source_alpha_present"] else None)
    if frame.binding["episode_id"] != episode_id or frame.binding["tick"] != tick or frame.binding["observer"] != observer:
        raise ValueError("requested episode/tick/observer differs from materialized frame")
    return replace(frame, binding={**frame.binding, "branch": branch, "cutoff": cutoff})


def source_entity_association(frame: ObservationFrame):
    """Project source coordinates; do not promote roster visibility to instances.

    Source Z may be AGL/ground-reference rather than capture-world geometry.
    This diagnostic preserves that limitation and uses no invented ground Z.
    """
    entities = frame.binding["source_entity_geometry"]
    xyz = np.array([e["position_enu_m"] for e in entities], dtype=np.float64)
    uv, optical, inside = project_world(xyz, frame.calibration)
    output = []
    for i, entity in enumerate(entities):
        item = dict(entity)
        item.update({"coordinate_interpretation": "source map coordinate; capture ground adjustment not proven for this entity",
                     "in_frustum_under_source_coordinate_projection": bool(inside[i]),
                     "instance_visibility": "not_measured_class_ids_only"})
        if inside[i]:
            u, v = np.floor(uv[i]).astype(int)
            item.update({"pixel_uv": uv[i].tolist(), "camera_axial_m": float(optical[i, 2]),
                         "sampled_depth_m": float(frame.depth_m[v, u]),
                         "sampled_segmentation_class_id": int(frame.segmentation_class_ids[v, u]),
                         "association_kind": "projected_source_point_and_pixel_support_only"})
        output.append(item)
    return output


def _visualize(frame: ObservationFrame, association: Mapping, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes[0, 0].imshow(frame.rgb); axes[0, 0].set_title("Actual RGB")
    depth = axes[0, 1].imshow(frame.depth_m, cmap="viridis")
    axes[0, 1].set_title("Actual depth (m); producer semantics unconfirmed")
    fig.colorbar(depth, ax=axes[0, 1], label="m", shrink=.8)
    axes[1, 0].imshow(_class_rgb(frame.segmentation_class_ids))
    axes[1, 0].set_title("Actual segmentation class IDs (no instance IDs)")
    axes[1, 1].imshow(frame.rgb)
    uv = association["pixel_uv"]
    keep = np.arange(0, len(uv), max(1, len(uv)//12000))
    scatter = axes[1, 1].scatter(uv[keep, 0], uv[keep, 1], s=.5,
                                c=association["axial_residual_m"][keep], cmap="coolwarm", vmin=-2, vmax=2)
    axes[1, 1].set_title("Real LiDAR hit projections; axial residual (m)")
    fig.colorbar(scatter, ax=axes[1, 1], shrink=.8, label="LiDAR axial - depth (m)")
    for axis in axes.flat:
        axis.set_xlim(0, frame.calibration.width); axis.set_ylim(frame.calibration.height, 0)
        axis.set_axis_off()
    fig.suptitle(f"{frame.binding['episode_id']} / {frame.binding['observer']} / tick {frame.binding['tick']}")
    from matplotlib.patches import Patch
    classes = frame.binding["segmentation"]["class_by_id"]
    axes[1, 0].legend(handles=[Patch(color=CLASS_COLORS[int(k)], label=f"{k}: {v}")
                             for k, v in sorted(classes.items(), key=lambda x: int(x[0]))],
                      fontsize=6, ncol=3, loc="lower left", framealpha=.9)
    fig.savefig(path, dpi=150); plt.close(fig)


def materialize_observations(archive: CaptureArchive, output, *, cutoff: int):
    """Persist selected true arrays and sanitized binding, once before training."""
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    inventory = {"source_archive": str(archive.path), "episode_id": archive.episode_id,
                 "observer": archive.observer, "capture_view_id": archive.view_id,
                 "selected_ticks": list(archive.ticks), "cutoff": cutoff,
                 "historical_ticks": [t for t in archive.ticks if t <= cutoff],
                 "target_ticks": [t for t in archive.ticks if t > cutoff],
                 "tick_0_available": {m: 0 in archive.inventory[m]["payload_ticks"] for m in MODALITIES},
                 "coverage": archive.inventory,
                 "numerical_workflow": "one source scan; cached immutable typed arrays; no fitted distribution/donor index; targets distinct from prefix",
                 "producer_source_limit": "Fixed-world depth producer code absent from repository Plugins; image_type alone does not confirm radial semantics"}
    all_metrics = {}
    for tick in archive.ticks:
        branch = "historical" if tick <= cutoff else "target"
        frame = load_observation_frame(archive, archive.episode_id, tick, archive.observer, branch=branch, cutoff=cutoff)
        binding = dict(frame.binding)
        binding.update({"branch": branch, "cutoff": cutoff, "calibration": frame.calibration.as_dict()})
        actual_arrays = {k: getattr(frame, k) for k in ARRAY_NAMES}
        if frame.rgb_source_alpha is not None:
            actual_arrays["rgb_source_alpha"] = frame.rgb_source_alpha
        np.savez_compressed(root/f"tick_{tick:06d}.npz", **actual_arrays)
        (root/f"tick_{tick:06d}.json").write_text(json.dumps(binding, indent=2)+"\n")
        association = lidar_depth_association(frame.points_sensor_ned_m, frame.hit_mask, frame.depth_m,
                                              frame.segmentation_class_ids, frame.calibration)
        np.savez_compressed(root/f"tick_{tick:06d}_association.npz", **association)
        metrics = association_metrics(association, frame.calibration, frame.points_sensor_ned_m)
        metrics.update({"full_scan_point_count": len(frame.points_sensor_ned_m),
                        "hit_count": int(frame.hit_mask.sum()), "rgb_shape": list(frame.rgb.shape),
                        "depth_shape": list(frame.depth_m.shape), "depth_min_m": float(np.min(frame.depth_m)),
                        "depth_max_m": float(np.max(frame.depth_m)),
                        "segmentation_histogram": {str(int(c)): int(n) for c, n in
                                                   zip(*np.unique(frame.segmentation_class_ids, return_counts=True))},
                        "source_entity_association": source_entity_association(frame)})
        all_metrics[str(tick)] = metrics
        _visualize(frame, association, root/f"tick_{tick:06d}_view.png")
    (root/"inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    (root/"geometry_metrics.json").write_text(json.dumps(all_metrics, indent=2)+"\n")
    return all_metrics


def _class_rgb(class_ids):
    palette = np.array([[int(c[i:i+2], 16) for i in (1, 3, 5)] for c in CLASS_COLORS], dtype=np.uint8)
    if int(class_ids.max()) >= len(palette):
        raise ValueError("actual segmentation ID has no fixed display color")
    return palette[class_ids]


def _png_data(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")


def _actor_candidates(metadata, frame):
    """Retain producer command/spawn centres, never infer actor world heights."""
    candidates = {}
    for identity, record in metadata["uav_debug"]["vehicles"].items():
        candidates[identity] = {"kind": "commanded_world_center_candidate",
                                "world_center_m": record["command_payload_target_enu_m"],
                                "source_field": f"uav_debug.vehicles.{identity}.command_payload_target_enu_m"}
    for record in metadata["event_semantic_objects"]:
        response = record["spawn_response"]
        if "position_world_cm" in response:
            xyz = response["position_world_cm"]
            candidates[record["entity_id"]] = {
                "kind": "spawn_response_world_center_candidate",
                "world_center_m": [xyz[k]/100. for k in ("x", "y", "z")],
                "ue_actor_name": response["actor_name"],
                "source_field": f"event_semantic_objects[entity_id={record['entity_id']}].spawn_response"}
    output = []
    for record in metadata["entity_records"]:
        identity = record["entity_id"]
        item = {"entity_id": identity, "category": record["entity_category"],
                "source_position_m": record["position_enu_m"],
                "measured_actor_world_transform": "not_recorded",
                "actor_world_bounds": "not_recorded", "instance_pixel_mapping": "not_recorded",
                "instance_visibility": "not_measured_class_segmentation_only"}
        if identity in candidates:
            item["candidate"] = candidates[identity]
            uv, _, inside = project_world(np.array([candidates[identity]["world_center_m"]]), frame.calibration)
            item["candidate"]["center_in_frustum"] = bool(inside[0])
            if inside[0]:
                item["candidate"]["pixel_uv"] = uv[0].tolist()
        elif record["entity_category"] == "airspace_corridor":
            item["geometry_status"] = "logical_region_only_not_rasterized"
        else:
            item["geometry_status"] = "source_height_to_capture_world_conversion_not_recorded"
        if identity == frame.binding["observer"]:
            item["measured_binding"] = "observer_capture_sensor_rig; not an instance pixel association"
        output.append(item)
    return output


def materialize_episode_view(output, *, model_result, model_bindings):
    """Export true full RGB coverage and actual-run nominal spatial supports.

    This presentation pass does not fit, train, replay or alter model input.
    The archive is streamed once for RGB and selected RGB metadata only.
    """
    root = Path(output)
    inventory = json.loads((root/"inventory.json").read_text())
    result = json.loads(Path(model_result).read_text())
    bindings = json.loads(Path(model_bindings).read_text())
    if result["episode"] != inventory["episode_id"]:
        raise ValueError("model result and capture inventory refer to different episodes")
    rgb_members = inventory["coverage"]["rgb"]["payload_members"]
    tick_by_member = dict(zip(rgb_members, inventory["coverage"]["rgb"]["payload_ticks"]))
    selected = set(inventory["selected_ticks"])
    selected_sidecars = {m: t for m, t in zip(inventory["coverage"]["rgb"]["sidecar_members"],
                                            inventory["coverage"]["rgb"]["sidecar_ticks"]) if t in selected}
    width, height, columns = 192, 108, 10
    sprite = Image.new("RGB", (width*columns, height*((len(rgb_members)+columns-1)//columns)))
    metadata, captured = {}, set()
    with Path(inventory["source_archive"]).open("rb") as source, \
            zstandard.ZstdDecompressor().stream_reader(source) as stream, \
            tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if member.name in tick_by_member:
                tick = tick_by_member[member.name]
                if tick in captured:
                    raise ValueError("duplicate RGB timeline member")
                with Image.open(io.BytesIO(archive.extractfile(member).read())) as image:
                    thumb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                index = inventory["coverage"]["rgb"]["payload_ticks"].index(tick)
                sprite.paste(thumb, ((index % columns)*width, (index//columns)*height))
                captured.add(tick)
            elif member.name in selected_sidecars:
                metadata[selected_sidecars[member.name]] = json.loads(archive.extractfile(member).read())
    if captured != set(tick_by_member.values()) or metadata.keys() != selected:
        raise ValueError("actual archive did not supply complete inventoried RGB/selected metadata")
    sprite.save(root/"rgb_timeline.jpg", quality=88)
    actual_layouts = [{**b, "role": "historical_input"} for b in bindings["rgb"]+bindings["sensor_suffix"]]
    actual_layouts += bindings["predicted_append"]+bindings["future_head_outputs"]
    settings = result["input"]["static_sensor_settings"]
    channels, azimuths = settings["NumberOfChannels"], settings["MeasurementsPerCycle"]
    column = np.arange(azimuths)[:, None]
    channel = np.arange(channels)[None, :]
    fine = np.minimum(channel*32//(channels-1), 31)*64+column*64//azimuths
    coarse = (fine//64//8)*8+(fine % 64)//8
    fine, coarse = fine.ravel().astype(np.uint16), coarse.ravel().astype(np.uint8)
    slot_arrays = {"fine_region_id": fine, "coarse_region_id": coarse,
                   "source_scan_index": np.arange(len(fine), dtype=np.uint32)}
    frames, embedded_images, packed_hits = {}, {}, {}
    semantic_classes = None
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    for tick in sorted(selected):
        role = "historical" if tick <= result["cutoff"] else "target"
        frame = load_observation_frame(root, inventory["episode_id"], tick, inventory["observer"],
                                       branch=role, cutoff=result["cutoff"])
        classes = frame.binding["segmentation"]["class_by_id"]
        if semantic_classes is None:
            semantic_classes = classes
        elif classes != semantic_classes:
            raise ValueError("selected capture frames have different semantic class registries")
        if any(int(k) >= len(CLASS_COLORS) for k in classes):
            raise ValueError("capture semantic registry exceeds explicit color table")
        if len(frame.hit_mask) != len(coarse):
            raise ValueError("actual scan slot count differs from executed sensor layout")
        hit = frame.hit_mask.astype(bool)
        slot_arrays[f"tick_{tick:06d}_hit_mask"] = frame.hit_mask
        packed_hits[str(tick)] = base64.b64encode(np.packbits(hit, bitorder="little").tobytes()).decode("ascii")
        embedded_images[str(tick)] = {"rgb": _png_data(frame.rgb),
                                       "seg": _png_data(_class_rgb(frame.segmentation_class_ids)),
                                       "depth": _png_data((colormaps["viridis"](np.log1p(np.clip(frame.depth_m, 0, 200.))/np.log1p(200.))[..., :3]*255).astype(np.uint8))}
        frames[str(tick)] = {
            "archive_sidecar": next(m for m, t in selected_sidecars.items() if t == tick),
            "raw_array_file": f"tick_{tick:06d}.npz", "image_size_wh": [frame.calibration.width, frame.calibration.height],
            "actor_associations": _actor_candidates(metadata[tick], frame),
            "scan_hit_count": int(hit.sum()),
            "coarse_hit_count": np.bincount(coarse[hit], minlength=32).tolist(),
            "fine_hit_count": np.bincount(fine[hit], minlength=2048).tolist(),
            "model_roles": sorted({b["role"] for b in actual_layouts if b["tick"] == tick}),
            "modality_sources": frame.binding["modality"]}
        with np.load(root/f"tick_{tick:06d}_association.npz", allow_pickle=False) as association:
            _visualize(frame, association, root/f"tick_{tick:06d}_view.png")
    regions = []
    for layout in actual_layouts:
        tick, modality = layout["tick"], layout["modality"]
        w, h = frames[str(tick)]["image_size_wh"]
        if modality == "rgb":
            grid = result["input"]["native_rgb_grid"]
            rows, cols = grid[1]//2, grid[2]//2
        elif modality == "lidar":
            rows, cols = 4, 8
        else:
            rows, cols = 8, 16
        if "token_start" in layout and layout["token_end"]-layout["token_start"] != rows*cols:
            raise ValueError("executed token count diverges from nominal spatial grid")
        entry = {"tick": tick, "modality": modality, "role": layout["role"], "rows": rows, "cols": cols,
                 "source": frames[str(tick)]["modality_sources"][modality],
                 "actual_source_role": "historical_input" if layout["role"] == "historical_input" else "target_label_only",
                 "support_semantics": "nominal spatial support; encoder attention mixes context; not an exclusive receptive field",
                 "regions": []}
        for i in range(rows*cols):
            row, col = divmod(i, cols)
            cell = {"region_id": i, "row": row, "column": col,
                    "actual_model_grid_position": bindings["future_layouts"][modality][i]}
            if "token_start" in layout:
                cell["token_index"] = layout["token_start"]+i
            else:
                cell["token_status"] = "head_output_only_no_sequence_token"
            if modality == "lidar":
                ids = np.flatnonzero(coarse == i)
                cell.update({"scan_slot_count": len(ids), "actual_hit_count": frames[str(tick)]["coarse_hit_count"][i],
                             "scan_index_array_file": "scan_regions.npz", "scan_index_rule": "source_scan_index[coarse_region_id == region_id]",
                             "azimuth_column_range_half_open": [int(ids[0]//channels), int(ids[-1]//channels)+1],
                             "channel_range_half_open": [int(ids[0] % channels), int(ids[-1] % channels)+1],
                             "azimuth_nominal_edges_deg": [-180.+col*45., -180.+(col+1)*45.],
                             "elevation_nominal_edges_deg": [settings["VerticalFOVLower"]+row*(settings["VerticalFOVUpper"]-settings["VerticalFOVLower"])/4.,
                                                              settings["VerticalFOVLower"]+(row+1)*(settings["VerticalFOVUpper"]-settings["VerticalFOVLower"])/4.],
                             "fine_region_rows_half_open": [row*8, (row+1)*8],
                             "fine_region_columns_half_open": [col*8, (col+1)*8]})
            elif modality == "rgb":
                cell.update({"processed_pixel_bbox_half_open": [col*32, row*32, (col+1)*32, (row+1)*32],
                             "processed_image_size_wh": [grid[2]*16, grid[1]*16],
                             "source_pixel_bbox_nominal_continuous": [col*w/cols, row*h/rows, (col+1)*w/cols, (row+1)*h/rows],
                             "resize_semantics": "full-image resize without crop; bbox is inverse mapped merged patch footprint; resampling mixes boundary pixels"})
            else:
                cell["source_pixel_bbox_half_open"] = [col*w//cols, row*h//rows, (col+1)*w//cols, (row+1)*h//rows]
            if modality != "lidar":
                box = cell["source_pixel_bbox_nominal_continuous"] if modality == "rgb" else cell["source_pixel_bbox_half_open"]
                cell["candidate_centers_in_nominal_pixel_bbox"] = [
                    {"entity_id": a["entity_id"], "kind": a["candidate"]["kind"]}
                    for a in frames[str(tick)]["actor_associations"] if "candidate" in a and "pixel_uv" in a["candidate"]
                    and box[0] <= a["candidate"]["pixel_uv"][0] < box[2] and box[1] <= a["candidate"]["pixel_uv"][1] < box[3]]
                cell["candidate_geometry_role"] = "capture sidecar centre candidates only; target geometry is label-only; no measured instance binding"
            entry["regions"].append(cell)
        regions.append(entry)
    np.savez_compressed(root/"scan_regions.npz", **slot_arrays)
    timeline = {"episode_id": inventory["episode_id"], "observer": inventory["observer"],
                "source_archive": inventory["source_archive"], "inventory_source": str(root/"inventory.json"),
                "model_result_source": str(model_result), "model_bindings_source": str(model_bindings),
                "run": {"cutoff": result["cutoff"], "targets": result["targets"],
                        "prefix_tokens": result["input"]["prefix_tokens"], "suffix_tokens": result["input"]["suffix_tokens"],
                        "predicted_append_tokens": result["input"]["predicted_append_tokens"], "final_sequence_tokens": result["rollout"]["cache_length"]},
                "sprite": {"path": "rgb_timeline.jpg", "width": width, "height": height, "columns": columns},
                "frames": [{"tick": t, "sprite_index": i, "rgb_payload_member": m,
                            "coverage": {kind: {"payload": t in inventory["coverage"][kind]["payload_ticks"],
                                                "sidecar": t in inventory["coverage"][kind]["sidecar_ticks"]} for kind in MODALITIES},
                            "model_roles": sorted({b["role"] for b in actual_layouts if b["tick"] == t})}
                           for i, (m, t) in enumerate(tick_by_member.items())],
                "tick_0_available": inventory["tick_0_available"],
                "sensor_layout": {"source": result["input"]["sensor_settings_source"], "settings": settings,
                                  "fine_rows_columns": [32, 64], "coarse_rows_columns": [4, 8],
                                  "slot_order": "azimuth-major; index=azimuth_column*NumberOfChannels+channel",
                                  "executed_lidar_feature_shape": result["input"]["modal_feature_shapes"]["lidar"]},
                "semantic_legend_source": frames[str(min(selected))]["modality_sources"]["seg"]["sidecar_member"]+"#semantic_class_by_id",
                "semantic_legend": [{"class_id": int(k), "name": v, "color": CLASS_COLORS[int(k)]}
                                    for k, v in sorted(classes.items(), key=lambda x: int(x[0]))],
                "selected_frames": frames, "support_semantics": "nominal spatial support; Qwen and Sonata mix encoder context",
                "actor_binding_missing_fields": ["capture-time measured UE actor transform", "UE world bounds/mesh extents", "instance pixel IDs or hit actor IDs", "per-image acquisition timestamp"],
                "numerical_workflow": "presentation only; no fit or split changes; raw immutable arrays reused; archive scan outside numerical hot path"}
    (root/"episode_timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (root/"token_regions.json").write_text(json.dumps({"model_bindings_source": str(model_bindings), "layouts": regions}, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    browser_data = {"timeline": timeline, "layouts": regions, "images": embedded_images,
                    "packed_hits": packed_hits, "channels": channels, "fine_ids": base64.b64encode(fine.tobytes()).decode("ascii")}
    _write_episode_html(root/"episode_view.html", browser_data)
    return {"rgb_frames": len(captured), "coverage": {m: {"payload": len(inventory["coverage"][m]["payload_ticks"]),
                                                            "sidecar": len(inventory["coverage"][m]["sidecar_ticks"])} for m in MODALITIES},
            "executed_layouts": len(regions), "run": timeline["run"], "view": str(root/"episode_view.html")}


def _write_episode_html(path, data):
    # Embed the small selected-frame presentation data so file:// needs no fetch.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实 episode 时间线与名义空间支持</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151c;color:#e5ecf3}
body{max-width:1500px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 8px}h2{font-size:18px;margin:12px 0}
p{line-height:1.55;color:#b9c7d6}a{color:#8bd2ff}button,select{background:#263342;color:#eef6ff;border:1px solid #526479;border-radius:5px;padding:7px;cursor:pointer}
button.active{border:2px solid #ffd268}code,pre{font-size:12px;overflow-wrap:anywhere;white-space:pre-wrap}pre{background:#0c1117;border-radius:6px;padding:12px;line-height:1.5}
.panel{background:#18212c;border:1px solid #344354;border-radius:8px;padding:16px;margin-top:16px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.workspace{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(330px,1fr);gap:16px}.workspace>.panel{min-width:0}
#viewer{width:100%;background:#080d13;display:block;cursor:crosshair;border:1px solid #516070}.label{font-size:12px;color:#a9bacb}.badge{border:1px solid #6d8b9f;padding:4px 7px;border-radius:5px;color:#ffd268}
#coverage{width:100%;height:115px;display:block;cursor:pointer}#timeline{display:grid;grid-template-columns:repeat(auto-fill,minmax(142px,1fr));gap:8px}
.frame{padding:0;text-align:left;overflow:hidden}.thumb{display:block;width:100%;aspect-ratio:16/9;background-repeat:no-repeat}.frame span{display:block;padding:5px;font-size:11px}
.legend{display:flex;gap:6px 16px;flex-wrap:wrap;font-size:12px}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px;vertical-align:middle}
.muted{color:#9bb0c2}.downloads{font-size:13px}.hidden{display:none}@media(max-width:850px){body{padding:12px}.workspace{grid-template-columns:1fr}}
</style>
<h1 id="title"></h1><p>180 帧真实 RGB 时间线；四模态归档覆盖与实际执行角色分开。点击时间线，再点击网格选择区域。</p>
<div class="row downloads"><a href="episode_timeline.json">时间线 / 来源 JSON</a><a href="token_regions.json">区域 / token JSON</a><a href="scan_regions.npz">精确 scan 下标 NPZ</a><a href="rgb_timeline.jpg">180 帧 RGB sprite</a><a href="../current_model/result.json">实际运行结果</a><a href="../current_model/bindings.json">实际运行绑定</a></div>
<section class="panel"><div class="row"><strong>归档与执行覆盖</strong><span class="label" id="summary"></span></div><canvas id="coverage"></canvas><div class="legend"><span>蓝色：payload + sidecar 已采集</span><span style="color:#ffd268">黄色：历史输入</span><span style="color:#e99bff">紫色：预测追加</span><span style="color:#90dca6">绿色：预测头输出</span><span>空白：未执行模型区域</span></div></section>
<div class="workspace"><section class="panel"><div class="row"><h2 id="frameTitle"></h2><span class="badge" id="role"></span></div>
<div class="row" id="modes"></div><p class="label" id="caption"></p><canvas id="viewer" width="1280" height="720"></canvas>
<p class="label">RGB / Sonata：nominal spatial support（名义空间网格）。编码器注意力已混合上下文，所标区域不是独占感受野。depth/seg 是原图聚合区域；目标原图只用于对照。</p>
<div id="legend" class="legend"></div></section>
<section class="panel"><h2>所选区域</h2><pre id="inspect">点击网格查看精确 bbox、token 角色或 scan 槽下标。</pre>
<div class="row"><button id="download">导出所选 scan 下标 CSV</button><label class="label"><input id="hitsOnly" type="checkbox">仅真实 hit</label><label class="label"><input id="fineGrid" type="checkbox">LiDAR 64×32 细网格</label></div>
<h2>实体几何</h2><select id="actor"></select><pre id="actorInspect"></pre><p class="label">十字只表示命令或 spawn 返回中心候选。类别分割没有实例 ID；未记录实测 actor transform / bounds 时，不给精确实例绑定。</p>
<details><summary>真实来源路径与缺项</summary><pre id="sources"></pre></details></section></div>
<section class="panel"><h2>完整 observer 时间线</h2><div id="timeline"></div></section>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent),T=D.timeline,$=id=>document.getElementById(id);
const names={historical_input:'历史输入',predicted_append:'预测追加',predicted_head_output_no_token_range:'预测头输出 / 无追加 token'};
const colors={historical_input:'#ffd268',predicted_append:'#e99bff',predicted_head_output_no_token_range:'#90dca6'};
let tick=T.run.cutoff,mode='rgb',region=0,fineRegion=null,currentImage=null,drawSerial=0;
const fineBytes=Uint8Array.from(atob(D.fine_ids),c=>c.charCodeAt(0));
const fineIds=new Uint16Array(fineBytes.buffer),hitBytes={};for(const [t,v] of Object.entries(D.packed_hits))hitBytes[t]=Uint8Array.from(atob(v),c=>c.charCodeAt(0));
const sprite=new Image();sprite.src=T.sprite.path;
$('title').textContent=T.episode_id+' / '+T.observer;
$('summary').textContent='ticks 5–900 / 180 RGB；tick 0 四模态缺失。prefix '+T.run.prefix_tokens+' + suffix '+T.run.suffix_tokens+' + predicted append '+T.run.predicted_append_tokens+' = '+T.run.final_sequence_tokens;
for(const entry of T.semantic_legend){const s=document.createElement('span'),box=document.createElement('i');box.className='swatch';box.style.background=entry.color;s.append(box,document.createTextNode(entry.class_id+': '+entry.name));$('legend').append(s)}
function frameEntry(){return T.frames.find(f=>f.tick===tick)}function selected(){return T.selected_frames[String(tick)]}
function layout(){return D.layouts.find(l=>l.tick===tick&&l.modality===mode)}
function hit(index){return (hitBytes[String(tick)][index>>3]>>(index&7))&1}
function roleLabel(){const roles=frameEntry().model_roles;return roles.length?roles.map(x=>names[x]).join(' / '):'归档帧 / 本轮模型未执行'}
function selectTick(t){tick=t;region=0;fineRegion=null;document.querySelectorAll('.frame').forEach(b=>b.classList.toggle('active',Number(b.dataset.tick)===tick));update()}
function coverage(){const c=$('coverage'),ratio=window.devicePixelRatio||1,w=c.clientWidth;c.width=w*ratio;c.height=115*ratio;const ctx=c.getContext('2d');ctx.scale(ratio,ratio);const start=62,cell=(w-start)/T.frames.length;
 ['rgb','depth','seg','lidar'].forEach((m,r)=>{ctx.fillStyle='#d6e4f1';ctx.font='11px sans-serif';ctx.fillText(m,2,20+r*18);T.frames.forEach((f,i)=>{ctx.fillStyle=f.coverage[m].payload&&f.coverage[m].sidecar?'#397ca4':'#b85757';ctx.fillRect(start+i*cell,8+r*18,Math.max(1,cell-1),12)})});
 T.frames.forEach((f,i)=>{if(f.model_roles.length){ctx.fillStyle=colors[f.model_roles[0]];ctx.fillRect(start+i*cell,82,Math.max(2,cell),10)}});ctx.strokeStyle='#fff';const idx=T.frames.findIndex(f=>f.tick===tick);ctx.strokeRect(start+idx*cell-1,6,cell+1,88);ctx.fillStyle='#b9c7d6';ctx.fillText('5',start,110);ctx.fillText('900',w-26,110)}
$('coverage').onclick=e=>{const x=e.offsetX,w=$('coverage').clientWidth;const i=Math.floor((x-62)/(w-62)*T.frames.length);if(i>=0&&i<T.frames.length)selectTick(T.frames[i].tick)};
for(const f of T.frames){const b=document.createElement('button');b.className='frame';b.dataset.tick=f.tick;const pic=document.createElement('i');pic.className='thumb';const col=f.sprite_index%T.sprite.columns,row=Math.floor(f.sprite_index/T.sprite.columns),rows=Math.ceil(T.frames.length/T.sprite.columns);pic.style.backgroundImage='url('+T.sprite.path+')';pic.style.backgroundSize=(T.sprite.columns*100)+'% '+(rows*100)+'%';pic.style.backgroundPosition=(col/(T.sprite.columns-1)*100)+'% '+(row/(rows-1)*100)+'%';const text=document.createElement('span');text.textContent='tick '+f.tick+' · '+(f.model_roles.length?names[f.model_roles[0]]:'采集');b.append(pic,text);b.onclick=()=>selectTick(f.tick);$('timeline').append(b)}
for(const m of ['rgb','depth','seg','lidar']){const b=document.createElement('button');b.textContent=m.toUpperCase();b.dataset.mode=m;b.onclick=()=>{mode=m;region=0;fineRegion=null;update()};$('modes').append(b)}
function actorUpdate(){const f=selected(),i=Number($('actor').value),a=f&&f.actor_associations[i];$('actorInspect').textContent=a?JSON.stringify(a,null,2):'此帧未材料化 actor 绑定。';draw()}
function update(){const f=selected();if(!f)mode='rgb';$('frameTitle').textContent='tick '+tick+' / '+mode.toUpperCase();$('role').textContent=roleLabel();$('role').style.color=frameEntry().model_roles.length?colors[frameEntry().model_roles[0]]:'#a6bac9';
 document.querySelectorAll('#modes button').forEach(b=>{b.classList.toggle('active',b.dataset.mode===mode);b.disabled=!f&&b.dataset.mode!=='rgb'});
 $('caption').textContent=!f?'真实归档 RGB 缩略预览 192×108；此帧无模型 token 记录。':mode==='lidar'?'真实 scan 命中计数的名义角度网格；点击粗区域或细网格。':frameEntry().model_roles.includes('historical_input')?'实际历史输入原图；网格读取实际执行绑定。':frameEntry().model_roles.length?'实际目标真值原图，仅作对照；覆盖预测的名义布局，不是预测 RGB 解码图。':'真实材料化归档帧；本轮模型未执行。';if(mode==='depth')$('caption').textContent+=' 深度显示：固定 viridis / log(1+m)，0–200 m；原始数值不变。';
 $('actor').replaceChildren();if(f){f.actor_associations.forEach((a,i)=>{const o=document.createElement('option');o.value=i;o.textContent=a.entity_id+(a.candidate?' · 中心候选':' · 几何缺失');$('actor').append(o)})}else{const o=document.createElement('option');o.textContent='未材料化';$('actor').append(o)}
 $('legend').classList.toggle('hidden',mode!=='seg');$('fineGrid').disabled=mode!=='lidar';$('download').disabled=mode!=='lidar'||!layout();
 $('sources').textContent=JSON.stringify({source_archive:T.source_archive,source_rgb:frameEntry().rgb_payload_member,model_result:T.model_result_source,model_bindings:T.model_bindings_source,selected_sources:f&&f.modality_sources,missing_actor_binding_fields:T.actor_binding_missing_fields},null,2);
 coverage();actorUpdate();inspect();loadImage()}
function loadImage(){const serial=++drawSerial;currentImage=null;if(mode==='lidar'){draw();return}const img=new Image();img.onload=()=>{if(serial===drawSerial){currentImage=img;draw()}};img.src=selected()?D.images[String(tick)][mode]:sprite.src}
function draw(){const canvas=$('viewer'),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);const f=selected(),l=layout();
 if(mode==='lidar'&&f){const max=Math.max(...f.fine_hit_count);for(let i=0;i<2048;i++){const n=f.fine_hit_count[i],v=n?Math.log1p(n)/Math.log1p(max):0;ctx.fillStyle=n?'hsl('+(215-v*170)+',70%,'+(25+v*35)+'%)':'#101923';ctx.fillRect(i%64*W/64,Math.floor(i/64)*H/32,W/64+1,H/32+1)}}
 else if(currentImage){if(f)ctx.drawImage(currentImage,0,0,W,H);else{const i=frameEntry().sprite_index;ctx.drawImage(currentImage,i%T.sprite.columns*T.sprite.width,Math.floor(i/T.sprite.columns)*T.sprite.height,T.sprite.width,T.sprite.height,0,0,W,H)}}
 if(l){const rows=mode==='lidar'&&$('fineGrid').checked?32:l.rows,cols=mode==='lidar'&&$('fineGrid').checked?64:l.cols;ctx.strokeStyle='rgba(255,255,255,.32)';ctx.lineWidth=1;for(let x=0;x<=cols;x++){ctx.beginPath();ctx.moveTo(x*W/cols,0);ctx.lineTo(x*W/cols,H);ctx.stroke()}for(let y=0;y<=rows;y++){ctx.beginPath();ctx.moveTo(0,y*H/rows);ctx.lineTo(W,y*H/rows);ctx.stroke()}const fineSelected=mode==='lidar'&&fineRegion!==null&&$('fineGrid').checked,hcols=fineSelected?64:l.cols,hrows=fineSelected?32:l.rows,c=fineSelected?fineRegion%64:region%l.cols,r=fineSelected?Math.floor(fineRegion/64):Math.floor(region/l.cols);ctx.strokeStyle='#ffcf53';ctx.lineWidth=4;ctx.strokeRect(c*W/hcols,r*H/hrows,W/hcols,H/hrows)}
 if(f&&mode!=='lidar'){const a=f.actor_associations[Number($('actor').value)];if(a&&a.candidate&&a.candidate.pixel_uv){const [x,y]=a.candidate.pixel_uv;ctx.strokeStyle='#ff7ad9';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-10,y);ctx.lineTo(x+10,y);ctx.moveTo(x,y-10);ctx.lineTo(x,y+10);ctx.stroke()}}}
function inspect(){const l=layout();if(!l){$('inspect').textContent='此帧本轮没有执行 token / 预测区域记录。';return}const c=l.regions[region];const info={tick,modality:mode,execution_role:l.role,actual_source_role:l.actual_source_role,support_semantics:l.support_semantics,...c};if(fineRegion!==null&&mode==='lidar'&&$('fineGrid').checked){info.selected_fine_region=fineRegion;info.fine_actual_hit_count=selected().fine_hit_count[fineRegion];info.fine_scan_index_rule='source_scan_index[fine_region_id == '+fineRegion+']';info.token_scope='对应粗 region token；细格没有独立 token'}$('inspect').textContent=JSON.stringify(info,null,2)}
$('viewer').onclick=e=>{const l=layout();if(!l)return;const rect=$('viewer').getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width,y=(e.clientY-rect.top)/rect.height;if(mode==='lidar'&&$('fineGrid').checked){const r=Math.min(31,Math.floor(y*32)),c=Math.min(63,Math.floor(x*64));fineRegion=r*64+c;region=Math.floor(r/8)*8+Math.floor(c/8)}else{region=Math.min(l.rows-1,Math.floor(y*l.rows))*l.cols+Math.min(l.cols-1,Math.floor(x*l.cols));fineRegion=null}inspect();draw()};
$('fineGrid').onchange=()=>{fineRegion=null;inspect();draw()};$('actor').onchange=actorUpdate;
$('download').onclick=()=>{if(mode!=='lidar'||!layout())return;const useFine=$('fineGrid').checked&&fineRegion!==null,chosen=useFine?fineRegion:region,rows=['source_scan_index,hit'];for(let i=0;i<fineIds.length;i++){const fine=fineIds[i],coarse=Math.floor(Math.floor(fine/64)/8)*8+Math.floor((fine%64)/8);if((useFine?fine:coarse)!==chosen)continue;if($('hitsOnly').checked&&!hit(i))continue;rows.push(i+','+hit(i))}const blob=new Blob([rows.join('\n')+'\n'],{type:'text/csv'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='tick_'+tick+'_'+(useFine?'fine':'coarse')+'_'+chosen+'_scan_indices.csv';a.click();URL.revokeObjectURL(url)};
window.addEventListener('resize',coverage);sprite.onload=()=>{if(!selected())loadImage()};selectTick(tick);
</script></html>'''
    path.write_text(template.replace("__DATA__", payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive"); parser.add_argument("--output", required=True)
    parser.add_argument("--episode"); parser.add_argument("--observer")
    parser.add_argument("--ticks", nargs="+", type=int)
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--episode-view", action="store_true", help="export full RGB timeline from existing observation inventory")
    parser.add_argument("--model-result"); parser.add_argument("--model-bindings")
    args = parser.parse_args()
    if args.episode_view:
        if args.model_result is None or args.model_bindings is None:
            parser.error("--episode-view requires --model-result and --model-bindings")
        print(json.dumps(materialize_episode_view(args.output, model_result=args.model_result,
                                                model_bindings=args.model_bindings), ensure_ascii=False))
        return
    if any(getattr(args, name) is None for name in ("archive", "episode", "observer", "ticks", "cutoff")):
        parser.error("array materialization requires --archive --episode --observer --ticks --cutoff")
    source = CaptureArchive(args.archive, episode_id=args.episode, observer=args.observer, ticks=args.ticks)
    metrics = materialize_observations(source, args.output, cutoff=args.cutoff)
    print(json.dumps({"coverage_counts": {m: {k: v for k, v in e.items() if k.endswith("_count")} for m, e in source.inventory.items()},
                      "metrics": {t: {k: v for k, v in m.items() if k != "source_entity_association"} for t, m in metrics.items()}}, indent=2))


if __name__ == "__main__":
    main()
