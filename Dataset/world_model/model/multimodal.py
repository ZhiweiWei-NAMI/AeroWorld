"""Frozen official 4M/RGB/Sonata with trainable typed spatial adapters.

Depth and UE labels use independent 196x32 quantized spatial grids. Original
4M weights and codebooks stay in eval; small metric-depth and declared-UE
adapters train from genuine images. Sonata emits 32x512 regional features.
The full acquired-ray decoder uses fixed calibrated directions, causal per-ray
range/return priors and shared learned residual laws, outside Qwen positions.
World clouds combine predicted proper NED6 body orientation/position with the
historical calibrated mount. Source identities and raw clocks remain sidecars.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from ..data.geometry import WORLD_FROM_NED

# Dense acquired-ray grid of the measured sensor configuration. The producer
# scan is azimuth-major with 2048 azimuth columns and 128 elevation channels
# (262144 acquired directions); configure_sensor_layout validates both numbers
# against the authentic settings file. The 4x8 regional grid is the separate
# 32-slot Sonata/support representation consumed by Qwen, never the ray grid.
AZIMUTH_CELLS, ELEVATION_CELLS = 64, 32
COARSE_AZ, COARSE_EL = 8, 4
REGIONS = COARSE_AZ * COARSE_EL
# Selected codec runtime resolution: 224 px with 16 px patches -> 14x14 tokens.
CODEC_IMAGE_SIZE, CODEC_PATCH_SIZE = 224, 16
CODEC_ROWS, CODEC_COLS = CODEC_IMAGE_SIZE // CODEC_PATCH_SIZE, CODEC_IMAGE_SIZE // CODEC_PATCH_SIZE
# Conditional range support of the dense decoder: 384 cells over [0, 200] m.
RANGE_CELLS = 384
RAY_CHUNK = 8192
MODALITY_IDS = {"rgb": 0, "depth": 1, "seg": 2, "lidar": 3}

EL_LOWER = EL_UPPER = None
SENSOR_RANGE_M = None
SCANNED_CELL_COUNTS = None
BEAM_CELL_INDICES = BEAM_DIRECTIONS = None


def configure_sensor_layout(binding_lidar, sensor_name):
    """Build the fixed dense ray geometry from an authentic frame binding.

    ``binding_lidar`` is the prepared frame binding's ``lidar`` document, i.e.
    the true producer sidecar's ``lidar`` section (sensor_name, sensor_settings,
    parameter_provenance). Validates sensor identity, channel/azimuth counts,
    full azimuth coverage, the declared uniform CPU ray pattern and positive
    range; raises on any divergence. Sets BEAM_DIRECTIONS [acquired, 3] (flat
    index azimuth*channels+channel, sensor-local NED unit vectors),
    BEAM_CELL_INDICES mapping every acquired ray onto the fixed aggregate grid,
    and SCANNED_CELL_COUNTS. Returns the raw declared settings dictionary.
    """
    global EL_LOWER, EL_UPPER, SENSOR_RANGE_M, SCANNED_CELL_COUNTS, BEAM_CELL_INDICES, BEAM_DIRECTIONS
    metadata = {"lidar": binding_lidar}
    if metadata["lidar"]["sensor_name"] != sensor_name:
        raise ValueError("static LiDAR settings refer to another sensor")
    settings = metadata["lidar"]["sensor_settings"]
    SENSOR_RANGE_M = float(settings["Range"])
    if not SENSOR_RANGE_M > 0.:
        raise ValueError("declared sensor range must be positive")
    pattern = metadata["lidar"]["parameter_provenance"]["parameter_basis"]["BeamPattern"]
    if "uniformly spaced" not in pattern or "periodic endpoint excluded" not in pattern:
        raise ValueError("dense ray geometry requires the declared uniform CPU ray pattern")
    if settings["HorizontalFOVStart"] != -180. or settings["HorizontalFOVEnd"] != 180.:
        raise ValueError("dense ray geometry requires full azimuth coverage")
    channels, azimuths = int(settings["NumberOfChannels"]), int(settings["MeasurementsPerCycle"])
    EL_LOWER, EL_UPPER = np.deg2rad(settings["VerticalFOVLower"]), np.deg2rad(settings["VerticalFOVUpper"])
    emitted_el = np.linspace(EL_LOWER, EL_UPPER, channels)
    emitted_az = np.linspace(-np.pi, np.pi, azimuths, endpoint=False)
    az, el = np.meshgrid(emitted_az, emitted_el, indexing="ij")
    columns, channel_index = np.arange(azimuths)[:, None], np.arange(channels)[None, :]
    ia = columns * AZIMUTH_CELLS // azimuths
    ie = np.minimum(channel_index * ELEVATION_CELLS // (channels - 1), ELEVATION_CELLS - 1)
    BEAM_CELL_INDICES = (ie * AZIMUTH_CELLS + ia).ravel()
    BEAM_DIRECTIONS = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1).reshape(-1, 3)
    SCANNED_CELL_COUNTS = np.bincount(BEAM_CELL_INDICES, minlength=AZIMUTH_CELLS * ELEVATION_CELLS)
    return settings


def image_grid(rows, cols):
    """[rows*cols, 4] nominal image positions: x, y in [-1, 1] plus squares."""
    yy, xx = np.meshgrid((np.arange(rows) + .5) / rows, (np.arange(cols) + .5) / cols, indexing="ij")
    x, y = xx.ravel() * 2 - 1, yy.ravel() * 2 - 1
    return torch.tensor(np.stack((x, y, x * x, y * y), -1), dtype=torch.float32)


def angular_grid(rows, cols):
    """[rows*cols, 4] region positions: cos/sin azimuth and sin/cos elevation."""
    if EL_LOWER is None:
        raise ValueError("sensor ray layout has not been loaded")
    el, az = np.meshgrid((np.arange(rows) + .5) / rows * (EL_UPPER - EL_LOWER) + EL_LOWER,
                         (np.arange(cols) + .5) / cols * 2 * np.pi - np.pi, indexing="ij")
    return torch.tensor(np.stack((np.cos(az), np.sin(az), np.sin(el), np.cos(el)), -1).reshape(-1, 4),
                        dtype=torch.float32)


def group_fine(value):
    """[..., azimuth, channel(, ...)] az-major values -> [32, rays_per_region(, ...)].

    Region order is row-major over the 4x8 support grid; inside a region the
    ray order stays elevation-major then azimuth inside each pooled support region.
    """
    shape = value.shape[1:]
    return value.reshape(COARSE_EL, ELEVATION_CELLS // COARSE_EL, COARSE_AZ,
                         AZIMUTH_CELLS // COARSE_AZ, *shape).swapaxes(1, 2).reshape(COARSE_EL * COARSE_AZ, -1, *shape)


def sensor_pose_from_calibration(calibration):
    """Authentic observer pose fields from the capture calibration.

    Returns program-side dictionaries (never model tensors):
    ``observer.capture_position_world_enu_m`` (three metre components) and
    ``observer.body_orientation_ned_6d`` (the first two columns of
    ``R_ned = F.T @ R_bodyworld``, column-major). The rig's body_to_world is
    improper by declared convention; R_ned is proper and reconstructs exactly.
    """
    body_to_world = np.asarray(calibration.body_to_world, dtype=np.float64)
    rotation = body_to_world[:3, :3]
    if abs(np.linalg.det(rotation) + 1.) > 1e-6:
        raise ValueError("authentic body_to_world does not follow the fixed rig reflection convention")
    # F = diag(1,1,-1) is its own transpose, so F.T @ R is exact.
    r_ned = WORLD_FROM_NED.T @ rotation
    if abs(np.linalg.det(r_ned) - 1.) > 1e-6:
        raise ValueError("F.T @ body_to_world rotation is not proper; check the mount convention")
    if abs(np.linalg.norm(r_ned[:, 0]) - 1.) > 1e-9 or abs(float(r_ned[:, 0] @ r_ned[:, 1])) > 1e-9:
        raise ValueError("authentic orientation columns are not orthonormal")
    return {"observer.capture_position_world_enu_m": body_to_world[:3, 3].tolist(),
            "observer.body_orientation_ned_6d": np.concatenate((r_ned[:, 0], r_ned[:, 1])).tolist(),
            "body_to_world": body_to_world.tolist(),
            "sensor_to_body": np.asarray(calibration.sensor_to_body, dtype=np.float64).tolist(),
            "convention": "R_ned=F.T@R_bodyworld with F=diag(1,1,-1); R_ned reconstructed by "
                          "Gram-Schmidt on the two carried columns; R_bodyworld=F@R_ned"}


class SpatialCodecs(nn.Module):
    """Strictly loaded frozen codecs and independently learned UE/metric adapters."""

    def __init__(self, models_path, device, decode_timesteps=25):
        super().__init__()
        root = Path(models_path)
        from fourm.vq.vqvae import DiVAE, VQVAE  # installed 4M runtime, imported lazily
        depth_config = json.loads((root / "depth_config.json").read_text())
        semantic_config = json.loads((root / "semantic_config.json").read_text())
        self.depth = DiVAE(**depth_config)
        self.depth.load_state_dict(torch.load(root / "depth.pt", map_location="cpu", weights_only=False), strict=True)
        self.semantic = VQVAE(**semantic_config)
        self.semantic.load_state_dict(torch.load(root / "semantic.pt", map_location="cpu", weights_only=False), strict=True)
        if int(depth_config["patch_size"]) != CODEC_PATCH_SIZE or int(semantic_config["patch_size"]) != CODEC_PATCH_SIZE:
            raise ValueError("codec patch size diverges from the selected runtime patch size")
        self.config_image_size = int(depth_config["image_size"])
        if int(semantic_config["image_size"]) != self.config_image_size:
            raise ValueError("depth and semantic codecs declare different image sizes")
        # Runtime resolution: 224 px (verified round trip). Config declares 448.
        self.image_size = CODEC_IMAGE_SIZE
        self.grid = (CODEC_ROWS, CODEC_COLS)
        self.depth_codebook_size = int(depth_config["codebook_size"])
        self.semantic_codebook_size = int(semantic_config["codebook_size"])
        self.semantic_labels = int(semantic_config["n_labels"])
        self.decode_timesteps = int(decode_timesteps)
        self.device = torch.device(device)
        self.ue_ids = ()
        for model in (self.depth, self.semantic):  # frozen by default
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.metadata = {
            "depth": {"architecture": "DiVAE", "config_image_size": self.config_image_size,
                      "runtime_image_size": self.image_size, "patch_size": CODEC_PATCH_SIZE,
                      "grid": list(self.grid), "codebook_size": self.depth_codebook_size,
                      "latent_dim": int(depth_config["latent_dim"]),
                      "input_convention": "bilinear runtime resize to 224 px, then d16=metres/65535; truncated (10% tail "
                                          "removed) mean/var of valid resized pixels; d_std=(d16-mean)/sqrt(var+1e-6); "
                                          "decode metres=(d_std*sqrt(var+1e-6)+mean)*65535; zero-depth entries follow the "
                                          "official invalid-entry convention and stay outside the statistics",
                      "normalization_recorded_per_frame": True, "decode_timesteps": self.decode_timesteps},
            "semantic": {"architecture": "VQVAE", "config_image_size": self.config_image_size,
                         "runtime_image_size": self.image_size, "patch_size": CODEC_PATCH_SIZE,
                         "grid": list(self.grid), "codebook_size": self.semantic_codebook_size,
                         "latent_dim": int(semantic_config["latent_dim"]),
                         "codec_label_classes": self.semantic_labels,
                         "input_convention": "declared UE class IDs resized to 224 px with nearest sampling "
                                             "(official SemsegTransform convention) through the explicit trainable "
                                             "adapter that replaces the official cls_emb embedding; "
                                             "no identity or COCO fallback exists"}}
        self.to(self.device)
        self.eval()

    def register_ue_classes(self, class_by_id):
        self.ue_ids = tuple(sorted(int(value) for value in class_by_id))
        if not self.ue_ids:
            raise ValueError("semantic source has no declared UE classes")
        self.ue_id_to_row = {value: row for row, value in enumerate(self.ue_ids)}
        self.ue_embedding = nn.Embedding(len(self.ue_ids), 64).to(self.device)
        self.ue_decoder = nn.Conv2d(self.semantic_labels, len(self.ue_ids), 1).to(self.device)
        self.depth_input = nn.Conv2d(1, 1, 1).to(self.device)
        nn.init.ones_(self.depth_input.weight); nn.init.zeros_(self.depth_input.bias)
        self.depth_output = nn.Conv2d(1, 1, 1).to(self.device)
        nn.init.ones_(self.depth_output.weight); nn.init.zeros_(self.depth_output.bias)
        self.teacher_depth_input = copy.deepcopy(self.depth_input).requires_grad_(False)
        self.teacher_ue_embedding = copy.deepcopy(self.ue_embedding).requires_grad_(False)
        self.metadata["semantic"]["ue_class_ids"] = list(self.ue_ids)
        self.metadata["semantic"]["adapter"] = "independent UE embedding; pretrained semantic logits to declared UE trainable decoder; no COCO mapping"
        self.metadata["semantic"]["trainable_adapter"]={
            "input_module":"ue_embedding","input_embedding_width":self.ue_embedding.embedding_dim,
            "output_module":"ue_decoder","pretrained_logit_channels":self.ue_decoder.in_channels,
            "ue_class_count":self.ue_decoder.out_channels,"output_kernel":list(self.ue_decoder.kernel_size),
            "ue_embedding_trainable":self.ue_embedding.weight.requires_grad,
            "ue_decoder_trainable":all(parameter.requires_grad for parameter in self.ue_decoder.parameters())}

    def freeze_pretrained(self):
        self.depth.eval(); self.semantic.eval()
        self.depth.requires_grad_(False); self.semantic.requires_grad_(False)

    def _quantize(self, model, image):
        latents = model.quant_proj(model.encoder(image))
        quant, _, codes = model.quantize(latents)
        # Eval freezes the codebook/EMA. Explicit straight-through derivatives
        # retain small-adapter learning through the fixed nearest-code choice.
        quant = latents+(quant-latents).detach()
        return {"features": quant[0].permute(1, 2, 0).reshape(-1, quant.shape[1]),
                "codes": codes[0].reshape(-1), "quant": quant}

    def depth_input_data(self, metres):
        plane = torch.as_tensor(np.array(metres, copy=True), device=self.device).float()[None, None]
        valid = torch.isfinite(plane) & (plane > 0)
        resized = F.interpolate(torch.where(valid, plane, torch.zeros_like(plane)), size=(self.image_size,)*2,
                                mode="bilinear", align_corners=False)
        mask = F.interpolate(valid.float(), size=(self.image_size,)*2, mode="nearest").bool()
        ordered = (resized[mask].double()/65535.).sort()[0]
        trimmed = ordered[int(.1*len(ordered)):int(.9*len(ordered))]
        if len(trimmed) < 2:
            raise ValueError("insufficient authentic depth normalization pixels")
        mean, std = trimmed.mean().float(), (trimmed.var()+1e-6).sqrt().float()
        normalization = {"mean_m": float(mean*65535.), "std_m": float(std*65535.),
                         "mean": float(mean), "std": float(std), "unit": "m"}
        return (resized/65535.-mean)/std, mask, normalization

    def encode_depth(self, metres, teacher=False):
        standardized, mask, normalization = self.depth_input_data(metres)
        adapter = self.teacher_depth_input if teacher else self.depth_input
        result = self._quantize(self.depth, adapter(standardized))
        result.update(normalization=normalization, standardized=standardized, mask=mask)
        return result

    def semantic_rows(self, class_ids):
        ids = torch.as_tensor(np.array(class_ids, copy=True), device=self.device).long()[None, None]
        ids = F.interpolate(ids.float(), size=(self.image_size,)*2, mode="nearest").long()[0,0]
        unique = set(ids.unique().tolist())
        if not unique.issubset(self.ue_id_to_row):
            raise ValueError(f"undeclared UE semantic class IDs {sorted(unique-set(self.ue_id_to_row))}")
        rows = torch.empty_like(ids)
        for value, row in self.ue_id_to_row.items():
            rows[ids == value] = row
        return rows

    def encode_seg(self, class_ids, teacher=False):
        rows = self.semantic_rows(class_ids)
        adapter = self.teacher_ue_embedding if teacher else self.ue_embedding
        result = self._quantize(self.semantic, adapter(rows).permute(2,0,1)[None])
        result["rows"] = rows
        return result

    def code_features(self, modality, codes):
        model = self.depth if modality == "depth" else self.semantic
        quant = model.tokens_to_embedding(torch.as_tensor(codes, device=self.device).reshape(1,*self.grid))
        return quant[0].permute(1,2,0).reshape(-1,quant.shape[1])

    def decode_depth_standardized(self,codes,generator=None):
        tokens=torch.as_tensor(codes,device=self.device).reshape(1,*self.grid)
        decoded=self.depth.decode_tokens(tokens,timesteps=self.decode_timesteps,image_size=self.image_size,generator=generator)
        return self.depth_output(decoded)[0,0]

    def decode_depth(self, codes, normalization, generator=None):
        decoded=self.decode_depth_standardized(codes,generator)
        return (decoded*float(normalization["std_m"])+float(normalization["mean_m"])).clamp_min(0.)

    def semantic_logits(self, quant):
        return self.ue_decoder(self.semantic.decode_quant(quant))

    def decode_seg(self, codes):
        tokens = torch.as_tensor(codes, device=self.device).reshape(1,*self.grid)
        logits = self.ue_decoder(self.semantic.decode_tokens(tokens))[0]
        ids = torch.tensor(self.ue_ids, device=self.device)[logits.argmax(0)]
        return ids, logits

    def adaptation_losses(self, metres, class_ids):
        depth = self.encode_depth(metres)
        noise = torch.randn_like(depth["standardized"])
        timesteps = torch.randint(0, self.depth.noise_scheduler.config.num_train_timesteps,
                                  (1,), device=self.device)
        noised = self.depth.noise_scheduler.add_noise(depth["standardized"], noise, timesteps)
        from torch.utils.checkpoint import checkpoint
        predicted = checkpoint(self.depth.decoder, noised, timesteps, depth["quant"], use_reentrant=False)
        predicted = self.depth_output(predicted)
        depth_loss = F.mse_loss(predicted[depth["mask"]], depth["standardized"][depth["mask"]])
        semantic = self.encode_seg(class_ids)
        semantic_logits = checkpoint(self.semantic.decode_quant, semantic["quant"], use_reentrant=False)
        semantic_loss = F.cross_entropy(self.ue_decoder(semantic_logits), semantic["rows"][None])
        return {"depth_adaptation": depth_loss, "ue_label_adaptation": semantic_loss}


CAMERA_DESCRIPTOR_FIELDS = ("fx/width", "fy/height", "cx/width", "cy/height",
                            "width/1024", "height/1024",
                            *(f"camera_to_body.rotation[{r},{c}]" for r in range(3) for c in range(3)),
                            *(f"camera_to_body.translation[{r}]_m/200" for r in range(3)))
LIDAR_DESCRIPTOR_FIELDS = ("range_m/200", "horizontal_start_deg/180", "horizontal_end_deg/180",
                           "vertical_lower_deg/180", "vertical_upper_deg/180", "channels/128", "azimuth_measurements/2048",
                           *(f"sensor_to_body.rotation[{r},{c}]" for r in range(3) for c in range(3)),
                           *(f"sensor_to_body.translation[{r}]_m/200" for r in range(3)))


def calibrated_descriptors(frame, grid, codecs, settings, device, native_patch_size):
    """Compile resized intrinsics and authentic mounts once per source frame."""
    calibration=frame.calibration
    patch_grid=grid[0].tolist()
    # Merger does not alter the image dimensions used for camera calibration.
    rgb_shape=(int(patch_grid[1])*native_patch_size,int(patch_grid[2])*native_patch_size)
    values={};bindings={}
    for name in ("rgb","depth","seg","lidar"):
        if name == "lidar":
            mount=calibration.sensor_to_body
            numeric=[settings["Range"]/200.,settings["HorizontalFOVStart"]/180.,settings["HorizontalFOVEnd"]/180.,
                     settings["VerticalFOVLower"]/180.,settings["VerticalFOVUpper"]/180.,
                     settings["NumberOfChannels"]/128.,settings["MeasurementsPerCycle"]/2048.,
                     *mount[:3,:3].reshape(-1),*(mount[:3,3]/200.)]
            fields=LIDAR_DESCRIPTOR_FIELDS
            physical={"sensor_settings":dict(settings),"sensor_to_body":mount.tolist()}
        else:
            height,width=rgb_shape if name == "rgb" else (codecs.image_size,codecs.image_size)
            intrinsic=calibration.intrinsic.copy()
            intrinsic[0,:]*=width/calibration.width;intrinsic[1,:]*=height/calibration.height
            mount=calibration.camera_to_body
            numeric=[intrinsic[0,0]/width,intrinsic[1,1]/height,intrinsic[0,2]/width,intrinsic[1,2]/height,
                     width/1024.,height/1024.,*mount[:3,:3].reshape(-1),*(mount[:3,3]/200.)]
            fields=CAMERA_DESCRIPTOR_FIELDS
            physical={"width":width,"height":height,"intrinsic":intrinsic.tolist(),"camera_to_body":mount.tolist(),
                      "resize":"full-frame anisotropic resize; no crop", "original_width":calibration.width,"original_height":calibration.height}
        if not np.isfinite(numeric).all():
            raise ValueError(f"non-finite authentic {name} calibration")
        values[name]=torch.tensor(numeric,device=device,dtype=torch.float32)[None]
        bindings[name]={"field_order":list(fields),"normalized_values":list(map(float,numeric)),"physical":physical,
                        "sensor_local_alias":f"sensor:{MODALITY_IDS[name]}","source_sensor":frame.binding["modality"][name]["sensor_id"],
                        "observer":frame.binding["observer"],"source_reference":frame.binding["modality"][name]}
    return values,bindings


@dataclass
class SensorFeatures:
    """One prepared capture frame: native/codec tensors, dense rays, pose fields."""
    tick: int
    descriptors: dict
    rgb: torch.Tensor
    image_grid_thw: torch.Tensor
    depth: torch.Tensor
    seg: torch.Tensor
    depth_codes: torch.Tensor
    seg_codes: torch.Tensor
    depth_normalization: dict
    lidar_coordinates: torch.Tensor
    lidar_normals: torch.Tensor
    lidar_geometry_channels: torch.Tensor
    occupied_cell_indices: torch.Tensor
    coarse_cell_indices: torch.Tensor
    lidar_region_stats: torch.Tensor
    ranges: torch.Tensor
    hits: torch.Tensor
    ray_ranges: torch.Tensor
    ray_hits: torch.Tensor
    ray_directions: torch.Tensor
    raw_hit_points: np.ndarray
    binding: dict
    cell_counts: torch.Tensor
    cell_range_std: torch.Tensor
    scanned_cells: torch.Tensor


def prepare_frame(frame, qwen, image_processor, device, codecs, settings):
    """Encode authentic spatial grids, full rays and Sonata support without future leakage."""
    processed = image_processor(images=frame.rgb.copy(), return_tensors="pt")
    grid = processed["image_grid_thw"].to(device)
    with torch.no_grad():
        rgb = torch.cat(qwen.get_image_features(processed["pixel_values"].to(device), grid, return_dict=True).pooler_output, 0).detach().float()
    encoded_depth = codecs.encode_depth(frame.depth_m, teacher=True)
    encoded_seg = codecs.encode_seg(frame.segmentation_class_ids, teacher=True)
    depth_value, seg_value = encoded_depth["features"], encoded_seg["features"]
    depth_codes, seg_codes = encoded_depth["codes"], encoded_seg["codes"]
    depth_normalization = encoded_depth["normalization"]
    if BEAM_DIRECTIONS is None:
        raise ValueError("configure_sensor_layout must run before preparing frames")
    points = frame.points_sensor_ned_m[frame.hit_mask.astype(bool)].astype(np.float64)
    ranges = np.linalg.norm(points, axis=-1)
    if len(frame.hit_mask) != int(SCANNED_CELL_COUNTS.sum()):
        raise ValueError("real scan point slots do not match the fixed emitted-ray layout")
    if not len(points):
        # A genuine scan with no returns (every emitted ray missed): encoded as an explicit empty cloud.
        return _empty_scan_features(frame, grid, rgb, depth_value, seg_value, depth_codes, seg_codes,
                                    depth_normalization, codecs, settings, device, qwen)
    if not (ranges > 0).all():
        raise ValueError("hit at invalid zero range")
    hit_indices = np.flatnonzero(frame.hit_mask)
    direction_error = np.linalg.norm(points / ranges[:, None] - BEAM_DIRECTIONS[hit_indices], axis=-1).max()
    if direction_error > 1e-4:
        raise ValueError(f"actual LiDAR ray ordering diverges from fixed geometry: {direction_error}")
    cell = BEAM_CELL_INDICES[hit_indices]
    ie, ia = cell // AZIMUTH_CELLS, cell % AZIMUTH_CELLS
    counts = np.bincount(cell, minlength=AZIMUTH_CELLS * ELEVATION_CELLS)
    occupied = np.flatnonzero(counts)
    if int(counts.sum()) != len(points):
        raise RuntimeError("LiDAR angular aggregation dropped acquired hits")
    coordinates = np.stack([np.bincount(cell, weights=points[:, k], minlength=len(counts)) for k in range(3)], -1)
    coordinates = coordinates[occupied] / counts[occupied, None]
    neighbors = cKDTree(coordinates).query(coordinates, k=min(16, len(coordinates)))[1]
    centered = coordinates[neighbors] - coordinates[neighbors].mean(1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors.shape[1]
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    normals *= np.where((normals * coordinates).sum(-1) > 0, -1., 1.)[:, None]
    norm = np.linalg.norm(coordinates, axis=-1, keepdims=True)
    geometry = np.concatenate((coordinates / norm, normals, np.log1p(norm)), -1)
    coarse = (occupied // AZIMUTH_CELLS // (ELEVATION_CELLS // COARSE_EL)) * COARSE_AZ + (occupied % AZIMUTH_CELLS) // (AZIMUTH_CELLS // COARSE_AZ)
    coarse_hits = (ie // (ELEVATION_CELLS // COARSE_EL)) * COARSE_AZ + ia // (AZIMUTH_CELLS // COARSE_AZ)
    stats = np.zeros((COARSE_EL * COARSE_AZ, 7), dtype=np.float32)
    for r in range(len(stats)):
        selected = coarse_hits == r
        if selected.any():
            pr, rr = points[selected], ranges[selected]
            stats[r] = [*pr.mean(0), float(rr.mean()), float(rr.std()), float(selected.sum() / len(points)), 1.]
    mean_range = np.zeros(len(counts), dtype=np.float32)
    mean_range[occupied] = np.bincount(cell, weights=ranges, minlength=len(counts))[occupied] / counts[occupied]
    squared = np.bincount(cell, weights=ranges * ranges, minlength=len(counts))
    range_std = np.zeros(len(counts), dtype=np.float32)
    range_std[occupied] = np.sqrt(np.maximum(0., squared[occupied] / counts[occupied] - mean_range[occupied].astype(np.float64) ** 2))
    # Dense full-resolution arrays over every acquired ray slot; misses keep
    # zero range and a zero hit flag, coverage stays explicit.
    dense_ranges = np.zeros(len(BEAM_DIRECTIONS), dtype=np.float32)
    dense_ranges[hit_indices] = ranges.astype(np.float32)
    dense_hits = np.zeros(len(BEAM_DIRECTIONS), dtype=np.float32)
    dense_hits[hit_indices] = 1.
    binding = dict(frame.binding)
    binding["lidar_sensor_pose"] = sensor_pose_from_calibration(frame.calibration)
    binding["codec_normalization"] = {"depth": depth_normalization}
    descriptors,descriptor_bindings=calibrated_descriptors(frame,grid,codecs,settings,device,qwen.config.vision_config.patch_size)
    binding["calibrated_descriptors"]=descriptor_bindings

    def tensor(v, dtype=torch.float32):
        return torch.tensor(v, dtype=dtype, device=device)

    return SensorFeatures(
        tick=frame.binding["tick"], descriptors=descriptors, rgb=rgb, image_grid_thw=grid,
        depth=depth_value, seg=seg_value, depth_codes=depth_codes, seg_codes=seg_codes,
        depth_normalization=depth_normalization,
        lidar_coordinates=tensor(coordinates), lidar_normals=tensor(normals), lidar_geometry_channels=tensor(geometry),
        occupied_cell_indices=tensor(occupied, torch.long), coarse_cell_indices=tensor(coarse, torch.long),
        lidar_region_stats=tensor(stats), ranges=tensor(group_fine(mean_range)),
        hits=tensor(group_fine((counts > 0).astype(np.float32))),
        ray_ranges=tensor(dense_ranges), ray_hits=tensor(dense_hits), ray_directions=tensor(BEAM_DIRECTIONS.astype(np.float32)),
        raw_hit_points=points.astype(np.float32), binding=binding,
        cell_counts=tensor(group_fine(counts)), cell_range_std=tensor(group_fine(range_std)),
        scanned_cells=tensor(group_fine(SCANNED_CELL_COUNTS > 0), torch.bool))


def _empty_scan_features(frame, grid, rgb, depth_value, seg_value, depth_codes, seg_codes, depth_normalization,
                         codecs, settings, device, qwen):
    """SensorFeatures of a frame whose LiDAR scan acquired no hits: zero-size clouds, zero coverage statistics,
    all dense rays marked as misses; camera modalities are encoded as usual."""
    binding = dict(frame.binding)
    binding["lidar_sensor_pose"] = sensor_pose_from_calibration(frame.calibration)
    binding["codec_normalization"] = {"depth": depth_normalization}
    binding["lidar_empty_scan"] = True
    descriptors, descriptor_bindings = calibrated_descriptors(frame, grid, codecs, settings, device,
                                                              qwen.config.vision_config.patch_size)
    binding["calibrated_descriptors"] = descriptor_bindings
    zeros = np.zeros(AZIMUTH_CELLS * ELEVATION_CELLS, dtype=np.float32)

    def tensor(v, dtype=torch.float32):
        return torch.tensor(v, dtype=dtype, device=device)
    return SensorFeatures(
        tick=frame.binding["tick"], descriptors=descriptors, rgb=rgb, image_grid_thw=grid,
        depth=depth_value, seg=seg_value, depth_codes=depth_codes, seg_codes=seg_codes,
        depth_normalization=depth_normalization,
        lidar_coordinates=tensor(np.zeros((0, 3))), lidar_normals=tensor(np.zeros((0, 3))),
        lidar_geometry_channels=tensor(np.zeros((0, 7))),
        occupied_cell_indices=tensor(np.zeros(0), torch.long), coarse_cell_indices=tensor(np.zeros(0), torch.long),
        lidar_region_stats=tensor(np.zeros((COARSE_EL * COARSE_AZ, 7))), ranges=tensor(group_fine(zeros)),
        hits=tensor(group_fine(zeros)),
        ray_ranges=tensor(np.zeros(len(BEAM_DIRECTIONS))), ray_hits=tensor(np.zeros(len(BEAM_DIRECTIONS))),
        ray_directions=tensor(BEAM_DIRECTIONS.astype(np.float32)),
        raw_hit_points=np.zeros((0, 3), dtype=np.float32), binding=binding,
        cell_counts=tensor(group_fine(zeros)), cell_range_std=tensor(group_fine(zeros)),
        scanned_cells=tensor(group_fine(SCANNED_CELL_COUNTS > 0), torch.bool))


class ModalityProjectors(nn.Module):
    """Separate spatial projections into Qwen2048 and graph256."""
    def __init__(self, sonata_dim=512, hidden=2048, context=256):
        super().__init__()
        self.geometry_adapter = nn.Sequential(nn.Linear(7,32), nn.SiLU(), nn.Linear(32,3), nn.Sigmoid())
        widths = {"rgb": hidden, "depth":32, "seg":32, "lidar":sonata_dim}
        self.qwen = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden))
                                   for m,width in widths.items() if m != "rgb"})
        self.graph = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, context))
                                   for m,width in widths.items()})
        self.position = nn.Linear(4,hidden)
        self.graph_position = nn.Linear(4,context)
        self.time = nn.Linear(2,hidden)
        self.graph_time = nn.Linear(2,context)
        self.modality = nn.Embedding(4,hidden)
        self.graph_modality = nn.Embedding(4,context)
        self.descriptor_qwen=nn.ModuleDict({name:nn.Sequential(nn.Linear(width+2,128),nn.SiLU(),nn.Linear(128,hidden))
                                           for name,width in (("camera",len(CAMERA_DESCRIPTOR_FIELDS)),("lidar",len(LIDAR_DESCRIPTOR_FIELDS)))})
        self.descriptor_graph=nn.ModuleDict({name:nn.Sequential(nn.Linear(width+2,128),nn.SiLU(),nn.Linear(128,context))
                                            for name,width in (("camera",len(CAMERA_DESCRIPTOR_FIELDS)),("lidar",len(LIDAR_DESCRIPTOR_FIELDS)))})

    def descriptor(self,modality,values,tau,observer_local_id):
        alias=values.new_tensor([[math.sin(observer_local_id),math.cos(observer_local_id)]])
        numeric=torch.cat((values,alias),-1)
        family="lidar" if modality == "lidar" else "camera"
        time=values.new_tensor([tau,float(tau>0)])
        return (self.descriptor_qwen[family](numeric)+self.time(time)+self.modality.weight[MODALITY_IDS[modality]],
                self.descriptor_graph[family](numeric)+self.graph_time(time)+self.graph_modality.weight[MODALITY_IDS[modality]])

    def project(self, modality, value, positions, tau):
        time = value.new_tensor([tau, float(tau > 0)])
        base = value if modality == "rgb" else self.qwen[modality](value)
        return base+self.position(positions)+self.time(time)+self.modality.weight[MODALITY_IDS[modality]]

    def project_graph(self, modality, value, positions, tau):
        time = value.new_tensor([tau, float(tau > 0)])
        return self.graph[modality](value)+self.graph_position(positions)+self.graph_time(time)+self.graph_modality.weight[MODALITY_IDS[modality]]


class FutureEmbeddingHead(nn.Module):
    def __init__(self, dimensions, hidden=2048, codebooks=None):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden+9+64,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.inputs = nn.ModuleDict({m:nn.Linear(width,64) for m,width in dimensions.items()})
        self.outputs = nn.ModuleDict({m:nn.Linear(128,dimensions[m]) for m in ("rgb","lidar")})
        self.code_heads = nn.ModuleDict({m:nn.Linear(128,codebooks[m]) for m in ("depth","seg")})
        self.depth_normalization = nn.Linear(hidden+64+3,2)
        for module in (*self.outputs.values(),*self.code_heads.values(),self.depth_normalization):
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)

    def forward(self, context, layouts, horizon_seconds, current_features, transition_seconds):
        result = {}
        for name,positions in layouts.items():
            ident = positions.new_zeros((len(positions),4));ident[:,MODALITY_IDS[name]]=1.
            tau = positions.new_full((len(positions),1),float(horizon_seconds))
            hidden = self.trunk(torch.cat((context.float().expand(len(positions),-1),positions,ident,tau,
                                          self.inputs[name](current_features[name])),dim=-1))
            if name in ("rgb","lidar"):
                result[name] = current_features[name]+self.outputs[name](hidden)*transition_seconds
            if name in self.code_heads:
                prior = F.one_hot(current_features[name+"_codes"],self.code_heads[name].out_features).float()*8.
                result[name+"_codes"] = self.code_heads[name](hidden)+prior
        normalization = current_features["depth_normalization"]
        summary = torch.cat((context.float(),self.inputs["depth"](current_features["depth"]).mean(0),
                             normalization/200.,normalization.new_tensor([horizon_seconds])),dim=-1)
        residual = self.depth_normalization(summary)
        result["depth_normalization"] = torch.stack((normalization[0]+residual[0]*transition_seconds*200.,
                                                     normalization[1]*torch.exp(residual[1]*transition_seconds*.1)))
        return result


class DenseLidarDecoder(nn.Module):
    """Bounded categorical range and Bernoulli return with causal per-ray priors."""
    def __init__(self,sensor_range,lidar_feature_dim,ray_chunk=RAY_CHUNK):
        super().__init__()
        self.sensor_range=float(sensor_range);self.ray_chunk=int(ray_chunk)
        self.range_local_sigma_m=3.;self.range_escape_mass=.05
        self.range_law={"support_m":[0.,self.sensor_range],"categorical_cells":RANGE_CELLS,
                        "prior":"normalized (1-escape_mass)*discrete Gaussian around historical returned range + escape_mass*uniform",
                        "local_sigma_m":self.range_local_sigma_m,"initial_escape_mass":self.range_escape_mass,
                        "historical_no_return_conditional_prior":"uniform over all range cells",
                        "learned_law":"softmax(shared ray residual logits + log normalized prior)",
                        "return_law":"separate Bernoulli residual plus historical logit"}
        self.network=nn.Sequential(nn.Linear(lidar_feature_dim+9,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.range_head=nn.Linear(128,RANGE_CELLS);self.return_head=nn.Linear(128,1)
        for module in (self.range_head,self.return_head):
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        cell=torch.as_tensor(BEAM_CELL_INDICES,dtype=torch.long)
        region=(cell//AZIMUTH_CELLS//(ELEVATION_CELLS//COARSE_EL))*COARSE_AZ+(cell%AZIMUTH_CELLS)//(AZIMUTH_CELLS//COARSE_AZ)
        self.register_buffer("region_indices",region,persistent=False)
        self.register_buffer("directions",torch.tensor(BEAM_DIRECTIONS,dtype=torch.float32),persistent=False)
        # Ordered beam coordinates are exact calibrated pattern coordinates;
        # no unmeasured per-ray timestamp is synthesized.
        az=torch.atan2(self.directions[:,1],self.directions[:,0])/torch.pi
        el=torch.asin(self.directions[:,2])
        self.register_buffer("beam_coordinates",torch.stack((az,el),dim=-1),persistent=False)

    def range_cell_centres(self,device):
        return (torch.arange(RANGE_CELLS,device=device,dtype=torch.float32)+.5)*self.sensor_range/RANGE_CELLS

    def chunk_law(self,features,prior_ranges,prior_probability,start,end,transition_seconds):
        p=prior_probability[start:end]
        direction=self.directions[start:end]
        elapsed=direction.new_full((end-start,1),transition_seconds)
        attributes=torch.cat((features[self.region_indices[start:end]],self.beam_coordinates[start:end],
                              elapsed,direction,prior_ranges[start:end,None]/self.sensor_range,p[:,None],
                              torch.ones_like(p[:,None])),dim=-1)
        hidden=self.network(attributes)
        centres=self.range_cell_centres(features.device)
        local_log_probability=F.log_softmax(-.5*((centres[None]-prior_ranges[start:end,None])/self.range_local_sigma_m).square(),dim=-1)
        uniform_log_probability=local_log_probability.new_full((),-math.log(RANGE_CELLS))
        range_prior=torch.logaddexp(local_log_probability+math.log1p(-self.range_escape_mass),
                                   uniform_log_probability+math.log(self.range_escape_mass))
        # A genuine earlier miss has no observed range; its conditional law is uniform.
        range_prior=torch.where((p>.5)[:,None],range_prior,uniform_log_probability)
        epsilon=torch.finfo(p.dtype).eps
        return self.range_head(hidden)+range_prior, self.return_head(hidden).squeeze(-1)+torch.logit(p.clamp(epsilon,1-epsilon))

    def training_losses(self,features,prior_ranges,prior_probability,target_ranges,target_hits,transition_seconds):
        from torch.utils.checkpoint import checkpoint
        hit_count=target_hits.sum()
        if not bool(hit_count > 0):
            raise ValueError("dense target scan has no genuine acquired hits")
        range_losses=[];return_losses=[]
        for start in range(0,len(target_hits),self.ray_chunk):
            end=min(start+self.ray_chunk,len(target_hits))
            def chunk_loss(value,prior_range,prior_p,_start=start,_end=end):
                ranges,returns=self.chunk_law(value,prior_range,prior_p,_start,_end,transition_seconds)
                hits=target_hits[_start:_end].bool()
                targets=(target_ranges[_start:_end]/self.sensor_range*RANGE_CELLS).long().clamp(0,RANGE_CELLS-1)
                range_loss=F.cross_entropy(ranges[hits],targets[hits],reduction="sum")/hit_count
                return_loss=F.binary_cross_entropy_with_logits(returns,target_hits[_start:_end],reduction="sum")/len(target_hits)
                return range_loss,return_loss
            losses=checkpoint(chunk_loss,features,prior_ranges,prior_probability,use_reentrant=False)
            range_losses.append(losses[0]);return_losses.append(losses[1])
        return {"dense_range_nll":torch.stack(range_losses).sum(),"dense_return_bce":torch.stack(return_losses).sum()}

    def law_decode(self,features,prior_ranges,prior_probability,transition_seconds):
        logits=[];returns=[];means=[];medians=[]
        centres=self.range_cell_centres(features.device)
        for start in range(0,len(prior_ranges),self.ray_chunk):
            end=min(start+self.ray_chunk,len(prior_ranges))
            ranges,return_logits=self.chunk_law(features,prior_ranges,prior_probability,start,end,transition_seconds)
            probability=ranges.softmax(-1)
            logits.append(ranges);returns.append(return_logits)
            means.append((probability*centres).sum(-1))
            medians.append(centres[(probability.cumsum(-1)>=.5).long().argmax(-1)])
        return {"range_logit":torch.cat(logits),"return_logit":torch.cat(returns),
                "expected_range_m":torch.cat(means),"median_range_m":torch.cat(medians),
                "return_probability":torch.cat(returns).sigmoid(),"direction":self.directions,
                "range_cell_centres":centres,"range_support_m":[0.,self.sensor_range]}


def pointcloud_metrics(predicted, target, thresholds=(.1, .5, 1.)):
    """Geometry metrics in metres; empty clouds keep counts and mark distance undefined."""
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    result = {"predicted_points": int(len(predicted)), "target_points": int(len(target)), "unit": "m"}
    if not len(predicted) or not len(target):
        result["symmetric_chamfer_l1_m"] = None
        result["distance_metrics_defined"] = False
        return result
    d_pt = cKDTree(target).query(predicted,workers=4)[0]
    d_tp = cKDTree(predicted).query(target,workers=4)[0]
    result["symmetric_chamfer_l1_m"] = float(d_pt.mean() + d_tp.mean())
    result["distance_metrics_defined"] = True
    for threshold in thresholds:
        precision = float((d_pt <= threshold).mean())
        recall = float((d_tp <= threshold).mean())
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.
        result[f"precision_at_{threshold}m"] = precision
        result[f"recall_at_{threshold}m"] = recall
        result[f"fscore_at_{threshold}m"] = f_score
    return result


class NativeCSRMax(torch.autograd.Function):
    """CSR maximum with the first maximizer's input derivative.

    The retained torch_scatter SegmentMaxCSR backward segfaults in this
    runtime. Native tensor operations retain the same contiguous CSR groups
    and maximum, including the single selected derivative at exact ties.
    """
    @staticmethod
    def forward(ctx, values, pointers):
        lengths = pointers.diff()
        maxima = torch.segment_reduce(values, "max", lengths=lengths, axis=0)
        segments = torch.repeat_interleave(torch.arange(len(lengths), device=values.device), lengths)
        rows = torch.arange(len(values), device=values.device).unsqueeze(1).expand_as(values)
        candidates = torch.where(values == maxima[segments], rows, len(values))
        indices = torch.full(maxima.shape, len(values), device=values.device, dtype=torch.long)
        indices.scatter_reduce_(0, segments.unsqueeze(1).expand_as(values), candidates, reduce="amin", include_self=True)
        ctx.save_for_backward(indices)
        ctx.input_shape = values.shape
        return maxima

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        gradient = grad_output.new_zeros(ctx.input_shape)
        gradient.scatter_add_(0, indices, grad_output)
        return gradient, None


def native_segment_csr(values, pointers, *, reduce):
    if reduce == "max":
        return NativeCSRMax.apply(values, pointers)
    if reduce == "mean":
        return torch.segment_reduce(values, reduce, lengths=pointers.diff(), axis=0)
    raise ValueError(f"undeclared Sonata CSR reduction {reduce}")


def load_sonata(root: Path, device):
    source = root / "source/sonata-main"
    dependencies = root / "dependencies"
    sys.path[:0] = [str(source), str(dependencies)]
    from sonata import PointTransformerV3
    import sonata.model as sonata_model
    from types import SimpleNamespace
    # Replace this model's CSR dependency explicitly, not an exception path
    # or a global torch_scatter mutation. Pooling architecture is unchanged.
    sonata_model.torch_scatter = SimpleNamespace(segment_csr=native_segment_csr)
    checkpoint = torch.load(root / "weights/point_transformer_v3_s/sonata_small.pth", map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    # Retained source implements exact reference attention, and removed the
    # old fused-kernel constructor switch; no architecture/weights change.
    del config["enable_flash"]
    config["shuffle_orders"] = False
    model = PointTransformerV3(**config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    from spconv.pytorch.conv import SparseConvolution
    convolutions = [m for m in model.modules() if isinstance(m, SparseConvolution)]
    # spconv's eval kernel does not register its input Jacobian. Only the
    # convolution execution flag changes; normalization/dropout stay frozen.
    for convolution in convolutions:
        convolution.training = True
    config["autograd_sparse_convolutions"] = len(convolutions)
    config["csr_reduction"] = "native tensor CSR; max derivative selects first maximizer"
    return model, config


def sonata_features(frame, sonata, adapter):
    """Frozen Sonata point features pooled into the 32 support regions.

    Returns pooled Sonata features [32, 512] and actual regional presence [32].
    Calibrated ray coordinates and geometry channels supply encoder support;
    dense range and return prediction belongs to the independent ray decoder.
    """
    coordinates = frame.lidar_coordinates
    centered = coordinates - coordinates.mean(0)
    slots = adapter(frame.lidar_geometry_channels)
    feat = torch.cat((centered, slots, frame.lidar_normals), -1)
    grid = torch.floor((centered - centered.min(0).values) / .1).int()
    point = sonata({"coord": centered, "grid_coord": grid, "feat": feat,
                    "offset": torch.tensor([len(centered)], dtype=torch.long, device=centered.device)})
    feature = point.feat
    while "pooling_parent" in point:
        feature = feature[point.pooling_inverse]
        point = point.pooling_parent
    if feature.shape != (len(centered), 512):
        raise RuntimeError("actual retained Sonata dimension diverged")
    region_sum = feature.new_zeros((COARSE_EL * COARSE_AZ, 512)).index_add(0, frame.coarse_cell_indices, feature)
    region_count = torch.bincount(frame.coarse_cell_indices, minlength=COARSE_EL * COARSE_AZ)
    present = region_count > 0
    region = region_sum / region_count.clamp_min(1).unsqueeze(-1)
    return region, present
