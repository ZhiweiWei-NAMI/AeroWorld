"""Independent real-sensor representations and native Qwen hidden-state prefill.

Sonata's three colour input slots receive a learned geometry adapter. They
are latent channels, never claimed as measured RGB. Every acquired hit
contributes to a fixed angular aggregate; no colour-calibration assumption
is used. Native RGB retains Qwen's patch/merge and M-RoPE protocol.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import inspect
from pathlib import Path
import sys
import json
from contextlib import contextmanager
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

AZIMUTH_CELLS, ELEVATION_CELLS = 64, 32
COARSE_AZ, COARSE_EL = 8, 4
IMAGE_ROWS, IMAGE_COLS = 8, 16
EL_LOWER = EL_UPPER = None
SENSOR_RANGE_M = None
SCANNED_CELL_COUNTS = None
BEAM_CELL_INDICES = BEAM_DIRECTIONS = None
MODALITY_IDS = {"rgb": 0, "depth": 1, "seg": 2, "lidar": 3}


def configure_sensor_layout(path, sensor_name):
    global EL_LOWER, EL_UPPER, SENSOR_RANGE_M, SCANNED_CELL_COUNTS, BEAM_CELL_INDICES, BEAM_DIRECTIONS
    metadata = json.loads(Path(path).read_text())
    if metadata["lidar"]["sensor_name"] != sensor_name:
        raise ValueError("static LiDAR settings refer to another sensor")
    settings = metadata["lidar"]["sensor_settings"]
    SENSOR_RANGE_M = float(settings["Range"])
    pattern = metadata["lidar"]["parameter_provenance"]["parameter_basis"]["BeamPattern"]
    if "uniformly spaced" not in pattern or "periodic endpoint excluded" not in pattern:
        raise ValueError("current angular-grid code requires the declared uniform CPU ray pattern")
    if settings["HorizontalFOVStart"] != -180. or settings["HorizontalFOVEnd"] != 180.:
        raise ValueError("the pilot grid requires full azimuth")
    EL_LOWER, EL_UPPER = np.deg2rad(settings["VerticalFOVLower"]), np.deg2rad(settings["VerticalFOVUpper"])
    emitted_el = np.linspace(EL_LOWER, EL_UPPER, settings["NumberOfChannels"])
    emitted_az = np.linspace(-np.pi, np.pi, settings["MeasurementsPerCycle"], endpoint=False)
    # Exact acquired ordering is azimuth-major; misses have zero XYZ, but
    # their flat slot indices retain the emitted direction and coverage.
    az, el = np.meshgrid(emitted_az, emitted_el, indexing="ij")
    columns = np.arange(settings["MeasurementsPerCycle"])[:, None]
    channels = np.arange(settings["NumberOfChannels"])[None, :]
    ia = columns*AZIMUTH_CELLS//settings["MeasurementsPerCycle"]
    ie = np.minimum(channels*ELEVATION_CELLS//(settings["NumberOfChannels"]-1), ELEVATION_CELLS-1)
    BEAM_CELL_INDICES = (ie*AZIMUTH_CELLS+ia).ravel()
    BEAM_DIRECTIONS = np.stack((np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)), -1).reshape(-1, 3)
    SCANNED_CELL_COUNTS = np.bincount(BEAM_CELL_INDICES, minlength=AZIMUTH_CELLS*ELEVATION_CELLS)
    return settings


def image_grid(rows, cols):
    yy, xx = np.meshgrid((np.arange(rows)+.5)/rows, (np.arange(cols)+.5)/cols, indexing="ij")
    x, y = xx.ravel()*2-1, yy.ravel()*2-1
    return torch.tensor(np.stack((x, y, x*x, y*y), -1), dtype=torch.float32)


def angular_grid(rows, cols):
    if EL_LOWER is None: raise ValueError("sensor ray layout has not been loaded")
    el, az = np.meshgrid((np.arange(rows)+.5)/rows*(EL_UPPER-EL_LOWER)+EL_LOWER,
                         (np.arange(cols)+.5)/cols*2*np.pi-np.pi, indexing="ij")
    return torch.tensor(np.stack((np.cos(az), np.sin(az), np.sin(el), np.cos(el)), -1).reshape(-1, 4), dtype=torch.float32)


def ray_grid():
    grid = angular_grid(ELEVATION_CELLS, AZIMUTH_CELLS).numpy()
    return np.stack((grid[:, 0]*grid[:, 3], grid[:, 1]*grid[:, 3], grid[:, 2]), -1)


def group_fine(value):
    shape = value.shape[1:]
    return value.reshape(COARSE_EL, ELEVATION_CELLS//COARSE_EL, COARSE_AZ,
                         AZIMUTH_CELLS//COARSE_AZ, *shape).swapaxes(1, 2).reshape(COARSE_EL*COARSE_AZ, -1, *shape)


def ungroup_fine(value):
    return value.reshape(COARSE_EL, COARSE_AZ, ELEVATION_CELLS//COARSE_EL,
                         AZIMUTH_CELLS//COARSE_AZ).swapaxes(1, 2).reshape(-1)


@dataclass
class SensorFeatures:
    tick: int
    rgb: torch.Tensor
    image_grid_thw: torch.Tensor
    depth: torch.Tensor
    seg: torch.Tensor
    lidar_coordinates: torch.Tensor
    lidar_normals: torch.Tensor
    lidar_geometry_channels: torch.Tensor
    occupied_cell_indices: torch.Tensor
    coarse_cell_indices: torch.Tensor
    lidar_region_stats: torch.Tensor
    ranges: torch.Tensor
    hits: torch.Tensor
    raw_hit_points: np.ndarray
    binding: dict
    cell_counts: torch.Tensor
    cell_range_std: torch.Tensor
    scanned_cells: torch.Tensor


def prepare_frame(frame, qwen, image_processor, device):
    processed = image_processor(images=frame.rgb.copy(), return_tensors="pt")
    grid = processed["image_grid_thw"].to(device)
    with torch.no_grad():
        rgb = torch.cat(qwen.get_image_features(processed["pixel_values"].to(device), grid, return_dict=True).pooler_output, 0).detach().float()
    h, w = frame.depth_m.shape
    depth_rows, seg_rows = [], []
    for i in range(IMAGE_ROWS):
        for j in range(IMAGE_COLS):
            d = frame.depth_m[i*h//IMAGE_ROWS:(i+1)*h//IMAGE_ROWS, j*w//IMAGE_COLS:(j+1)*w//IMAGE_COLS]
            valid = np.isfinite(d) & (d > 0)
            if not valid.any():
                depth_rows.append([0., 0., 0., 0., 0.])  # isolated by valid fraction
            else:
                v = d[valid]; depth_rows.append([float(v.mean()), float(v.std()), float(v.min()), float(v.max()), float(valid.mean())])
            s = frame.segmentation_class_ids[i*h//IMAGE_ROWS:(i+1)*h//IMAGE_ROWS, j*w//IMAGE_COLS:(j+1)*w//IMAGE_COLS]
            seg_rows.append(np.bincount(s.ravel(), minlength=256)/s.size)
    points = frame.points_sensor_ned_m[frame.hit_mask.astype(bool)].astype(np.float64)
    if not len(points):
        raise ValueError("real pilot LiDAR has no acquired hits")
    ranges = np.linalg.norm(points, axis=-1)
    if len(frame.hit_mask) != int(SCANNED_CELL_COUNTS.sum()):
        raise ValueError("real scan point slots do not match the fixed emitted-ray layout")
    if not (ranges > 0).all(): raise ValueError("hit at invalid zero range")
    hit_indices = np.flatnonzero(frame.hit_mask)
    direction_error = np.linalg.norm(points/ranges[:, None]-BEAM_DIRECTIONS[hit_indices], axis=-1).max()
    if direction_error > 1e-4:
        raise ValueError(f"actual LiDAR ray ordering diverges from fixed geometry: {direction_error}")
    cell = BEAM_CELL_INDICES[hit_indices]
    ie, ia = cell//AZIMUTH_CELLS, cell%AZIMUTH_CELLS
    counts = np.bincount(cell, minlength=AZIMUTH_CELLS*ELEVATION_CELLS)
    occupied = np.flatnonzero(counts)
    if int(counts.sum()) != len(points): raise RuntimeError("LiDAR angular aggregation dropped acquired hits")
    coordinates = np.stack([np.bincount(cell, weights=points[:, k], minlength=len(counts)) for k in range(3)], -1)
    coordinates = coordinates[occupied]/counts[occupied, None]
    neighbors = cKDTree(coordinates).query(coordinates, k=min(16, len(coordinates)))[1]
    centered = coordinates[neighbors]-coordinates[neighbors].mean(1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)/neighbors.shape[1]
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    normals *= np.where((normals*coordinates).sum(-1) > 0, -1., 1.)[:, None]
    norm = np.linalg.norm(coordinates, axis=-1, keepdims=True)
    geometry = np.concatenate((coordinates/norm, normals, np.log1p(norm)), -1)
    coarse = (occupied//AZIMUTH_CELLS//(ELEVATION_CELLS//COARSE_EL))*COARSE_AZ + (occupied%AZIMUTH_CELLS)//(AZIMUTH_CELLS//COARSE_AZ)
    coarse_hits = (ie//(ELEVATION_CELLS//COARSE_EL))*COARSE_AZ + ia//(AZIMUTH_CELLS//COARSE_AZ)
    stats = np.zeros((COARSE_EL*COARSE_AZ, 7), dtype=np.float32)
    for r in range(len(stats)):
        selected = coarse_hits == r
        if selected.any():
            pr, rr = points[selected], ranges[selected]
            stats[r] = [*pr.mean(0), float(rr.mean()), float(rr.std()), float(selected.sum()/len(points)), 1.]
    mean_range = np.zeros(len(counts), dtype=np.float32)
    mean_range[occupied] = np.bincount(cell, weights=ranges, minlength=len(counts))[occupied]/counts[occupied]
    squared = np.bincount(cell, weights=ranges*ranges, minlength=len(counts))
    range_std = np.zeros(len(counts), dtype=np.float32)
    range_std[occupied] = np.sqrt(np.maximum(0., squared[occupied]/counts[occupied]-mean_range[occupied].astype(np.float64)**2))
    def tensor(v, dtype=torch.float32): return torch.tensor(v, dtype=dtype, device=device)
    return SensorFeatures(frame.binding["tick"], rgb, grid, tensor(depth_rows), tensor(np.asarray(seg_rows)),
                          tensor(coordinates), tensor(normals), tensor(geometry), tensor(occupied, torch.long),
                          tensor(coarse, torch.long), tensor(stats), tensor(group_fine(mean_range)),
                          tensor(group_fine((counts > 0).astype(np.float32))), points.astype(np.float32), dict(frame.binding),
                          tensor(group_fine(counts)), tensor(group_fine(range_std)), tensor(group_fine(SCANNED_CELL_COUNTS > 0), torch.bool))


class ModalityProjectors(nn.Module):
    def __init__(self, sonata_dim=512, hidden=2048):
        super().__init__()
        self.geometry_adapter = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 3), nn.Sigmoid())
        self.depth = nn.Linear(5, hidden)
        self.class_embeddings = nn.Embedding(256, 32)
        self.seg = nn.Linear(32, hidden)
        self.lidar = nn.Linear(sonata_dim+7+128, hidden)
        self.position = nn.Linear(4, hidden)
        self.modality = nn.Embedding(4, hidden)

    def project(self, modality, value):
        if modality == "rgb": return value
        if modality == "depth": return self.depth(value)
        if modality == "seg": return self.seg(value @ self.class_embeddings.weight)
        if modality == "lidar": return self.lidar(value)
        raise ValueError(modality)


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
    if reduce == "max": return NativeCSRMax.apply(values, pointers)
    if reduce == "mean":
        return torch.segment_reduce(values, reduce, lengths=pointers.diff(), axis=0)
    raise ValueError(f"undeclared Sonata CSR reduction {reduce}")


def load_sonata(root: Path, device):
    source = root/"source/sonata-main"
    dependencies = root/"dependencies"
    sys.path[:0] = [str(source), str(dependencies)]
    from sonata import PointTransformerV3
    import sonata.model as sonata_model
    from types import SimpleNamespace
    # Replace this model's CSR dependency explicitly, not an exception path
    # or a global torch_scatter mutation. Pooling architecture is unchanged.
    sonata_model.torch_scatter = SimpleNamespace(segment_csr=native_segment_csr)
    checkpoint = torch.load(root/"weights/point_transformer_v3_s/sonata_small.pth", map_location="cpu", weights_only=False)
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
    for convolution in convolutions: convolution.training = True
    config["autograd_sparse_convolutions"] = len(convolutions)
    config["csr_reduction"] = "native tensor CSR; max derivative selects first maximizer"
    return model, config


def sonata_features(frame, sonata, adapter):
    coordinates = frame.lidar_coordinates
    centered = coordinates-coordinates.mean(0)
    slots = adapter(frame.lidar_geometry_channels)
    feat = torch.cat((centered, slots, frame.lidar_normals), -1)
    grid = torch.floor((centered-centered.min(0).values)/.1).int()
    point = sonata({"coord": centered, "grid_coord": grid, "feat": feat,
                    "offset": torch.tensor([len(centered)], dtype=torch.long, device=centered.device)})
    feature = point.feat
    while "pooling_parent" in point:
        feature = feature[point.pooling_inverse]
        point = point.pooling_parent
    if feature.shape != (len(centered), 512): raise RuntimeError("actual retained Sonata dimension diverged")
    region_sum = feature.new_zeros((COARSE_EL*COARSE_AZ, 512)).index_add(0, frame.coarse_cell_indices, feature)
    region_count = torch.bincount(frame.coarse_cell_indices, minlength=COARSE_EL*COARSE_AZ)
    present = region_count > 0
    region = region_sum/region_count.clamp_min(1).unsqueeze(-1)
    fine_ranges = frame.ranges.reshape(COARSE_EL*COARSE_AZ, -1)
    fine_hits = frame.hits.reshape(COARSE_EL*COARSE_AZ, -1)
    # Conditional distance logit code is invertible to metres within the
    # declared 200m sensor range. No-return slots are masked storage0.
    eps = torch.finfo(fine_ranges.dtype).eps
    distance_code = torch.where(fine_hits.bool(), torch.logit((fine_ranges/SENSOR_RANGE_M).clamp(eps, 1.-eps)), torch.zeros_like(fine_ranges))
    return torch.cat((region, frame.lidar_region_stats, distance_code, fine_hits), -1), present


class FutureEmbeddingHead(nn.Module):
    def __init__(self, dimensions, hidden=2048):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden+9+64, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.inputs = nn.ModuleDict({k: nn.Linear(dim, 64) for k, dim in dimensions.items()})
        self.outputs = nn.ModuleDict({k: nn.Linear(128, dim) for k, dim in dimensions.items()})
        for output in self.outputs.values():
            nn.init.zeros_(output.weight); nn.init.zeros_(output.bias)
        self.modalities = tuple(dimensions)

    def forward(self, context, layouts, horizon_seconds, current_features):
        result = {}
        for m, positions in layouts.items():
            ident = positions.new_zeros((len(positions), 4)); ident[:, MODALITY_IDS[m]] = 1.
            # Physical horizon changes query identity, independent target set.
            horizon = positions.new_full((len(positions), 1), horizon_seconds)
            x = torch.cat((context.float().expand(len(positions), -1), positions, ident, horizon, self.inputs[m](current_features[m])), -1)
            result[m] = current_features[m]+self.outputs[m](self.trunk(x))*horizon_seconds
        return result


class CoarseLidarDecoder(nn.Module):
    def __init__(self, range_scale, sensor_range):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(512+7+128, 128), nn.SiLU(), nn.Linear(128, 128))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        self.range_scale = float(range_scale)
        self.sensor_range = float(sensor_range)

    def forward(self, predicted_feature):
        raw = self.network(predicted_feature)
        ranges = torch.sigmoid(predicted_feature[:,519:583]+raw[:,:64]) * self.sensor_range
        hit_logits = (predicted_feature[:,583:647]*2.-1.)*8.+raw[:,64:]
        return ranges, hit_logits


def native_prefix(qwen, tokenizer, text, historical, chunk_size):
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    text_ids = tokens["input_ids"]
    ids = list(text_ids); rgb_ranges = []; grids = []; rgb_features = []
    for frame in historical:
        start = len(ids)
        header = tokenizer(f"\nRGB tick={frame.tick}; observer={frame.binding['observer']}; sensor={frame.binding['modality']['rgb']['sensor_id']}\n", add_special_tokens=False)["input_ids"]
        ids.extend(header); ids.append(qwen.config.vision_start_token_id)
        image_start = len(ids); ids.extend([qwen.config.image_token_id]*len(frame.rgb)); image_end = len(ids)
        ids.append(qwen.config.vision_end_token_id)
        rgb_ranges.append({"tick": frame.tick, "modality": "rgb", "token_start": image_start, "token_end": image_end,
                           "block_start": start, "image_grid_thw": frame.image_grid_thw[0].tolist(), "binding": frame.binding["modality"]["rgb"]})
        grids.append(frame.image_grid_thw); rgb_features.append(frame.rgb)
    device = next(qwen.parameters()).device
    input_ids = torch.tensor([ids], device=device)
    mm_types = torch.zeros_like(input_ids); mm_types[input_ids == qwen.config.image_token_id] = 1
    image_grid = torch.cat(grids, 0)
    native_positions, delta = qwen.get_rope_index(input_ids, image_grid_thw=image_grid, mm_token_type_ids=mm_types)
    positions = torch.cat((torch.arange(len(ids), device=device).view(1, 1, -1), native_positions), 0)
    with torch.no_grad():
        embeddings = qwen.get_input_embeddings()(input_ids)
        mask = (input_ids == qwen.config.image_token_id).unsqueeze(-1)
        embeddings = embeddings.masked_scatter(mask, torch.cat(rgb_features, 0).to(embeddings.dtype))
        hidden, cache = prefill_embeddings(qwen, embeddings, positions, chunk_size)
    return embeddings.detach(), positions, hidden.detach(), cache, rgb_ranges, tokens["offset_mapping"], int(native_positions.max())+1


def prefill_embeddings(qwen, embeddings, positions, chunk_size, cache=None, collect_hidden=True):
    hidden = []
    for start in range(0, embeddings.shape[1], chunk_size):
        out = qwen(inputs_embeds=embeddings[:, start:start+chunk_size], position_ids=positions[:, :, start:start+chunk_size],
                   past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if collect_hidden: hidden.append(out.last_hidden_state)
    return torch.cat(hidden, 1) if collect_hidden else None, cache


def appended_positions(device, length, token_start, rope_start):
    seq = torch.arange(length, device=device)
    return torch.cat(((seq+token_start).view(1, 1, -1), (seq+rope_start).view(1, 1, -1).expand(3, 1, -1)), 0)


def cloned_cache(cache):
    # Native DynamicCache carries KV plus convolution/recurrent state. Copy
    # every mutable native tensor; branching never shares mutable states.
    return copy.deepcopy(cache)


SAVED_ATTENTION_LAYOUTS = []


@contextmanager
def cache_backward_values(cache):
    """Keep native recurrent initial values before native cache copy_ updates.

    Only saved tensors sharing native mutable-state storage are copied.
    This does not install a global dispatch mode on Sonata's inference-only
    integer serialization operations.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    pointers = {t.untyped_storage().data_ptr() for layer in cache.layers
                if isinstance(layer, LinearAttentionCacheLayerMixin)
                for t in (*layer.conv_states.values(), *layer.recurrent_states.values()) if t is not None}
    def pack(tensor):
        if tensor.untyped_storage().data_ptr() in pointers:
            return tensor.detach().clone()
        if tensor.ndim == 4 and tensor.numel() > 10_000_000:
            # Masked native SDPA repeats historic KV to query heads. Keeping
            # all six expanded KV/matrix saves on GPU exceeded actual VRAM.
            # CPU storage is exact, not compression or a numerical fallback.
            # Native efficient attention saves a padded/broadcast bias whose
            # row stride has kernel alignment requirements. CPU round trips
            # compact that view, so only dense contiguous K/V are offloaded.
            layout = {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                      "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset(),
                      "offloaded": tensor.is_contiguous()}
            if layout not in SAVED_ATTENTION_LAYOUTS:
                SAVED_ATTENTION_LAYOUTS.append(layout)
                print(json.dumps({"phase": "saved_attention_layout", **layout}), flush=True)
            if tensor.is_contiguous():
                return tensor.detach().to("cpu", copy=True), tensor.device
        return tensor
    def unpack(saved):
        if isinstance(saved, tuple): return saved[0].to(saved[1])
        return saved
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield
