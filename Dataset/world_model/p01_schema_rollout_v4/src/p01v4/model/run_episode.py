"""Actual frozen Qwen/Sonata -> trained typed laws/features -> cached rollout."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import copy
import bisect
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from p01v4.data.observations import load_observation_frame
from .causal_input import load_index, semantic_text, query_tensors, labels, dumps, MODEL_INPUT_EXCLUDED
from .distributions import SharedDistributionHead, typed_loss, predictions
from .multimodal import (prepare_frame, load_sonata, sonata_features, ModalityProjectors,
    FutureEmbeddingHead, CoarseLidarDecoder, image_grid, angular_grid, native_prefix,
    appended_positions, cloned_cache, prefill_embeddings, ungroup_fine, ray_grid,
    IMAGE_ROWS, IMAGE_COLS, COARSE_EL, COARSE_AZ)
from .multimodal import configure_sensor_layout, MODALITY_IDS, cache_backward_values, SAVED_ATTENTION_LAYOUTS


def device_tensors(values, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in values.items()}


def span_tokens(spans, offsets):
    starts, ends = [x[0] for x in offsets], [x[1] for x in offsets]
    out = {}
    for key, (lo, hi) in spans.items():
        start, end = bisect.bisect_right(ends, lo), bisect.bisect_left(starts, hi)
        if end <= start: raise ValueError(f"empty native-token binding for {key}")
        out[key] = (start, end)
    return out


def contexts(queries, hidden, token_spans):
    means = {k: hidden[0, lo:hi].float().mean(0) for k, (lo, hi) in token_spans.items()}
    entity = torch.stack([means["entity", a.entity_id] for a in queries["addresses"]])
    field = torch.stack([means["field", a.field_family] for a in queries["addresses"]])
    return entity, field


def feature_scales(values, masks):
    result = {}
    for modality, v in values.items():
        mask = masks[modality]
        if mask.ndim == 1: mask = mask[:,None].expand_as(v)
        count = mask.sum(0)
        mean = (v*mask).sum(0)/count.clamp_min(1)
        observed_std = (((v-mean).square()*mask).sum(0)/count.clamp_min(1)).sqrt()
        std = torch.where(observed_std > 0, observed_std, torch.ones_like(observed_std))
        result[modality] = mean.detach(), std.detach(), count.detach()
    return result


def normalize(values, scales):
    # Exact constant-feature scale=1 is a mathematical representation, not
    # a missing-value repair. Empty cells carry explicit separate masks.
    result = {m: (v-scales[m][0])/scales[m][1] for m, v in values.items()}
    if "lidar" in values:
        result["lidar"][:,519:583] = torch.where(values["lidar"][:,583:647].bool(), result["lidar"][:,519:583], torch.zeros_like(result["lidar"][:,519:583]))
    return result


def suffix(qwen, tokenizer, projectors, values, layouts, tick, binding, token_start, rope_start):
    blocks, bindings = [], []
    offset = token_start
    for modality, value in values.items():
        if modality == "rgb": continue  # Native RGB is already in the prefix.
        sensor = binding["modality"][modality]["sensor_id"]
        text = f"\nOBSERVED {modality} tick={tick}; sensor={sensor}; fixed_region_grid;\n"
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(value.device)
        header = qwen.get_input_embeddings()(ids).detach()
        region = projectors.project(modality, value)+projectors.position(layouts[modality])+projectors.modality.weight[MODALITY_IDS[modality]]
        blocks.extend((header, region.to(qwen.dtype).unsqueeze(0)))
        start = offset+header.shape[1]
        bindings.append({"modality": modality, "tick": tick, "sensor": sensor,
                         "token_start": start, "token_end": start+len(value), "positions": layouts[modality].tolist(),
                         "source": binding["modality"][modality]})
        offset = start+len(value)
    embeddings = torch.cat(blocks, 1)
    positions = appended_positions(embeddings.device, embeddings.shape[1], token_start, rope_start)
    return embeddings, positions, bindings


def loss_values(output, query, target, future, target_features, masks, decoder, scales, target_frame):
    losses = typed_loss(output, query, target)
    feature_losses = {}
    for m in future:
        feature_losses[m] = F.mse_loss(future[m][masks[m]], target_features[m][masks[m]])
    lidar_components = {}
    for name, lo, hi in (("sonata",0,512),("geometry",512,519),("conditional_distance_logit",519,583),("return_marker",583,647)):
        mask = masks["lidar"][:,lo:hi]
        lidar_components[name] = F.mse_loss(future["lidar"][:,lo:hi][mask], target_features["lidar"][:,lo:hi][mask])
    feature_losses["lidar"] = sum(lidar_components.values())/len(lidar_components)
    decoded_input = future["lidar"]*scales["lidar"][1]+scales["lidar"][0]
    ranges, logits = decoder(decoded_input)
    hits = target_frame.hits.bool()
    range_loss = F.smooth_l1_loss(ranges[hits]/decoder.range_scale, target_frame.ranges[hits]/decoder.range_scale)
    hit_loss = F.binary_cross_entropy_with_logits(logits[target_frame.scanned_cells], target_frame.hits[target_frame.scanned_cells])
    losses.update({"future_"+m: v for m, v in feature_losses.items()})
    losses.update({"future_lidar_"+m:v for m,v in lidar_components.items()})
    losses.update({"coarse_range_huber": range_loss, "captured_return_bce": hit_loss})
    losses["total"] = losses["typed"]+sum(feature_losses.values())+range_loss+hit_loss
    return losses


def floats(values): return {k: float(v.detach()) for k, v in values.items()}


def feature_metrics(predicted, target, target_masks, previous, previous_masks):
    result = {}
    for m in predicted:
        mask, prior_mask = target_masks[m], previous_masks[m]
        if mask.ndim == 1: mask = mask[:,None].expand_as(predicted[m])
        if prior_mask.ndim == 1: prior_mask = prior_mask[:,None].expand_as(predicted[m])
        overlap = mask & prior_mask
        result[m] = {"model_mse_on_target_support":float((predicted[m][mask]-target[m][mask]).square().mean()),
                     "model_mse_on_common_support":float((predicted[m][overlap]-target[m][overlap]).square().mean()),
                     "persistence_mse_on_common_support":float((previous[m][overlap]-target[m][overlap]).square().mean()),
                     "target_scalar_count":int(mask.sum()),"common_scalar_count":int(overlap.sum()),
                     "unit":"squared frozen TRAIN-normalized feature units"}
    return result


def typed_metrics(index, rows):
    """Same-address physical errors and causal persistence/constant velocity."""
    fields = {(f.family, f.component): f for f in index.fields}
    groups = {}
    for row in rows:
        e, name, component, tick = row["entity"], row["field"], row["component"], row["target_tick"]
        f = fields[name, component]
        state = index.states.get((tick, e))
        if state is None or name not in state: continue
        from .causal_input import cell
        kind, truth = cell(state[name], component)
        if kind != "present": continue
        past = [(t, cell(s[name], component)) for (t, actor), s in index.states.items()
                if t <= index.cutoff and actor == e and name in s]
        past.sort()
        past = [(t, value) for t, (k, value) in past if k == "present"]
        if name not in groups: groups[name] = {"model": [], "persistence": [], "constant_velocity": [], "selected_present": [], "unit": f.unit, "categorical": f.logical_dtype != "real"}
        group = groups[name]; value = row["conditional_point"]
        group["selected_present"].append(row["predicted_cell_kind"] == "present" and row["record_probability"] >= .5)
        def error(predicted):
            if f.logical_dtype != "real": return float(predicted == truth)
            if f.circular: return abs((float(predicted)-float(truth)+180.)%360.-180.)
            return abs(float(predicted)-float(truth))
        group["model"].append(error(value))
        if past: group["persistence"].append(error(past[-1][1]))
        if name == "pose.position_enu_m":
            vhist = [(t, cell(s["pose.velocity_enu_mps"], component)) for (t, actor), s in index.states.items()
                     if t <= index.cutoff and actor == e and "pose.velocity_enu_mps" in s]
            vhist.sort(); vhist = [(t, v) for t, (k, v) in vhist if k == "present"]
            if past and vhist:
                group["constant_velocity"].append(error(float(past[-1][1])+float(vhist[-1][1])*(tick-past[-1][0])*.1))
    return {name: {"unit": g["unit"], "metric": "accuracy" if g["categorical"] else ("circular_absolute_error" if (name, None) in fields and fields[name, None].circular else "mean_absolute_error"),
                   "present_component_count": len(g["model"]), "model": float(np.mean(g["model"])),
                   "selected_present_count": sum(g["selected_present"]), "selected_present_coverage": float(np.mean(g["selected_present"])),
                   "selected_present_model_metric": float(np.mean([v for v,selected in zip(g["model"],g["selected_present"]) if selected])) if any(g["selected_present"]) else None,
                   "persistence": float(np.mean(g["persistence"])) if g["persistence"] else None,
                   "persistence_count": len(g["persistence"]),
                   "constant_velocity": float(np.mean(g["constant_velocity"])) if g["constant_velocity"] else None,
                   "constant_velocity_count": len(g["constant_velocity"])} for name, g in groups.items()}


def lidar_metrics(ranges, logits, target_frame, reference_frame):
    predicted_range = ungroup_fine(ranges.detach().cpu().numpy())
    probability = ungroup_fine(logits.detach().sigmoid().cpu().numpy())
    target_range = ungroup_fine(target_frame.ranges.cpu().numpy())
    truth = ungroup_fine(target_frame.hits.cpu().numpy()).astype(bool)
    scanned = ungroup_fine(target_frame.scanned_cells.cpu().numpy()).astype(bool)
    previous_range = ungroup_fine(reference_frame.ranges.cpu().numpy())
    previous_hits = ungroup_fine(reference_frame.hits.cpu().numpy()).astype(bool)
    selected = (probability >= .5) & scanned
    rays = ray_grid()
    predicted_cloud = rays[selected]*predicted_range[selected, None]
    actual_coarse_cloud = rays[truth]*target_range[truth, None]
    if len(predicted_cloud):
        d1 = cKDTree(target_frame.raw_hit_points).query(predicted_cloud)[0]
        d2 = cKDTree(predicted_cloud).query(target_frame.raw_hit_points)[0]
        chamfer = float(d1.mean()+d2.mean())
    else:
        chamfer = None  # An empty reconstruction has no finite Chamfer score.
    overlap = truth & previous_hits
    return {"range_mae_on_target_hit_cells_m": float(np.abs(predicted_range[truth]-target_range[truth]).mean()),
            "captured_cell_occupancy_accuracy": float((selected[scanned] == truth[scanned]).mean()),
            "captured_cell_brier": float(((probability[scanned]-truth[scanned])**2).mean()),
            "captured_cell_bce": float(F.binary_cross_entropy_with_logits(logits[target_frame.scanned_cells], target_frame.hits[target_frame.scanned_cells])),
            "scanned_cell_count": int(scanned.sum()),
            "predicted_occupied_cells": int(selected.sum()), "target_occupied_cells": int(truth.sum()),
            "symmetric_raw_cloud_chamfer_sum_m": chamfer,
            "persistence_range_mae_on_common_hit_cells_m": float(np.abs(previous_range[overlap]-target_range[overlap]).mean()),
            "model_range_mae_on_common_hit_cells_m": float(np.abs(predicted_range[overlap]-target_range[overlap]).mean()),
            "common_hit_cell_count": int(overlap.sum()), "newly_hit_cell_count": int((truth & ~previous_hits).sum()),
            "persistence_occupancy_accuracy": float((previous_hits[scanned] == truth[scanned]).mean()),
            "persistence_brier": float(((previous_hits[scanned].astype(np.float32)-truth[scanned].astype(np.float32))**2).mean()),
            "target_cell_hit_count": ungroup_fine(target_frame.cell_counts.cpu().numpy()).tolist(),
            "target_cell_range_std_m": ungroup_fine(target_frame.cell_range_std.cpu().numpy()).tolist(),
            "range_evaluation": "64x32 angular captured-return cells within the loaded uniform ray FOV, conditional mean range; not emitted-beam probability",
            "chamfer_evaluation": "coarse prediction vs every actual target acquired hit, two directional mean distances added"}, predicted_cloud, actual_coarse_cloud


def plot_clouds(path, clouds, target_tick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    predicted, previous, target = clouds
    all_points = np.concatenate([p for p in clouds if len(p)])
    lo, hi = all_points.min(0), all_points.max(0)
    center = (lo+hi)/2.; extent = float((hi-lo).max())/2.
    color_max = float(np.linalg.norm(all_points, axis=1).max())
    fig = plt.figure(figsize=(15, 4.5))
    for i, (points, title) in enumerate(((predicted, "Predicted coarse returns"), (previous, "Historical persistence"), (target, "Actual future angular means")), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        if len(points): ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, c=np.linalg.norm(points, axis=1), cmap="viridis", vmin=0., vmax=color_max)
        ax.set_xlim(center[0]-extent,center[0]+extent); ax.set_ylim(center[1]-extent,center[1]+extent); ax.set_zlim(center[2]-extent,center[2]+extent)
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=25.,azim=-60.)
        ax.set(xlabel="Sensor forward (m)", ylabel="Sensor right (m)", zlabel="Sensor down (m)", title=title)
    fig.suptitle(f"tick {target_tick}; fixed 64x32 scan-FOV grid; conditional range/occupancy decoder")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def run(args):
    started = time.perf_counter()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    device = torch.device("cuda:0")
    target_ticks = (args.cutoff+args.step, args.cutoff+2*args.step)
    index = load_index(args.canonical, args.communication, args.episode, args.cutoff, target_ticks, args.graph_base)
    text, character_spans = semantic_text(index)
    from transformers import AutoTokenizer, AutoConfig, Qwen3_5Model, Qwen2VLImageProcessor
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    original_lines = text.splitlines()
    wire_symbols = json.loads(original_lines[1].split("=", 1)[1])
    original_lines[1] = "STRINGS="+dumps([index.identity_aliases.get(v, v) for v in wire_symbols])
    unaliased_tokens = len(tokenizer("\n".join(original_lines)+"\n", add_special_tokens=False)["input_ids"])
    print(dumps({"phase": "semantic_input", "characters": len(text), "tokens": text_tokens,
                 "history_ticks": sorted({r["tick"] for r in index.semantic_records if isinstance(r["tick"], int)}),
                 "records": dict(Counter(r["kind"] for r in index.semantic_records)), "causal_actor_count": len(index.entities)}), flush=True)
    historical_ticks = tuple(range(5, args.cutoff+1, 5))
    if not historical_ticks: raise ValueError("actual multimodal pilot requires at least capture tick5")
    frames = {tick: load_observation_frame(args.observations, args.episode, tick, args.observer,
                                          branch="historical" if tick <= args.cutoff else "target", cutoff=args.cutoff)
              for tick in (*historical_ticks, *target_ticks)}
    sensor_settings = configure_sensor_layout(args.lidar_settings, frames[args.cutoff].binding["modality"]["lidar"]["sensor_id"])
    # Only historical geometry reaches Qwen. Future calibration remains on
    # the target branch and is not used by sensor-frame forecasting.
    static_text = ""
    for tick in historical_ticks:
        frame = frames[tick]
        sensor_template = {"episode": args.episode, "tick": tick, "observer": args.observer,
                           "calibration": frame.calibration.as_dict(), "source_pose_agl_m": frame.binding["source_pose_agl_m"],
                           "capture_position_world_m": frame.binding["capture_position_world_m"],
                           "ground_reference": frame.binding["ground_reference"],
                           "depth_semantics": frame.binding["depth"], "sensor_settings": sensor_settings,
                           "geometry_policy": "source AGL and capture world altitude differ; no calibrated instance visibility; independent sensor-frame forecasts"}
        static_text += "SENSOR_TEMPLATE="+dumps(sensor_template)+"\n"
    aliases = {source: alias for alias, source in index.identity_aliases.items()}
    static_text += "GRAPH_IDENTITIES="+dumps([[aliases.get(identity, identity), node["ontology_classes"]] for identity, node in sorted(index.graph_nodes.items())])+"\n"
    static_text += "PREDICATE_VOCABULARY="+dumps(index.graph_predicates)+"\n"
    character_spans = {k:(lo+len(static_text),hi+len(static_text)) for k,(lo,hi) in character_spans.items()}
    text = static_text+text
    final_text_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    # Reserve explicit future state/latent capacity before model allocation.
    # Exact final length is checked again before append; no truncation.
    append_reserve = 40000  # Explicit predicted-append token capacity.
    token_ceiling = final_text_tokens+len(historical_ticks)*600+append_reserve
    if token_ceiling > config.text_config.max_position_embeddings:
        raise ValueError(f"full input+declared append budget {token_ceiling} exceeds native context {config.text_config.max_position_embeddings}")
    kv_bytes = token_ceiling*6*2*2*256*2
    expanded_kv_saved_bytes = token_ceiling*6*2*8*256*2
    print(dumps({"phase": "resource_budget", "final_structured_text_tokens": final_text_tokens,
                 "max_tokens_including_append_reserve": token_ceiling, "native_context": config.text_config.max_position_embeddings,
                 "kv_theoretical_bytes": kv_bytes, "frozen_weight_bytes": 4548144832,
                 "cpu_expanded_kv_saved_estimate_bytes": expanded_kv_saved_bytes,
                 "training_memory_ceiling_estimate_bytes": 4548144832+2*kv_bytes+token_ceiling*4096+6*1024**3,
                 "assumptions": "BF16;6full attention layers,2KVheads,256head dim;2cachecopies;chunk1024;6GiBactivation/workspace reserve;contiguous expanded8head savedK/V on CPU, padded bias stays GPU;no LM logits"}), flush=True)
    qwen = Qwen3_5Model.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    image_processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
                                          min_pixels=256*448, max_pixels=256*448)
    prepared = {tick: prepare_frame(frame, qwen, image_processor, device) for tick, frame in frames.items()}
    sonata, sonata_config = load_sonata(args.sonata, device)
    projectors = ModalityProjectors().to(device)
    teacher_adapter = copy.deepcopy(projectors.geometry_adapter).eval().requires_grad_(False)
    raw_features, masks = {}, {}
    with torch.no_grad():
        for tick, frame in prepared.items():
            lidar, occupied = sonata_features(frame, sonata, teacher_adapter)
            raw_features[tick] = {"rgb": frame.rgb, "depth": frame.depth, "seg": frame.seg, "lidar": lidar}
            masks[tick] = {"rgb": torch.ones(len(frame.rgb), dtype=torch.bool, device=device),
                           "depth": frame.depth[:, 4] > 0, "seg": torch.ones(len(frame.seg), dtype=torch.bool, device=device),
                           "lidar": torch.cat((occupied[:,None].expand(-1,519), frame.hits.reshape(32,64).bool(), frame.scanned_cells.reshape(32,64)), -1)}
    # Fit once on complete declared historical TRAIN observations only.
    fitted = {m: torch.cat([raw_features[t][m] for t in historical_ticks]) for m in raw_features[args.cutoff]}
    fit_masks = {m: torch.cat([masks[t][m] for t in historical_ticks]) for m in masks[args.cutoff]}
    scales = feature_scales(fitted, fit_masks)
    target_features = {t: normalize(raw_features[t], scales) for t in target_ticks}
    current_features = normalize(raw_features[args.cutoff], scales)
    current_features["lidar"] = current_features["lidar"].clone()
    # Unknown conditional ranges use the frozen TRAIN conditional-distance
    # prior (normalized mean0), with the separate acquired-return plane kept.
    current_features["lidar"][:,519:583] = torch.where(prepared[args.cutoff].hits.reshape(32,64).bool(), current_features["lidar"][:,519:583], torch.zeros_like(current_features["lidar"][:,519:583]))
    prefix_embeddings, prefix_positions, prefix_hidden, prefix_cache, rgb_bindings, offsets, next_rope = native_prefix(
        qwen, tokenizer, text, [prepared[t] for t in historical_ticks], args.chunk)
    if prefix_embeddings.shape[1] > qwen.config.text_config.max_position_embeddings:
        raise ValueError("complete declared input exceeds native model context; no truncation is performed")
    token_spans = span_tokens(character_spans, offsets)
    query = device_tensors(query_tensors(index, target_ticks[0]), device)
    target = device_tensors(labels(index, query), device)
    entity_context, field_context = contexts(query, prefix_hidden, token_spans)
    del prefix_hidden
    current = prepared[args.cutoff]
    rgb_shape = current.image_grid_thw[0].cpu().tolist()
    layouts = {"rgb": image_grid(rgb_shape[1]//2, rgb_shape[2]//2).to(device),
               "depth": image_grid(IMAGE_ROWS, IMAGE_COLS).to(device), "seg": image_grid(IMAGE_ROWS, IMAGE_COLS).to(device),
               "lidar": angular_grid(COARSE_EL, COARSE_AZ).to(device)}
    typed_head = SharedDistributionHead().to(device)
    future_head = FutureEmbeddingHead({m: v.shape[-1] for m, v in raw_features[args.cutoff].items()}).to(device)
    decoder = CoarseLidarDecoder(float(current.ranges[current.hits.bool()].mean()), sensor_settings["Range"]).to(device)
    trainables = nn.ModuleDict({"projectors": projectors, "typed": typed_head, "future": future_head, "decoder": decoder})
    optimizer = torch.optim.AdamW(trainables.parameters(), lr=args.lr)
    history = []; gradient_norms = {}; gradient_presence = {}; first_rows = None
    if args.resume_head is not None:
        saved = torch.load(args.resume_head, map_location=device, weights_only=False)
        if saved["seed"] != args.seed or saved["steps"] != args.steps:
            raise ValueError("resume checkpoint has different declared training seed/steps")
        trainables.load_state_dict(saved["trainable_state"], strict=True)
        history = saved["loss_history"]
        first_rows = saved["initial_predictions"]

    def forward_current():
        values, binding_rows = [], []
        # All declared historical modalities are consumed, never just the
        # current frame. Each suffix branch starts from an isolated cache.
        cache = cloned_cache(prefix_cache); token_start = prefix_embeddings.shape[1]; rope_start = next_rope
        suffix_embs, suffix_positions = [], []
        last_hidden = None
        for tick in historical_ticks:
            lidar, _ = sonata_features(prepared[tick], sonata, projectors.geometry_adapter)
            raw = {**raw_features[tick], "lidar": lidar}
            value = normalize(raw, scales)
            emb, pos, binding = suffix(qwen, tokenizer, projectors, value, layouts, tick, prepared[tick].binding, token_start, rope_start)
            with cache_backward_values(cache):
                out = qwen(inputs_embeds=emb, position_ids=pos, past_key_values=cache, use_cache=True)
            cache, last_hidden = out.past_key_values, out.last_hidden_state
            suffix_embs.append(emb); suffix_positions.append(pos); binding_rows.extend(binding)
            token_start += emb.shape[1]; rope_start += emb.shape[1]
        context = last_hidden[0, -1].float()
        output = typed_head(query, entity_context, field_context, context)
        future = future_head(context, layouts, args.step*.1, current_features)
        return output, future, context, cache, torch.cat(suffix_embs, 1), torch.cat(suffix_positions, 2), binding_rows

    print(dumps({"phase": "training_launch", "prefix_tokens": int(prefix_embeddings.shape[1]),
                 "queries": len(query["addresses"]), "trainable_parameters": sum(p.numel() for p in trainables.parameters()),
                 "steps": args.steps, "native_cache": type(prefix_cache).__name__}), flush=True)
    for step in range(1 if args.resume_head is not None else args.steps):
        optimizer.zero_grad(set_to_none=True)
        # Native cache-specific saved-tensor hooks in forward_current protect
        # only mutable initial recurrent/conv values needed by backward.
        with torch.enable_grad():
            output, future, _, _, _, _, _ = forward_current()
            losses = loss_values(output, query, target, future, target_features[target_ticks[0]], masks[target_ticks[0]], decoder, scales, prepared[target_ticks[0]])
            if not bool(torch.isfinite(losses["total"])): raise FloatingPointError("actual loss is non-finite")
            if step == 0 and args.resume_head is None:
                first_rows, _ = predictions(output, query)
            losses["total"].backward()
            for name, module in trainables.items():
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                if not all(bool(torch.isfinite(g).all()) for g in grads): raise FloatingPointError(f"non-finite {name} gradient")
                gradient_norms[name] = float(torch.sqrt(sum(g.square().sum() for g in grads))) if grads else None
                gradient_presence[name] = {"parameter_tensors_with_grad": len(grads), "parameter_tensors": len(list(module.parameters()))}
            for name in ("geometry_adapter", "depth", "class_embeddings", "seg", "lidar", "position", "modality"):
                grads = [p.grad for p in getattr(projectors, name).parameters() if p.grad is not None]
                gradient_norms["projectors."+name] = float(torch.sqrt(sum(g.square().sum() for g in grads))) if grads else None
                gradient_presence["projectors."+name] = {"parameter_tensors_with_grad": len(grads), "parameter_tensors": len(list(getattr(projectors, name).parameters()))}
            global_gradient = nn.utils.clip_grad_norm_(trainables.parameters(), args.grad_clip)
            if args.resume_head is None:
                history.append({"step": step, **floats(losses), "gradient_global_norm_before_clip":float(global_gradient), "gradient_clip_norm":args.grad_clip})
                optimizer.step()
        print(dumps({"phase": "gradient_replay" if args.resume_head is not None else "optimizer", **floats(losses), "step": step, "gradient_global_norm_before_clip":float(global_gradient), "gradient_clip_norm":args.grad_clip}), flush=True)
    with torch.no_grad():
        output, future, context, full_cache, suffix_embeddings, suffix_positions, suffix_bindings = forward_current()
        final_losses = loss_values(output, query, target, future, target_features[target_ticks[0]], masks[target_ticks[0]], decoder, scales, prepared[target_ticks[0]])
        first_predictions, rollout = predictions(output, query)
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable_state": trainables.state_dict(), "seed": args.seed, "steps": args.steps,
                    "loss_history": history, "initial_predictions": first_rows}, args.output/"head.pt")
        (args.output/"predictions.json").write_text(json.dumps({"initial": first_rows, "first_step": first_predictions}, indent=2, allow_nan=False)+"\n")
        # Append explicit predicted cells/recording probabilities, not a
        # fabricated observed frame. Future truth remains only in labels.
        predicted_entity_ids = sorted(index.entities)
        predicted_field_ids = sorted({f.family for f in index.fields})
        parameter_names = {"normal":["location","scale"],"von_mises":["location_deg","concentration"],
                           "boundary_inflated_gamma":["boundary_mass","concentration","rate"],
                           "endpoint_inflated_beta":["endpoint_masses","alpha","beta"]}
        compact_rows = [[predicted_entity_ids.index(r["entity"]), predicted_field_ids.index(r["field"]), r["component"],
                         r["conditional_point"], r["record_probability"],
                         [r["cell_kind_probabilities"][k] for k,legal in zip(r["cell_kind_probabilities"],query["allowed_status"][i].tolist()) if legal],
                         [r["parameters"][name] for name in parameter_names[r["law"]]] if r["law"] != "categorical" else r["probabilities"]]
                        for i,r in enumerate(first_predictions)]
        predicted_header = "PREDICTED_STATE entity_ids="+dumps(predicted_entity_ids)+"; field_ids="+dumps(predicted_field_ids)+"; columns=[entity_index,field_index,component,conditional_point,record_probability,probabilities_for_legal_cell_kinds_in_schema_order,law_parameters]; law_parameter_names="+dumps(parameter_names)+"; bounds/classes/legal_kinds come from FIELD declarations; categorical law_parameters=class probabilities\n"
        predicted_text = predicted_header+"\n".join(dumps(r) for r in compact_rows)+"\n"
        state_tokens = tokenizer(predicted_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        append_start = prefix_embeddings.shape[1]+suffix_embeddings.shape[1]
        rgb_header = tokenizer(f"PREDICTED RGB tick={target_ticks[0]}; fixed native grid\n", add_special_tokens=False)["input_ids"]
        image_ids = [*rgb_header, qwen.config.vision_start_token_id, *([qwen.config.image_token_id]*len(future["rgb"])), qwen.config.vision_end_token_id]
        native_append_ids = torch.cat((state_tokens, torch.tensor([image_ids], device=device)), 1)
        native_append_embeddings = qwen.get_input_embeddings()(native_append_ids)
        predicted_rgb = (future["rgb"]*scales["rgb"][1]+scales["rgb"][0]).to(qwen.dtype)
        image_mask = (native_append_ids == qwen.config.image_token_id).unsqueeze(-1)
        native_append_embeddings = native_append_embeddings.masked_scatter(image_mask, predicted_rgb)
        predicted_bindings = [{"modality":"rgb","tick":target_ticks[0],"role":"predicted_append",
                               "token_start":append_start+state_tokens.shape[1]+len(rgb_header)+1,
                               "token_end":append_start+state_tokens.shape[1]+len(rgb_header)+1+len(predicted_rgb),
                               "sensor":current.binding["modality"]["rgb"]["sensor_id"],"positions":layouts["rgb"].cpu().tolist()}]
        mm_types = torch.zeros_like(native_append_ids); mm_types[native_append_ids == qwen.config.image_token_id] = 1
        native_positions, _ = qwen.get_rope_index(native_append_ids, image_grid_thw=current.image_grid_thw, mm_token_type_ids=mm_types)
        native_positions += next_rope+suffix_embeddings.shape[1]
        native_append_positions = torch.cat((torch.arange(native_append_ids.shape[1], device=device).view(1, 1, -1)+append_start, native_positions), 0)
        blocks = [native_append_embeddings]; pos_blocks = [native_append_positions]
        token_offset = append_start+native_append_ids.shape[1]; rope_offset = int(native_positions.max())+1
        for m in ("depth", "seg", "lidar"):
            header_ids = tokenizer(f"PREDICTED {m} tick={target_ticks[0]}; sensor={current.binding['modality'][m]['sensor_id']}; fixed_region_grid\n",
                                   add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            block = torch.cat((qwen.get_input_embeddings()(header_ids),
                               (projectors.project(m, future[m])+projectors.position(layouts[m])+projectors.modality.weight[MODALITY_IDS[m]]).to(qwen.dtype).unsqueeze(0)), 1)
            blocks.append(block); pos_blocks.append(appended_positions(device, block.shape[1], token_offset, rope_offset))
            predicted_bindings.append({"modality":m,"tick":target_ticks[0],"role":"predicted_append",
                                       "token_start":token_offset+header_ids.shape[1],"token_end":token_offset+block.shape[1],
                                       "sensor":current.binding["modality"][m]["sensor_id"],"positions":layouts[m].cpu().tolist()})
            token_offset += block.shape[1]; rope_offset += block.shape[1]
        append_embeddings, append_positions = torch.cat(blocks, 1), torch.cat(pos_blocks, 2)
        if append_embeddings.shape[1] > append_reserve:
            raise ValueError("actual predicted append exceeds reserved token capacity; no truncation")
        if append_start+append_embeddings.shape[1] > config.text_config.max_position_embeddings:
            raise ValueError("actual predicted append exceeds native context; no truncation")
        append_started = time.perf_counter()
        appended_hidden, rollout_cache = prefill_embeddings(qwen, append_embeddings, append_positions, args.chunk, cache=cloned_cache(full_cache))
        torch.cuda.synchronize(); cache_seconds = time.perf_counter()-append_started
        query2 = device_tensors(query_tensors(index, target_ticks[1], rollout_values=rollout, cutoff=target_ticks[0]), device)
        target2 = device_tensors(labels(index, query2), device)
        # Address identity stays causal. Updated entity context reads that
        # entity's own predicted-state rows, not target-frame rows.
        state_offsets = tokenizer(predicted_text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        cursor = len(predicted_header); predicted_spans = {}
        for row, compact in zip(first_predictions, compact_rows):
            line = dumps(compact); predicted_spans["entity", row["entity"]] = (cursor, cursor+len(line)); cursor += len(line)+1
        predicted_token_spans = span_tokens(predicted_spans, state_offsets)
        entity2 = torch.stack([appended_hidden[0, slice(*predicted_token_spans["entity", a.entity_id])].float().mean(0) for a in query2["addresses"]])
        context2 = appended_hidden[0, -1].float()
        output2 = typed_head(query2, entity2, field_context, context2)
        future2 = future_head(context2, layouts, args.step*.1, future)
        second_predictions, _ = predictions(output2, query2)
        second_losses = loss_values(output2, query2, target2, future2, target_features[target_ticks[1]], masks[target_ticks[1]], decoder, scales, prepared[target_ticks[1]])
        direct_query = device_tensors(query_tensors(index, target_ticks[1]), device)
        direct_output = typed_head(direct_query, entity_context, field_context, context)
        direct_future = future_head(context, layouts, args.step*.2, current_features)
        direct_losses = loss_values(direct_output, direct_query, target2, direct_future, target_features[target_ticks[1]], masks[target_ticks[1]], decoder, scales, prepared[target_ticks[1]])
        direct_predictions, _ = predictions(direct_output, direct_query)
        rollout_cache_metadata = {"native_cache": type(rollout_cache).__name__, "cache_layers": [type(layer).__name__ for layer in rollout_cache.layers],
                                  "cache_length": rollout_cache.get_seq_length()}
        del prefix_cache, full_cache, rollout_cache
        # Re-execute identical complete input with one independent native
        # cache; the full history and projections are unchanged.
        recompute_started = time.perf_counter()
        _, recomputed_cache = prefill_embeddings(qwen, prefix_embeddings, prefix_positions, args.chunk, collect_hidden=False)
        frame_endpoints = [max(b["token_end"] for b in suffix_bindings if b["tick"] == tick)-prefix_embeddings.shape[1] for tick in historical_ticks]
        begin = 0
        for end in frame_endpoints:
            repeated = qwen(inputs_embeds=suffix_embeddings[:,begin:end], position_ids=suffix_positions[:,:,begin:end],
                            past_key_values=recomputed_cache, use_cache=True)
            recomputed_cache = repeated.past_key_values; begin = end
        recomputed_hidden, recomputed_cache = prefill_embeddings(qwen, append_embeddings, append_positions, args.chunk, cache=recomputed_cache)
        torch.cuda.synchronize(); recompute_seconds = time.perf_counter()-recompute_started
        cache_difference = float((appended_hidden-recomputed_hidden[:, -append_embeddings.shape[1]:]).abs().max())
        cache_rms = float((appended_hidden.float()-recomputed_hidden.float()).square().mean().sqrt())
        recomputed_entity2 = torch.stack([recomputed_hidden[0,slice(*predicted_token_spans["entity",a.entity_id])].float().mean(0) for a in query2["addresses"]])
        recomputed_output2 = typed_head(query2,recomputed_entity2,field_context,recomputed_hidden[0,-1].float())
        cache_head_difference = {"parameters_max_abs":float((output2["parameters"]-recomputed_output2["parameters"]).abs().max()),
                                 "class_probabilities_max_abs":float((output2["classes"].softmax(-1)-recomputed_output2["classes"].softmax(-1)).abs().max()),
                                 "cell_kind_probabilities_max_abs":float((output2["status"].softmax(-1)-recomputed_output2["status"].softmax(-1)).abs().max()),
                                 "record_probability_max_abs":float((output2["recorded"].sigmoid()-recomputed_output2["recorded"].sigmoid()).abs().max())}
        reconstructed = []
        reconstruction_metrics = {}
        for tick, features in ((target_ticks[0], future), (target_ticks[1], future2)):
            physical_lidar = features["lidar"]*scales["lidar"][1]+scales["lidar"][0]
            ranges, logits = decoder(physical_lidar)
            metrics, predicted_cloud, target_cloud = lidar_metrics(ranges, logits, prepared[tick], current)
            reconstruction_metrics[str(tick)] = metrics
            previous_hits = ungroup_fine(current.hits.cpu().numpy()).astype(bool)
            previous_ranges = ungroup_fine(current.ranges.cpu().numpy())
            previous_cloud = ray_grid()[previous_hits]*previous_ranges[previous_hits,None]
            reconstructed.append((tick, (predicted_cloud, previous_cloud, target_cloud)))
    args.output.mkdir(parents=True, exist_ok=True)
    for tick, clouds in reconstructed: plot_clouds(args.output/f"lidar_tick_{tick:06d}.png", clouds, tick)
    result = {"episode": args.episode, "cutoff": args.cutoff, "targets": target_ticks,
              "scope": "one authentic TRAIN episode, frozen Qwen and Sonata, shared heads and sensor projections; gradient path only, not generalization/calibration",
              "numerical_workflow": "TRAIN prefix-only frozen scales; one canonical/communication index; future labels separate; no provenance/source/catalog scans in numerical hot path",
              "canonical": str(args.canonical), "communication": str(args.communication), "model": str(args.model), "observations": str(args.observations),
              "input": {"graph_semantic_tokens": text_tokens, "unaliased_full_graph_tokens_measured": unaliased_tokens,
                        "final_structured_text_tokens": final_text_tokens, "prefix_tokens": int(prefix_embeddings.shape[1]), "suffix_tokens": int(suffix_embeddings.shape[1]),
                        "predicted_append_tokens": int(append_embeddings.shape[1]), "history_ticks": list(range(args.cutoff+1)), "historical_capture_ticks": historical_ticks,
                        "semantic_records": dict(Counter(r["kind"] for r in index.semantic_records)), "semantic_roundtrip_exact": True,
                        "semantic_roundtrip_scope": "retained model-input semantic projection; archive bookkeeping and renderer hardcoded network defaults excluded; complete causal worldtruth retained",
                        "excluded_model_fields": sorted(MODEL_INPUT_EXCLUDED),
                        "communication_replacement_source": str(args.communication),
                        "provenance_in_model_text": False, "causal_actor_count": len(index.entities), "query_count": len(query["addresses"]),
                        "native_rgb_grid": rgb_shape, "modal_feature_shapes": {m: list(v.shape) for m, v in raw_features[args.cutoff].items()},
                        "static_sensor_settings": sensor_settings, "sensor_settings_source": str(args.lidar_settings),
                        "codec": "static frozen-prefix reversible symbol/change text; re-encoding an extended real prefix is not cache-preserving",
                        "graph_scope": "complete tick0 initial worldtruth assertions and all causal delta operations; graph identities from causal references; query actors remain frame-prefix scoped",
                        "graph_identity_count": len(index.graph_nodes), "kv_theoretical_budget_bytes": kv_bytes, "max_context_budget_tokens": token_ceiling,
                        "predicted_append_reserved_capacity_tokens": append_reserve},
              "training": {"seed": args.seed, "steps": args.steps, "lr": args.lr, "loss_history": history, "loss_after": floats(final_losses), "gradient_norms": gradient_norms,
                           "parameterization":"shared contextual trunk, zero-initialized separate continuous-type outputs; causal constant-velocity position prior; TRAIN temporal increment RMS scales; latest-anchored gamma/Beta means; feature persistence plus learned residual",
                           "gradient_clip_norm":args.grad_clip,
                           "gradient_evaluation": "post-training replay, no optimizer update" if args.resume_head is not None else "last optimizer step",
                           "resumed_trained_head": args.resume_head is not None,
                           "gradient_presence": gradient_presence,
                           "trainable_parameters": sum(p.numel() for p in trainables.parameters()), "qwen_parameters": sum(p.numel() for p in qwen.parameters()),
                           "sonata_parameters": sum(p.numel() for p in sonata.parameters()), "qwen_frozen": True, "sonata_frozen": True,
                           "sonata_geometry_adapter": "learned3geometry channels in checkpoint RGB slots; actual coordinates+calculated PCA normals; no measured colours or pretrained colour efficacy claim",
                           "teacher_adapter": "initial geometry adapter frozen for all feature labels", "sonata_config": sonata_config},
              "rollout": {"second_step_loss": floats(second_losses), "direct_second_target_loss": floats(direct_losses), "future_truth_inputs": False, **rollout_cache_metadata,
                          "full_recompute_length": recomputed_cache.get_seq_length(), "hidden_max_abs_difference": cache_difference,
                          "hidden_rms_difference":cache_rms,"head_output_difference":cache_head_difference,
                          "full_recompute_schedule":"empty cache; identical prefix chunks, each historical sensor suffix, then identical predicted append chunks",
                          "cached_append_seconds": cache_seconds, "full_recompute_seconds": recompute_seconds,
                          "sensor_pose_policy": "fixed sensor-frame grids; no future world-pose input or world-calibrated fusion; cutoff rig is immutable",
                          "no_lm_head_or_generate": True},
              "coarse_lidar": reconstruction_metrics,
              "lidar_representation":{"dimensions":647,"components":{"Sonata":512,"geometry_statistics":7,"conditional_range_logit":64,"captured_return_marker":64},
                                      "range_support_m":[0.,sensor_settings["Range"]],"range_code":"logit(metres/sensorRange); finite FP32 epsilon at range endpoints only; no-return ranges masked out of normalization/loss; conditional TRAIN prior in model latent inputs",
                                      "loss":"equal average of separate TRAIN-normalized Sonata, geometry, conditional-distance and return-plane MSE; additional decoded physical range Huber and return BCE"},
              "future_feature_metrics":{"first_step":feature_metrics(future,target_features[target_ticks[0]],masks[target_ticks[0]],current_features,masks[args.cutoff]),
                                        "rollout_second_step":feature_metrics(future2,target_features[target_ticks[1]],masks[target_ticks[1]],current_features,masks[args.cutoff]),
                                        "direct_second_target":feature_metrics(direct_future,target_features[target_ticks[1]],masks[target_ticks[1]],current_features,masks[args.cutoff])},
              "typed_physical_metrics": {"initial_first_step": typed_metrics(index, first_rows), "trained_first_step": typed_metrics(index, first_predictions),
                                         "rollout_second_step": typed_metrics(index, second_predictions), "direct_second_target": typed_metrics(index, direct_predictions)},
              "saved_attention_layouts": SAVED_ATTENTION_LAYOUTS,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "wall_seconds": time.perf_counter()-started,
              "limitations": ["one sample training loss does not establish prediction accuracy or calibration", "depth distance semantics and RGB/LiDAR acquisition alignment remain unconfirmed",
                              "class-ID segmentation is not instance identity", "coarse learned angular reconstruction is not fullscan Sonata inversion", "communication is controlled simulation, not measured radio telemetry",
                              "observed_ue_world_cm is null for22actors per supplied frame; measured per-actor world transforms/bounds and instance-pixel masks are unavailable",
                              "12corridors are sidecar-only logical regions and are not rasterized"]}
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (args.output/"predictions.json").write_text(json.dumps({"initial": first_rows, "first_step": first_predictions, "second_step": second_predictions, "direct_second_target": direct_predictions}, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.output/"future_features.npz", **{f"tick_{tick}_{m}_train_normalized":v.float().cpu().numpy() for tick,feature in ((target_ticks[0],future),(target_ticks[1],future2)) for m,v in feature.items()},
                        tick_first_rgb_native_append=predicted_rgb.float().cpu().numpy())
    bindings = {"rgb": rgb_bindings, "sensor_suffix": suffix_bindings,
                "predicted_append":predicted_bindings,
                "future_head_outputs":[{"modality":m,"tick":target_ticks[1],"role":"predicted_head_output_no_token_range","positions":positions.cpu().tolist()} for m,positions in layouts.items()],
                "future_coordinate_policy":"predicted tick observer sensor-local regions on the fixed cutoff sensor layout; target poses are supervision only; independent modality/state forecasts do not establish world-grid fusion or physical instance visibility",
                "semantic_token_spans": [{"kind": k[0], "identity": k[1], "start": v[0], "end": v[1]} for k, v in token_spans.items()],
                "source_records": index.source_bindings,
                "communication": [b for b in index.source_bindings if b.get("path") == str(args.communication)],
                "frozen_numeric_scales": [vars(v) for v in index.scales.values()],
                "opaque_identity_aliases": index.identity_aliases, "graph_nodes": index.graph_nodes,
                "frozen_forecast_scales": [{"family":k[0],"component":k[1],**v} for k,v in index.forecast_scales.items()],
                "frozen_feature_scales": {m: {"mean": x[0].cpu().tolist(), "std": x[1].cpu().tolist(), "present_count":x[2].cpu().tolist()} for m, x in scales.items()},
                "future_layouts": {m: x.cpu().tolist() for m, x in layouts.items()}, "query_binding": "entity span + exact field declaration span + component + physical horizon",
                "excluded_model_fields": sorted(MODEL_INPUT_EXCLUDED),
                "semantic_projection": "complete causal worldtruth and prefix frame/entity semantics, excluding archive bookkeeping and three renderer hardcoded network defaults; authentic communication.* supplied separately; future-selected roster nodes excluded; source/evidence bookkeeping is sidecar; roundtrip applies to this retained projection"}
    (args.output/"bindings.json").write_text(json.dumps(bindings, indent=2, allow_nan=False)+"\n")
    (args.output/"semantic_input.txt").write_text(text)
    torch.save({"trainable_state": trainables.state_dict(), "seed": args.seed, "steps": args.steps,
                "loss_history": history, "initial_predictions": first_rows}, args.output/"head.pt")
    print(dumps({"phase": "complete", "loss_before": history[0]["total"], "loss_after": float(final_losses["total"]),
                 "cache_difference": cache_difference, "gpu_peak_bytes": result["peak_gpu_allocated_bytes"], "output": str(args.output)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True); parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True); parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--graph-base", type=Path, default=Path("aw_data/objective_semantic_truth/L4-1_v1__seed00/world_truth_graph_base.json"))
    parser.add_argument("--sonata", type=Path, default=Path("Dataset/world_model/runs/pretrained_encoders_20260822"))
    parser.add_argument("--lidar-settings", type=Path, default=Path("design/p01_generic/structured_mm_20261006/host_data_v2/L4-1_v1/seed00/uav_view_000__u_inspect_l4_1_v1/lidar/tick_000250.json"))
    parser.add_argument("--output", type=Path, default=Path("design/p01_generic/schema_rollout_v4/derived/current_model"))
    parser.add_argument("--resume-head", type=Path, help="evaluate an existing trained head, preserving its measured training history")
    parser.add_argument("--episode", default="L4-1_v1__seed00"); parser.add_argument("--observer", default="u_inspect_l4_1_v1")
    parser.add_argument("--cutoff", type=int, default=5); parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--steps", type=int, default=8); parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--chunk", type=int, default=1024); parser.add_argument("--seed", type=int, default=20261007)
    run(parser.parse_args())


if __name__ == "__main__": main()
