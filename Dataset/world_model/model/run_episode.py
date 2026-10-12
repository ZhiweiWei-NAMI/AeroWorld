"""Authentic typed graph and spatial modalities through a frozen shared Qwen."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from Dataset.world_model.data.geometry import WORLD_FROM_NED, transform_points
from Dataset.world_model.data.observations import (load_observation_frame,
                                                   load_observation_inventory,
                                                   observation_token_bindings,
                                                   source_entity_association)
from .causal_input import (TypedGraphEncoder, STATUS, cell, dumps, fit_observation_scales, fit_state_normalization,
                           forecast_anchor, labels, load_index, query_tensors, restrict_query_fields,
                           read_tick_seconds, MODEL_INPUT_EXCLUDED)
from .distributions import SharedDistributionHead, predictions, typed_loss
from .query_head import AddressQueryAttention, QueryAddress, fit_feature_scale
from . import derivations
from . import multimodal as mm


def device_tensors(values, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in values.items()}


def floats(values):
    return {name: float(value.detach()) for name, value in values.items()}


def normalize_features(values, scales):
    return {name: scales[name].transform(value) if name in scales else value
            for name, value in values.items()}


def predicted_sensor_transform(observer_pose, calibration):
    """Build the world transform solely from the derived observer body pose and cutoff mount."""
    if observer_pose is None:
        return None
    position = np.array(observer_pose["observer.capture_position_world_enu_m"], dtype=np.float64)
    orientation = np.array(observer_pose["observer.body_orientation_ned_6d"], dtype=np.float64)
    a, b = orientation[:3], orientation[3:]
    an = np.linalg.norm(a)
    if not np.isfinite(an) or an == 0:
        raise ValueError("predicted orientation has an undefined first axis")
    a = a/an
    b = b-a*np.dot(a, b)
    bn = np.linalg.norm(b)
    if not np.isfinite(bn) or bn == 0:
        raise ValueError("predicted orientation has an undefined second axis")
    b = b/bn
    rotation = np.column_stack((a, b, np.cross(a, b)))
    body = np.eye(4)
    body[:3, :3] = WORLD_FROM_NED @ rotation
    body[:3, 3] = position
    return body @ calibration.sensor_to_body


def complete_constant_feedback(feedback, index, constant, target_tick):
    """Add TRAIN-constant components (never queried) to rollout feedback at their latest present value.

    Every TRAIN residual of these components is zero, so the latest present
    value is their TRAIN-supported prediction; vector consumers need all components.
    """
    for entity, cells in feedback.items():
        for family, component in constant:
            if (family, component) in cells or not any(key[0] == family for key in cells):
                continue
            anchor, _, observed = forecast_anchor(index, QueryAddress(index.episode, entity, family, target_tick, component))
            if not observed:
                raise ValueError(f"constant component {family}[{component}] of {entity} has no present history")
            cells[family, component] = {"value": anchor, "kind": "present", "record_probability": 1.}
    return feedback


def derived_observer_pose(feedback, index, observer):
    """Camera pose derived from the observer UAV's predicted position and velocity.

    The camera position equals the UAV position and the level body follows the
    derived yaw (derivations.py); None when a predicted primary is not present.
    """
    cells = feedback[observer]
    def vector(family):
        rows = [cells.get((family, i)) for i in range(3)]
        return None if any(r is None or r["kind"] != "present" or r["value"] is None for r in rows) else [r["value"] for r in rows]
    primary = {family: vector(family) for family in ("pose.position_enu_m", "pose.velocity_enu_mps")}
    if any(v is None for v in primary.values()):
        return None
    yaws = [cell(state["pose.yaw_deg"], None) for (t, entity), state in sorted(index.states.items())
            if entity == observer and t <= index.cutoff and "pose.yaw_deg" in state]
    context = {"observer": True, "latest_yaw": next((v for kind, v in reversed(yaws) if kind == "present"), None)}
    derived = derivations.derive(primary, context)
    return {family: derived[family] for family in ("observer.capture_position_world_enu_m", "observer.body_orientation_ned_6d")}


def configure_qwen_checkpointing(qwen):
    """Keep native layer metadata while explicitly checkpointing each block."""
    layers = qwen.language_model.layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Qwen language_model.layers must be its native ModuleList")
    for layer in layers:
        original = layer.forward
        def forward(*args, _original=original, _layer=layer, **kwargs):
            if _layer.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)
        layer.forward = forward
    qwen.requires_grad_(False)
    return len(layers)


def native_rgb_block(qwen, features, grid, token_start, rope_start):
    """Use the native visual marker/grid positions without clock-bearing text."""
    ids = torch.tensor([[qwen.config.vision_start_token_id,
                         *([qwen.config.image_token_id]*len(features)),
                         qwen.config.vision_end_token_id]], device=features.device)
    types = (ids == qwen.config.image_token_id).long()
    native, _ = qwen.get_rope_index(ids, image_grid_thw=grid, mm_token_type_ids=types)
    positions = torch.cat((torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1)+token_start,
                           native+rope_start), dim=0)
    marker = qwen.get_input_embeddings()(ids).detach()
    mask = (ids == qwen.config.image_token_id).unsqueeze(-1)
    embeddings = marker.masked_scatter(mask, features.to(marker.dtype))
    return embeddings, positions, int(native.max())+rope_start+1


def scalar_positions(length, token_start, rope_start, device):
    seq = torch.arange(length, device=device).view(1, 1, -1)
    return torch.cat((seq+token_start, (seq+rope_start).expand(3, 1, length)), dim=0)


def chronological_prefix(qwen, structure, modal_blocks, *, token_start=0, rope_start=0):
    """Interleave emitted records and available capture blocks by relative time."""
    embeddings, positions, entities, fields, binding = [], [], [], [], []
    times = structure["times"]
    unique = sorted(set(times.detach().cpu().tolist()) | set(modal_blocks))
    for tau in unique:
        selected = torch.where(times == tau)[0]
        if len(selected):
            block = structure["embeddings"][:, selected].to(qwen.dtype)
            embeddings.append(block)
            positions.append(scalar_positions(len(selected), token_start, rope_start, block.device))
            entities.extend(structure["token_entities"][int(i)] for i in selected)
            fields.extend(structure["token_fields"][int(i)] for i in selected)
            binding.append({"kind": "structure", "relative_time_s": tau,
                            "token_start": token_start, "token_end": token_start+len(selected)})
            token_start += len(selected); rope_start += len(selected)
        if tau in modal_blocks:
            for name, value, grid, owners in modal_blocks[tau]:
                start = token_start
                if name == "rgb":
                    block, pos, rope_start = native_rgb_block(qwen, value, grid, token_start, rope_start)
                    owners = [[], *owners, []]
                else:
                    block = value.to(qwen.dtype)[None]
                    pos = scalar_positions(len(value), token_start, rope_start, value.device)
                    rope_start += len(value)
                embeddings.append(block); positions.append(pos)
                entities.extend(owners); fields.extend([[] for _ in owners])
                token_start += block.shape[1]
                binding.append({"kind": "modality", "modality": name, "relative_time_s": tau,
                                "token_start": start, "token_end": token_start})
    return {"embeddings": torch.cat(embeddings, dim=1), "positions": torch.cat(positions, dim=2),
            "token_entities": entities, "token_fields": fields, "bindings": binding,
            "next_token": token_start, "next_rope": rope_start}


def cached_forward(qwen, prefix, chunk_size, cache=None):
    hidden = []
    for start in range(0, prefix["embeddings"].shape[1], chunk_size):
        out = qwen(inputs_embeds=prefix["embeddings"][:, start:start+chunk_size],
                   position_ids=prefix["positions"][:, :, start:start+chunk_size],
                   past_key_values=cache, use_cache=True)
        hidden.append(out.last_hidden_state)
        cache = out.past_key_values
    return torch.cat(hidden, dim=1), cache


def typed_read(head, attention, query, hidden, prefix):
    global_context = hidden[0].float().mean(0)
    entity, field = attention(query, hidden[0], prefix["token_entities"], prefix["token_fields"], global_context)
    return head(query, entity, field, global_context), global_context


def typed_metrics(index, rows, anchor_rows=None):
    fields = {(field.family, field.component): field for field in index.fields}
    groups = {}
    for position, row in enumerate(rows):
        name, component, entity = row["field"], row["component"], row["entity"]
        state = index.states.get((row["target_tick"], entity))
        if state is None or name not in state:
            continue
        kind, truth = cell(state[name], component)
        if kind != "present":
            continue
        field = fields[name, component]
        past = sorted((tick, cell(value[name], component)) for (tick, actor), value in index.states.items()
                      if tick <= index.cutoff and actor == entity and name in value)
        past = [(tick, value) for tick, (status, value) in past if status == "present"]
        if name not in groups:
            groups[name] = {"model": [], "persistence": [], "constant_velocity": [], "anchor": [], "selected": [],
                            "unit": field.unit, "circular": field.circular, "categorical": field.logical_dtype != "real"}
        group = groups[name]
        def error(prediction):
            if field.logical_dtype != "real":
                return float(prediction == truth)
            if field.circular:
                return abs((float(prediction)-float(truth)+180)%360-180)
            return abs(float(prediction)-float(truth))
        group["model"].append(error(row["conditional_point"]))
        if anchor_rows is not None:
            group["anchor"].append(error(anchor_rows[position]["conditional_point"]))
        group["selected"].append(row["predicted_cell_kind"] == "present" and row["record_probability"] >= .5)
        if past:
            group["persistence"].append(error(past[-1][1]))
        if name == "pose.position_enu_m" and past:
            velocity = sorted((tick, cell(value["pose.velocity_enu_mps"], component))
                              for (tick, actor), value in index.states.items()
                              if tick <= index.cutoff and actor == entity and "pose.velocity_enu_mps" in value)
            velocity = [value for _, (status, value) in velocity if status == "present"]
            if velocity:
                group["constant_velocity"].append(error(float(past[-1][1])+float(velocity[-1])*
                                                       (row["target_tick"]-past[-1][0])*index.tick_seconds))
    result = {}
    for name, group in groups.items():
        selected = [value for value, include in zip(group["model"], group["selected"]) if include]
        result[name] = {"unit": group["unit"], "metric": "accuracy" if group["categorical"] else
                        "circular_absolute_error" if group["circular"] else "mean_absolute_error",
                        "present_component_count": len(group["model"]), "model": float(np.mean(group["model"])),
                        "selected_present_count": len(selected), "selected_present_metric": float(np.mean(selected)) if selected else None,
                        "persistence": float(np.mean(group["persistence"])) if group["persistence"] else None,
                        "persistence_count": len(group["persistence"]),
                        "constant_velocity": float(np.mean(group["constant_velocity"])) if group["constant_velocity"] else None,
                        "constant_velocity_count": len(group["constant_velocity"]),
                        "anchor_only": float(np.mean(group["anchor"])) if group["anchor"] else None,
                        "anchor_only_count": len(group["anchor"])}
    return result


def anchor_output(query, device):
    """Zero learned residual: the hand-coded CV anchor law is the whole model."""
    n = len(query["addresses"])
    classes = F.one_hot(query["category_prior"], 5).float()*4.
    status = torch.zeros((n, len(STATUS)), device=device)
    status[:, 0] = 4.
    return {"parameters": torch.zeros((n, 6), device=device), "classes": classes,
            "status": status.masked_fill(~query["allowed_status"].to(device), -torch.inf),
            "recorded": torch.full((n,), 4., device=device)}


def common_metrics(predicted, target, previous, target_mask=None, previous_mask=None):
    target_mask = torch.ones_like(target, dtype=torch.bool) if target_mask is None else target_mask
    previous_mask = torch.ones_like(target, dtype=torch.bool) if previous_mask is None else previous_mask
    common = target_mask & previous_mask
    def mean(values, mask):
        return float(values[mask].mean()) if bool(mask.any()) else None
    return {"model_mae_target_support": mean((predicted-target).abs(), target_mask),
            "model_mae_common_support": mean((predicted-target).abs(), common),
            "persistence_mae_common_support": mean((previous-target).abs(), common),
            "target_support_count": int(target_mask.sum()), "common_support_count": int(common.sum())}


def gradient_report(modules):
    result = {}
    for name, module in modules.items():
        named = list(module.named_parameters())
        gradients = [parameter.grad for _, parameter in named if parameter.grad is not None]
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError(f"non-finite participating gradient: {name}")
        result[name] = {"parameter_tensors": len(named), "participating_tensors": len(gradients),
                        "nonparticipating_tensors": sum(1 for _, parameter in named if parameter.grad is None),
                        "gradient_norm": float(torch.sqrt(sum(gradient.square().sum() for gradient in gradients))) if gradients else None}
    return result


def modality_data(features, frame):
    return {**features, "depth_codes": frame.depth_codes, "seg_codes": frame.seg_codes,
            "depth_normalization": frame.depth.new_tensor([frame.depth_normalization["mean_m"], frame.depth_normalization["std_m"]])}


def modality_inputs(projectors, features, prepared, ticks, layouts, index, scales, token_bindings):
    """Project each capture's modality tensors under observer/entity ownership.

    ``token_bindings[tick]`` maps each modality token to its owning observer
    and the entities projected into that region (``observation_token_bindings``).
    Every token therefore keeps its source observer and capture tau.
    """
    contexts, blocks = {}, {}
    for tick in ticks:
        tau = (tick-index.cutoff)*index.tick_seconds
        binding = token_bindings[tick]
        values, owners, row_blocks = [], [], []
        for name in ("rgb", "depth", "seg", "lidar"):
            value = features[tick][name]
            owners_now = binding[name]
            values.append(projectors.project_graph(name, value, layouts[name], tau)); owners.extend(owners_now)
            native_value = scales["rgb"].inverse(value) if name == "rgb" else value
            projected = projectors.project(name, native_value, layouts[name], tau)
            row_blocks.append((name, projected, prepared[tick].image_grid_thw if name == "rgb" else None, owners_now))
        for name in ("rgb","depth","seg","lidar"):
            observer=prepared[tick].binding["observer"]
            descriptor,graph_descriptor=projectors.descriptor(name,prepared[tick].descriptors[name],tau,index.local_identities[observer])
            values.append(graph_descriptor);owners.append([observer])
            row_blocks.append((name+"_descriptor",descriptor,None,[[observer]]))
        norm = prepared[tick].depth.new_tensor([prepared[tick].depth_normalization["mean_m"], prepared[tick].depth_normalization["std_m"]])
        # Physical normalization is numerical capture context, never source-clock text.
        norm_feature = prepared[tick].depth.new_zeros((1,32)); norm_feature[0,:2] = norm/200.
        values.append(projectors.project_graph("depth",norm_feature,layouts["depth"].mean(0,keepdim=True),tau))
        owners.append([prepared[tick].binding["observer"]])
        contexts[tick] = {"features": torch.cat(values), "entity_ids": owners}
        row_blocks.append(("depth",projectors.project("depth",norm_feature,layouts["depth"].mean(0,keepdim=True),tau),None,
                           [[prepared[tick].binding["observer"]]]))
        blocks[tau] = row_blocks
    return contexts, blocks


def forecast_loss(output, query, truth, future, teacher, prepared_target, decoder, prior_ranges, prior_probability, transition):
    losses = typed_loss(output,query,truth)
    for name in ("rgb","lidar"):
        losses[name+"_feature_mse"] = F.mse_loss(future[name],teacher[name])
    losses["depth_code_ce"] = F.cross_entropy(future["depth_codes"],teacher["depth_codes"])
    losses["semantic_code_ce"] = F.cross_entropy(future["seg_codes"],teacher["seg_codes"])
    losses["depth_normalization_mse"] = F.mse_loss(future["depth_normalization"]/200.,teacher["depth_normalization"]/200.)
    losses.update(decoder.training_losses(future["lidar"],prior_ranges,prior_probability,
                                          prepared_target.ray_ranges,prepared_target.ray_hits,transition))
    losses["total"] = (losses["typed"]+losses["rgb_feature_mse"]+losses["lidar_feature_mse"]+
                       losses["depth_code_ce"]+losses["semantic_code_ce"]+losses["depth_normalization_mse"]+
                       losses["dense_range_nll"]+losses["dense_return_bce"])
    return losses


def feedback_features(future, codecs):
    result={name:future[name] for name in ("rgb","lidar")}
    for name in ("depth","seg"):
        codes=future[name+"_codes"].argmax(-1)
        result[name+"_codes"]=codes
        result[name]=codecs.code_features(name,codes)
    result["depth_normalization"]=future["depth_normalization"]
    return result


def active_parameter_state(modules):
    """The trained head, saved once at the end of the run."""
    return {name:parameter.detach().cpu() for name,parameter in modules.named_parameters() if parameter.requires_grad}


def cloud_view(path,clouds,tick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=("Prediction","Causal persistence","Future truth")
    occupied=[value for value in clouds if len(value)]
    if not occupied:
        raise ValueError("point-cloud visualization has no acquired support")
    points=np.concatenate(occupied)
    lo,hi=points.min(0),points.max(0);center=(lo+hi)/2;radius=max(float((hi-lo).max()/2),.1)
    fig=plt.figure(figsize=(15,4.5))
    for number,(name,value) in enumerate(zip(names,clouds),1):
        ax=fig.add_subplot(1,3,number,projection="3d")
        if len(value):
            # Drawing stride affects visualization only; metrics/cloud files retain every selected ray.
            shown=value[::max(1,len(value)//12000)]
            ax.scatter(*shown.T,s=1,c=np.linalg.norm(shown,axis=1),cmap="viridis")
        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_zlim(center[2]-radius,center[2]+radius);ax.set_box_aspect((1,1,1));ax.set_title(name)
    fig.suptitle(f"Full-scan reconstruction, target tick {tick}");fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def evaluate_prediction(args,name,tick,future,observer_pose,law,codecs,frames,prepared,current,persistence_cloud_metrics=None):
    from PIL import Image
    from Dataset.world_model.data.observations import _class_rgb
    print(dumps({"phase":"evaluate_start","branch":name,"target_tick":tick}),flush=True)
    target=prepared[tick]
    hits=target.ray_hits.bool();previous_hits=current.ray_hits.bool();selected=law["return_probability"]>=.5
    ranges=law["expected_range_m"]
    point_ranges=law["median_range_m"]
    sensor=(law["direction"][selected]*point_ranges[selected,None]).cpu().numpy()
    target_sensor=frames[tick].points_sensor_ned_m[frames[tick].hit_mask.astype(bool)]
    previous_sensor=frames[args.cutoff].points_sensor_ned_m[frames[args.cutoff].hit_mask.astype(bool)]
    transform=predicted_sensor_transform(observer_pose,frames[args.cutoff].calibration)
    world=transform_points(sensor,transform) if transform is not None else None
    target_world=transform_points(target_sensor,frames[tick].calibration.sensor_to_world)
    previous_world=transform_points(previous_sensor,frames[args.cutoff].calibration.sensor_to_world)
    metrics={"acquired_rays":len(hits),"target_hits":int(hits.sum()),"predicted_returns":int(selected.sum()),
             "return_brier":float((law["return_probability"]-target.ray_hits).square().mean()),
             "return_accuracy":float((selected==hits).float().mean()),
             "persistence_return_accuracy":float((previous_hits==hits).float().mean()),
             "range_m":common_metrics(ranges,target.ray_ranges,current.ray_ranges,hits,previous_hits),
             "median_range_m":common_metrics(point_ranges,target.ray_ranges,current.ray_ranges,hits,previous_hits),
             "sensor_cloud":mm.pointcloud_metrics(sensor,target_sensor),
             "persistence_sensor_cloud":mm.pointcloud_metrics(previous_sensor,target_sensor) if persistence_cloud_metrics is None else persistence_cloud_metrics["persistence_sensor_cloud"],
             "world_cloud":mm.pointcloud_metrics(world,target_world) if world is not None else None,
             "persistence_world_cloud":mm.pointcloud_metrics(previous_world,target_world) if persistence_cloud_metrics is None else persistence_cloud_metrics["persistence_world_cloud"],
             "predicted_pose_available":transform is not None,"cloud_point_rule":"conditional range median and return probability>=0.5"}
    arrays={"predicted_sensor_m":sensor,"target_sensor_m":target_sensor,"persistence_sensor_m":previous_sensor,
            "target_world_m":target_world,"persistence_world_m":previous_world,
            "range_mean_m":ranges.cpu().numpy(),"range_median_m":point_ranges.cpu().numpy(),"return_probability":law["return_probability"].cpu().numpy()}
    if world is not None: arrays["predicted_world_m"]=world
    np.savez_compressed(args.output/f"{name}_tick_{tick:06d}_clouds.npz",**arrays)
    cloud_view(args.output/f"{name}_tick_{tick:06d}_sensor.png",(sensor,previous_sensor,target_sensor),tick)
    if world is not None:
        cloud_view(args.output/f"{name}_tick_{tick:06d}_world.png",(world,previous_world,target_world),tick)
    print(dumps({"phase":"geometry_complete","branch":name,"target_tick":tick}),flush=True)
    norm={"mean_m":float(future["depth_normalization"][0]),"std_m":float(future["depth_normalization"][1])}
    depth_codes=future["depth_codes"].argmax(-1)
    decoded_standardized=codecs.decode_depth_standardized(depth_codes,generator=torch.Generator(device="cpu").manual_seed(args.seed+tick))
    decoded_depth=(decoded_standardized*norm["std_m"]+norm["mean_m"]).clamp_min(0.)
    decoded_seg,_=codecs.decode_seg(future["seg_codes"].argmax(-1))
    height,width=frames[tick].depth_m.shape
    depth=F.interpolate(decoded_depth[None,None],size=(height,width),mode="bilinear",align_corners=False)[0,0]
    seg=F.interpolate(decoded_seg[None,None].float(),size=(height,width),mode="nearest")[0,0].long()
    truth_depth=torch.tensor(np.array(frames[tick].depth_m,copy=True),device=depth.device)
    previous_depth=torch.tensor(np.array(frames[args.cutoff].depth_m,copy=True),device=depth.device)
    depth_valid=torch.isfinite(truth_depth)&(truth_depth>0);previous_valid=torch.isfinite(previous_depth)&(previous_depth>0)
    metrics["decoded_depth_m"]=common_metrics(depth,truth_depth,previous_depth,depth_valid,previous_valid)
    target_norm=target.depth_normalization
    supervised_norm_depth=(decoded_standardized*target_norm["std_m"]+target_norm["mean_m"]).clamp_min(0.)
    oracle_depth=codecs.decode_depth(target.depth_codes,target_norm,generator=torch.Generator(device="cpu").manual_seed(args.seed+tick))
    supervised_norm_depth=F.interpolate(supervised_norm_depth[None,None],size=(height,width),mode="bilinear",align_corners=False)[0,0]
    oracle_depth=F.interpolate(oracle_depth[None,None],size=(height,width),mode="bilinear",align_corners=False)[0,0]
    metrics["depth_diagnostics"]={"predicted_normalization_m":norm,"target_normalization_m":{key:target_norm[key] for key in ("mean_m","std_m")},
                                  "predicted_codes_target_agreement":float((depth_codes==target.depth_codes).float().mean()),
                                  "persistence_codes_target_agreement":float((current.depth_codes==target.depth_codes).float().mean()),
                                  "predicted_codes_true_normalization_mae_m":float((supervised_norm_depth-truth_depth).abs()[depth_valid].mean()),
                                  "target_codes_true_normalization_mae_m":float((oracle_depth-truth_depth).abs()[depth_valid].mean()),
                                  "scope":"target normalization/codes used only for decoder diagnosis; forecast arrays use predicted normalization"}
    truth_seg=torch.tensor(np.array(frames[tick].segmentation_class_ids,copy=True),device=seg.device)
    previous_seg=torch.tensor(np.array(frames[args.cutoff].segmentation_class_ids,copy=True),device=seg.device)
    metrics["decoded_semantic"]={"full_resolution_pixels":truth_seg.numel(),"model_accuracy":float((seg==truth_seg).float().mean()),
                                 "persistence_accuracy":float((previous_seg==truth_seg).float().mean()),"instance_supervision":False}
    np.savez_compressed(args.output/f"{name}_tick_{tick:06d}_spatial.npz",depth_m=depth.cpu().numpy(),
                        segmentation_ue_ids=seg.cpu().numpy().astype(np.uint8),
                        depth_normalization_m=np.array([norm["mean_m"],norm["std_m"]]),depth_standardized=decoded_standardized.cpu().numpy(),
                        rgb_features=future["rgb"].cpu().numpy(),lidar_features=future["lidar"].cpu().numpy(),
                        depth_codes=future["depth_codes"].argmax(-1).cpu().numpy(),seg_codes=future["seg_codes"].argmax(-1).cpu().numpy())
    Image.fromarray(_class_rgb(seg.cpu().numpy().astype(np.uint8))).save(args.output/f"{name}_tick_{tick:06d}_seg.png")
    print(dumps({"phase":"evaluate_complete","branch":name,"target_tick":tick}),flush=True)
    return metrics


def run(args):
    started=time.perf_counter();torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(4)
    if args.step<=0 or args.steps<=0 or args.chunk<=0:
        raise ValueError("step, training steps and cache chunk must be positive")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Relative time comes from the authentic source clock, never a literal.
    tick_seconds=read_tick_seconds(args.canonical)
    train_targets=(args.cutoff+args.step,)
    eval_targets=(args.eval_cutoff+args.step,args.eval_cutoff+2*args.step)
    left=args.cutoff-round(args.history_s/tick_seconds)
    eval_left=args.eval_cutoff-round(args.history_s/tick_seconds)
    selected=load_observation_inventory(args.observations)["selected_ticks"]
    historical=tuple(t for t in selected if left<=t<=args.cutoff)
    eval_historical=tuple(t for t in selected if eval_left<=t<=args.eval_cutoff)
    train_ticks=tuple(historical)+train_targets
    eval_ticks=tuple(eval_historical)+eval_targets
    all_ticks=tuple(sorted(set(train_ticks)|set(eval_ticks)))
    def load(tick):
        in_train=tick in train_ticks
        cutoff=args.cutoff if in_train else args.eval_cutoff
        return load_observation_frame(args.observations,args.episode,tick,args.observer,
                                      branch="historical" if tick<=cutoff else "target",cutoff=cutoff)
    frames={tick:load(tick) for tick in all_ticks}
    index=load_index(args.canonical,args.communication,args.episode,args.cutoff,train_targets,args.graph_base,
                     history_s=args.history_s,observer_ids=(args.observer,))
    eval_index=load_index(args.canonical,args.communication,args.episode,args.eval_cutoff,eval_targets,args.graph_base,
                          history_s=args.history_s,observer_ids=(args.observer,))
    # All fitted statistics are training-window authority; the held-out window
    # reuses them unchanged.
    _,_,constant_targets=fit_state_normalization([index])
    restrict_query_fields(index,constant_targets);restrict_query_fields(eval_index,constant_targets)
    eval_index.scales=index.scales
    eval_index.forecast_scales=index.forecast_scales
    settings=mm.configure_sensor_layout(frames[args.cutoff].binding["lidar"],
                                        frames[args.cutoff].binding["modality"]["lidar"]["sensor_id"])
    from transformers import Qwen3_5Model,AutoImageProcessor
    qwen=Qwen3_5Model.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    checkpointed_layers=configure_qwen_checkpointing(qwen)
    processor=AutoImageProcessor.from_pretrained(args.model,local_files_only=True,min_pixels=256*448,max_pixels=256*448)
    codecs=mm.SpatialCodecs(args.codecs,device)
    codecs.register_ue_classes(frames[args.cutoff].binding["segmentation"]["class_by_id"])
    with torch.no_grad():
        prepared={tick:mm.prepare_frame(frame,qwen,processor,device,codecs,settings) for tick,frame in frames.items()}
    # Observer/entity/region association with an explicit geometric consistency
    # test, and per-token ownership (observer source + projected entities).
    association={tick:source_entity_association(frames[tick]) for tick in all_ticks}
    token_bindings={tick:observation_token_bindings(frames[tick],association[tick]) for tick in all_ticks}
    with torch.no_grad():
        teacher_semantic_logits={tick:codecs.semantic.decode_tokens(prepared[tick].seg_codes.reshape(1,*codecs.grid)).detach()
                                 for tick in historical}
        teacher_semantic_rows={tick:codecs.semantic_rows(frames[tick].segmentation_class_ids)[None] for tick in historical}
    sonata,sonata_config=mm.load_sonata(args.sonata,device)
    projectors=mm.ModalityProjectors().to(device)
    teacher_geometry=copy.deepcopy(projectors.geometry_adapter).requires_grad_(False).eval()
    # Frozen encoder outputs (Qwen vision tower, codec teacher encodes) are
    # cached once per capture before the training loop.  The LiDAR Sonata
    # backbone is frozen too, but the trainable ``projectors.geometry_adapter``
    # sits BEFORE it, so the Sonata features are rebuilt from the live adapter
    # on every training step (see ``trainable_history``); only the heartbeat
    # target supervision below uses the frozen adapter copy.
    raw={}
    with torch.no_grad():
        for tick,frame in prepared.items():
            lidar,_=mm.sonata_features(frame,sonata,teacher_geometry)
            raw[tick]={"rgb":frame.rgb,"depth":frame.depth,"seg":frame.seg,"lidar":lidar.detach()}
    # One normalization path, fitted only on the training window's captures.
    feature_scales={name:fit_feature_scale(torch.cat([raw[t][name] for t in historical],0).float())
                    for name in ("rgb","lidar")}
    normalized={tick:normalize_features(values,feature_scales) for tick,values in raw.items()}
    current=prepared[args.cutoff]
    shape=current.image_grid_thw[0].tolist()
    layouts={"rgb":mm.image_grid(shape[1]//2,shape[2]//2).to(device),
             "depth":mm.image_grid(*codecs.grid).to(device),"seg":mm.image_grid(*codecs.grid).to(device),
             "lidar":mm.angular_grid(mm.COARSE_EL,mm.COARSE_AZ).to(device)}
    graph=TypedGraphEncoder(fit_observation_scales([index]),mode=args.input_mode).to(device)
    attention=AddressQueryAttention().to(device)
    typed=SharedDistributionHead().to(device)
    future_head=mm.FutureEmbeddingHead({name:value.shape[-1] for name,value in normalized[args.cutoff].items()},
                                      codebooks={"depth":codecs.depth_codebook_size,"seg":codecs.semantic_codebook_size}).to(device)
    decoder=mm.DenseLidarDecoder(settings["Range"],512).to(device)
    modules=nn.ModuleDict({"graph":graph,"query":attention,"typed":typed,"future":future_head,
                           "projectors":projectors,"decoder":decoder,"codecs":codecs})
    active=[parameter for parameter in modules.parameters() if parameter.requires_grad]
    optimizer=torch.optim.AdamW(active,lr=args.lr)
    query=device_tensors(query_tensors(index,train_targets[0]),device)
    truth=device_tensors(labels(index,query),device)
    teacher={tick:modality_data(normalized[tick],prepared[tick]) for tick in frames}
    prior_ranges=current.ray_ranges
    epsilon=torch.finfo(prior_ranges.dtype).eps
    prior_probability=current.ray_hits.clamp(epsilon,1-epsilon)
    transition=args.step*tick_seconds
    args.output.mkdir(parents=True,exist_ok=True)
    config={name:str(value) if isinstance(value,Path) else value for name,value in vars(args).items()}
    history=[];gradients={};completed_steps=0
    codec_adapter={"completed_steps":0,"planned_steps":args.codec_decoder_steps,"loss_history":[],"train_tick":args.cutoff,
                   "learning_rate":args.codec_decoder_lr,"pixel_batch_size_per_plane":8192,
                   "before":None,"after":None,"depth_affine_diagnostic":None,"depth_oracle_25step_mae_m":None,"depth_output_after":None,"elapsed_seconds":None,
                   "scope":"UE decoder and depth output affine only; authentic TRAIN cutoff; frozen teacher/current codec logits and raw depth decode; no future frames"}
    def save_head():
        payload={"trained_parameters":active_parameter_state(modules),"state_scales":index.scales,
                 "forecast_scales":index.forecast_scales,"graph_observation_scales":graph.observation_scales,
                 "feature_scales":{name:{"mean":scale.mean.cpu(),"std":scale.std.cpu()} for name,scale in feature_scales.items()},
                 "completed_steps":completed_steps,"planned_steps":args.steps,"loss_history":history,
                 "source_config":config,"frozen_assets":{"qwen":str(args.model),"sonata":str(args.sonata),"codecs":str(args.codecs)}}
        temporary=args.output/"head.pt.tmp"
        with temporary.open("wb") as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,args.output/"head.pt")
    def compile_features(ticks,graph_index,graph_module):
        """Rebuild the trainable feature/prefix path for one training step.

        Only the frozen Qwen vision-tower features and codec teacher encodes
        (``normalized``) are cached; the LiDAR adapter -> frozen Sonata path,
        the graph encoders, the modality projections and the reconstruction
        losses are all rebuilt here so their autograd graph is never reused
        across optimizer steps.
        """
        features={tick:dict(normalized[tick]) for tick in ticks}
        for tick in ticks:
            # geometry_adapter sits BEFORE the frozen Sonata backbone, so it
            # must run on the trainable path or it never receives a gradient.
            lidar,_=mm.sonata_features(prepared[tick],sonata,projectors.geometry_adapter)
            features[tick]["lidar"]=feature_scales["lidar"].transform(lidar)
        contexts,blocks=modality_inputs(projectors,features,prepared,ticks,layouts,graph_index,feature_scales,token_bindings)
        structure=graph_module.encode_history(graph_index,contexts)
        prefix=chronological_prefix(qwen,structure,blocks)
        if prefix["embeddings"].shape[1]>qwen.config.text_config.max_position_embeddings:
            raise ValueError("authentic complete multimodal prefix exceeds native Qwen context")
        return features,contexts,structure,prefix
    print(dumps({"phase":"training_launch","steps":args.steps,"tick_seconds":tick_seconds,
                 "train_origin":args.cutoff,"heldout_origin":args.eval_cutoff,
                 "train_target_ticks":train_targets,"heldout_target_ticks":eval_targets,
                 "train_history_ticks":historical,"heldout_history_ticks":eval_historical,
                 "frozen_encoder_cache":"Qwen vision tower and codec teacher encodes cached once per capture; LiDAR adapter+Sonata and graph/projector prefix rebuilt every step",
                 "cache_rebuild_conditions":"new origin, new/changed inputs, or any parameter update that produces prefix embeddings"}),flush=True)
    for step in range(completed_steps,args.steps):
        optimizer.zero_grad(set_to_none=True);qwen.train();codecs.freeze_pretrained()
        # Rebuild the trainable prefix and reconstruction losses every step.
        history_features,history_contexts,structure,prefix=compile_features(historical,index,graph)
        input_features=modality_data(history_features[args.cutoff],current)
        hidden=qwen(inputs_embeds=prefix["embeddings"],position_ids=prefix["positions"],use_cache=False).last_hidden_state
        output,context=typed_read(typed,attention,query,hidden,prefix)
        future=future_head(context,layouts,transition,input_features,transition)
        losses=forecast_loss(output,query,truth,future,teacher[train_targets[0]],prepared[train_targets[0]],decoder,prior_ranges,prior_probability,transition)
        reconstruction=structure["reconstruction_losses"]
        losses.update({"reconstruct_"+name:value for name,value in reconstruction.items() if name!="total"})
        adaptations=codecs.adaptation_losses(frames[args.cutoff].depth_m,frames[args.cutoff].segmentation_class_ids)
        adaptations["ue_teacher_code_adaptation"]=torch.stack([F.cross_entropy(codecs.ue_decoder(teacher_semantic_logits[tick]),teacher_semantic_rows[tick])
                                                               for tick in historical]).mean()
        losses.update(adaptations);losses["total"]=losses["total"]+reconstruction["total"]+sum(adaptations.values())
        if not bool(torch.isfinite(losses["total"])):
            raise FloatingPointError("authentic multimodal objective is non-finite")
        losses["total"].backward();gradients=gradient_report(modules)
        gradient_norm=nn.utils.clip_grad_norm_(active,args.grad_clip)
        optimizer.step()
        row={"step":step,**floats(losses),"gradient_norm_before_clip":float(gradient_norm),
             "elapsed_seconds":time.perf_counter()-started,"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated()}
        history.append(row);completed_steps=step+1
        print(dumps({"phase":"optimizer","checkpoint_completed_steps":completed_steps,**row}),flush=True)
        del hidden,output,future,losses,adaptations
    # Snapshot the last training step's read-only prefix metadata so no training
    # autograd graph is retained across the inference/decoder phases.
    train_ledger=copy.deepcopy(structure["ledger"])
    train_reconstruction=floats(structure["reconstruction_losses"])
    train_prefix_bindings=copy.deepcopy(prefix["bindings"])
    train_token_entities=list(prefix["token_entities"]);train_token_fields=list(prefix["token_fields"])
    train_tokens=prefix["embeddings"].shape[1]
    del prefix,structure,history_contexts,history_features
    # Supplementary TRAIN-only decoder optimization keeps all core parameters fixed.
    # The reconstruction losses were consumed by the last training backward; the
    # snapshot above keeps only detached metadata for the result record.
    adapter_start=time.perf_counter()
    with torch.no_grad():
        current_semantic=codecs.encode_seg(frames[args.cutoff].segmentation_class_ids)
        current_logits=codecs.semantic.decode_tokens(current_semantic["codes"].reshape(1,*codecs.grid)).detach()
        adapter_planes=(teacher_semantic_logits[args.cutoff],current_logits)
        adapter_rows=teacher_semantic_rows[args.cutoff].reshape(-1)
        adapter_features=tuple(plane[0].permute(1,2,0).reshape(-1,plane.shape[1]) for plane in adapter_planes)
        def decoder_diagnostics():
            diagnostics=[]
            for plane in adapter_planes:
                logits=codecs.ue_decoder(plane)
                diagnostics.append({"ce":float(F.cross_entropy(logits,adapter_rows.reshape(1,224,224))),
                                    "accuracy":float((logits.argmax(1).reshape(-1)==adapter_rows).float().mean())})
            return dict(zip(("fixed_teacher_codes","current_codes"),diagnostics))
        codec_adapter["before"]=decoder_diagnostics()
        oracle=codecs.encode_depth(frames[args.cutoff].depth_m)
        raw_depth_decode=codecs.depth.decode_tokens(oracle["codes"].reshape(1,*codecs.grid),timesteps=25,image_size=224,
                                                   generator=torch.Generator(device="cpu").manual_seed(args.seed+args.cutoff)).detach()
        oracle_depth=(codecs.depth_output(raw_depth_decode)[0,0]*oracle["normalization"]["std_m"]+oracle["normalization"]["mean_m"]).clamp_min(0.)
        depth_mask=oracle["mask"][0,0]
        depth_x=raw_depth_decode[0,0][depth_mask];depth_y=oracle["standardized"][0,0][depth_mask]
        dx=depth_x.double();dy=depth_y.double()
        slope=((dx-dx.mean())*(dy-dy.mean())).mean()/dx.var(unbiased=False);offset=dy.mean()-slope*dx.mean()
        oracle_truth=F.interpolate(torch.tensor(np.array(frames[args.cutoff].depth_m,copy=True),device=device)[None,None],size=(224,224),mode="bilinear",align_corners=False)[0,0]
        codec_adapter["depth_affine_diagnostic"]={"correlation":float(torch.corrcoef(torch.stack((dx,dy)))[0,1]),
                "best_slope":float(slope),"best_offset":float(offset)}
        codec_adapter["depth_oracle_25step_mae_m"]=float((oracle_depth-oracle_truth).abs()[depth_mask].mean())
    adapter_parameters=[*codecs.ue_decoder.parameters(),*codecs.depth_output.parameters()]
    adapter_optimizer=torch.optim.AdamW(adapter_parameters,lr=args.codec_decoder_lr)
    for adapter_step in range(args.codec_decoder_steps):
        adapter_optimizer.zero_grad(set_to_none=True)
        indices=torch.randint(len(adapter_rows),(8192,),device=device)
        adapter_loss=torch.stack([F.cross_entropy(F.linear(values[indices],codecs.ue_decoder.weight[:,:,0,0],codecs.ue_decoder.bias),adapter_rows[indices])
                                  for values in adapter_features]).mean()
        depth_indices=torch.randint(len(depth_x),(8192,),device=device)
        depth_affine_loss=F.mse_loss(depth_x[depth_indices]*codecs.depth_output.weight.reshape(())+codecs.depth_output.bias.reshape(()),depth_y[depth_indices])
        adapter_loss=adapter_loss+depth_affine_loss
        adapter_loss.backward()
        if not bool(torch.isfinite(adapter_loss)) or any(not bool(torch.isfinite(parameter.grad).all()) for parameter in adapter_parameters):
            raise FloatingPointError("non-finite authentic TRAIN decoder optimization")
        adapter_optimizer.step()
        codec_adapter["completed_steps"]=adapter_step+1
        codec_adapter["loss_history"].append(float(adapter_loss))
    with torch.no_grad():
        codec_adapter["after"]=decoder_diagnostics()
        optimized_depth=(codecs.depth_output(raw_depth_decode)[0,0]*oracle["normalization"]["std_m"]+oracle["normalization"]["mean_m"]).clamp_min(0.)
        codec_adapter["depth_output_after"]={"slope":float(codecs.depth_output.weight.reshape(())),"offset":float(codecs.depth_output.bias.reshape(())),
                                           "resized_mae_m":float((optimized_depth-oracle_truth).abs()[depth_mask].mean())}
    codec_adapter["elapsed_seconds"]=time.perf_counter()-adapter_start
    print(dumps({"phase":"codec_adapter_complete","completed_steps":codec_adapter["completed_steps"]}),flush=True)
    qwen.eval();codecs.freeze_pretrained()
    def predict(graph_index,graph_module,attention_module,ticks,target_tick,rollout_tick,label):
        with torch.no_grad():
            features,contexts,structure_,prefix_=compile_features(ticks,graph_index,graph_module)
            prefix_hidden,prefix_cache=cached_forward(qwen,prefix_,args.chunk)
            target_query=device_tensors(query_tensors(graph_index,target_tick),device)
            output,context=typed_read(typed,attention_module,target_query,prefix_hidden,prefix_)
            rows,feedback=predictions(output,target_query)
            feedback=complete_constant_feedback(feedback,graph_index,constant_targets,target_tick)
            observer_pose=derived_observer_pose(feedback,graph_index,args.observer)
            frame=prepared[graph_index.cutoff]
            input_feat=modality_data(features[graph_index.cutoff],frame)
            prior_r=frame.ray_ranges;prior_p=frame.ray_hits.clamp(epsilon,1-epsilon)
            future=future_head(context,layouts,transition,input_feat,transition)
            law=decoder.law_decode(future["lidar"],prior_r,prior_p,transition)
            branch_args=copy.copy(args);branch_args.cutoff=graph_index.cutoff
            metrics=evaluate_prediction(branch_args,label,target_tick,future,observer_pose,law,codecs,frames,prepared,frame,None)
            anchor_rows,_=predictions(anchor_output(target_query,device),target_query)
            rollout=None
            if rollout_tick is not None:
                next_ranges=law["expected_range_m"].clone();next_probability=law["return_probability"].clone()
                predicted_features=feedback_features(future,codecs)
                contexts_r,blocks_r=modality_inputs(projectors,{target_tick:predicted_features},{target_tick:frame},
                                                    (target_tick,),layouts,graph_index,feature_scales,
                                                    {target_tick:token_bindings[graph_index.cutoff]})
                tau=transition;norm_features=frame.depth.new_zeros((1,32));norm_features[0,:2]=predicted_features["depth_normalization"]/200.
                contexts_r[target_tick]["features"][-1]=projectors.project_graph("depth",norm_features,layouts["depth"].mean(0,keepdim=True),tau)[0]
                blocks_r[tau][-1]=("depth",projectors.project("depth",norm_features,layouts["depth"].mean(0,keepdim=True),tau),None,[[args.observer]])
                full=graph_module._encode(graph_index,((target_tick,feedback),),{**contexts,**contexts_r})
                historical_slot_count=structure_["embeddings"].shape[1]
                history_structure_difference=float((full["embeddings"][:,:historical_slot_count]-structure_["embeddings"]).abs().max())
                append_count=full["append_lengths"][-1]
                appended=dict(full)
                appended["embeddings"]=full["embeddings"][:,-append_count:]
                appended["times"]=full["times"][-append_count:]
                appended["positions"]=full["positions"][:,:,-append_count:]
                appended["token_entities"]=full["token_entities"][-append_count:]
                appended["token_fields"]=full["token_fields"][-append_count:]
                append=chronological_prefix(qwen,appended,blocks_r,token_start=prefix_["next_token"],rope_start=prefix_["next_rope"])
                cache_start=time.perf_counter()
                append_hidden,rollout_cache=cached_forward(qwen,append,args.chunk,copy.deepcopy(prefix_cache))
                cache_seconds=time.perf_counter()-cache_start
                combined={"token_entities":prefix_["token_entities"]+append["token_entities"],"token_fields":prefix_["token_fields"]+append["token_fields"]}
                combined_hidden=torch.cat((prefix_hidden,append_hidden),dim=1)
                second_query=device_tensors(query_tensors(graph_index,rollout_tick,rollout_values=feedback,cutoff=target_tick),device)
                second,second_context=typed_read(typed,attention_module,second_query,combined_hidden,combined)
                second_rows,second_feedback=predictions(second,second_query)
                second_feedback=complete_constant_feedback(second_feedback,graph_index,constant_targets,rollout_tick)
                second_pose=derived_observer_pose(second_feedback,graph_index,args.observer)
                # The second step starts from the predicted tick-10 features, so
                # the residual scale is one step while the horizon 2*tau is
                # measured from the original origin (reviewer-confirmed).
                second_future=future_head(second_context,layouts,2*transition,predicted_features,transition)
                second_law=decoder.law_decode(second_future["lidar"],next_ranges,next_probability,transition)
                second_metrics=evaluate_prediction(branch_args,label+"_rollout",rollout_tick,second_future,second_pose,second_law,codecs,frames,prepared,frame,None)
                recompute_start=time.perf_counter()
                full_hidden,new_cache=cached_forward(qwen,prefix_,args.chunk)
                recomputed_tail,new_cache=cached_forward(qwen,append,args.chunk,new_cache)
                recomputed,_=typed_read(typed,attention_module,second_query,torch.cat((full_hidden,recomputed_tail),1),combined)
                cache_metrics={"historical_structure_prefix_max_abs":history_structure_difference,
                               "same_prefix_hidden_max_abs":float((append_hidden-recomputed_tail).abs().max()),
                               "same_prefix_hidden_rms":float((append_hidden.float()-recomputed_tail.float()).square().mean().sqrt()),
                               "typed_parameters_max_abs":float((second["parameters"]-recomputed["parameters"]).abs().max()),
                               "class_probability_max_abs":float((second["classes"].softmax(-1)-recomputed["classes"].softmax(-1)).abs().max()),
                               "status_probability_max_abs":float((second["status"].softmax(-1)-recomputed["status"].softmax(-1)).abs().max()),
                               "cached_append_seconds":cache_seconds,"same_boundary_full_recompute_seconds":time.perf_counter()-recompute_start,
                               "origin_tick":graph_index.cutoff,"second_query_tau_s":sorted(set(second_query["features"][:,4].cpu().tolist())),
                               "transition_seconds":transition,"cache_length":rollout_cache.get_seq_length(),
                               "prefix_tokens":prefix_["embeddings"].shape[1],"appended_tokens":append["embeddings"].shape[1],
                               "chunk_boundary_policy":"same original prefix chunks, then same each append chunks; no concatenation rechunking"}
                rollout={"rows":second_rows,"metrics":second_metrics,"cache":cache_metrics,
                         "anchor_rows":predictions(anchor_output(second_query,device),second_query)[0],
                         "append":append}
            return {"rows":rows,"metrics":metrics,"anchor_rows":anchor_rows,"prefix":prefix_,
                    "structure":structure_,"prefix_tokens":prefix_["embeddings"].shape[1],"rollout":rollout}
    # Training-fit numbers (the supervised tick) clearly labelled as such.
    train_fit=predict(index,graph,attention,historical,train_targets[0],None,"training_fit")
    # Held-out evaluation on a temporally disjoint origin whose targets were
    # never supervised; the same fitted statistics are reused unchanged.
    heldout=predict(eval_index,graph,attention,eval_historical,eval_targets[0],eval_targets[1],"heldout")
    resource={"qwen_checkpointed_blocks":checkpointed_layers,"ray_chunk":decoder.ray_chunk,
              "frozen_encoder_cache":"Qwen vision tower features and codec teacher encodes cached once per capture before optimization; LiDAR geometry_adapter+Sonata, graph encoders, modality projections and reconstruction losses rebuilt every optimizer step",
              "cache_rebuild_conditions":["new origin","new/changed inputs","any parameter update that produces prefix embeddings"]}
    entity_binding={"per_capture":{str(tick):{"roster_entities":len(association[tick]),
                     "associated_entities":sum(1 for e in association[tick] if e["association_supported"]),
                     "camera_supported":sum(1 for e in association[tick] if e["camera_region_supported"]),
                     "lidar_supported":sum(1 for e in association[tick] if e["lidar_region_supported"]),
                     "tokens_with_entity":{name:sum(1 for owners in token_bindings[tick][name] if len(owners)>1)
                                           for name in ("rgb","depth","seg","lidar")},
                     "token_count":{name:len(token_bindings[tick][name]) for name in ("rgb","depth","seg","lidar")}}
                     for tick in all_ticks}}
    def prefix_token_counts(binding_rows):
        """Per-category Qwen token counts from the chronological prefix."""
        counts={"structure":0,"modality":0,"rgb":0,"depth":0,"seg":0,"lidar":0,
                "rgb_descriptor":0,"depth_descriptor":0,"seg_descriptor":0,"lidar_descriptor":0}
        for row in binding_rows:
            span=row["token_end"]-row["token_start"]
            if row["kind"]=="structure":
                counts["structure"]+=span
            else:
                counts["modality"]+=span
                counts[row["modality"]]=counts.get(row["modality"],0)+span
        return counts
    structure_categories=("entity","state","relation","update","communication","global")
    def qwen_token_counts(ledger,binding_rows,total):
        counts={name:ledger[name] for name in structure_categories}
        counts.update(prefix_token_counts(binding_rows));counts["total"]=total
        return counts
    static_reuse={"entity_static_slots_emitted_once_per_window":train_ledger["entity"],
                  "new_entity_classifications":train_ledger["new_entity"],
                  "rule":"each window entity emits its identity/roster static slot exactly once (first tick it appears, spec category 'entity'); later ticks reference the cached static node embedding in the graph without re-emitting a Qwen token",
                  "relation_static_reuse":"unchanged relation bodies keep their cached role-port/predicate/evidence parts and are only re-dated per tick",
                  "cached_frozen":"Qwen vision tower features and codec teacher encodes cached once per capture; graph/projector prefix and reconstruction losses rebuilt every optimizer step"}
    token_counts={"training_prefix":qwen_token_counts(train_ledger,train_prefix_bindings,train_tokens),
                  "heldout_prefix":qwen_token_counts(heldout["structure"]["ledger"],heldout["prefix"]["bindings"],
                                                     heldout["prefix_tokens"]),
                  "heldout_predicted_append":prefix_token_counts(heldout["rollout"]["append"]["bindings"]),
                  "categories":"entity, state, relation, update, communication, global are structure tokens; modality rgb/depth/seg/lidar patch tokens per capture; *_descriptor is one calibrated sensor descriptor token per capture"}
    result={"episode":args.episode,"origin_tick":args.cutoff,"eval_origin_tick":args.eval_cutoff,
            "target_ticks":train_targets,"eval_target_ticks":eval_targets,"source_config":config,
            "scope":"train on one authentic TRAIN window with >=2 history captures; evaluate on a temporally disjoint held-out origin; no held-out target was supervised",
            "numerical_workflow":"sources compiled once; TRAIN-only fitted immutable scales reused by the held-out window; future frames supervision only",
            "resource_assumptions":resource,"entity_observation_binding":entity_binding,
            "token_counts":token_counts,"static_reuse":static_reuse,
            "input":{"mode":args.input_mode,"history_s":args.history_s,"tick_seconds":tick_seconds,
            "historical_capture_ticks":historical,"eval_historical_capture_ticks":eval_historical,
            "structural_ledger":train_ledger,"prefix_tokens":train_tokens,
            "qwen_tokens_by_category":{**{name:train_ledger[name] for name in structure_categories},
                                       "modality":sum(row["token_end"]-row["token_start"] for row in train_prefix_bindings
                                                      if row["kind"]=="modality")},
            "eval_prefix_tokens":heldout["prefix_tokens"],
            "query_count":len(query["addresses"]),"native_rgb_grid":shape,"native_rgb_positions":len(current.rgb),
            "modal_feature_shapes":{name:list(value.shape) for name,value in raw[args.cutoff].items()},
            "graph_identity_count":len(index.graph_nodes),"actor_count":len(index.entities),"excluded_fields":sorted(MODEL_INPUT_EXCLUDED)},
            "training":{"steps":args.steps,"completed_steps":completed_steps,"loss_history":history,"participating_gradients":gradients,
                        "qwen_frozen":True,"sonata_frozen":True,"codec_weights_frozen":True,"codec_adapters_trained":True,
                        "supplemental_codec_decoder":codec_adapter,"reconstruction_losses":train_reconstruction},
            "training_fit":{"typed_metrics":typed_metrics(index,train_fit["rows"],train_fit["anchor_rows"]),
                            "multimodal_metrics":train_fit["metrics"],
                            "note":"target tick is the supervised training target; these are training-fit numbers, not held-out prediction"},
            "heldout":{"typed_metrics":typed_metrics(eval_index,heldout["rows"],heldout["anchor_rows"]),
                       "rollout_typed_metrics":typed_metrics(eval_index,heldout["rollout"]["rows"],heldout["rollout"]["anchor_rows"]),
                       "multimodal_metrics":heldout["metrics"],"rollout_multimodal_metrics":heldout["rollout"]["metrics"],
                       "cache":heldout["rollout"]["cache"]},
            "dense_lidar_probability_law":decoder.range_law,"codec":codecs.metadata,"sonata":sonata_config,
            "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),"wall_seconds":time.perf_counter()-started,
            "limitations":["future RGB is a learned embedding target, no future RGB pixel decoder",
                           "class-ID segmentation has no instance identity; entity/observation binding uses a geometric consistency test",
                           "one TRAIN window does not establish generalization or calibration across episodes",
                           "point feedback does not propagate the full previous continuous uncertainty",
                           "depth producer axial/radial semantics remain unconfirmed; metric raster reconstruction is evaluated directly",
                           "conditional range categorical bin resolution is finite; per-ray no-return mask is separate"]}
    save_head()
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (args.output/"source_config.json").write_text(json.dumps(config,indent=2,allow_nan=False)+"\n")
    (args.output/"predictions.json").write_text(json.dumps({"training_fit":train_fit["rows"],
        "training_fit_anchor_only":train_fit["anchor_rows"],"heldout":heldout["rows"],
        "heldout_anchor_only":heldout["anchor_rows"],"heldout_rollout":heldout["rollout"]["rows"]},indent=2,allow_nan=False)+"\n")
    append=heldout["rollout"]["append"]
    bindings={"train_history":train_prefix_bindings,"train_token_entities":train_token_entities,
              "train_token_fields":train_token_fields,"eval_history":heldout["prefix"]["bindings"],
              "eval_token_entities":heldout["prefix"]["token_entities"],"eval_token_fields":heldout["prefix"]["token_fields"],
              "heldout_predicted_append":append["bindings"],
              "heldout_predicted_append_token_entities":append["token_entities"],
              "heldout_predicted_append_token_fields":append["token_fields"],
              "training_fit_target_tick":train_targets[0],"heldout_target_ticks":list(eval_targets),
              "source_records":index.source_bindings,"identity_aliases":index.identity_aliases,"local_identities":index.local_identities,
              "sensor_calibration":frames[args.cutoff].calibration.as_dict(),
              "sensor_settings_source":f"{args.observations}/tick_{args.cutoff:06d}.npz#binding_json.lidar",
              "calibrated_sensor_descriptors":{str(tick):prepared[tick].binding["calibrated_descriptors"] for tick in all_ticks},
              "entity_observation_association":{str(tick):association[tick] for tick in all_ticks},
              "token_observer_entity_ownership":{str(tick):token_bindings[tick] for tick in all_ticks},
              "descriptor_observer_alias_encoding":"sin/cos of window-local identity ordinal; modality embedding is local sensor alias",
              "source_config":config,"geometry_policy":"predicted pose plus cutoff mount; future calibrated pose used only for target metric"}
    (args.output/"bindings.json").write_text(json.dumps(bindings,indent=2,allow_nan=False)+"\n")
    print(dumps({"phase":"complete","output":str(args.output),"wall_seconds":result["wall_seconds"],
                 "peak_gpu_allocated_bytes":result["peak_gpu_allocated_bytes"]}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--graph-base", type=Path, default=Path("aw_data/objective_semantic_truth/L4-1_v1__seed00/world_truth_graph_base.json"))
    parser.add_argument("--sonata", type=Path, default=Path("Dataset/world_model/models/sonata"))
    parser.add_argument("--codecs", type=Path, default=Path("Dataset/world_model/models"))
    parser.add_argument("--output", type=Path, default=Path("Dataset/world_model/runs/prediction"))
    parser.add_argument("--codec-decoder-steps",type=int,default=128)
    parser.add_argument("--codec-decoder-lr",type=float,default=.01)
    parser.add_argument("--episode", default="L4-1_v1__seed00")
    parser.add_argument("--observer", default="u_inspect_l4_1_v1")
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--eval-cutoff", type=int, default=255)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--history-s", type=float, default=.5)
    parser.add_argument("--input-mode", choices=("graph_full", "graph_summary", "record_ablation"), default="graph_full")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261007)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
