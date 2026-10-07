"""Real-pilot CPU checkpoint for dynamic typed entity-field queries."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

from p01v4.data.normalization import fit_scaler
from p01v4.data.serialization import to_typed_input
from p01v4.data.windows import build_supervision_window, scan_input_leak
from p01v4.model.query_head import (
    InputScale,
    LOGICAL_DTYPES,
    STATUS_KINDS,
    SharedTypedQueryHead,
    attach_typed_labels,
    compile_query_batch,
    compile_query_inventory,
    declared_component_axis,
    fit_input_scales,
    typed_supervision_loss,
)


EPISODE_ID = "L4-1_v1__seed00"
CUTOFF = 250
TARGET_TICKS = tuple(range(300, 351, 5))


def _records(path: Path, counts: Counter[str]) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                counts[record["record_kind"]] += 1
                yield record


def _max_difference(left, right) -> float:
    tensors = (
        (left.numeric, right.numeric),
        (left.class_logits, right.class_logits),
        (left.status_logits, right.status_logits),
    )
    return max(float((a - b).abs().max().item()) for a, b in tensors)


def _loss_values(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().item()) for name, value in losses.items()}


def run(canonical: Path, receipt_path: Path, *, seed: int, threads: int) -> dict[str, Any]:
    started = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    stage_seconds: dict[str, float] = {}
    source_counts: Counter[str] = Counter()

    stage = time.perf_counter()
    sample = build_supervision_window(
        _records(canonical, source_counts),
        episode_id=EPISODE_ID,
        split="TRAIN",
        cutoff=CUTOFF,
        target_times=TARGET_TICKS,
    )
    scan_input_leak(sample)
    stage_seconds["canonical_stream_and_window"] = time.perf_counter() - stage

    stage = time.perf_counter()
    typed_input = to_typed_input(sample.input_records)
    typed_targets = to_typed_input(sample.target_records)
    stage_seconds["typed_views"] = time.perf_counter() - stage

    stage = time.perf_counter()
    inventory = compile_query_inventory(
        typed_input, episode_id=EPISODE_ID, target_ticks=TARGET_TICKS)
    speed = fit_scaler(
        sample.input_records,
        field="motion.speed_mps",
        episode_id=EPISODE_ID,
        split="TRAIN",
        cutoff=CUTOFF,
    )
    speed_override = InputScale(
        family=speed.field,
        component=None,
        mean=speed.mean,
        std=speed.std,
        empirical_std=speed.std,
        count=speed.count,
        max_tick=speed.max_fit_tick,
    )
    scales = fit_input_scales(
        typed_input,
        inventory.fields,
        overrides={(speed.field, None): speed_override},
    )
    batch = compile_query_batch(inventory, typed_input, scales, cutoff=CUTOFF)
    labels = attach_typed_labels(batch, typed_targets, inventory.fields, scales)
    stage_seconds["address_features_and_labels"] = time.perf_counter() - stage

    stage = time.perf_counter()
    max_classes = max(field.class_count for field in inventory.fields)
    model = SharedTypedQueryHead(
        batch.features.shape[-1], hidden_dim=64, max_classes=max_classes).cpu()
    model.eval()
    with torch.no_grad():
        before_output = model(batch)
        permutation = torch.randperm(len(batch.addresses))
        permuted_output = model(batch.index_select(permutation))
        permutation_difference = _max_difference(
            permuted_output, before_output.index_select(permutation))

        removed = len(batch.addresses) // 2
        retained = torch.cat((
            torch.arange(removed),
            torch.arange(removed + 1, len(batch.addresses)),
        ))
        reduced_output = model(batch.index_select(retained))
        inventory_difference = _max_difference(
            reduced_output, before_output.index_select(retained))
    stage_seconds["forward_and_query_invariance"] = time.perf_counter() - stage

    stage = time.perf_counter()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    before_losses = typed_supervision_loss(output, labels, batch)
    before_losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    gradient_elements = sum(gradient.numel() for gradient in gradients)
    nonzero_gradient_elements = sum(
        int(torch.count_nonzero(gradient).item()) for gradient in gradients)
    gradients_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    gradient_norm = float(torch.sqrt(sum(
        gradient.detach().pow(2).sum() for gradient in gradients)).item())
    optimizer.step()
    model.eval()
    with torch.no_grad():
        after_losses = typed_supervision_loss(model(batch), labels, batch)
    stage_seconds["backward_and_optimizer_step"] = time.perf_counter() - stage

    target_status_counts = {
        kind: int(mask.sum().item())
        for kind, mask in labels.status_masks().items()
    }
    address_type_counts = Counter(
        next(field.logical_dtype for field in inventory.fields
             if (field.family, field.component) ==
             (address.field_family, address.component))
        for address in inventory.addresses
    )
    field_documents = []
    for field in inventory.fields:
        scale = scales.get((field.family, field.component))
        field_documents.append({
            "family": field.family,
            "component": field.component,
            "component_axis": declared_component_axis(field),
            "logical_dtype": field.logical_dtype,
            "unit": field.unit,
            "frame": field.frame,
            "axis_role": field.axis_role,
            "vector_length": field.vector_length,
            "circular": field.circular,
            "bounded": [field.lower, field.upper],
            "enum_values": list(field.enum_values),
            "allowed_non_present": list(field.allowed_non_present),
            "input_scale": None if scale is None else {
                "mean": scale.mean,
                "std": scale.std,
                "empirical_std": scale.empirical_std,
                "count": scale.count,
                "max_tick": scale.max_tick,
                "constant": scale.empirical_std == 0.0,
            },
        })

    wall_seconds = time.perf_counter() - started
    receipt = {
        "receipt": "B02/dynamic-entity-field-shared-head-cpu-preliminary",
        "stage": "S02",
        "status": "completed",
        "episode_id": EPISODE_ID,
        "canonical_jsonl": str(canonical),
        "command": (
            "CUDA_VISIBLE_DEVICES= PYTHONPATH=Dataset/world_model/"
            "p01_schema_rollout_v4/src "
            "/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python "
            "-m p01v4.model.cpu_preliminary --canonical "
            f"{canonical} --receipt {receipt_path} --threads {threads} --seed {seed}"
        ),
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "threads": threads,
            "seed": seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_seconds": wall_seconds,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "stage_seconds": stage_seconds,
        },
        "window": {
            "split": sample.split,
            "cutoff": sample.cutoff,
            "target_ticks": sample.target_times,
            "input_record_count": len(sample.input_records),
            "target_record_count": len(sample.target_records),
            "source_counts": dict(sorted(source_counts.items())),
            "typed_input_counts": {
                key: len(value) for key, value in typed_input.items()
            },
            "typed_target_counts": {
                key: len(value) for key, value in typed_targets.items()
            },
        },
        "inventory": {
            "compiled_from": "cutoff-legal typed input entities plus FIELD_REGISTRY",
            "future_used_for_membership": False,
            "entity_count": len(inventory.entities),
            "field_family_count": len({field.family for field in inventory.fields}),
            "field_component_count": len(inventory.fields),
            "target_tick_count": len(inventory.target_ticks),
            "address_count": len(inventory.addresses),
            "address_type_counts": dict(sorted(address_type_counts.items())),
            "excluded_open_fields": list(inventory.excluded_open_fields),
            "fields": field_documents,
        },
        "tensors": {
            "features": list(batch.features.shape),
            "numeric_output": list(before_output.numeric.shape),
            "class_logits": list(before_output.class_logits.shape),
            "status_logits": list(before_output.status_logits.shape),
            "feature_count": len(batch.feature_names),
            "feature_names": list(batch.feature_names),
        },
        "supervision": {
            "target_status_counts": target_status_counts,
            "present_numeric": int(labels.numeric_mask.sum().item()),
            "present_categorical": int(labels.categorical_mask.sum().item()),
            "labels_passed_to_forward": False,
        },
        "model": {
            "class": "SharedTypedQueryHead",
            "hidden_dim": 64,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "entity_identity_embedding": False,
            "per_entity_field_or_query_modules": False,
            "optimizer": "AdamW",
            "optimizer_steps": 1,
            "learning_rate": 1.0e-3,
        },
        "measured_results": {
            "loss_before": _loss_values(before_losses),
            "loss_after_one_step": _loss_values(after_losses),
            "gradient_norm": gradient_norm,
            "gradient_elements": gradient_elements,
            "nonzero_gradient_elements": nonzero_gradient_elements,
            "gradients_finite": gradients_finite,
            "query_permutation_max_abs_difference": permutation_difference,
            "query_inventory_remove_one_max_abs_difference": inventory_difference,
        },
        "establishes": [
            "real canonical -> causal window -> typed records -> dynamic exact addresses",
            "one CPU forward/backward/AdamW update through a shared row-wise typed head",
            "future labels remain outside forward inputs and do not select query membership",
            "measured query permutation equivariance and remove-one inventory independence",
        ],
        "remaining_native_model_stages": [
            "graph context and rule-trace integration",
            "pretrained Qwen backbone and approved LoRA training",
            "rgb/lidar multimodal token integration",
            "autoregressive rollout and multi-episode train/validation evaluation",
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    receipt = run(args.canonical, args.receipt, seed=args.seed, threads=args.threads)
    print(json.dumps({
        "status": receipt["status"],
        "addresses": receipt["inventory"]["address_count"],
        "loss_before": receipt["measured_results"]["loss_before"]["total"],
        "loss_after": receipt["measured_results"]["loss_after_one_step"]["total"],
        "wall_seconds": receipt["runtime"]["wall_seconds"],
        "receipt": str(args.receipt),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
