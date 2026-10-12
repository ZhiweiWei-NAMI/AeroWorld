"""Prepare declared samples in isolation, then publish them serially."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from Dataset.world_model.graph.arm import materialize_arm
from Dataset.world_model.graph.contract import CONTRACT_VERSION, REPO, declared_samples
from Dataset.world_model.graph.formal import materialize_formal
from Dataset.world_model.graph.records import DEFAULT_ROOT, load_scope


def source_samples() -> tuple[list[str], list[Path]]:
    formal, arm = declared_samples()
    if len(formal) != 210 or len(arm) != 690:
        raise ValueError(f"source scope differs: formal={len(formal)}, ARM={len(arm)}")
    return formal, arm


def _planned(kind: str, samples: tuple[str, ...]) -> list[tuple[str, str, str]]:
    formal, arm = source_samples()
    planned = []
    if kind in {"all", "formal"}:
        planned.extend(("formal", episode, f"formal:{episode}") for episode in formal)
    if kind in {"all", "arm"}:
        planned.extend(("arm", str(branch), "arm:" + branch.relative_to(REPO / "arms").as_posix())
                       for branch in arm)
    if samples:
        requested = set(samples)
        missing = requested - {item[2] for item in planned}
        if missing:
            raise ValueError(f"requested samples absent from {kind} scope: {sorted(missing)}")
        planned = [item for item in planned if item[2] in requested]
    return planned


def _relative_partition(source_kind: str, label: str) -> Path:
    return Path(source_kind) / label.split(":", 1)[1]


def _prepare_graph(item: tuple[str, str, str], root: str) -> dict:
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source_kind, source, label = item
    started = time.monotonic()
    root_path = Path(root)
    if source_kind == "formal":
        materialize_formal(source, root_path)
    else:
        materialize_arm(Path(source), root_path)
    partition = root_path / _relative_partition(source_kind, label)
    scope = load_scope(partition)
    return {"sample_key": label, "prepared": str(partition), "counts": scope["counts"],
            "materialization_status": scope["materialization_status"],
            "unresolved_count": len(scope["unresolved"]),
            "seconds": round(time.monotonic() - started, 3)}


def _prepare_tensor(item: tuple[str, str, str], graph_root: str, output_root: str) -> dict:
    import resource
    import torch
    import pyarrow as pa
    from Dataset.world_model.graph.model_data import export_partition
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    started = time.monotonic()
    source_kind, _, label = item
    partition = Path(graph_root) / _relative_partition(source_kind, label)
    output = Path(output_root) / _relative_partition(source_kind, label)
    result = export_partition(partition, output)
    return {**result, "prepared": str(output), "seconds": round(time.monotonic() - started, 3),
            "worker_peak_memory_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}


def _publish(prepared: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists() and not destination.exists():
        backup.replace(destination)
    if destination.exists():
        if backup.exists():
            shutil.rmtree(backup)
        destination.replace(backup)
    prepared.replace(destination)
    if backup.exists():
        shutil.rmtree(backup)


def _save_coverage(path: Path, report: dict, completed: dict[str, dict]) -> None:
    report["materialized"] = [completed[key] for key in sorted(completed)]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tensor_memory_estimate(label: str, counts: dict, completed: dict[str, dict]) -> float:
    """Reserve memory from real formal seeds with the same producer profile."""
    estimate = (counts["nodes"] * 60 * 1024 + counts["edges"] * 512) / 1024**3
    if not label.startswith("formal:"):
        return estimate
    profile = label.rsplit("__", 1)[0]
    for row in completed.values():
        if (row["sample_key"].rsplit("__", 1)[0] != profile or
                "worker_peak_memory_mib" not in row or
                not row.get("nodes") or not row.get("edges")):
            continue
        scale = max(1., counts["nodes"] / row["nodes"], counts["edges"] / row["edges"])
        # The reported lifetime peak is at least this seed's own peak. Keep
        # 35% headroom for seed variation; the worker baseline is reserved
        # separately. Larger node and edge counts both increase the estimate.
        observed = row["worker_peak_memory_mib"] / 1024 * scale * 1.35
        estimate = min(estimate, observed)
    return estimate


def run_corpus(*, kind: str = "all", samples: tuple[str, ...] = (), workers: int = 4,
               graph_root: Path = DEFAULT_ROOT,
               tensor_root: Path = REPO / "Dataset/world_model/runs/belief_pyg",
               phase: str = "all", replace_generated: bool = False,
               tensor_memory_gib: float = 160) -> dict:
    """Checkpoint each completed sample; children never write into the worktree."""
    formal, arm = source_samples()
    planned = _planned(kind, samples)
    reports = {}
    for stage, output_root in (("graph", graph_root), ("tensor", tensor_root)):
        if phase not in {"all", stage}:
            continue
        if stage == "tensor":
            from Dataset.world_model.graph.model_data import _tensor_inputs, validate_saved_tensor
        output_root.mkdir(parents=True, exist_ok=True)
        report_path = output_root / "coverage.json"
        report = {"contract_version": CONTRACT_VERSION, "declared_formal": len(formal),
                  "declared_arm": len(arm), "stage": stage, "materialized": []}
        completed = {}
        if report_path.is_file():
            old = json.loads(report_path.read_text(encoding="utf-8"))
            if old["contract_version"] != CONTRACT_VERSION:
                raise ValueError("coverage belongs to another contract")
            completed.update((row["sample_key"], row) for row in old["materialized"])
        pending = []
        for item in planned:
            source_kind, _, label = item
            partition = output_root / _relative_partition(source_kind, label)
            marker = partition / ("scope.parquet" if stage == "graph" else "metadata.json")
            if marker.is_file() and not replace_generated:
                if stage == "graph":
                    scope = load_scope(partition)
                    if scope["sample_key"] != label or scope["contract_version"] != CONTRACT_VERSION:
                        raise ValueError(f"existing graph sample differs: {partition}")
                    completed[label] = {"sample_key": label, "counts": scope["counts"],
                        "materialization_status": scope["materialization_status"],
                        "unresolved_count": len(scope["unresolved"])}
                else:
                    metadata = json.loads(marker.read_text(encoding="utf-8"))
                    inputs = _tensor_inputs(graph_root / _relative_partition(source_kind, label),
                                           REPO / "design/belief/field_catalog.jsonl")
                    if metadata["sample_key"] != label or metadata.get("artifact_inputs") != inputs:
                        # Changes invalidate only their existing tensor outputs;
                        # source grammar scanning is not repeated.
                        pending.append(item)
                        completed.pop(label, None)
                        continue
                    if label not in completed:
                        completed[label] = {"sample_key": label, **validate_saved_tensor(partition)}
                continue
            pending.append(item)
            completed.pop(label, None)
        _save_coverage(report_path, report, completed)
        total = len(completed) + len(pending)
        failures = []
        queued = []
        # Reserve 2 GiB per worker. Unmeasured profiles use 60 KiB per node
        # plus 512 bytes per edge; completed formal seeds calibrate their
        # own profile with an additional 35% margin.
        capacity = tensor_memory_gib - 2 * workers
        sample_counts = {}
        for item in pending:
            weight = 0.
            if stage == "tensor":
                scope = load_scope(graph_root / _relative_partition(item[0], item[2]))
                sample_counts[item[2]] = scope["counts"]
                weight = _tensor_memory_estimate(item[2], scope["counts"], completed)
                if weight > capacity:
                    raise ValueError(f"tensor sample exceeds configured memory budget: "
                                     f"{item[2]} needs an estimated {weight + 2 * workers:.1f} GiB")
            queued.append((item, weight))
        # Private job directories stay with the project's generated data.
        # Only the parent publishes into the shared sample destinations.
        preparation_root = output_root / ".preparation"
        preparation_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="batch-", dir=preparation_root) as temporary:
            with ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn")) as pool:
                jobs = {}
                active_weight, ordinal = 0., 0
                while queued or jobs:
                    while queued and len(jobs) < workers:
                        eligible = next((i for i, (_, weight) in enumerate(queued)
                                         if stage != "tensor" or active_weight + weight <= capacity), None)
                        if eligible is None:
                            break
                        item, weight = queued.pop(eligible)
                        preparation_root = str(Path(temporary) / str(ordinal))
                        ordinal += 1
                        future = (pool.submit(_prepare_graph, item, preparation_root) if stage == "graph"
                                  else pool.submit(_prepare_tensor, item, str(graph_root), preparation_root))
                        jobs[future] = item, weight
                        active_weight += weight
                    done, _ = wait(jobs, return_when=FIRST_COMPLETED)
                    for future in done:
                        item, weight = jobs.pop(future)
                        active_weight -= weight
                        label = item[2]
                        try:
                            result = future.result()
                        except Exception as error:
                            failures.append({"sample_key": label, "error": f"{type(error).__name__}: {error}"})
                            print(json.dumps({"stage": stage, "failed": failures[-1]}, ensure_ascii=False), flush=True)
                            continue
                        prepared = Path(result.pop("prepared"))
                        _publish(prepared, output_root / _relative_partition(item[0], label))
                        if stage == "tensor":
                            result["admission_memory_estimate_gib"] = round(weight, 3)
                        completed[label] = result
                        _save_coverage(report_path, report, completed)
                        print(json.dumps({"stage": stage, "completed": len(completed),
                                          "total": total, **result}, ensure_ascii=False), flush=True)
                    if stage == "tensor":
                        # A first seed can calibrate the queued seeds of its
                        # profile while other admitted jobs finish unchanged.
                        queued = [(item, _tensor_memory_estimate(
                            item[2], sample_counts[item[2]], completed)) for item, _ in queued]
        report["failures"] = failures
        _save_coverage(report_path, report, completed)
        reports[stage] = {"completed": len(completed), "failed": len(failures)}
        if failures:
            raise ValueError(f"{stage} failed for {len(failures)} samples; details: {report_path}")
        if stage == "graph" and phase == "all":
            from Dataset.world_model.graph.vocabulary import build_vocabulary
            print(json.dumps({"vocabulary": build_vocabulary(root=graph_root)}), flush=True)
    return reports


def run(kind: str = "all", *, samples: tuple[str, ...] = (),
        replace_generated: bool = False, output_root: Path = DEFAULT_ROOT) -> dict:
    formal, arm = source_samples()
    planned: list[tuple[str, str | Path]] = []
    if kind in {"all", "formal"}:
        planned.extend(("formal", episode) for episode in formal)
    if kind in {"all", "arm"}:
        planned.extend(("arm", branch) for branch in arm)
    if samples:
        requested = set(samples)
        available = {f"formal:{sample}" if source_kind == "formal" else
                     "arm:" + Path(sample).relative_to(REPO / "arms").as_posix()
                     for source_kind, sample in planned}
        missing = requested - available
        if missing:
            raise ValueError(f"requested samples absent from {kind} scope: {sorted(missing)}")
        planned = [(source_kind, sample) for source_kind, sample in planned
                   if (f"formal:{sample}" if source_kind == "formal" else
                       "arm:" + Path(sample).relative_to(REPO / "arms").as_posix()) in requested]
    report = {"contract_version": CONTRACT_VERSION, "declared_formal": len(formal),
              "declared_arm": len(arm), "requested_kind": kind,
              "materialized": []}
    report_path = output_root / "coverage.json"
    output_root.mkdir(parents=True, exist_ok=True)
    for source_kind, sample in planned:
        if source_kind == "formal":
            episode = str(sample)
            partition = output_root / "formal" / episode
            label = f"formal:{episode}"
        else:
            branch = Path(sample)
            partition = output_root / "arm" / branch.relative_to(REPO / "arms")
            label = "arm:" + branch.relative_to(REPO / "arms").as_posix()
        print(f"START {label} {partition}", flush=True)
        scope_path = partition / "scope.parquet"
        if scope_path.exists() and not replace_generated:
            scope = load_scope(partition)
            if scope["contract_version"] != CONTRACT_VERSION or scope["sample_key"] != label:
                raise ValueError("existing partition differs from current contract/sample")
        else:
            if source_kind == "formal":
                materialize_formal(str(sample), output_root, replace_generated=replace_generated)
            else:
                materialize_arm(Path(sample), output_root, replace_generated=replace_generated)
            scope = load_scope(partition)
        report["materialized"].append({"sample_key": label, "counts": scope["counts"],
                                       "materialization_status": scope["materialization_status"],
                                       "unresolved_count": len(scope["unresolved"])})
        print(f"{len(report['materialized'])}/{len(planned)} {label} {scope['counts']}", flush=True)
        temporary = report_path.with_name("coverage.json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)
    return {"materialized": len(report["materialized"]), "failed": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("all", "formal", "arm"), default="all")
    parser.add_argument("--sample", action="append", default=[],
                        help="exact formal:<episode> or arm:<profile>/<seed>/<group>/<branch> sample key")
    parser.add_argument("--replace-generated", action="store_true")
    parser.add_argument("--corpus", action="store_true", help="prepare samples in isolated worker processes")
    parser.add_argument("--phase", choices=("all", "graph", "tensor"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tensor-memory-gib", type=float, default=160,
                        help="estimated aggregate tensor worker memory budget")
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tensor-root", type=Path, default=REPO / "Dataset/world_model/runs/belief_pyg")
    args = parser.parse_args()
    if args.corpus:
        print(run_corpus(kind=args.kind, samples=tuple(args.sample), workers=args.workers,
                         phase=args.phase, graph_root=args.graph_root, tensor_root=args.tensor_root,
                         tensor_memory_gib=args.tensor_memory_gib,
                         replace_generated=args.replace_generated))
    else:
        print(run(args.kind, samples=tuple(args.sample), replace_generated=args.replace_generated,
                  output_root=args.graph_root))
