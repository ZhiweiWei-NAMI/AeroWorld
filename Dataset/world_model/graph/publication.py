"""Relocate saved tensor provenance when publishing prepared graph partitions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from Dataset.world_model.graph.contract import CONTRACT_VERSION, REPO


def relocate_tensor(output: Path, *, previous_graph_root: Path, graph_root: Path) -> bool:
    """Change only provenance paths; retain tensor values and record indices.

    The source graph files must be moved without rewriting them. Verify their
    saved sizes and modification times before changing references. A restart
    can finish the JSON write if the tensor was already replaced.
    """
    previous_graph_root, graph_root = previous_graph_root.resolve(), graph_root.resolve()
    old = previous_graph_root.relative_to(REPO).as_posix() + "/"
    new = graph_root.relative_to(REPO).as_posix() + "/"
    marker = output / "metadata.json"
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"tensor version differs: {output}")

    def relocate(value: dict) -> dict:
        result = dict(value)
        source = result["source_graph"]
        if not source.startswith((old, new)):
            raise ValueError(f"tensor source outside publication roots: {output}")
        if source.startswith(old):
            result["source_graph"] = new + source[len(old):]
        result["artifact_inputs"] = [
            {**item, "path": new + item["path"][len(old):]}
            if item["path"].startswith(old) else dict(item)
            for item in result["artifact_inputs"]]
        return result

    updated = relocate(metadata)
    if updated == metadata:
        return False
    for item in metadata["artifact_inputs"]:
        path = REPO / item["path"]
        if item["path"].startswith(old) and not path.exists():
            path = REPO / (new + item["path"][len(old):])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
            raise ValueError(f"tensor input changed before publication: {path}")
    tensor_path = output / "target_graph.pt"
    data = torch.load(tensor_path, map_location="cpu", weights_only=False, mmap=True)
    tensor_metadata = relocate(data.graph_metadata)
    # External resume stamps can have been refreshed after proving that an
    # append-only field catalog change did not affect this sample's values.
    if ({k: v for k, v in tensor_metadata.items() if k != "artifact_inputs"} !=
            {k: v for k, v in updated.items() if k != "artifact_inputs"}):
        raise ValueError(f"saved tensor metadata differs: {output}")
    if data.num_nodes != updated["nodes"] or data.num_edges != updated["edges"]:
        raise ValueError(f"saved tensor dimensions differ: {output}")
    data.graph_metadata = updated
    temporary = output / "target_graph.pt.publishing"
    torch.save(data, temporary)
    temporary.replace(tensor_path)
    del data
    gc.collect()
    temporary_marker = marker.with_suffix(".json.publishing")
    temporary_marker.write_text(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n",
                                encoding="utf-8")
    temporary_marker.replace(marker)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--previous-graph-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    partitions = sorted(p.parent for p in args.tensor_root.rglob("metadata.json")
                        if ".preparation" not in p.relative_to(args.tensor_root).parts)
    changed = 0
    for index, partition in enumerate(partitions, 1):
        changed += relocate_tensor(partition, previous_graph_root=args.previous_graph_root,
                                   graph_root=args.graph_root)
        print(json.dumps({"publication_paths_completed": index, "requested": len(partitions),
                          "rewritten": changed, "partition": str(partition)}), flush=True)


if __name__ == "__main__":
    main()
