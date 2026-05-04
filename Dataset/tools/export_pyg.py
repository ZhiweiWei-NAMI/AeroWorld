"""
PyG HeteroData 导出工具。

将 frame_graphs.jsonl 转换为 PyTorch Geometric HeteroData 对象。

使用方式:
    python export_pyg.py --episode-dir Dataset/episodes/L4-5_v1__seed00/
    python export_pyg.py --dataset-dir Dataset/episodes/ --output-dir Dataset/pyg_export/
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path


def parse_edge_type(edge_key: str) -> tuple[str, str, str]:
    """
    解析边类型: "spatial:near" → ("uav", "near", "uav")
    注意: 实际 src/dst node type 需要在构建图时从节点列表推导，
    这里返回关系名，调用者负责确定 src/dst node types。
    """
    parts = edge_key.split(":", 1)
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return edge_key, "", ""


def build_node_index(frame_graph: dict) -> dict[str, tuple[str, int]]:
    """
    构建 (node_id → (node_type, index)) 映射。
    """
    index: dict[str, tuple[str, int]] = {}
    for ntype, nodes in frame_graph.get("nodes", {}).items():
        for i, node in enumerate(nodes):
            node_id = node["node_id"]
            index[node_id] = (ntype, i)
    return index


def frame_graph_to_pyg_dict(frame_graph: dict) -> dict:
    """
    将一帧的 JSON 图标签转换为 PyG HeteroData 兼容的 dict。

    返回 dict 包含:
      - node_types: {type: {attr: tensor_like_list}}
      - edge_types: {(src_type, rel, dst_type): {edge_index: [[src_idx, dst_idx], ...], edge_attr: {attr: [...]}}}
    """
    node_types: dict[str, dict[str, list]] = {}
    node_index: dict[str, tuple[str, int]] = {}

    # 构建节点特征
    for ntype, nodes in frame_graph.get("nodes", {}).items():
        if ntype not in node_types:
            node_types[ntype] = {}

        for i, node in enumerate(nodes):
            nid = node.get("node_id", "")
            node_index[nid] = (ntype, i)

            # 收集数值特征
            for key, val in node.items():
                if key in ("node_id", "type", "status", "phase", "topic",
                          "lane_id", "mode", "state", "severity", "mission_phase"):
                    continue  # 分类特征单独处理或跳过
                if isinstance(val, (int, float)):
                    if key not in node_types[ntype]:
                        node_types[ntype][key] = []
                    # 确保所有节点此属性长度一致 (填充缺失)
                    while len(node_types[ntype][key]) < i:
                        node_types[ntype][key].append(0.0)
                    node_types[ntype][key].append(float(val))

            # position 特殊处理: 展开为 pos_x, pos_y, pos_z
            pos = node.get("pos_enu", [0.0, 0.0, 0.0])
            if not isinstance(pos, list) or len(pos) < 3:
                pos = [0.0, 0.0, 0.0]
            for axis, pval in zip(["x", "y", "z"], pos):
                pkey = f"pos_{axis}"
                if pkey not in node_types[ntype]:
                    node_types[ntype][pkey] = []
                while len(node_types[ntype][pkey]) < i:
                    node_types[ntype][pkey].append(0.0)
                node_types[ntype][pkey].append(float(pval))

            # velocity 特殊处理
            vel = node.get("vel_mps", [0.0, 0.0, 0.0])
            if not isinstance(vel, list) or len(vel) < 3:
                vel = [0.0, 0.0, 0.0]
            for axis, vval in zip(["x", "y", "z"], vel):
                vkey = f"vel_{axis}"
                if vkey not in node_types[ntype]:
                    node_types[ntype][vkey] = []
                while len(node_types[ntype][vkey]) < i:
                    node_types[ntype][vkey].append(0.0)
                node_types[ntype][vkey].append(float(vval))

    # 构建边
    edge_types: dict[tuple[str, str, str], dict] = {}

    for edge_key, edges in frame_graph.get("edges", {}).items():
        for edge in edges:
            src = edge.get("src", "")
            dst = edge.get("dst", "")
            if src not in node_index or dst not in node_index:
                continue

            src_type, src_idx = node_index[src]
            dst_type, dst_idx = node_index[dst]

            # Edge relation: spatial:near → "near", causal:triggers → "triggers"
            rel = edge_key.split(":", 1)[1] if ":" in edge_key else edge_key

            etype = (src_type, rel, dst_type)
            if etype not in edge_types:
                edge_types[etype] = {"edge_index": [], "edge_attr": {}}

            edge_types[etype]["edge_index"].append([src_idx, dst_idx])

            # 收集边属性
            for attr_key, attr_val in edge.items():
                if attr_key in ("src", "dst"):
                    continue
                if isinstance(attr_val, (int, float)):
                    if attr_key not in edge_types[etype]["edge_attr"]:
                        edge_types[etype]["edge_attr"][attr_key] = []
                    edge_types[etype]["edge_attr"][attr_key].append(float(attr_val))

    # 转换为 tensor 格式的 dict (无实际 torch 依赖，输出为列表)
    result: dict = {
        "node_types": node_types,
        "edge_types": {
            f"{src}__{rel}__{dst}": {
                "edge_index": data["edge_index"],
                "edge_attr": data["edge_attr"],
            }
            for (src, rel, dst), data in edge_types.items()
        },
    }

    return result


def episode_graphs_to_pyg_list(episode_dir: Path, strict: bool = False) -> list[dict]:
    """加载整个 episode 的 frame_graphs.jsonl 并转换为 PyG dict 列表。"""
    graphs_path = episode_dir / "graphs" / "frame_graphs.jsonl"
    if not graphs_path.exists():
        # Try alternative location
        graphs_path = episode_dir / "frame_graphs.jsonl"

    if not graphs_path.exists():
        print(f"WARNING: no frame_graphs.jsonl found in {episode_dir}", file=sys.stderr)
        return []

    frames = []
    skipped = 0
    with open(graphs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    if strict:
                        raise
                    skipped += 1

    if skipped:
        print(f"WARNING: skipped {skipped} corrupted frames in {graphs_path}", file=sys.stderr)

    pyg_list = []
    conversion_skipped = 0
    for fg in frames:
        try:
            pyg_dict = frame_graph_to_pyg_dict(fg)
            pyg_dict["frame_id"] = fg.get("frame_id", "")
            pyg_dict["tick"] = fg.get("tick", 0)
            pyg_dict["episode_id"] = fg.get("episode_id", "")
            pyg_list.append(pyg_dict)
        except Exception:
            if strict:
                raise
            conversion_skipped += 1

    if conversion_skipped:
        print(f"WARNING: skipped {conversion_skipped} corrupted frames", file=sys.stderr)

    return pyg_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frame graphs to PyG format")
    parser.add_argument("--episode-dir", default=None, help="Single episode directory")
    parser.add_argument("--dataset-dir", default=None, help="Dataset episodes directory (batch)")
    parser.add_argument("--output-dir", default="Dataset/pyg_export", help="Output directory")
    parser.add_argument("--single-file", action="store_true", help="Write all frames to a single JSON file")
    parser.add_argument("--strict", action="store_true", help="Abort on the first corrupted frame or episode")
    args = parser.parse_args()

    if not args.episode_dir and not args.dataset_dir:
        print("ERROR: specify --episode-dir or --dataset-dir", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.episode_dir:
        # Single episode
        episode_dir = Path(args.episode_dir).resolve()
        pyg_list = episode_graphs_to_pyg_list(episode_dir, strict=args.strict)
        out_path = output_dir / f"{episode_dir.name}_pyg.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pyg_list, f, ensure_ascii=False)
        print(f"Exported {len(pyg_list)} frames → {out_path}")

    elif args.dataset_dir:
        # All episodes
        dataset_dir = Path(args.dataset_dir).resolve()
        all_data = []
        for episode_dir in sorted(dataset_dir.iterdir()):
            if episode_dir.is_dir():
                try:
                    pyg_list = episode_graphs_to_pyg_list(episode_dir, strict=args.strict)
                except Exception as exc:
                    if args.strict:
                        raise
                    print(f"WARNING: skipping {episode_dir}: {exc}", file=sys.stderr)
                    continue
                if pyg_list:
                    all_data.extend(pyg_list)
                    print(f"  {episode_dir.name}: {len(pyg_list)} frames")

        if args.single_file:
            out_path = output_dir / "all_frames_pyg.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False)
            print(f"Exported {len(all_data)} total frames → {out_path}")
        else:
            print(f"Processed {len(all_data)} frames (use --single-file to export)")


if __name__ == "__main__":
    main()
