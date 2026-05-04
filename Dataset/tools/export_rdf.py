"""
RDF 知识图谱导出工具。

将 frame_graphs.jsonl 转换为 RDF NTriples 格式，与 ontology.ttl 的 TBox 对齐。

使用方式:
    python export_rdf.py --episode-dir Dataset/episodes/L4-5_v1__seed00/
    python export_rdf.py --dataset-dir Dataset/episodes/ --output merged_abox.nt
"""

from __future__ import annotations
import json, sys, argparse
from pathlib import Path

# Base namespace
NS = "http://uam-dataset.org/"

# 类映射: frame_graph node_type → OWL class
NODE_TYPE_TO_CLASS = {
    "uav": "ontology/UAV",
    "ground_vehicle": "ontology/GroundVehicle",
    "pedestrian": "ontology/Pedestrian",
    "infrastructure": "ontology/Infrastructure",
    "zone": "ontology/Zone",
    "weather": "ontology/EnvironmentalCondition",
    "event": "ontology/Event",
}

# 边映射: frame_graph edge_key → OWL object property
EDGE_TO_PROPERTY = {
    "spatial:near": "ontology/spatial_near",
    "spatial:approaching": "ontology/spatial_approaching",
    "spatial:inside": "ontology/spatial_inside",
    "spatial:on": "ontology/spatial_on",
    "causal:triggers": "ontology/causal_triggers",
    "causal:prevents": "ontology/causal_prevents",
    "causal:amplifies": "ontology/causal_amplifies",
    "temporal:before": "ontology/temporal_before",
    "temporal:overlaps": "ontology/temporal_overlaps",
    "temporal:during": "ontology/temporal_during",
    "operational:controls": "ontology/operational_controls",
    "operational:monitors": "ontology/operational_monitors",
    "operational:assigned_to": "ontology/operational_assigned_to",
    "operational:responds_to": "ontology/operational_responds_to",
}


def escape_ntriples(s: str) -> str:
    """转义 NTriples 字符串。"""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    return s


def node_to_uri(node_id: str, episode_id: str, tick: int = 0,
                node_type: str = "Agent") -> str:
    """构建实体的 RDF URI。"""
    safe_id = node_id.replace(" ", "_").replace("/", "_")
    return f"<{NS}{episode_id}/tick/{tick}/entity/{safe_id}>"


def event_to_uri(event_id: str, episode_id: str) -> str:
    """构建事件的 RDF URI。"""
    safe_id = event_id.replace(" ", "_").replace("/", "_")
    return f"<{NS}{episode_id}/event/{safe_id}>"


def class_uri(node_type: str) -> str:
    """获取节点类型对应的 OWL class URI。"""
    cls = NODE_TYPE_TO_CLASS.get(node_type, "ontology/Agent")
    return f"<{NS}{cls}>"


def property_uri(edge_key: str) -> str:
    """获取边类型对应的 OWL property URI。"""
    prop = EDGE_TO_PROPERTY.get(edge_key, f"ontology/{edge_key}")
    return f"<{NS}{prop}>"


def frame_graph_to_triples(frame_graph: dict) -> list[str]:
    """将一帧图标签转换为 RDF triples (NTriples 格式)。"""
    episode_id = frame_graph.get("episode_id", "unknown")
    tick = frame_graph.get("tick", 0)
    triples = []

    # 节点 → rdf:type 声明
    for ntype, nodes in frame_graph.get("nodes", {}).items():
        for node in nodes:
            nid = node.get("node_id", "")
            if ntype == "event":
                s = event_to_uri(nid, episode_id)
            else:
                s = node_to_uri(nid, episode_id, tick, ntype)
            p = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
            o = class_uri(ntype)
            triples.append(f"{s} {p} {o} .")

            # 数值属性: has_state, has_position_x, 等
            state = node.get("state", "")
            if state:
                triples.append(
                    f'{s} <{NS}ontology/has_state> "{escape_ntriples(str(state))}" .')

            pos = node.get("pos_enu", [])
            for i, axis in enumerate(["x", "y", "z"]):
                if i < len(pos):
                    triples.append(
                        f'{s} <{NS}ontology/has_position_{axis}> "{pos[i]}"^^<http://www.w3.org/2001/XMLSchema#float> .')

            # severity (for events)
            if ntype == "event":
                sev = node.get("severity", "")
                if sev:
                    triples.append(
                        f'{s} <{NS}ontology/has_severity> "{escape_ntriples(str(sev))}" .')

    # 边 → object property 断言
    for edge_key, edges in frame_graph.get("edges", {}).items():
        prop = property_uri(edge_key)
        for edge in edges:
            src = edge.get("src", "")
            dst = edge.get("dst", "")

            # 判断 src/dst 是 event 还是 entity
            nodes_dict = frame_graph.get("nodes", {})
            src_is_event = any(
                n.get("node_id") == src for n in nodes_dict.get("event", []))
            dst_is_event = any(
                n.get("node_id") == dst for n in nodes_dict.get("event", []))

            if src_is_event:
                s = event_to_uri(src, episode_id)
            else:
                s = node_to_uri(src, episode_id, tick)

            if dst_is_event:
                o = event_to_uri(dst, episode_id)
            else:
                o = node_to_uri(dst, episode_id, tick)

            triples.append(f"{s} {prop} {o} .")

    return triples


def episode_to_ntriples(episode_dir: Path, strict: bool = False) -> str:
    """将整个 episode 导出为 NTriples 字符串。"""
    graphs_path = episode_dir / "graphs" / "frame_graphs.jsonl"
    if not graphs_path.exists():
        graphs_path = episode_dir / "frame_graphs.jsonl"
    if not graphs_path.exists():
        print(f"WARNING: no frame_graphs.jsonl in {episode_dir}", file=sys.stderr)
        return ""

    all_triples = []
    skipped = 0
    with open(graphs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                fg = json.loads(line)
                triples = frame_graph_to_triples(fg)
                all_triples.extend(triples)
            except Exception:
                if strict:
                    raise
                skipped += 1

    if skipped:
        print(f"WARNING: skipped {skipped} corrupted frames in {graphs_path}", file=sys.stderr)

    return "\n".join(all_triples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frame graphs to RDF NTriples")
    parser.add_argument("--episode-dir", default=None, help="Single episode directory")
    parser.add_argument("--dataset-dir", default=None, help="Dataset episodes directory (batch merge)")
    parser.add_argument("--output", default="merged_abox.nt", help="Output file")
    parser.add_argument("--per-episode", action="store_true", help="Write per-episode files")
    parser.add_argument("--strict", action="store_true", help="Abort on the first corrupted frame or episode")
    args = parser.parse_args()

    if not args.episode_dir and not args.dataset_dir:
        print("ERROR: specify --episode-dir or --dataset-dir", file=sys.stderr)
        sys.exit(1)

    if args.episode_dir:
        episode_dir = Path(args.episode_dir).resolve()
        ntriples = episode_to_ntriples(episode_dir, strict=args.strict)
        out_path = Path(args.output).resolve()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(ntriples)
        n_lines = ntriples.count("\n") + 1 if ntriples else 0
        print(f"Exported {n_lines} triples → {out_path}")

    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).resolve()
        out_path = Path(args.output).resolve()
        all_ntriples = []
        total = 0

        for episode_dir in sorted(dataset_dir.iterdir()):
            if episode_dir.is_dir():
                try:
                    ntriples = episode_to_ntriples(episode_dir, strict=args.strict)
                except Exception as exc:
                    if args.strict:
                        raise
                    print(f"WARNING: skipping {episode_dir}: {exc}", file=sys.stderr)
                    continue
                if ntriples:
                    total += ntriples.count("\n") + 1
                    if args.per_episode:
                        ep_out = episode_dir / "graphs" / "episode_graph.nt"
                        with open(ep_out, "w", encoding="utf-8") as f:
                            f.write(ntriples)
                        print(f"  {episode_dir.name}: {ntriples.count(chr(10)) + 1} triples → {ep_out}")
                    all_ntriples.append(ntriples)

        if not args.per_episode:
            merged = "\n".join(all_ntriples)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(merged)
            print(f"Merged {total} triples from {len(all_ntriples)} episodes → {out_path}")
        else:
            print(f"Exported {total} total triples across episodes")


if __name__ == "__main__":
    main()
