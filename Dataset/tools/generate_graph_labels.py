"""
图标签生成器。

从 event_trace.jsonl + trajectories.jsonl + global_entity_roster.json 生成
每帧的异构图标签 (frame_graphs.jsonl) 和 episode 级 RDF 图。

使用方式:
    python generate_graph_labels.py --episode-dir Dataset/episodes/L4-5_v1__seed00/
"""

from __future__ import annotations
import json, sys, argparse, math
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Spatial thresholds for edge creation
SPATIAL_NEAR_THRESHOLD_M = 50.0      # entities within 50m get spatial:near edge
SPATIAL_APPROACHING_THRESHOLD_M = 30.0  # entities within 30m AND closing get spatial:approaching
SPATIAL_INSIDE_TOLERANCE_M = 2.0     # tolerance for spatial:inside check


def load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL 文件。"""
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Optional[dict]:
    """加载 JSON 文件。"""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_distance(pos_a: list[float], pos_b: list[float]) -> float:
    """计算两点间欧几里得距离。"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)))


def is_inside_bbox(pos: list[float], bbox_min: list[float], bbox_max: list[float]) -> bool:
    """检查点是否在 bbox 内。"""
    for i in range(min(len(pos), len(bbox_min), len(bbox_max))):
        if pos[i] < bbox_min[i] - SPATIAL_INSIDE_TOLERANCE_M:
            return False
        if pos[i] > bbox_max[i] + SPATIAL_INSIDE_TOLERANCE_M:
            return False
    return True


def is_closing(pos_a: list[float], vel_a: list[float],
               pos_b: list[float], vel_b: list[float]) -> bool:
    """检查两实体是否正在靠近 (点积 < 0)。"""
    rel_pos = [pos_b[i] - pos_a[i] for i in range(3)]
    rel_vel = [vel_b[i] - vel_a[i] for i in range(min(3, len(vel_a), len(vel_b)))]
    dot = sum(rp * rv for rp, rv in zip(rel_pos, rel_vel))
    return dot < 0


def build_node_id(entity_id: str, entity_type: str, tick: int) -> str:
    """构建 frame-level 节点 ID。"""
    return f"{entity_id}"


def classify_node_type(entity_id: str, label_class: str,
                       entity_roster: dict) -> str:
    """
    将 label_class 映射到图节点类型。

    图节点类型: uav, ground_vehicle, pedestrian, infrastructure, zone, weather, event
    """
    roster_entry = entity_roster.get(entity_id, {})
    asset_id = roster_entry.get("asset_id", "")

    if label_class in ("uav",):
        return "uav"
    elif label_class in ("vehicle",):
        return "ground_vehicle"
    elif label_class in ("pedestrian",):
        return "pedestrian"
    elif label_class in ("landing_pad", "charging_pile", "radio_tower",
                          "traffic_light", "barrier", "construction_fence",
                          "police_sign", "traffic_cone", "backpack",
                          "delivery_bag", "phone", "umbrella"):
        return "infrastructure"
    elif label_class in ("no_fly_zone", "hazard_zone", "spawn_zone", "asset_anchor"):
        return "zone"
    elif label_class in ("camera",):
        return "infrastructure"
    else:
        return "ground_vehicle"  # fallback


def classify_zone_type(zone_id: str, entity_roster: dict) -> str:
    """推断 zone 的类型。"""
    entry = entity_roster.get(zone_id, {})
    asset_id = entry.get("asset_id", "")
    if "no_fly" in asset_id:
        return "no_fly_zone"
    elif "hazard" in asset_id:
        return "hazard_zone"
    elif "spawn" in asset_id:
        return "spawn_zone"
    return "zone"


def generate_frame_graphs(episode_dir: Path) -> tuple[list[dict], dict]:
    """
    主函数：生成所有帧的图标签。

    返回: (frame_graphs_list, episode_level_summary)
    """
    # 加载数据
    event_trace = load_jsonl(episode_dir / "event_trace.jsonl")
    trajectories = load_jsonl(episode_dir / "trajectories.jsonl")
    roster = load_json(episode_dir / "global_entity_roster.json") or {}
    weather_data = load_jsonl(episode_dir / "weather_meta.jsonl")

    # 索引: tick → events
    events_by_tick: dict[int, list[dict]] = defaultdict(list)
    for evt in event_trace:
        tick = evt.get("tick", evt.get("source_tick", 0))
        events_by_tick[tick].append(evt)

    # 索引: tick → entity states
    entities_by_tick: dict[int, dict[str, dict]] = defaultdict(dict)
    for traj in trajectories:
        tick = traj.get("tick", 0)
        entity_id = traj.get("entity_id", traj.get("id", ""))
        if entity_id:
            entities_by_tick[tick][entity_id] = traj

    # 索引: tick → weather
    weather_by_tick: dict[int, dict] = {}
    for w in weather_data:
        tick = w.get("tick", 0)
        weather_by_tick[tick] = w

    # 收集所有 tick
    all_ticks = sorted(set(
        list(entities_by_tick.keys()) +
        list(events_by_tick.keys()) +
        list(weather_by_tick.keys())
    ))

    if not all_ticks:
        print("WARNING: no data found in episode directory")
        return [], {}

    # 推断 entities_roster 中实体的类型
    entity_types: dict[str, str] = {}
    for entity_id, entry in roster.items():
        label_class = entry.get("label_class", "")
        entity_types[entity_id] = classify_node_type(entity_id, label_class, roster)

    # 同时从 trajectory 数据中推断
    for tick_data in entities_by_tick.values():
        for entity_id, traj in tick_data.items():
            if entity_id not in entity_types:
                label_class = traj.get("label_class", traj.get("class", ""))
                entity_types[entity_id] = classify_node_type(entity_id, label_class, {})

    # zone 类型
    zone_types: dict[str, str] = {}
    for entity_id, ntype in entity_types.items():
        if ntype == "zone":
            zone_types[entity_id] = classify_zone_type(entity_id, roster)

    frame_graphs = []
    prev_entity_positions: dict[str, list[float]] = {}

    for tick in all_ticks:
        frame = {
            "frame_id": f"tick:{tick:06d}",
            "tick": tick,
            "episode_id": episode_dir.name,
            "nodes": {
                "uav": [],
                "ground_vehicle": [],
                "pedestrian": [],
                "infrastructure": [],
                "zone": [],
                "weather": [],
                "event": [],
            },
            "edges": {
                "spatial:near": [],
                "spatial:approaching": [],
                "spatial:inside": [],
                "causal:triggers": [],
                "temporal:before": [],
            },
            "scene_context": {},
        }

        # --- 实体节点 ---
        tick_entities = entities_by_tick.get(tick, {})
        current_positions: dict[str, list[float]] = {}

        for entity_id, traj in tick_entities.items():
            ntype = entity_types.get(entity_id, "ground_vehicle")
            pos = traj.get("pos_enu", traj.get("position_enu_m",
                    traj.get("position", [0.0, 0.0, 0.0])))
            vel = traj.get("vel_mps", traj.get("velocity_enu_mps",
                    traj.get("velocity", [0.0, 0.0, 0.0])))
            state = traj.get("state", traj.get("status", "unknown"))

            current_positions[entity_id] = pos

            node = {
                "node_id": entity_id,
                "pos_enu": pos,
                "vel_mps": vel,
                "state": state,
            }

            # Add type-specific fields
            if ntype == "uav":
                node["battery_pct"] = traj.get("battery_pct", 100.0)
                node["mission_phase"] = traj.get("mission_phase", "cruise")
            elif ntype == "ground_vehicle":
                node["lane_id"] = traj.get("lane_id", "")
            elif ntype == "infrastructure":
                node["type"] = traj.get("label_class", traj.get("class", ""))
                node["status"] = traj.get("status", "active")
            elif ntype == "zone":
                node["type"] = zone_types.get(entity_id, "zone")
                node["status"] = traj.get("status", "active")
                bbox = traj.get("bbox", traj.get("extent", None))
                if bbox:
                    if isinstance(bbox, list) and len(bbox) == 2:
                        node["bbox_min"] = bbox[0]
                        node["bbox_max"] = bbox[1]
                    elif isinstance(bbox, dict):
                        node["bbox_min"] = bbox.get("min", [0, 0, 0])
                        node["bbox_max"] = bbox.get("max", [0, 0, 0])

            frame["nodes"][ntype].append(node)

        # --- 天气节点 ---
        weather = weather_by_tick.get(tick, {})
        if weather:
            frame["nodes"]["weather"].append({
                "node_id": "weather",
                "rain": weather.get("rain", 0.0),
                "fog": weather.get("fog", 0.0),
                "wind": weather.get("wind_speed", weather.get("wind", 0.0)),
                "visibility_m": weather.get("visibility_m", 20000.0),
                "temperature_c": weather.get("temperature_c", 20.0),
            })

        # --- 事件节点 ---
        tick_events = events_by_tick.get(tick, [])
        for evt in tick_events:
            event_node = {
                "node_id": evt.get("source_event_id", evt.get("instance_id", f"evt_{tick}")),
                "type": evt.get("category", "unknown"),
                "severity": evt.get("render_hints", {}).get("severity", "info"),
                "phase": evt.get("status", "active"),
                "topic": evt.get("topic", ""),
            }
            frame["nodes"]["event"].append(event_node)

        # --- Spatial edges ---
        all_agent_nodes = []
        for ntype in ["uav", "ground_vehicle", "pedestrian"]:
            for node in frame["nodes"][ntype]:
                all_agent_nodes.append((ntype, node))

        for i in range(len(all_agent_nodes)):
            ntype_a, node_a = all_agent_nodes[i]
            pos_a = node_a["pos_enu"]
            vel_a = node_a.get("vel_mps", [0, 0, 0])

            for j in range(i + 1, len(all_agent_nodes)):
                ntype_b, node_b = all_agent_nodes[j]
                pos_b = node_b["pos_enu"]
                vel_b = node_b.get("vel_mps", [0, 0, 0])

                dist = compute_distance(pos_a, pos_b)

                if dist < SPATIAL_NEAR_THRESHOLD_M:
                    frame["edges"]["spatial:near"].append({
                        "src": node_a["node_id"],
                        "dst": node_b["node_id"],
                        "distance_m": round(dist, 2),
                    })

                if dist < SPATIAL_APPROACHING_THRESHOLD_M:
                    if is_closing(pos_a, vel_a, pos_b, vel_b):
                        frame["edges"]["spatial:approaching"].append({
                            "src": node_a["node_id"],
                            "dst": node_b["node_id"],
                            "distance_m": round(dist, 2),
                        })

        # --- spatial:inside edges (agent in zone) ---
        for ntype in ["uav", "ground_vehicle", "pedestrian"]:
            for agent_node in frame["nodes"][ntype]:
                pos = agent_node["pos_enu"]
                for zone_node in frame["nodes"]["zone"]:
                    bbox_min = zone_node.get("bbox_min")
                    bbox_max = zone_node.get("bbox_max")
                    if bbox_min and bbox_max:
                        if is_inside_bbox(pos, bbox_min, bbox_max):
                            frame["edges"]["spatial:inside"].append({
                                "src": agent_node["node_id"],
                                "dst": zone_node["node_id"],
                            })

        # --- Causal edges (from event_trace parent_event_id) ---
        for evt in tick_events:
            parent = evt.get("parent_event_id", "")
            if parent:
                parent_evt_id = parent
                current_evt_id = evt.get("source_event_id", "")
                if current_evt_id and parent_evt_id:
                    frame["edges"]["causal:triggers"].append({
                        "src": parent_evt_id,
                        "dst": current_evt_id,
                    })

        # --- Temporal edges (events in same frame that are causally linked) ---
        event_ids_in_frame = [e["node_id"] for e in frame["nodes"]["event"]]
        for i in range(len(event_ids_in_frame)):
            for j in range(i + 1, len(event_ids_in_frame)):
                # Check if there's already a causal link
                already_linked = any(
                    e["src"] == event_ids_in_frame[i] and e["dst"] == event_ids_in_frame[j]
                    for e in frame["edges"]["causal:triggers"]
                )
                if already_linked:
                    frame["edges"]["temporal:before"].append({
                        "src": event_ids_in_frame[i],
                        "dst": event_ids_in_frame[j],
                    })

        # --- Scene context ---
        frame["scene_context"] = {
            "tick": tick,
            "weather_profile": weather.get("profile", "clear") if weather else "clear",
            "n_agents": sum(len(frame["nodes"][t]) for t in ["uav", "ground_vehicle", "pedestrian"]),
            "n_active_events": len(tick_events),
        }

        # Remove empty edge types
        frame["edges"] = {k: v for k, v in frame["edges"].items() if v}
        # Remove empty node types
        frame["nodes"] = {k: v for k, v in frame["nodes"].items() if v}

        frame_graphs.append(frame)
        prev_entity_positions = current_positions

    # Episode-level summary
    summary = {
        "episode_id": episode_dir.name,
        "n_frames": len(frame_graphs),
        "n_events_total": len(event_trace),
        "n_unique_entities": len(entity_types),
        "entity_types": entity_types,
        "zone_types": zone_types,
    }

    return frame_graphs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frame-level graph labels")
    parser.add_argument("--episode-dir", required=True, help="Path to episode directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: episode_dir/graphs/)")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir).resolve()
    if not episode_dir.exists():
        print(f"ERROR: episode directory not found: {episode_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else episode_dir / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {episode_dir}")
    frame_graphs, summary = generate_frame_graphs(episode_dir)

    # Write frame_graphs.jsonl
    output_path = output_dir / "frame_graphs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for fg in frame_graphs:
            f.write(json.dumps(fg, ensure_ascii=False) + "\n")

    # Write summary
    summary_path = output_dir / "graph_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(frame_graphs)} frame graphs")
    print(f"  → {output_path}")
    print(f"  → {summary_path}")
    print(f"Summary: {summary['n_frames']} frames, "
          f"{summary['n_events_total']} events, "
          f"{summary['n_unique_entities']} unique entities")


if __name__ == "__main__":
    main()
