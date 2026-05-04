"""
批量 episode 生成器。

遍历 scenarios/ 下所有 event_script.json，使用 EventScriptInterpreter (mock handler 模式)
生成 event_trace.jsonl 和合成 trajectories/weather 数据，然后调用 generate_graph_labels.py
生成图标签。

使用方式:
    python batch_generate.py --dataset-root Dataset/ --seeds 2
"""

from __future__ import annotations
import json, sys, argparse, random
from pathlib import Path


def _resolve_param_value(value: object, params: dict) -> object:
    if isinstance(value, str) and value.startswith("$param."):
        return params.get(value[len("$param."):], value)
    if isinstance(value, list):
        return [_resolve_param_value(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_param_value(v, params) for k, v in value.items()}
    return value


def _infer_label_class(entity_id: str, asset_id: str = "") -> str:
    text = f"{entity_id} {asset_id}".lower()
    if "uav" in text or "drone" in text:
        return "uav"
    if "ped" in text or "crowd" in text:
        return "pedestrian"
    if "zone" in text or "nfz" in text or "hazard" in text:
        return "no_fly_zone"
    if "light" in text or "tower" in text or "station" in text or "barrier" in text:
        return "traffic_light" if "light" in text else "radio_tower"
    return "vehicle"


def generate_synthetic_trajectories(script: dict, duration_ticks: int,
                                    seed: int = 0) -> list[dict]:
    """
    从 event_script.json 推断实体布局，生成合成轨迹。
    在实际运行中，trajectory 由 UE 引擎生成；这里生成合理的合成数据用于图标签生成。
    """
    rng = random.Random(seed)
    trajectories: list[dict] = []
    params = script.get("parameters", {})

    # 从 actions 中提取实体引用
    entities: dict[str, dict] = {}  # entity_id → {pos, vel, type}

    for event in script.get("events", []):
        for action in event.get("actions", []):
            atype = action.get("type", "")
            eid = _resolve_param_value(action.get("entity_id", ""), params)

            if atype == "spawn_entity":
                pos = _resolve_param_value(
                    action.get("position_enu_m", [rng.uniform(40, 60), rng.uniform(15, 25), 0.0]),
                    params,
                )
                asset = _resolve_param_value(action.get("asset_id", ""), params)
                label_class = _infer_label_class(str(eid), str(asset))
                if label_class == "uav":
                    label_class = "uav"
                    pos[2] = rng.uniform(20, 40)
                elif label_class == "vehicle":
                    label_class = "vehicle"
                    pos[2] = 0.0

                entities[eid] = {
                    "pos": list(pos),
                    "vel": [0.0, 0.0, 0.0],
                    "label_class": label_class,
                    "asset_id": asset,
                    "state": "idle",
                    "spawn_tick": event.get("trigger_ref", ""),
                }

            elif atype == "move_entity":
                waypoints = _resolve_param_value(
                    action.get("waypoints_enu_m", action.get("waypoints", [])),
                    params,
                )
                if waypoints and eid not in entities:
                    label_class = _infer_label_class(str(eid))
                    entities[eid] = {
                        "pos": list(waypoints[0]) if waypoints else [50, 20, 30],
                        "vel": [1.0, 0.0, 0.0],
                        "label_class": label_class,
                        "asset_id": "uav.inspect.quad.v1",
                        "state": "moving",
                    }

    # Add entities from actions that target entity_ids not yet spawned
    for event in script.get("events", []):
        for action in event.get("actions", []):
            eid = _resolve_param_value(action.get("entity_id", action.get("ped_id", "")), params)
            if eid and eid not in entities:
                atype = action.get("type", "")
                label = "pedestrian" if atype == "play_animation" else _infer_label_class(str(eid))
                entities[eid] = {
                    "pos": [50 + rng.uniform(-5, 5), 20 + rng.uniform(-5, 5),
                            30 if label == "uav" else 0],
                    "vel": [0.0, 0.0, 0.0],
                    "label_class": label,
                    "asset_id": "unknown",
                    "state": "idle",
                }

    # Add entities that appear only in proximity triggers. Keep the pair close
    # so synthetic runs can exercise entity_proximity and composite triggers.
    for trigger in script.get("triggers", []):
        if trigger.get("type") != "entity_proximity":
            continue
        for offset, eid in enumerate([
            _resolve_param_value(trigger.get("entity_a", ""), params),
            _resolve_param_value(trigger.get("entity_b", ""), params),
        ]):
            if not eid or eid in entities:
                continue
            label = _infer_label_class(str(eid))
            z = 30.0 if label == "uav" else 0.0
            entities[eid] = {
                "pos": [50.0 + offset * 2.0, 20.0 + offset, z],
                "vel": [0.0, 0.0, 0.0],
                "label_class": label,
                "asset_id": "unknown",
                "state": "idle",
            }

    # Generate trajectories for each tick
    for tick in range(0, duration_ticks, 10):  # 每 10 ticks 一个轨迹点
        for eid, ent in entities.items():
            pos = list(ent["pos"])

            # Simple motion: constant velocity with small noise
            vel = ent.get("vel", [0.0, 0.0, 0.0])
            dt = 10  # ticks
            pos[0] += vel[0] * dt * 0.1 + rng.uniform(-0.1, 0.1)
            pos[1] += vel[1] * dt * 0.1 + rng.uniform(-0.1, 0.1)
            pos[2] += vel[2] * dt * 0.1

            ent["pos"] = pos
            ent["vel"] = [vel[0] + rng.uniform(-0.05, 0.05),
                         vel[1] + rng.uniform(-0.05, 0.05),
                         vel[2] + rng.uniform(-0.01, 0.01)]

            trajectories.append({
                "tick": tick,
                "entity_id": eid,
                "label_class": ent["label_class"],
                "pos_enu": list(ent["pos"]),
                "vel_mps": list(ent["vel"]),
                "state": ent.get("state", "moving"),
            })

    return trajectories


def generate_synthetic_weather(script: dict, duration_ticks: int) -> list[dict]:
    """生成合成天气数据（基于 event_script 中的 set_weather 动作）。"""
    weather_log = []
    current_weather = {"rain": 0.0, "fog": 0.0, "wind_speed": 2.0, "visibility_m": 20000.0}
    params = script.get("parameters", {})

    # 收集天气变更事件
    weather_changes: dict[int, dict] = {}
    for event in script.get("events", []):
        trigger_ref = event.get("trigger_ref", "")
        # 查找对应的 tick trigger
        for t in script.get("triggers", []):
            if t.get("trigger_id") == trigger_ref and t.get("type") == "tick":
                change_tick = _resolve_param_value(t.get("tick", 0), params)
                if isinstance(change_tick, str):
                    try:
                        change_tick = int(change_tick)
                    except ValueError:
                        change_tick = 300
                # 查找该事件的 set_weather 动作
                for action in event.get("actions", []):
                    if action.get("type") == "set_weather":
                        weather_changes[change_tick] = action.get("overrides", {})
                break

    for trigger in script.get("triggers", []):
        if trigger.get("type") != "weather_state":
            continue
        parameter = str(trigger.get("parameter", "rain"))
        operator = str(trigger.get("operator", "gte"))
        value = _resolve_param_value(trigger.get("value", 0.0), params)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if operator in ("gte", "gt"):
            synthetic_value = value + (0.01 if operator == "gt" else 0.0)
        elif operator in ("lte", "lt"):
            synthetic_value = value - (0.01 if operator == "lt" else 0.0)
        else:
            synthetic_value = value
        weather_changes.setdefault(300, {})[parameter] = synthetic_value

    for tick in range(0, duration_ticks, 10):
        if tick in weather_changes:
            current_weather.update(weather_changes[tick])
        weather_log.append({"tick": tick, **current_weather})

    return weather_log


def generate_episode(script_path: Path, episode_dir: Path, seed: int = 0) -> tuple[dict, dict]:
    """为一个 event_script.json 生成完整的 episode 数据。"""
    episode_dir.mkdir(parents=True, exist_ok=True)

    # 加载脚本
    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    scenario_id = script.get("scenario_id", "unknown")
    duration = script.get("parameters", {}).get("duration_ticks", 900)
    if isinstance(duration, str):
        try:
            duration = int(duration)
        except ValueError:
            duration = 900

    # 1. 使用 EventScriptInterpreter 生成 event_trace
    trajectories = generate_synthetic_trajectories(script, duration, seed)
    trajectory_by_tick: dict[int, list[dict]] = {}
    for row in trajectories:
        trajectory_by_tick.setdefault(int(row.get("tick", 0)), []).append(row)

    weather = generate_synthetic_weather(script, duration)
    weather_by_tick = {int(row.get("tick", 0)): row for row in weather}

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                           "Plugins/SumoImporter/Scripts"))
    from donghu_core.event_script_interpreter import EventScriptInterpreter

    interpreter = EventScriptInterpreter(Path(script_path))
    for t in range(duration):
        for row in trajectory_by_tick.get(t, []):
            interpreter.update_entity_state(
                row["entity_id"],
                row.get("pos_enu", [0.0, 0.0, 0.0]),
                {},
                row.get("vel_mps", [0.0, 0.0, 0.0]),
            )
        if t in weather_by_tick:
            interpreter.update_weather_state(weather_by_tick[t])
        interpreter.tick(t)

    event_log = interpreter.get_event_log()

    # 写入 event_trace.jsonl
    trace_path = episode_dir / "event_trace.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        for entry in event_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2. 生成合成轨迹
    traj_path = episode_dir / "trajectories.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 3. 生成合成天气
    weather_path = episode_dir / "weather_meta.jsonl"
    with open(weather_path, "w", encoding="utf-8") as f:
        for w in weather:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    # 4. 生成 global_entity_roster
    entity_ids = set(t["entity_id"] for t in trajectories)
    roster = {}
    for eid in entity_ids:
        traj_sample = next((t for t in trajectories if t["entity_id"] == eid), {})
        roster[eid] = {
            "entity_id": eid,
            "label_class": traj_sample.get("label_class", "unknown"),
            "asset_id": traj_sample.get("asset_id", "unknown"),
        }
    roster_path = episode_dir / "global_entity_roster.json"
    with open(roster_path, "w", encoding="utf-8") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)

    # 5. 写入 episode_manifest
    manifest = {
        "episode_id": episode_dir.name,
        "scenario_id": scenario_id,
        "duration_ticks": duration,
        "seed": seed,
        "n_events": len(event_log),
        "n_entities": len(entity_ids),
    }
    manifest_path = episode_dir / "episode_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 6. 生成图标签
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_graph_labels import generate_frame_graphs

    frame_graphs, summary = generate_frame_graphs(episode_dir)
    graphs_dir = episode_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graphs_path = graphs_dir / "frame_graphs.jsonl"
    with open(graphs_path, "w", encoding="utf-8") as f:
        for fg in frame_graphs:
            f.write(json.dumps(fg, ensure_ascii=False) + "\n")

    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate episodes")
    parser.add_argument("--dataset-root", default="Dataset", help="Path to Dataset/")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds per scenario")
    parser.add_argument("--skip-graphs", action="store_true", help="Skip graph label generation")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    scenarios_dir = root / "scenarios"
    episodes_dir = root / "episodes"

    # 找到所有 event_script.json
    script_paths = list(scenarios_dir.rglob("event_script.json"))
    print(f"Found {len(script_paths)} scenarios")

    total_episodes = 0
    total_events = 0

    for sp in sorted(script_paths):
        scenario_name = sp.parent.name
        for seed in range(args.seeds):
            episode_name = f"{scenario_name}__seed{seed:02d}"
            episode_dir = episodes_dir / episode_name

            try:
                manifest, summary = generate_episode(sp, episode_dir, seed)
                total_episodes += 1
                total_events += manifest["n_events"]
                print(f"  [OK] {episode_name}: {manifest['n_events']} events, "
                      f"{summary.get('n_frames', 0)} frames, "
                      f"{summary.get('n_unique_entities', 0)} entities")
            except Exception as e:
                print(f"  [FAIL] {episode_name}: {e}")

    print(f"\nGenerated {total_episodes} episodes with {total_events} total events")


if __name__ == "__main__":
    main()
