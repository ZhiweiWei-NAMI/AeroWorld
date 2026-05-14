#!/usr/bin/env python3
"""Repair derived render-ready truth/event alignment.

The scenario scripts remain the semantic source.  This tool only repairs
render-ready packages whose derived truth frames or event trace rows lost that
semantic alignment during conversion/filtering.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover
    orjson = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "Dataset" / "render_ready_episodes_capture_filtered"
TICK_HZ = 10

if str(ROOT / "Dataset" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "Dataset" / "tools"))

from pedestrian_activity_catalog import activity_annotations, normalize_activity_type  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def loads_jsonl(raw: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8-sig"))


def dumps_jsonl(payload: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield loads_jsonl(stripped)


def resolve_manifest_path(value: Any, episode_dir: Path) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    for candidate in ((ROOT / path).resolve(), (episode_dir / path).resolve()):
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def event_id_from_trace(row: dict[str, Any], scenario_id: str) -> str:
    raw = str(row.get("source_event_id") or row.get("topic") or row.get("event_id") or "")
    prefix = f"evt_{scenario_id}_"
    return raw[len(prefix) :] if raw.startswith(prefix) else raw


def trace_tick(row: dict[str, Any]) -> int:
    return int(row.get("tick", row.get("activated_tick", 0)) or 0)


def build_action_overrides(script: dict[str, Any], trace_rows: Sequence[dict[str, Any]], scenario_id: str) -> dict[str, list[tuple[int, str, str]]]:
    ticks = {event_id_from_trace(row, scenario_id): trace_tick(row) for row in trace_rows}
    overrides: dict[str, list[tuple[int, str, str]]] = {}
    for event in script.get("events") or []:
        event_id = str(event.get("event_id") or "")
        if event_id not in ticks:
            continue
        tick = ticks[event_id]
        for action in event.get("actions") or []:
            action_type = str(action.get("type") or "")
            entity_id = str(action.get("entity_id") or action.get("ped_id") or "")
            value = ""
            if action_type == "set_pedestrian_activity":
                value = str(action.get("activity_type") or "").strip()
            elif action_type == "set_visual_state":
                value = str((action.get("visual_state") or {}).get("mode") or action.get("mode") or "").strip()
            elif action_type == "move_entity":
                value = str(action.get("activity_type") or "").strip()
            if entity_id and value:
                overrides.setdefault(entity_id, []).append((tick, value, action_type))
    for values in overrides.values():
        values.sort(key=lambda item: item[0])
    return overrides


def position(entity: dict[str, Any]) -> list[float] | None:
    pose = entity.get("truth_pose") or {}
    raw = pose.get("position_enu_m") or pose.get("position_m") or entity.get("pos_enu")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 2:
        return None
    return [float(raw[0]), float(raw[1]), float(raw[2] if len(raw) > 2 else 0.0)]


def entity_category(entity: dict[str, Any]) -> str:
    return str(entity.get("entity_category") or entity.get("category") or entity.get("label_class") or "").strip().lower()


def simple_annotations(value: str, entity: dict[str, Any]) -> dict[str, Any]:
    annotations = dict(entity.get("annotations") or {})
    speed = float(annotations.get("speed_mps") or 0.0)
    if entity_category(entity) == "pedestrian":
        try:
            return activity_annotations(normalize_activity_type(value), speed_mps=speed)
        except ValueError:
            pass
    facets = dict(annotations.get("state_facets") or {})
    activity = dict(facets.get("activity") or {})
    activity["activity_type"] = value
    activity["animation_hint"] = value
    facets["activity"] = activity
    annotations["activity_type"] = value
    annotations["state_facets"] = facets
    return annotations


def apply_override(entity: dict[str, Any], value: str) -> None:
    entity["state"] = value
    sequence = list(entity.get("state_sequence") or [])
    if value not in sequence:
        sequence.append(value)
    entity["state_sequence"] = sequence
    entity["annotations"] = simple_annotations(value, entity)


def repair_truth_frames(episode_dir: Path, overrides: dict[str, list[tuple[int, str, str]]]) -> int:
    if not overrides:
        return 0
    truth_path = episode_dir / "truth_frames.jsonl"
    temp_path = truth_path.with_suffix(".jsonl.tmp")
    changed = 0
    with truth_path.open("rb") as source, temp_path.open("wb") as out:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            frame = loads_jsonl(stripped)
            tick = int(frame.get("tick", frame.get("frame_seq", 0)) or 0)
            for entity in frame.get("entities") or []:
                entity_id = str(entity.get("entity_id") or "")
                candidates = [item for item in overrides.get(entity_id, []) if item[0] <= tick]
                if not candidates:
                    continue
                sequence = list(entity.get("state_sequence") or [])
                for _candidate_tick, candidate_value, _candidate_type in candidates:
                    if candidate_value not in sequence:
                        sequence.append(candidate_value)
                entity["state_sequence"] = sequence
                _override_tick, value, _action_type = candidates[-1]
                before = (entity.get("state"), (entity.get("annotations") or {}).get("activity_type"))
                apply_override(entity, value)
                after = (entity.get("state"), (entity.get("annotations") or {}).get("activity_type"))
                if after != before:
                    changed += 1
            out.write(dumps_jsonl(frame))
    temp_path.replace(truth_path)
    return changed


def distance(a: Sequence[float], b: Sequence[float], metric: str) -> float:
    dxy = math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    if metric in {"3d", "xyz"}:
        return math.sqrt(dxy * dxy + (float(a[2]) - float(b[2])) ** 2)
    return dxy


def condition_ok(value: float, operator: str, threshold: float) -> bool:
    if operator == "lt":
        return value < threshold
    if operator == "lte":
        return value <= threshold
    if operator == "gt":
        return value > threshold
    if operator == "gte":
        return value >= threshold
    return abs(value - threshold) <= 1e-9


def positions_by_tick(episode_dir: Path, entity_ids: set[str]) -> dict[str, dict[int, list[float]]]:
    values = {entity_id: {} for entity_id in entity_ids}
    for frame in iter_jsonl(episode_dir / "truth_frames.jsonl"):
        tick = int(frame.get("tick", frame.get("frame_seq", 0)) or 0)
        for entity in frame.get("entities") or []:
            entity_id = str(entity.get("entity_id") or "")
            if entity_id in values:
                pos = position(entity)
                if pos is not None:
                    values[entity_id][tick] = pos
    return values


def repair_proximity_trace(episode_dir: Path, script: dict[str, Any], trace_rows: list[dict[str, Any]], scenario_id: str) -> int:
    triggers = {str(trigger.get("trigger_id") or ""): trigger for trigger in script.get("triggers") or []}
    events = {str(event.get("event_id") or ""): event for event in script.get("events") or []}
    trace_by_event = {event_id_from_trace(row, scenario_id): row for row in trace_rows}
    needed_entities: set[str] = set()
    proximity_events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for event_id, event in events.items():
        trigger = triggers.get(str(event.get("trigger_ref") or ""))
        if not trigger or str(trigger.get("type") or "") != "entity_proximity":
            continue
        entity_a = str(trigger.get("entity_a") or "")
        entity_b = str(trigger.get("entity_b") or "")
        if not entity_a or not entity_b:
            continue
        if event_id not in trace_by_event:
            continue
        needed_entities.update([entity_a, entity_b])
        proximity_events.append((event_id, event, trigger))
    if not proximity_events:
        return 0
    positions = positions_by_tick(episode_dir, needed_entities)
    changed = 0
    for event_id, _event, trigger in proximity_events:
        row = trace_by_event[event_id]
        current_tick = trace_tick(row)
        entity_a = str(trigger.get("entity_a") or "")
        entity_b = str(trigger.get("entity_b") or "")
        metric = str(trigger.get("metric") or "xy").lower()
        threshold = float(trigger.get("distance_m") or 0.0)
        operator = str(trigger.get("operator") or "lte")
        common_ticks = sorted(set(positions.get(entity_a, {})) & set(positions.get(entity_b, {})))
        candidate_tick = None
        for tick in common_ticks:
            if tick < current_tick:
                continue
            dist = distance(positions[entity_a][tick], positions[entity_b][tick], metric)
            if condition_ok(dist, operator, threshold):
                candidate_tick = tick
                break
        if candidate_tick is None or candidate_tick == current_tick:
            continue
        for key in ("tick", "activated_tick", "source_tick"):
            if key in row:
                row[key] = candidate_tick
        row["sim_time_s"] = round(candidate_tick / TICK_HZ, 6)
        for key in ("frame_id", "activated_frame_id", "source_frame_id"):
            if key in row:
                row[key] = f"tick:{candidate_tick}"
        row["sample_id"] = f"event_trace:{candidate_tick}:{event_id}"
        payload = row.get("payload")
        if isinstance(payload, dict):
            for key in ("activated_tick", "source_tick", "start_tick", "end_tick"):
                if key in payload:
                    payload[key] = candidate_tick
        changed += 1
    if changed:
        with (episode_dir / "event_trace.jsonl").open("wb") as out:
            for row in sorted(trace_rows, key=lambda item: (trace_tick(item), str(item.get("topic") or ""))):
                out.write(dumps_jsonl(row))
    return changed


def set_trace_tick(row: dict[str, Any], event_id: str, tick: int) -> None:
    for key in ("tick", "activated_tick", "source_tick"):
        if key in row:
            row[key] = tick
    row["sim_time_s"] = round(tick / TICK_HZ, 6)
    for key in ("frame_id", "activated_frame_id", "source_frame_id"):
        if key in row:
            row[key] = f"tick:{tick}"
    row["sample_id"] = f"event_trace:{tick}:{event_id}"
    payload = row.get("payload")
    if isinstance(payload, dict):
        for key in ("activated_tick", "source_tick", "start_tick", "end_tick"):
            if key in payload:
                payload[key] = tick


def repair_event_fired_after_trace(episode_dir: Path, script: dict[str, Any], trace_rows: list[dict[str, Any]], scenario_id: str) -> int:
    triggers = {str(trigger.get("trigger_id") or ""): trigger for trigger in script.get("triggers") or []}
    rows = {event_id_from_trace(row, scenario_id): row for row in trace_rows}
    changed = 0
    for _ in range(4):
        updated = False
        for event in script.get("events") or []:
            event_id = str(event.get("event_id") or "")
            row = rows.get(event_id)
            trigger = triggers.get(str(event.get("trigger_ref") or ""))
            if row is None or not trigger or str(trigger.get("type") or "") != "event_fired_after":
                continue
            predecessor = rows.get(str(trigger.get("event_id") or ""))
            if predecessor is None:
                continue
            expected_tick = trace_tick(predecessor) + int(trigger.get("delay_ticks") or 0)
            if trace_tick(row) != expected_tick:
                set_trace_tick(row, event_id, expected_tick)
                changed += 1
                updated = True
        if not updated:
            break
    if changed:
        with (episode_dir / "event_trace.jsonl").open("wb") as out:
            for row in sorted(trace_rows, key=lambda item: (trace_tick(item), str(item.get("topic") or ""))):
                out.write(dumps_jsonl(row))
    return changed


def repair_episode(episode_dir: Path) -> dict[str, Any]:
    manifest = read_json(episode_dir / "episode_manifest.json")
    scenario_id = str(manifest.get("scenario_id") or episode_dir.name.split("__seed")[0])
    script = read_json(resolve_manifest_path(manifest.get("source_event_script_path"), episode_dir))
    trace_rows = list(iter_jsonl(episode_dir / "event_trace.jsonl"))
    overrides = build_action_overrides(script, trace_rows, scenario_id)
    truth_changes = repair_truth_frames(episode_dir, overrides)
    trace_changes = repair_proximity_trace(episode_dir, script, trace_rows, scenario_id)
    trace_changes += repair_event_fired_after_trace(episode_dir, script, trace_rows, scenario_id)
    return {"episode": episode_dir.name, "truth_changes": truth_changes, "trace_changes": trace_changes}


def parse_episode_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                names.append(item)
    return names


def select_episode_dirs(input_root: Path, names: list[str]) -> list[Path]:
    all_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "truth_frames.jsonl").exists())
    if not names:
        return all_dirs
    by_name = {path.name: path for path in all_dirs}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown episode(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair derived truth/event semantic alignment in render-ready packages.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--episode", action="append", default=[])
    args = parser.parse_args()
    episode_dirs = select_episode_dirs(args.input_root.resolve(), parse_episode_names(args.episode))
    results = [repair_episode(path) for path in episode_dirs]
    summary = {
        "ok": True,
        "episodes": len(results),
        "truth_changes": sum(int(item["truth_changes"]) for item in results),
        "trace_changes": sum(int(item["trace_changes"]) for item in results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
