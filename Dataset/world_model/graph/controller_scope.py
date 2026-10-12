"""Prove the source namespace of a domain-state controller key per tick."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from Dataset.world_model.graph.contract import REPO, iter_jsonl, source_path


@dataclass(frozen=True)
class ControllerEvidence:
    source_file: str
    by_tick: dict[int, dict[str, str]]
    sumo_first_line: dict[str, int]

    def namespaces(self, tick: int, keys: set[str]) -> dict[str, str]:
        current = self.by_tick.get(tick)
        if current is None:
            raise ValueError(f"controller observation tick has no truth frame: {self.source_file}:{tick}")
        missing = keys - current.keys()
        if missing:
            raise ValueError(f"controller output has no current-frame source: {self.source_file}:{tick}:{sorted(missing)}")
        return {key: current[key] for key in keys}


@lru_cache(maxsize=4)
def controller_evidence(episode_id: str) -> ControllerEvidence:
    path = REPO / "aw_data/render_ready_episodes_capture_filtered" / episode_id / "truth_frames.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    by_tick: dict[int, dict[str, str]] = {}
    first: dict[str, int] = {}
    stable: dict[str, str] = {}
    for line, frame in iter_jsonl(path):
        tick = frame["tick"]
        if type(tick) is not int or tick in by_tick:
            raise ValueError(f"truth-frame tick missing or duplicated: {path}:{line}")
        scene = {entity["entity_id"] for entity in frame["entities"]
                 if entity.get("entity_category") == "traffic_light"}
        sumo = set(frame.get("sumo_traffic_light_states") or {})
        if scene & sumo:
            raise ValueError(f"scene signal and SUMO controller keys collide: {path}:{line}:{sorted(scene & sumo)}")
        current = {key: "sample" for key in scene}
        current.update({key: "sumo_tls_controller" for key in sumo})
        for key, namespace in current.items():
            if key in stable and stable[key] != namespace:
                raise ValueError(f"controller key changed source namespace: {path}:{line}:{key}")
            stable[key] = namespace
            if namespace == "sumo_tls_controller":
                first.setdefault(key, line)
        by_tick[tick] = current
    return ControllerEvidence(source_path(path), by_tick, first)


def controller_namespaces_for_record(source_file: str | Path,
                                     record: Mapping[str, Any]) -> dict[str, str] | None:
    keys = (record.get("values") or {}).get("red_duration_ticks_by_controller")
    if keys is None:
        return None
    if not isinstance(keys, dict):
        raise TypeError(f"controller durations are not an object: {source_file}:{record.get('tick')}")
    return controller_evidence(Path(source_file).parent.name).namespaces(record["tick"], set(keys))
