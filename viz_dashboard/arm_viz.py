"""ARM 可视化组装层：每臂事件链、窗口、切换节点轴、谓词图与语义说明。

只读组装，不做推断或兜底：
- 事件链来自本臂 objective 三类行 + raw/actions.jsonl 实际动作，全部带真实
  tick 与 evidence {path, line, raw}；窗口来自 manifest.window。
- 谓词图：实体为节点、谓词为边。静态边来自窗口内 predicate_transitions，
  动态边来自所选 tick 的 objective/predicate_truth.jsonl（四值真值、绑定、
  数值证据）。truth 文件只按 tick 建字节偏移索引，随后按需 seek。
- 说明来自 design/<profile>/<seed>/<group>/experiment.json 的既有字段与
  本臂配置 YAML 的 expected/rationale/falsified_if；文件或字段缺失直接报错，
  绝不生成模板文案。
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

from arm_review import ArmReviewError, _load_configuration

_DESCRIPTION_TTL_SECONDS = 2.0


# ── predicate_truth.jsonl 偏移索引（模块级缓存） ────────────────────────

class PredicateTruthIndex:
    """按 tick 的字节偏移索引；文件指纹变化时整体重建。"""

    __slots__ = ("path", "fingerprint", "ticks", "offsets")

    def __init__(self, path: Path, fingerprint: tuple[int, int],
                 ticks: list[int], offsets: dict[int, list[int]]) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.ticks = ticks
        self.offsets = offsets

    @classmethod
    def build(cls, path: Path, relative: str) -> "PredicateTruthIndex":
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        ticks: list[int] = []
        offsets: dict[int, list[int]] = {}
        try:
            with path.open("rb") as stream:
                offset = 0
                for raw_line in stream:
                    length = len(raw_line)
                    text = raw_line.strip()
                    offset += length
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ArmReviewError(
                            f"Malformed JSON: {relative}: {exc}") from exc
                    tick = row.get("tick")
                    if not isinstance(tick, int) or isinstance(tick, bool):
                        raise ArmReviewError(
                            f"{relative}: predicate_truth row lacks integer tick")
                    offsets.setdefault(tick, []).append(offset - length)
                    ticks.append(tick)
        except OSError as exc:
            raise ArmReviewError(f"Cannot read {relative}: {exc}") from exc
        return cls(path, fingerprint, ticks, offsets)

    def rows(self, tick: int) -> list[dict]:
        start_offsets = self.offsets.get(tick)
        if not start_offsets:
            return []
        rows: list[dict] = []
        with self.path.open("rb") as stream:
            for start in start_offsets:
                stream.seek(start)
                line = stream.readline()
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ArmReviewError(
                        f"Malformed JSON: {self.path}:{start}: {exc}") from exc
        return rows


class PredicateTruthIndexCache:
    """arm_key -> 索引；文件 mtime/size 变化时重建，进程内共享。"""

    def __init__(self) -> None:
        self._indexes: dict[str, PredicateTruthIndex] = {}

    def get(self, root: Path, relative: str) -> PredicateTruthIndex:
        path = root / relative
        try:
            stat = path.stat()
        except OSError as exc:
            raise ArmReviewError(f"Cannot stat {relative}: {exc}") from exc
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        cached = self._indexes.get(relative)
        if cached is not None and cached.fingerprint == fingerprint:
            return cached
        index = PredicateTruthIndex.build(path, relative)
        self._indexes[relative] = index
        return index


PREDICATE_TRUTH_INDEX_CACHE = PredicateTruthIndexCache()


# ── 说明：design 实验定义 + 配置 expected，逐字段实读 ────────────────────

def _experiment_document(root: Path, profile: str, seed: str, group: str) -> dict:
    relative = f"design/{profile}/{seed}/{group}/experiment.json"
    path = root / relative
    if not path.is_file():
        raise ArmReviewError(f"Missing design experiment definition: {relative}")
    try:
        content = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArmReviewError(f"Cannot read design experiment {relative}: {exc}") from exc
    if not isinstance(content, dict):
        raise ArmReviewError(f"{relative}: experiment definition must be a JSON object")
    return {"path": relative, "content": content}


def _config_description(arm: dict) -> dict:
    """配置文件里的 expected / hypothesis / mechanism（存在才返回）。"""
    configuration = arm.get("configuration") or {}
    content = configuration.get("content")
    if not isinstance(content, dict):
        return {}
    expected = content.get("expected")
    fields: dict = {}
    if isinstance(expected, dict):
        for key in ("rationale", "falsified_if", "statement"):
            value = expected.get(key)
            if isinstance(value, str) and value.strip():
                fields[key] = value
    elif isinstance(expected, str) and expected.strip():
        fields["expected"] = expected
    for key in ("mechanism", "intent"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            fields[key] = value
    # L6 JSON 配置：spec.hypothesis / spec.falsified_if。
    spec = content.get("spec")
    if isinstance(spec, dict):
        for key in ("hypothesis", "falsified_if"):
            value = spec.get(key)
            if isinstance(value, str) and value.strip():
                fields.setdefault(key, value)
    if fields and configuration.get("path"):
        fields["source"] = configuration["path"]
    return fields


def _arm_design_description(root: Path, design: dict, arm: dict, family: str) -> dict:
    """按家族 schema 从 experiment.json 抽取本臂描述字段；字段缺失保持缺失。"""
    content = design["content"]
    arm_id = arm["arm_id"]
    fields: dict = {"source": design["path"]}

    def put(key: str, value) -> None:
        if isinstance(value, str) and value.strip():
            fields[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            fields[key] = value

    for key in ("question", "hypothesis", "window_basis", "limits"):
        value = content.get(key)
        if isinstance(value, dict):
            inner = {k: v for k, v in value.items()
                     if isinstance(v, str) and v.strip()}
            if inner:
                fields[key] = inner
        else:
            put(key, value)

    interventions = content.get("interventions")
    intervention_refs: list[str] = []
    if isinstance(interventions, list):
        intervention_refs = [v for v in interventions if isinstance(v, str)]
    if isinstance(interventions, dict):
        intervention_refs = [v for v in interventions.values() if isinstance(v, str)]
    if intervention_refs:
        fields["intervention_refs"] = intervention_refs

    arm_defs = content.get("arm_definitions")
    if isinstance(arm_defs, dict):
        definition = arm_defs.get(arm_id)
        if isinstance(definition, dict):
            rewrites = definition.get("rewrites")
            if isinstance(rewrites, dict) and isinstance(rewrites.get("detail"), str):
                fields["mechanism_detail"] = rewrites["detail"]
            expected = definition.get("expected")
            if isinstance(expected, dict):
                for key in ("statement", "rationale", "falsified_if"):
                    put(f"design_{key}", expected.get(key))

    # 部分 L4 profile 的 design 用 arms 列表（元素含 arm_id）而非 arm_definitions。
    arms_list = content.get("arms")
    if isinstance(arms_list, list):
        for entry in arms_list:
            if not isinstance(entry, dict) or entry.get("arm_id") != arm_id:
                continue
            rewrites = entry.get("rewrites")
            if isinstance(rewrites, dict) and isinstance(rewrites.get("detail"), str):
                fields.setdefault("mechanism_detail", rewrites["detail"])
            expected = entry.get("expected")
            if isinstance(expected, dict):
                put("design_target_event", expected.get("target_event"))
                for key in ("statement", "rationale", "falsified_if"):
                    put(f"design_{key}", expected.get(key))
            break

    branches = content.get("branch_definitions")
    if isinstance(branches, dict) and isinstance(branches.get(arm_id), dict):
        definition = branches[arm_id]
        fields["branch_definition_present"] = True
        expected = definition.get("expected")
        if isinstance(expected, dict):
            for key in ("statement", "rationale", "falsified_if"):
                put(f"design_{key}", expected.get(key))
        elif isinstance(expected, str) and expected.strip():
            fields["design_statement"] = expected

    if family == "L5":
        arm_ref = None
        arms = content.get("arms")
        if isinstance(arms, dict):
            arm_ref = arms.get(arm_id)
        if isinstance(arm_ref, dict) and isinstance(arm_ref.get("experiment_ref"), str):
            ref_relative = arm_ref["experiment_ref"]
            ref_path = root / ref_relative
            if not ref_path.is_file():
                raise ArmReviewError(
                    f"Missing L5 arm experiment definition: {ref_relative}")
            try:
                ref = json.loads(ref_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArmReviewError(
                    f"Cannot read L5 arm experiment {ref_relative}: {exc}") from exc
            if not isinstance(ref, dict):
                raise ArmReviewError(f"{ref_relative}: L5 arm experiment must be an object")
            fields["source"] = f"{design['path']} + {ref_relative}"
            hazard = ref.get("hazard")
            if isinstance(hazard, dict) and isinstance(hazard.get("premise"), str):
                fields["mechanism_detail"] = hazard["premise"]
            target_event = ref.get("target_event")
            if isinstance(target_event, dict) and isinstance(target_event.get("basis"), str):
                fields["design_target_basis"] = target_event["basis"]

    if family == "X":
        arms = content.get("arms")
        if isinstance(arms, dict) and isinstance(arms.get(arm_id), dict):
            definition = arms[arm_id]
            put("research_question", definition.get("research_question"))
            put("design_window_basis", definition.get("window_basis"))

    return fields


def arm_description(root: Path, profile: str, seed: str, group: str,
                    arm: dict, family: str, design: dict) -> dict:
    description: dict = {}
    design_fields = _arm_design_description(root, design, arm, family)
    if design_fields:
        description["design"] = design_fields
    config_fields = _config_description(arm)
    if config_fields:
        description["configuration"] = config_fields
    if not description:
        raise ArmReviewError(
            f"arm {arm['arm_id']} ({profile}/{seed}/{group}): no description field "
            f"found in {design['path']} nor in its configuration expected block")
    return description


# ── 事件链：objective 三类行 + 实际动作 ─────────────────────────────────

def _action_row_fields(raw: dict) -> dict:
    action = raw.get("action") or {}
    result = raw.get("result") or {}
    action_id = action.get("action_id") or result.get("action_id")
    return {
        "action_id": action_id,
        "type": action.get("type"),
        "entity_id": action.get("entity_id") or result.get("entity_id"),
        "status": result.get("status"),
        "origin": raw.get("origin"),
    }


def _append_chain_item(items: list[dict], *, kind: str, tick: int | None,
                       label: str, detail: str, evidence: dict,
                       raw_ref: dict, censored: bool = False) -> None:
    items.append({
        "kind": kind,
        "tick": tick,
        "censored": bool(censored),
        "label": label,
        "detail": detail,
        "evidence": {"path": evidence["path"], "line": evidence["line"]},
        "raw": raw_ref,
    })


def arm_event_chain(arm: dict, family: str) -> list[dict]:
    items: list[dict] = []
    window = arm.get("window") or {}
    start = window.get("start_tick")
    end = window.get("end_tick")

    def in_window(tick) -> bool:
        return isinstance(tick, int) and (
            start is None or tick >= start) and (
            end is None or tick <= end)

    for row in arm["objective"]["transitions"]:
        raw = row["evidence"]["raw"]
        tick = raw.get("to_tick")
        if not in_window(tick):
            continue
        before = raw.get("before", raw.get("from_value"))
        after = raw.get("after", raw.get("to_value"))
        bindings = raw.get("bindings") or {}
        subject = ", ".join(str(v) for v in bindings.values()) if bindings else ""
        _append_chain_item(
            items, kind="predicate_transition", tick=tick,
            label=str(raw.get("predicate_id") or "?"),
            detail=f"{before} → {after}" + (f" · {subject}" if subject else ""),
            evidence=row["evidence"], raw_ref={"bindings": bindings,
                                               "predicate_id": raw.get("predicate_id"),
                                               "direction": raw.get("direction")})

    for row in arm["objective"]["occurrences"]:
        raw = row["evidence"]["raw"]
        tick = raw.get("trigger_tick") if family != "X" else raw.get("tick")
        if not in_window(tick):
            continue
        _append_chain_item(
            items, kind="event_occurrence", tick=tick,
            label=str(raw.get("event_family_id") or raw.get("event_id") or "?"),
            detail=str(raw.get("event_type_id") or ""),
            evidence=row["evidence"],
            raw_ref={"event_id": raw.get("event_id"),
                     "event_type_id": raw.get("event_type_id"),
                     "event_family_id": raw.get("event_family_id"),
                     "bindings": raw.get("bindings") or {}})

    for row in arm["objective"]["outcomes"]:
        raw = row["evidence"]["raw"]
        terminal_tick = raw.get("terminal_tick")
        if terminal_tick is None:
            if not in_window(end):
                continue
            censored = raw.get("status") == "right_censored"
            _append_chain_item(
                items, kind="event_outcome", tick=None, censored=censored,
                label=str(raw.get("event_family_id") or raw.get("event_id") or "?"),
                detail=("窗口删失" if censored else "终态待定"),
                evidence=row["evidence"],
                raw_ref={"event_id": raw.get("event_id"), "status": raw.get("status"),
                         "terminal_tick": None,
                         "window_censor_tick": raw.get("window_censor_tick")})
            continue
        if not in_window(terminal_tick):
            continue
        _append_chain_item(
            items, kind="event_outcome", tick=terminal_tick,
            label=str(raw.get("event_family_id") or raw.get("event_id") or "?"),
            detail=str(raw.get("status") or ""),
            evidence=row["evidence"],
            raw_ref={"event_id": raw.get("event_id"), "status": raw.get("status"),
                     "terminal_tick": terminal_tick})

    seen_actions: set[tuple] = set()
    for row in arm.get("raw_actions", []):
        raw = row["evidence"]["raw"]
        tick = raw.get("tick")
        fields = _action_row_fields(raw)
        key = (tick, fields["action_id"], fields["entity_id"])
        if key in seen_actions:
            continue
        seen_actions.add(key)
        if not in_window(tick):
            continue
        _append_chain_item(
            items, kind="action", tick=tick,
            label=str(fields["action_id"] or "?"),
            detail=" · ".join(str(v) for v in
                              (fields["type"], fields["entity_id"],
                               fields["status"], fields["origin"]) if v),
            evidence=row["evidence"],
            raw_ref={"action_id": fields["action_id"], "type": fields["type"],
                     "entity_id": fields["entity_id"],
                     "status": fields["status"], "origin": fields["origin"]})

    items.sort(key=lambda item: (
        item["tick"] if item["tick"] is not None else 10 ** 9,
        0 if item["kind"] == "action" else 1, item["kind"]))
    return items


def arm_switch_nodes(chains_by_arm: dict[str, list[dict]]) -> list[dict]:
    """所有事件链发生切换的时间节点（跨臂去重合并，供单条紧凑时间轴）。"""
    by_tick: dict[int, dict] = {}
    for arm_id, items in chains_by_arm.items():
        for item in items:
            tick = item["tick"]
            if tick is None:
                continue
            entry = by_tick.setdefault(tick, {"tick": tick, "events": []})
            entry["events"].append({"arm": arm_id, "kind": item["kind"],
                                    "label": item["label"], "detail": item["detail"]})
    return [by_tick[tick] for tick in sorted(by_tick)]


# ── 谓词图：实体为节点、谓词为边 ────────────────────────────────────────

def _scope_entity_ids(arm: dict) -> list[str]:
    manifest = arm.get("manifest", {}).get("content", {}) or {}
    scope = manifest.get("scope") or {}
    ids: list[str] = []
    for key in ("entity_ids", "aircraft_ids", "station_ids", "gcs_ids",
                "home_pad_ids", "simulated_entity_ids"):
        value = scope.get(key)
        if isinstance(value, list):
            ids.extend(str(v) for v in value
                       if isinstance(v, str) and not isinstance(v, bool))
    return ids


def _entity_label(entity_id: str) -> str:
    for token in entity_id.split("|"):
        if token:
            return token
    return entity_id


def _evidence_numbers(truth_row: dict) -> list[dict]:
    numbers: list[dict] = []
    for observation in (truth_row.get("evidence") or {}).get("observations") or []:
        if not isinstance(observation, dict):
            continue
        value = observation.get("value")
        if not isinstance(value, dict):
            continue
        for world in value.get("world_observations") or []:
            if not isinstance(world, dict):
                continue
            path = world.get("path")
            if isinstance(path, str) and isinstance(world.get("value"), (int, float)):
                numbers.append({"path": path, "value": world["value"]})
    return numbers


def arm_predicate_graph(root: Path, arm: dict, tick: int,
                        arm_dir_relative: str) -> dict:
    """谓词名词为边、实体为节点：静态结构（窗口全貌）+ 本 tick 动态状态。"""
    window = arm.get("window") or {}
    start, end = window.get("start_tick"), window.get("end_tick")

    def in_window(value) -> bool:
        return isinstance(value, int) and (
            start is None or value >= start) and (
            end is None or value <= end)

    nodes: dict[str, dict] = {}
    for entity_id in _scope_entity_ids(arm):
        nodes[entity_id] = {"id": entity_id, "label": _entity_label(entity_id),
                            "source": "manifest.scope"}
    static_edges: dict[str, dict] = {}
    dynamic_edges: list[dict] = []

    for row in arm["objective"]["transitions"]:
        raw = row["evidence"]["raw"]
        to_tick = raw.get("to_tick")
        if not in_window(to_tick) and not in_window(raw.get("from_tick")):
            continue
        predicate_id = raw.get("predicate_id")
        bindings = raw.get("bindings") or {}
        for value in bindings.values():
            if isinstance(value, str) and value not in nodes:
                nodes[value] = {"id": value, "label": _entity_label(value),
                                "source": "predicate_binding"}
        static_edges.setdefault(predicate_id, {
            "predicate_id": predicate_id,
            "roles": sorted(bindings.keys()),
            "transitions": [],
        })["transitions"].append({
            "from_tick": raw.get("from_tick"), "to_tick": to_tick,
            "from_value": raw.get("before", raw.get("from_value")),
            "to_value": raw.get("after", raw.get("to_value")),
            "bindings": bindings,
        })

    action_entities: list[str] = []
    for row in arm.get("raw_actions", []):
        raw = row["evidence"]["raw"]
        if not in_window(raw.get("tick")):
            continue
        fields = _action_row_fields(raw)
        entity_id = fields["entity_id"]
        if isinstance(entity_id, str) and entity_id:
            action_entities.append(entity_id)
            if entity_id not in nodes:
                nodes[entity_id] = {"id": entity_id, "label": _entity_label(entity_id),
                                    "source": "action_entity"}

    index = PREDICATE_TRUTH_INDEX_CACHE.get(
        root, f"{arm_dir_relative}/objective/predicate_truth.jsonl")
    truth_available = bool(index.ticks) and index.ticks[0] <= tick <= index.ticks[-1]
    if truth_available:
        nearest = index.ticks[bisect_right(index.ticks, tick) - 1]
        if nearest != tick:
            raise ArmReviewError(
                f"{arm_dir_relative}/objective/predicate_truth.jsonl: tick {tick} "
                f"is not a materialized sample (nearest {nearest})")
        for truth_row in index.rows(tick):
            predicate_id = truth_row.get("predicate_id")
            bindings = truth_row.get("bindings") or {}
            binding_values = [v for v in bindings.values() if isinstance(v, str)]
            if not binding_values:
                continue
            if not any(entity in nodes for entity in binding_values):
                continue
            for value in binding_values:
                if value not in nodes:
                    nodes[value] = {"id": value, "label": _entity_label(value),
                                    "source": "predicate_binding"}
            dynamic_edges.append({
                "predicate_id": predicate_id,
                "value": truth_row.get("value"),
                "bindings": bindings,
                "tick": tick,
                "numbers": _evidence_numbers(truth_row),
                "evidence": {
                    "path": f"{arm_dir_relative}/objective/predicate_truth.jsonl",
                    "truth_id": truth_row.get("truth_id"),
                },
            })

    dynamic_edges.sort(key=lambda edge: (
        edge["predicate_id"] or "",
        sorted(f"{k}={v}" for k, v in edge["bindings"].items())))
    return {
        "tick": tick,
        "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
        "static_edges": sorted(static_edges.values(),
                               key=lambda edge: edge["predicate_id"] or ""),
        "dynamic_edges": dynamic_edges,
        "action_entities": sorted(set(action_entities)),
        "truth_tick_available": truth_available,
    }


# ── 组装入口 ───────────────────────────────────────────────────────────

def arm_viz_payload(group_detail_payload: dict, root: Path) -> dict:
    if not isinstance(tick_holder := group_detail_payload.get("arms"), list) or not tick_holder:
        raise ArmReviewError("group detail has no arms")
    profile = group_detail_payload["profile"]
    seed = group_detail_payload["seed"]
    group = group_detail_payload["group"]
    family = group_detail_payload.get("family")
    design = _experiment_document(root, profile, seed, group)

    arms_out = []
    chains_by_arm: dict[str, list[dict]] = {}
    for arm in group_detail_payload["arms"]:
        chain = arm_event_chain(arm, family)
        chains_by_arm[arm["arm_key"]] = chain
        arms_out.append({
            "arm_key": arm["arm_key"],
            "arm_id": arm["arm_id"],
            "mode": arm.get("mode"),
            "window": arm.get("window"),
            "episode_id": arm.get("episode_id"),
            "chain": chain,
            "description": arm_description(root, profile, seed, group, arm, family, design),
        })

    return {
        "profile": profile,
        "seed": seed,
        "group": group,
        "family": family,
        "design": {"path": design["path"],
                   "chain_nodes": design["content"].get("chain_nodes"),
                   "anchor_event": design["content"].get("anchor_event")},
        "arms": arms_out,
        "switch_nodes": arm_switch_nodes(chains_by_arm),
    }
