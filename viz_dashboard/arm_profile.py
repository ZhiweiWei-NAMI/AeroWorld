"""Profile-level ARM overview: one row per real intervention arm in a profile+seed.

Public API:
- ``arm_profile.`` ``ArmProfileError``: raised on missing/malformed/incomparable input.
- ``profile_overview(root, profile, seed, catalog_data=None)``: walk the
  catalog groups of one profile+seed and, for every arm, read only the light
  artifacts (configuration, manifest, window_status, raw/actions.jsonl,
  objective/event_occurrences.jsonl, objective/event_outcomes.jsonl). The huge
  per-tick states/UE files and per-predicate truth are never read here; use
  ``arm_review.group_detail`` for the full group view.

Every mechanism description is derived from the actual configuration/action
rows, never from the arm_id. Chinese labels for known arm_ids are display
shorthands only; unknown arm_ids fall back to the raw id. Measured results
come from occurrences + outcomes only; right-censored rows and "did not occur
in window" stay distinct, and configuration expectations are reported in a
separate field.
"""
from __future__ import annotations

import json
from pathlib import Path

from arm_review import (
    ArmReviewError,
    _load_configuration,
    _read_json,
    _read_jsonl,
    _validated_outcome_row,
    _validate_path_segment,
    catalog,
)

# Display shorthand only. The mechanism column always comes from the real
# configuration/action files, never from this table.
ARM_LABELS = {
    "arm_factual": "事实观察",
    "arm_early_backup_link_recovery": "备用链路提前恢复",
    "arm_backup_link_unavailable": "备用链路不可用（任务中断）",
    "arm_early_relocalization_before_drift": "漂移前提前视觉重定位",
    "arm_relocalization_unavailable": "重定位不可用（原地保持）",
    "arm_primary_charger_repaired_before_reroute": "检查后主站快速修复",
    "arm_backup_charger_unavailable": "备用充电站不可用（原地保持）",
    "arm_offset_path_climb": "改变接近路径（抬高）",
    "arm_offset_path_lateral": "改变接近路径（侧移）",
    "arm_staggered_pad_admission": "错峰进场",
    "arm_arbitration_delayed": "仲裁推迟（争用持续）",
    "arm_early_signal_repair": "信号提前维修（无需人工指挥）",
    "arm_manual_control_absent": "人工指挥缺席（车辆滞留）",
    "arm_road_access_reopened": "提前恢复道路通行",
    "arm_evacuation_command_withheld": "撤去疏散指令",
    "arm_exterior_corridor_assignment": "改走区域外侧走廊",
    "arm_region_deactivated_before_entry": "入场前解除区域限制",
    "arm_response_vehicle_staged": "取消救护车出动",
    "arm_fault_absent": "删除故障补丁",
    "arm_fault_early_recovery": "提前修复故障",
    "arm_hold_at_fork": "分叉处保持位置",
    "arm_lifecycle_hold": "暂停起降动作",
    "arm_move_absent": "撤去移动指令",
    "arm_recovery_patch_absent": "删除恢复补丁",
    "arm_response_patch_absent": "删除响应补丁",
    "arm_delay": "延迟触发",
    "arm_suppress": "抑制触发",
    "arm_omit_anchor_move": "省略锚点移动",
    "arm_env_absent": "移除环境危害",
    "arm_env_weak": "减弱环境危害",
    "arm_time_delay": "延迟环境危害",
    "suppress_root": "移除根事件",
    "suppress_direct_consequence": "切断直接后果",
}


class ArmProfileError(ValueError):
    pass


def _err(message: str) -> ArmProfileError:
    return ArmProfileError(message)


def _label(arm_id: str) -> str:
    return ARM_LABELS.get(arm_id, arm_id)


def _patch_text(patch) -> str:
    if not isinstance(patch, dict):
        return json.dumps(patch, ensure_ascii=False)
    return json.dumps(patch, ensure_ascii=False, sort_keys=True)


def _action_brief(action: dict) -> dict:
    """One injected action row -> display fields, verbatim values."""
    tick = action.get("tick")
    inner = action.get("action") if isinstance(action.get("action"), dict) else action
    brief = {
        "action_id": inner.get("action_id"),
        "type": inner.get("type"),
        "entity_id": inner.get("entity_id"),
        "tick": tick,
        "delay_ticks": inner.get("delay_ticks"),
    }
    if isinstance(inner.get("state_patch"), dict):
        brief["state_patch"] = inner["state_patch"]
        brief["state_patch_text"] = _patch_text(inner["state_patch"])
    if "waypoints_enu_m" in inner:
        brief["waypoints_enu_m"] = inner["waypoints_enu_m"]
        brief["velocity_mps"] = inner.get("velocity_mps")
    if "profile" in inner and inner.get("type") == "set_weather":
        brief["weather_profile"] = inner.get("profile")
    return brief


def _describe_execution(config: dict, factual_actions: list[dict] | None) -> dict:
    """L1/L2/L4 ``execution`` configs: injected actions, omitted ids, retimes."""
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise _err(f"configuration lacks an execution mapping: {config.get('arm_id')}")
    injected = [_action_brief(a) for a in execution.get("actions") or []]
    omitted = []
    for action_id in execution.get("omit_action_ids") or []:
        resolved = None
        for row in factual_actions or []:
            raw = row["evidence"]["raw"]
            if raw.get("action", {}).get("action_id") == action_id:
                resolved = {
                    "action_id": action_id,
                    "tick": raw.get("tick"),
                    "entity_id": raw.get("action", {}).get("entity_id"),
                    "type": raw.get("action", {}).get("type"),
                    "state_patch": raw.get("action", {}).get("state_patch"),
                }
                break
        omitted.append({
            "action_id": action_id,
            "resolved": resolved is not None,
            "factual_action": resolved,
        })
    retimes = []
    for rewrite in execution.get("trigger_rewrites") or []:
        if isinstance(rewrite, dict):
            retimes.append({
                "trigger_id": rewrite.get("trigger_id"),
                "delay_ticks": rewrite.get("delay_ticks"),
                "mechanism": rewrite.get("mechanism"),
            })
    return {"injected": injected, "omitted": omitted, "retimes": retimes}


def _resolve_omitted(action_ids, factual_actions: list[dict] | None) -> list[dict]:
    omitted = []
    for action_id in action_ids or []:
        resolved = None
        for row in factual_actions or []:
            raw = row["evidence"]["raw"]
            if raw.get("action", {}).get("action_id") == action_id:
                resolved = {
                    "action_id": action_id,
                    "tick": raw.get("tick"),
                    "entity_id": raw.get("action", {}).get("entity_id"),
                    "type": raw.get("action", {}).get("type"),
                    "state_patch": raw.get("action", {}).get("state_patch"),
                }
                break
        omitted.append({
            "action_id": action_id,
            "resolved": resolved is not None,
            "factual_action": resolved,
        })
    return omitted


def _describe_l3(config: dict, factual_actions: list[dict] | None) -> dict:
    """L3 top-level ArmSpec: mechanism/inject/omit/sumo/retime verbatim."""
    parts = []
    mechanism = config.get("mechanism")
    if isinstance(mechanism, dict) and mechanism:
        parts.append(json.dumps(mechanism, ensure_ascii=False, sort_keys=True))
    return {
        "injected": [_action_brief(a) for a in config.get("inject_actions") or []],
        "omitted": _resolve_omitted(config.get("omit_action_ids"), factual_actions),
        "retimes": ([{
            "mechanism": json.dumps(config["retime"], ensure_ascii=False, sort_keys=True),
        }] if config.get("retime") else []),
        "sumo_intervention": config.get("sumo_intervention"),
        "state_overrides": config.get("state_overrides") or [],
        "mechanism_declared": parts[0] if parts else None,
    }


def _describe_l6(config: dict, factual_actions: list[dict] | None) -> dict:
    spec = config.get("spec")
    if not isinstance(spec, dict):
        raise _err("L6 configuration lacks a top-level spec mapping")
    mechanism = spec.get("mechanism")
    return {
        "injected": [_action_brief(a) for a in spec.get("inject_actions") or []],
        "omitted": _resolve_omitted(spec.get("omit_action_ids"), factual_actions),
        "retimes": ([{
            "mechanism": json.dumps(spec["retime"], ensure_ascii=False, sort_keys=True),
        }] if spec.get("retime") else []),
        "mechanism_declared": (
            json.dumps(mechanism, ensure_ascii=False, sort_keys=True)
            if isinstance(mechanism, dict) and mechanism else None),
        "cancelled_event_action_sets": (
            mechanism.get("cancelled_event_action_sets")
            if isinstance(mechanism, dict) else None),
    }


def _describe_l5(root: Path, config: dict, factual_config: dict) -> dict:
    sources = config["sources"]
    baseline = factual_config["sources"]
    branch_scene, _ = _read_json(root, sources["scene"])
    source_scene, _ = _read_json(root, baseline["scene"])
    branch_script, _ = _read_json(root, sources["script"])
    source_script, _ = _read_json(root, baseline["script"])
    def event_map(script):
        result = {}
        for event in script["events"]:
            key = event["event_id"]
            if key in result:
                raise _err(f"duplicate declared event {key}")
            result[key] = event
        return result
    source_events, branch_events = event_map(source_script), event_map(branch_script)
    source_triggers = {t["trigger_id"]: t for t in source_script["triggers"]}
    branch_triggers = {t["trigger_id"]: t for t in branch_script["triggers"]}
    trigger_changes = []
    action_changes = []
    for event_id, event in branch_events.items():
        if event_id not in source_events:
            continue
        before = source_events[event_id]
        bt = source_triggers[before["trigger_ref"]]
        at = branch_triggers[event["trigger_ref"]]
        if bt != at:
            trigger_changes.append({"event_id": event_id, "before": bt, "after": at})
        if before["actions"] != event["actions"]:
            action_changes.append({"event_id": event_id, "before": before["actions"], "after": event["actions"]})
    return {
        "declared_scene": sources["scene"], "declared_script": sources["script"],
        "source_scene": baseline["scene"], "source_script": baseline["script"],
        "weather_before": source_scene["weather_profile"],
        "weather_after": branch_scene["weather_profile"],
        "removed_event_ids": [e for e in source_events if e not in branch_events],
        "added_event_ids": [e for e in branch_events if e not in source_events],
        "trigger_changes": trigger_changes, "action_changes": action_changes,
        "status": config["status"],
    }


def _describe_x(config: dict) -> dict:
    removed = config.get("removed_event_ids")
    if not isinstance(removed, list):
        if config.get("mode") != "intervention":
            return {"removed_event_ids": [], "injected": [], "cut": None}
        raise _err("X configuration lacks removed_event_ids")
    return {
        "removed_event_ids": removed,
        "injected": [_action_brief(a) for a in config.get("intervention_actions") or []],
        "cut": (config.get("definition") or {}).get("cut"),
    }


def _expected(config: dict, family: str) -> dict | None:
    if family == "L6":
        spec = config.get("spec") or {}
        keys = ("target_event", "hypothesis", "falsified_if")
        out = {k: spec[k] for k in keys if k in spec}
        return out or None
    if family == "L3":
        keys = ("hypothesis", "falsified_if")
        out = {k: config[k] for k in keys if k in config}
        return out or None
    if family == "X":
        out = {}
        if "hypothesis" in config:
            out["hypothesis"] = config["hypothesis"]
        return out or None
    expected = config.get("expected")
    return expected if isinstance(expected, dict) else None


def _target_families(family: str, config: dict, manifest: dict) -> list[str]:
    if family == "L2":
        targets = config.get("target_families")
        if not isinstance(targets, list) or not targets:
            if config.get("mode") != "intervention":
                return []  # 对照臂没有 target_families，按业务观察列出全部记录
            raise _err(f"L2 configuration lacks target_families: {config.get('arm_id')}")
        return [str(t) for t in targets]
    if family in ("L3", "L4"):
        value = manifest.get("event_family") or config.get("event_family")
        if not value:
            if config.get("mode") != "intervention":
                return []
            raise _err(f"{family} arm lacks event_family in manifest and config")
        return [str(value)]
    if family == "L6":
        value = manifest.get("declared_anchor_family")
        if not value:
            if config.get("mode") != "intervention":
                return []
            raise _err("L6 manifest lacks declared_anchor_family")
        return [str(value)]
    if family == "L5":
        targets = config.get("event_families")
        if not isinstance(targets, list) or not targets:
            if config.get("mode") != "intervention":
                return []
            raise _err(f"L5 configuration lacks event_families: {config.get('arm_id')}")
        return [str(t) for t in targets]
    if family == "X":
        return []  # X rows are business observations keyed by predicate_id
    if family == "L1":
        return []  # L1 arms are factual-only observations; no target family declared
    raise _err(f"unknown family {family!r}")


def _measured(family: str, targets: list[str], occurrences: list[dict],
              outcomes: list[dict], window: dict) -> list[dict]:
    """Per target family: every occurrence with its own outcome; never only
    the first event; censored vs absent stay distinct."""
    outcome_by_event = {}
    for row in outcomes:
        _validated_outcome_row(row, family, window)
        raw = row["evidence"]["raw"]
        event_id = raw.get("event_id")
        if event_id is not None:
            if event_id in outcome_by_event:
                raise _err(f"duplicate outcome for event_id={event_id}")
            outcome_by_event[event_id] = row
    if family == "X":
        groups = [("业务观察（X 原始谓词）", None)]
        rows = occurrences
    elif family == "L1":
        groups = [("业务观察（L1 原始记录）", None)]
        rows = occurrences
    elif not targets:
        groups = [("业务观察（全部记录）", None)]
        rows = occurrences
    else:
        groups = [(t, t) for t in targets]
        rows = occurrences
    measured = []
    for label, target in groups:
        events = []
        for occ in rows:
            raw = occ["evidence"]["raw"]
            if target is not None:
                occ_family = raw["event_family_id"]
                if occ_family != target:
                    continue
            event_id = raw.get("event_id")
            outcome = outcome_by_event.get(event_id)
            terminal_tick = (outcome["evidence"]["raw"].get("terminal_tick")
                             if outcome else None)
            status = (outcome["evidence"]["raw"].get("status")
                      if outcome else None)
            has_outcome_row = outcome is not None
            censored = has_outcome_row and status == "right_censored"
            events.append({
                "event_id": event_id,
                "tick": raw["tick" if family == "X" else "trigger_tick"],
                "predicate_id": raw.get("predicate_id") or raw.get("trigger_predicate_id"),
                "outcome_id": (outcome["evidence"]["raw"].get("outcome_id")
                               if outcome else None),
                "terminal_tick": terminal_tick,
                "status": status,
                "window_censor_tick": (outcome["evidence"]["raw"].get("window_censor_tick")
                                       if outcome else None),
                "censored": censored,
                "has_outcome_row": has_outcome_row,
            })
        events.sort(key=lambda e: (e["tick"] is None, e["tick"]))
        if not events:
            text = "窗口内未发生"
        else:
            parts = []
            for e in events:
                head = f"发生 t={e['tick']}"
                if e["censored"]:
                    tail = f"右删失（窗口边界 t={e['window_censor_tick']}，terminal_tick=null）"
                elif e["status"] == "pending":
                    tail = "终态待定（terminal_tick=null）"
                elif e["has_outcome_row"] and e["terminal_tick"] is None:
                    tail = f"终态未记载（status={e['status']}）"
                elif e["has_outcome_row"]:
                    tail = f"终态 t={e['terminal_tick']} status={e['status']}"
                else:
                    tail = "无终态记录行"
                parts.append(f"{head}，{tail}")
            text = "；".join(parts)
        measured.append({
            "family": target,
            "label": label,
            "occurred": bool(events),
            "events": events,
            "text": text,
        })
    return measured


def _mechanism(family: str, config: dict, factual_actions: list[dict] | None,
               root: Path, factual_config: dict | None) -> dict:
    if family in ("L1", "L2", "L4"):
        return _describe_execution(config, factual_actions)
    if family == "L3":
        return _describe_l3(config, factual_actions)
    if family == "L6":
        return _describe_l6(config, factual_actions)
    if family == "L5":
        return _describe_l5(root, config, factual_config)
    if family == "X":
        return _describe_x(config)
    raise _err(f"unknown family {family!r}")


def _mechanism_summary(family: str, mechanism: dict) -> str:
    parts = []
    for action in mechanism.get("injected") or []:
        aid = action.get('action_id')
        head = (f"t={action['tick']} "
                + (f"动作 {aid}：" if aid else "")
                + f"对 {action.get('entity_id') or '?'} 注入 {action.get('type') or '动作'}")
        if "state_patch_text" in action:
            head += f"（{action['state_patch_text']}）"
        if "waypoints_enu_m" in action:
            head += (f"（路径改为 {json.dumps(action['waypoints_enu_m'])}"
                     f"，速度 {action.get('velocity_mps')} m/s）")
        if "weather_profile" in action:
            head += f"（weather profile={action['weather_profile']}）"
        parts.append(head)
    for omitted in mechanism.get("omitted") or []:
        resolved = omitted.get("factual_action")
        if omitted.get("resolved") and resolved:
            patch = resolved.get("state_patch")
            patch_text = f"，补丁 {_patch_text(patch)}" if patch else ""
            parts.append(
                f"删除 factual 动作 {omitted['action_id']}"
                f"（t={resolved.get('tick')}，对象 {resolved.get('entity_id')}"
                f"，类型 {resolved.get('type')}{patch_text}）")
        else:
            parts.append(
                f"声明删除动作 {omitted['action_id']}（factual raw/actions 中未找到该 action_id）")
    for retime in mechanism.get("retimes") or []:
        trigger = retime.get("trigger_id")
        head = f"延迟重定时 {trigger or ''}".strip()
        parts.append(f"{head}：{json.dumps(retime, ensure_ascii=False, sort_keys=True)}")
    sumo = mechanism.get("sumo_intervention")
    if isinstance(sumo, dict) and sumo:
        # 已知键给中文短句；未知键保留原文，不猜语义。release_tick=null 表示
        # factual 臂未声明释放时刻，保留 null 原文。
        head = "SUMO 道路变化：release_tick=" + json.dumps(sumo.get("release_tick"))
        if sumo.get("kind"):
            head += f"（{sumo['kind']}）"
        extra = {k: v for k, v in sumo.items()
                 if k not in ("release_tick", "kind")}
        if extra:
            head += "；" + json.dumps(extra, ensure_ascii=False, sort_keys=True)
        parts.append(head)
    if mechanism.get("mechanism_declared"):
        parts.append(f"声明机制：{mechanism['mechanism_declared']}")
    if mechanism.get("removed_event_ids"):
        parts.append("移除事件：" + ", ".join(mechanism["removed_event_ids"]))
    if mechanism.get("declared_scene"):
        parts.insert(0, "配置声明：天气 " + _patch_text(mechanism["weather_before"])
                     + " → " + _patch_text(mechanism["weather_after"]))
        if mechanism["added_event_ids"]:
            parts.append("新增事件：" + ", ".join(mechanism["added_event_ids"]))
        for change in mechanism["trigger_changes"]:
            parts.append(f"{change['event_id']} 触发条件："
                         + _patch_text(change["before"]) + " → " + _patch_text(change["after"]))
        for change in mechanism["action_changes"]:
            parts.append(f"{change['event_id']} 改写动作：" + ", ".join(a["type"] for a in change["after"]))
    if not parts:
        return "无干预改动（factual/对照臂）"
    return "；".join(parts)


def _factual_arm(arm_entries: list[dict]) -> tuple[dict | None, str | None]:
    factuals = [a for a in arm_entries if a["mode"] == "factual_observation"]
    if len(factuals) > 1:
        return None, f"组内有 {len(factuals)} 个 factual_observation 臂，无法确定唯一基线"
    return (factuals[0], None) if factuals else (None, "组内没有 factual_observation 臂")


def _profile_overview(root: Path, profile: str, seed: str,
                     catalog_data: dict | None = None) -> dict:
    """Overview of every group and arm in one profile+seed (light files only)."""
    root = Path(root)
    _validate_path_segment(profile, "profile")
    _validate_path_segment(seed, "seed")
    try:
        cat = catalog_data if catalog_data is not None else catalog(root)
    except ArmReviewError as exc:
        raise ArmProfileError(f"ARM 目录校验失败: {exc}") from exc
    groups = [g for g in cat["groups"]
              if g["profile"] == profile and g["seed"] == seed]
    if not groups:
        raise ArmProfileError(f"index 中没有 profile={profile} seed={seed} 的组")

    groups_out = []
    for group_entry in groups:
        arm_entries = group_entry["arms"]
        factual_entry, factual_note = _factual_arm(arm_entries)
        factual_actions: list[dict] | None = None
        factual_config = None
        factual_occurrences = factual_outcomes = None
        if factual_entry is not None:
            factual_config = _load_configuration(root, factual_entry["configuration"])["content"]
            factual_occurrences = _read_jsonl(root, f"{factual_entry['path']}/objective/event_occurrences.jsonl")
            factual_outcomes = _read_jsonl(root, f"{factual_entry['path']}/objective/event_outcomes.jsonl")
            factual_actions = _read_jsonl(
                root, f"{factual_entry['path']}/raw/actions.jsonl")

        arms_out = []
        for arm_entry in arm_entries:
            arm_dir = arm_entry["path"]
            try:
                manifest, manifest_evidence = _read_json(root, f"{arm_dir}/manifest.json")
                window_status, _ = _read_json(
                    root, f"{arm_dir}/objective/window_status.json")
                config = _load_configuration(root, arm_entry["configuration"])
                occurrences = _read_jsonl(
                    root, f"{arm_dir}/objective/event_occurrences.jsonl")
                outcomes = _read_jsonl(
                    root, f"{arm_dir}/objective/event_outcomes.jsonl")
                actions = _read_jsonl(root, f"{arm_dir}/raw/actions.jsonl")
            except ArmReviewError as exc:
                raise ArmProfileError(
                    f"{arm_entry['arm_key']} 产物缺失或损坏: {exc}") from exc

            family = arm_entry["family"]
            targets = _target_families(family, config["content"], manifest)
            mechanism = (_mechanism(family, config["content"], factual_actions, root, factual_config)
                         if arm_entry["mode"] == "intervention" else {})
            comparable = factual_entry is not None and arm_entry["mode"] == "intervention"
            note = None
            if arm_entry["mode"] != "intervention":
                note = "对照臂，不作为干预展示"
            elif factual_entry is None:
                note = factual_note

            arms_out.append({
                "group_key": f"{profile}/{seed}/{group_entry['group']}",
                "group": group_entry["group"],
                "arm_key": arm_entry["arm_key"],
                "arm_id": arm_entry["arm_id"],
                "family": family,
                "mode": arm_entry["mode"],
                "name": _label(arm_entry["arm_id"]),
                "window": manifest.get("window"),
                "mechanism": mechanism,
                "mechanism_summary": _mechanism_summary(family, mechanism),
                "is_business_observation": family == "X",
                "expected": _expected(config["content"], family),
                "factual_measured": (_measured(family, targets, factual_occurrences, factual_outcomes,
                                                factual_entry["window"])
                                     if factual_entry is not None else None),
                "factual_window": (next(g["window"] for g in arm_entries if g["mode"] == "factual_observation")
                                   if factual_entry is not None else None),
                "measured": _measured(
                    family, targets, occurrences, outcomes, manifest.get("window")),
                "evidence": {
                    "configuration": config["path"],
                    "manifest": f"{arm_dir}/manifest.json",
                    "window_status": f"{arm_dir}/objective/window_status.json",
                    "raw_actions": f"{arm_dir}/raw/actions.jsonl",
                    "event_occurrences": f"{arm_dir}/objective/event_occurrences.jsonl",
                    "event_outcomes": f"{arm_dir}/objective/event_outcomes.jsonl",
                },
                "comparable": comparable,
                "comparability_note": note,
                "window_status_status": window_status.get("status"),
            })

        interventions = [a for a in arms_out if a["mode"] == "intervention"]
        groups_out.append({
            "group_key": f"{profile}/{seed}/{group_entry['group']}",
            "group": group_entry["group"],
            "family": group_entry["family"],
            "arms": arms_out,
            "factual": next(
                (a for a in arms_out if a["mode"] == "factual_observation"), None),
            "interventions": interventions,
            "note": ("仅保留 factual，干预待重新设计" if not interventions
                     else factual_note),
        })

    return {
        "profile": profile,
        "seed": seed,
        "groups": groups_out,
        "counts": {
            "groups": len(groups_out),
            "arms": sum(len(g["arms"]) for g in groups_out),
            "interventions": sum(len(g["interventions"]) for g in groups_out),
        },
    }


def profile_overview(root: Path, profile: str, seed: str,
                     catalog_data: dict | None = None) -> dict:
    try:
        return _profile_overview(root, profile, seed, catalog_data)
    except (ArmReviewError, KeyError, TypeError) as exc:
        raise ArmProfileError(f"{profile}/{seed}: invalid or missing profile evidence: {exc}") from exc
