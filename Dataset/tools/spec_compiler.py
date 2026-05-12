"""
ScenarioSpec → event_script_v1 JSON 编译器。

将高层声明式的 ScenarioSpec 编译为 EventScriptInterpreter 可直接加载的
event_script.json 格式。编译过程不改变解释器语义——只是提供更紧凑的
场景描述方式以减少 61 个脚本的重复性手工 JSON 编写。

使用方式:
    from spec_compiler import ScenarioSpec, EntitySpec, EventStepSpec, TriggerSpec, ActionSpec, SpecCompiler
    spec = ScenarioSpec(...)
    compiler = SpecCompiler()
    json_dict = compiler.compile(spec)
    # json_dict 可直接传给 EventScriptInterpreter(json_dict)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import json


# ---------------------------------------------------------------------------
# Data classes — 场景描述
# ---------------------------------------------------------------------------

@dataclass
class WaypointSpec:
    """路径点"""
    pos_enu_m: list[float]  # [x, y, z]
    hold_ticks: int = 0     # 在此点停留的帧数


@dataclass
class EntitySpec:
    """初始实体定义"""
    entity_id: str
    asset_id: str                     # 引用 asset_catalog.json
    initial_pos_enu: list[float]      # [x, y, z]
    initial_rotation_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    movement_waypoints: list[WaypointSpec] = field(default_factory=list)
    visual_state: Optional[dict] = None  # {mode: "idle", lights_on: true, ...}


@dataclass
class TriggerSpec:
    """
    触发器描述。根据 type 不同携带不同参数。

    Types:
      - "tick": 需要 tick
      - "weather_state": 需要 weather_parameter, weather_operator, weather_value, [sustain_ticks]
      - "entity_proximity": 需要 entity_a, entity_b, distance_m, [proximity_operator], [min_true_ticks]
      - "event_fired": 需要 event_ref (指向另一个 event 的 event_id)
      - "composite": 需要 composite_operator ("AND"/"OR"), composite_children (trigger_ids 列表)
    """
    type: str
    tick: Optional[int] = None
    # weather_state
    weather_parameter: Optional[str] = None
    weather_operator: str = "gte"
    weather_value: Optional[float] = None
    sustain_ticks: int = 0
    # entity_proximity
    entity_a: Optional[str] = None
    entity_b: Optional[str] = None
    distance_m: Optional[float] = None
    proximity_operator: str = "lte"
    min_true_ticks: int = 1
    metric: str = "xy"
    horizontal_distance_m: Optional[float] = None
    vertical_distance_m: Optional[float] = None
    # event_fired
    event_ref: Optional[str] = None
    delay_ticks: int = 0
    # composite
    composite_operator: Optional[str] = None   # "AND" | "OR"
    composite_children: Optional[list[str]] = None  # trigger_id 列表


@dataclass
class ActionSpec:
    """动作描述。params 直接映射到 event_script_v1 action dict 的非 type 字段。"""
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class EventStepSpec:
    """
    事件链中的一个步骤。

    每个步骤 = trigger + actions + 可选的 emit_events。
    编译器将 trigger 映射为 trigger_ref，将 on_fire_emit 映射为事件链传播。
    """
    event_id: str
    trigger: TriggerSpec
    actions: list[ActionSpec] = field(default_factory=list)
    on_fire_emit: list[str] = field(default_factory=list)  # 触发下游 event_id 列表
    priority: int = 10
    max_fire_count: int = 1
    cooldown_ticks: int = 0
    require_conditions: list[str] = field(default_factory=list)  # 额外 trigger_id 条件
    log_topic: str = ""
    log_category: str = ""
    log_title: str = ""
    log_severity: str = "info"      # info | warning | critical
    log_overlay: str = ""
    log_target_ids: list[str] = field(default_factory=list)
    intent: str = ""
    intent_stage: str = ""
    causal_chain_id: str = ""
    causal_predecessor_intent: str = ""
    target_roles: list[str] = field(default_factory=list)


@dataclass
class ScenarioSpec:
    """一个场景的完整声明式描述。"""
    scenario_id: str
    category: str              # 如 "agent_interaction.uav_pedestrian"
    description: str
    duration_ticks: int = 900
    entities: list[EntitySpec] = field(default_factory=list)
    event_chain: list[EventStepSpec] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 编译器
# ---------------------------------------------------------------------------

class SpecCompiler:
    """将 ScenarioSpec 编译为 event_script_v1 JSON dict。"""

    def compile(self, spec: ScenarioSpec) -> dict:
        self._spec = spec
        self._trigger_map: dict[str, str] = {}   # event_id -> trigger_id
        self._compiled_triggers: list[dict] = []
        self._used_trigger_ids: set[str] = set()

        triggers = self._compile_triggers()
        events = self._compile_events()

        # 去重 triggers (相同内容的 trigger 合并) 并收集实际被引用的
        events = self._resolve_param_refs(events, spec.parameters)

        return {
            "$schema": "event_script_v1",
            "scenario_id": spec.scenario_id,
            "description": spec.description,
            "parameters": spec.parameters,
            "triggers": triggers,
            "events": events,
        }

    # ---- triggers ----

    def _compile_triggers(self) -> list[dict]:
        """从 event_chain 中提取所有触发器。"""
        triggers: list[dict] = []
        seen: dict[str, int] = {}  # trigger signature -> index in triggers

        for step in self._spec.event_chain:
            trigger_dict = self._trigger_spec_to_dict(step.trigger, step.event_id)
            sig = self._trigger_signature(trigger_dict)

            if sig in seen:
                # 复用已有 trigger
                self._trigger_map[step.event_id] = triggers[seen[sig]]["trigger_id"]
            else:
                seen[sig] = len(triggers)
                triggers.append(trigger_dict)
                self._trigger_map[step.event_id] = trigger_dict["trigger_id"]

            # 同时注册 require_conditions 中用到的 trigger (composite children)
            # composite 的 children 是 trigger_id，需要在编译前就存在
            # 这里先注册，后续 _compile_events 时再处理

        # 第二遍：处理 composite trigger 中的 children 引用
        for i, t in enumerate(triggers):
            if t.get("type") == "composite" and t.get("children"):
                resolved = []
                for child in t["children"]:
                    # child 可能是 event_id (需要映射) 或已经是 trigger_id
                    if child in self._trigger_map:
                        resolved.append(self._trigger_map[child])
                    else:
                        resolved.append(child)
                triggers[i]["children"] = resolved

        return triggers

    def _trigger_spec_to_dict(self, ts: TriggerSpec, event_id: str) -> dict:
        """将 TriggerSpec 转换为 event_script_v1 trigger dict。"""
        tid = self._make_trigger_id(event_id, ts)
        base = {"trigger_id": tid, "type": ts.type}

        if ts.type == "tick":
            base["tick"] = ts.tick

        elif ts.type == "weather_state":
            base["parameter"] = ts.weather_parameter
            base["operator"] = ts.weather_operator
            base["value"] = ts.weather_value
            if ts.sustain_ticks:
                base["sustain_ticks"] = ts.sustain_ticks

        elif ts.type == "entity_proximity":
            base["entity_a"] = ts.entity_a
            base["entity_b"] = ts.entity_b
            base["distance_m"] = ts.distance_m
            base["operator"] = ts.proximity_operator
            if ts.metric:
                base["metric"] = ts.metric
            if ts.horizontal_distance_m is not None:
                base["horizontal_distance_m"] = ts.horizontal_distance_m
            if ts.vertical_distance_m is not None:
                base["vertical_distance_m"] = ts.vertical_distance_m
            if ts.min_true_ticks > 1:
                base["min_true_ticks"] = ts.min_true_ticks

        elif ts.type == "event_fired":
            base["event_id"] = ts.event_ref

        elif ts.type == "event_fired_after":
            base["event_id"] = ts.event_ref
            base["delay_ticks"] = ts.delay_ticks

        elif ts.type == "composite":
            base["operator"] = ts.composite_operator
            # children: 先存原始引用（event_id 或 trigger_id），compile_triggers 第二遍解析
            base["children"] = list(ts.composite_children) if ts.composite_children else []

        return base

    def _make_trigger_id(self, event_id: str, ts: TriggerSpec) -> str:
        """生成唯一的 trigger_id。"""
        if ts.type == "event_fired":
            return f"trig_after_{ts.event_ref}"
        if ts.type == "event_fired_after":
            return f"trig_after_{ts.event_ref}_delay_{ts.delay_ticks}"
        return f"trig_{event_id}"

    def _trigger_signature(self, td: dict) -> str:
        """计算 trigger 的去重签名。排除 trigger_id 本身。"""
        items = sorted(
            (k, str(v)) for k, v in td.items()
            if k != "trigger_id"
        )
        return json.dumps(items, sort_keys=True)

    # ---- events ----

    def _compile_events(self) -> list[dict]:
        """编译事件列表。"""
        events = []
        for i, step in enumerate(self._spec.event_chain):
            trigger_id = self._trigger_map.get(step.event_id, f"trig_{step.event_id}")

            # 构建 require_conditions: 将 event_id 引用转换为 trigger_id
            require_conds = []
            for cond in step.require_conditions:
                if cond in self._trigger_map:
                    require_conds.append(self._trigger_map[cond])
                else:
                    require_conds.append(cond)

            event = {
                "event_id": step.event_id,
                "trigger_ref": trigger_id,
                "priority": step.priority if step.priority else (i + 1),
                "max_fire_count": step.max_fire_count,
                "actions": [self._action_spec_to_dict(a) for a in step.actions],
            }

            if step.cooldown_ticks:
                event["cooldown_ticks"] = step.cooldown_ticks
            if require_conds:
                event["require_conditions"] = require_conds
            if step.on_fire_emit:
                event["on_fire"] = {"emit_events": list(step.on_fire_emit)}
            if step.intent:
                event["intent"] = step.intent
            if step.intent_stage:
                event["intent_stage"] = step.intent_stage
            if step.causal_chain_id:
                event["causal_chain_id"] = step.causal_chain_id
            event["causal_predecessor_intent"] = step.causal_predecessor_intent
            if step.target_roles:
                event["target_roles"] = list(step.target_roles)
            if step.log_topic:
                event["log_event"] = {
                    "topic": step.log_topic,
                    "category": step.log_category,
                    "title": step.log_title,
                    "severity": step.log_severity,
                    "overlay": step.log_overlay,
                    "target_ids": list(step.log_target_ids),
                }
                if step.intent:
                    event["log_event"]["intent"] = step.intent
                if step.intent_stage:
                    event["log_event"]["intent_stage"] = step.intent_stage
                if step.causal_chain_id:
                    event["log_event"]["causal_chain_id"] = step.causal_chain_id
                if step.causal_predecessor_intent:
                    event["log_event"]["causal_predecessor_intent"] = step.causal_predecessor_intent
                if step.target_roles:
                    event["log_event"]["target_roles"] = list(step.target_roles)

            events.append(event)

        return events

    def _action_spec_to_dict(self, a: ActionSpec) -> dict:
        """将 ActionSpec 转换为 event_script_v1 action dict。"""
        result = {"type": a.type}
        result.update(a.params)
        return result

    # ---- $param 引用 ----

    def _resolve_param_refs(self, events: list[dict], params: dict) -> list[dict]:
        """
        将参数值中的 "$param.xxx" 引用替换为实际值。

        如果值以 "$param." 开头，且参数表中存在对应 key，
        则替换为参数表中指定的值。否则保留原样（延迟绑定，
        由 EventScriptInterpreter 在运行时解析）。
        """
        for param_key, param_value in params.items():
            ref = f"$param.{param_key}"
            events = self._replace_in_events(events, ref, param_value)
        return events

    def _replace_in_events(self, events: list[dict], ref: str, value: Any) -> list[dict]:
        """在 events 结构中递归替换 $param 引用。"""
        # Replace only exact string values, never substrings inside larger strings.
        def _replace(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_replace(v) for v in obj]
            if isinstance(obj, str) and obj == ref:
                return value
            return obj

        return _replace(events)


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------

def make_tick_trigger(tick: int) -> TriggerSpec:
    return TriggerSpec(type="tick", tick=tick)

def make_weather_trigger(parameter: str, operator: str, value: float,
                         sustain_ticks: int = 0) -> TriggerSpec:
    return TriggerSpec(
        type="weather_state",
        weather_parameter=parameter,
        weather_operator=operator,
        weather_value=value,
        sustain_ticks=sustain_ticks,
    )

def make_proximity_trigger(entity_a: str, entity_b: str, distance_m: float,
                           operator: str = "lte", min_true_ticks: int = 1,
                           metric: str = "xy",
                           horizontal_distance_m: float | None = None,
                           vertical_distance_m: float | None = None) -> TriggerSpec:
    return TriggerSpec(
        type="entity_proximity",
        entity_a=entity_a,
        entity_b=entity_b,
        distance_m=distance_m,
        proximity_operator=operator,
        min_true_ticks=min_true_ticks,
        metric=metric,
        horizontal_distance_m=horizontal_distance_m,
        vertical_distance_m=vertical_distance_m,
    )

def make_event_fired_trigger(event_ref: str) -> TriggerSpec:
    return TriggerSpec(type="event_fired", event_ref=event_ref)

def make_event_fired_after_trigger(event_ref: str, delay_ticks: int) -> TriggerSpec:
    return TriggerSpec(type="event_fired_after", event_ref=event_ref, delay_ticks=delay_ticks)

def make_composite_trigger(operator: str, children: list[str]) -> TriggerSpec:
    return TriggerSpec(
        type="composite",
        composite_operator=operator,
        composite_children=children,
    )
