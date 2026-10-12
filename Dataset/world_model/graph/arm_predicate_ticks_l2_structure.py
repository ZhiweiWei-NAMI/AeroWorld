"""L2 ARM 密集谓词 tick 的封闭记录文法（source_family: arm_predicate_truth_ticks）。

物理文件 objective/predicate_truth_ticks.jsonl 在 L2 只承载一种记录类型。
生产方 Dataset/tools/l2_window.py:452 将 l2_arm_semantics.recompute 的
semantic['predicate_truth_ticks'] 原样写出；密集行由 l2_arm_semantics.py:453-455
的固定 8 键 dict 构造：tick / predicate_id / bindings / binding_ontology_classes /
value / context / source_refs / missing_source_record。本层没有任何写
arm_domain_state_ticks 的入口：领域模拟行由 domain_state._observation_row
（domain_state.py:3769-3809）写入 domain_state_observations.jsonl，不进入本文件。
本模块因此只声明谓词行文法；先前版本中的 arm_domain_state_ticks_schema 及其专用
常量/helper/导出已按当前生产代码移除，本层有效 effective_family 只有
arm_predicate_truth_ticks。若任务冻结来源日后实际出现领域行，先报告具体来源与
记录位置，再依据该来源声明结构。

- tick：l2_arm_semantics.py:336 为 ``range(0, config['window']['end_tick']+1)``，
  而 read_config（l2_window.py:100-103）要求窗口端点 ≤900 且落在 5 tick 网格，
  故取值为 0..900 的整数（本轮修复补上此前缺失的 maximum:900）。
- predicate_id：15 个封闭枚举（l2_arm_semantics.py:82-96），oneOf 分支一一对应。
- bindings：角色键集即 scope_for（l2_arm_semantics.py:168-253）逐 add() 调用的
  实参键；值是真实实体/服务/派生区 ID（如 uav_l2_1_v1、tower_l2_1_v1、
  gnss_service:aeroworld_global_navigation、assigned_landing_zone:uav_l2_3_v1），
  只约束字符串形态，不枚举样本实体、不猜正则。
- binding_ontology_classes：add()（194-199 行）按注册表
  Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json 各模板
  argument_roles 写 ``'world:' + class``，并断言与 bindings 键集完全一致，故每个
  角色取值是固定 const（本轮修复：由原来的任意字符串改为逐角色 const）。分支内
  角色→类映射逐一转录自该注册表；类名本体仍以注册表为权威：
    actor/aircraft/first_aircraft/second_aircraft → UnmannedAircraft；
    home_pad/pad → LandingPad；landing_zone → LandingArea；
    structure → BuildingStructure；corridor → AirspaceCorridor；
    station（C2 两谓词）→ CommunicationStation；station（充电两谓词）→
    ChargingStation；service → GnssService；signal → TrafficSignal。
- value：evaluate_known（113-119 行）只返回 "true"/"false"/"unknown"；
  out_of_scope 只进入矩阵层，不进入密集行。
- context 各 kind 形状（l2_arm_semantics.py:391-449）：
    altitude   geometry.position_z_m（行缺测为 null）+ assigned_altitude_m；
    aircraft_ground / landing  geometry 取 _ground_contexts（256-301 行）行或
      空对象，键 z_agl_m / local_ground_reference_z_m /
      xy_distance_to_assigned_landing_zone_m / speed_mps 全可省但不得有其他键
      （冻结行里两 kind 出现过键子集，如 L2-1_v1 行 4/7；当前 writer 写全 4 键）；
    home       geometry.xy_distance_to_home_pad_m（行或 pad 缺测为 null）；
    pair       derived.nearest_aircraft_distance_m（任一端缺行为 null）；
    structure  derived.nearest_building_distance_m（任一端缺测为 null）；
    corridor   scene.maximum_corridor_occupancy_count（scope['uav_ids'] 缺测时
      为 null）+ maximum_corridor_capacity（_airspace_corridors，
      predicate_state_computers.py:2927-2934 的整数容量）；
    communication  仅 direct_predicate_value，取值来自
      compute_comm._truth_from_link_unavailable / _truth_from_link_degraded
      （compute_comm.py:2498-2517），域为 "true"/"false"/"unknown"；
    gnss       domain.gnss_navigation.quality_level：正式 tick 取
      domain_state._gnss_values 的 "nominal"/"degraded"/"spoofing_suspect"，
      回退/缺例行 "unknown"（345/349 行），非正式 tick 为 null；
    charger    domain.pad_facility 六键（recompute 358-384 行）：availability
      可空，命中计划时为 "available"/"unavailable"；fault_active 为布尔、null
      （设施无 fault 字段）或 "unknown"；requester_count/capacity 为整数；
      requester_ids/missing 为字符串数组；
    pad        domain.pad_facility 三键 pair_contention（requester_ids 非 list
      时为 null）/ requester_ids（可空字符串数组）/ capacity（437-446 行）；
    signal     domain.signal_queue.controller_health：null / "all_red_fault" /
      "nominal"（447-449 行）。
- source_refs：389-390 行保证恒非空（无动态实体引用时回退 raw/metadata.json）；
  corridor 行固定追加全部 uav 引用与 raw/metadata.json（415 行）。元素为
  "raw/states.jsonl#tick=…&entity=…" 或精确的 "raw/metadata.json"；tick 是
  0..900 的规范十进制整数，entity 来自非空 entity_ids，字符集不另作假设。
  dynamic 由 set 去重，scope['uav_ids'] 由去重的 entity_ids 生成；引用不重复。
- missing_source_record：字符串数组（如 "facility_state.fault_missing"、GNSS
  非正式 tick 说明），可为空。

无实例分支（尚无实例验证）：本轮抽样未观测到任何 null/unknown 子分支取值
（geometry.position_z_m=null、home/pair/structure 距离 null、corridor 计数 null、
availability=null、fault_active=null 与 "unknown"、pair_contention=null、
controller_health=null、quality_level=null 与 "unknown"、communication
"unknown"、landing/aircraft_ground 空对象）。这些分支由上述生产代码路径证明
可写，保持声明并标「尚无实例验证」；bindings 值、样本实体清单与各 profile 实际
出现的谓词集合不作为封闭依据。

完整性元数据：密集行不含 digest/hash/sha256 字段；
structure_schema._PUBLICATION_EXCLUDED_PATHS 与本模块声明字段无交集，本任务不
修改该全局排除集合，validate_record 也不应用发布投影。

校验范围：本次用 Python -B 编译，并从 L2-1_v1、L2-2_v1、L2-3_v1、
L2-4_v1、L2-4_v2、L2-5_v1 各一份真实 JSONL 中分别验证 12 行，共 72 行；
样本含 77 个 states 引用及 1 个 metadata 引用。此抽样只核对本次改动的
source_refs 约束，不代替整库重建。
"""

from __future__ import annotations

from typing import Mapping

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

_ALT = "airspace.aircraft_outside_assigned_altitude_band"
_AIR = "airspace.aircraft_airborne"
_TOUCH = "airspace.aircraft_touchdown"
_HOME = "airspace.aircraft_at_home_pad"
_LAND = "airspace.aircraft_in_landing_zone"
_PAIR = "safety.aircraft_pair_below_separation"
_STRUCTURE = "safety.aircraft_near_building_structure"
_CORRIDOR = "airspace.airspace_corridor_over_capacity"
_C2_LOSS = "communication.link_unavailable"
_C2_DEG = "communication.link_degraded"
_GNSS = "positioning.gnss_quality_degraded"
_CHARGER = "facility.charging_station_unavailable"
_QUEUE = "facility.charging_queue_over_capacity"
_PAD = "facility.landing_pad_contention"
_SIGNAL = "ground.traffic_signal_malfunction_active"

# 谓词 → (context kind, 角色→本体类)。角色键序即 scope_for 的 add() 实参键序，
# 类名逐一转录自 core_semantic_predicate_templates.json 各模板 argument_roles。
_VARIANTS: dict[str, tuple[str, dict[str, str]]] = {
    _ALT: ("altitude", {"actor": "UnmannedAircraft"}),
    _AIR: ("aircraft_ground", {"aircraft": "UnmannedAircraft"}),
    _TOUCH: ("aircraft_ground", {"aircraft": "UnmannedAircraft"}),
    _HOME: ("home", {"aircraft": "UnmannedAircraft", "home_pad": "LandingPad"}),
    _LAND: ("landing", {"aircraft": "UnmannedAircraft",
                        "landing_zone": "LandingArea"}),
    _PAIR: ("pair", {"first_aircraft": "UnmannedAircraft",
                     "second_aircraft": "UnmannedAircraft"}),
    _STRUCTURE: ("structure", {"aircraft": "UnmannedAircraft",
                               "structure": "BuildingStructure"}),
    _CORRIDOR: ("corridor", {"corridor": "AirspaceCorridor"}),
    _C2_LOSS: ("communication", {"actor": "UnmannedAircraft",
                                 "station": "CommunicationStation"}),
    _C2_DEG: ("communication", {"actor": "UnmannedAircraft",
                                "station": "CommunicationStation"}),
    _GNSS: ("gnss", {"actor": "UnmannedAircraft", "service": "GnssService"}),
    _CHARGER: ("charger", {"station": "ChargingStation"}),
    _QUEUE: ("charger", {"station": "ChargingStation"}),
    _PAD: ("pad", {"first_aircraft": "UnmannedAircraft", "pad": "LandingPad",
                   "second_aircraft": "UnmannedAircraft"}),
    _SIGNAL: ("signal", {"signal": "TrafficSignal"}),
}


def _object(properties: Mapping[str, dict],
            *, required: tuple[str, ...] | None = None) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": dict(properties),
    }


def _strings(*, min_items: int = 0) -> dict:
    return {"type": "array", "items": {"type": "string"}, "minItems": min_items}


def _source_refs() -> dict:
    return {
        "type": "array",
        "items": {"oneOf": [
            {"type": "string", "pattern": (
                r"^raw/states\.jsonl#tick=(0|[1-9][0-9]?|[1-8][0-9]{2}|900)"
                r"&entity=[\s\S]+$")},
            {"const": "raw/metadata.json"},
        ]},
        "minItems": 1,
        "uniqueItems": True,
    }


def _nullable(*branches: dict) -> dict:
    return {"oneOf": [{"type": "null"}, *branches]}


def _truth_value() -> dict:
    return {"enum": ["true", "false", "unknown"]}


def _tick() -> dict:
    # read_config 要求窗口端点 ≤900；重算从 tick 0 起。
    return {"type": "integer", "minimum": 0, "maximum": 900}


def _altitude_context() -> dict:
    return _object({"geometry": _object({
        "position_z_m": _nullable({"type": "number"}),
        "assigned_altitude_m": {"type": "number"},
    })})


def _ground_geometry() -> dict:
    # aircraft_ground/landing 取 _ground_contexts 行或空对象；四键全可省但不得有其他键。
    return _object({
        "z_agl_m": {"type": "number"},
        "local_ground_reference_z_m": {"type": "number"},
        "xy_distance_to_assigned_landing_zone_m": {"type": "number"},
        "speed_mps": {"type": "number"},
    }, required=())


def _ground_context() -> dict:
    return _object({"geometry": _ground_geometry()})


def _home_context() -> dict:
    return _object({"geometry": _object({
        "xy_distance_to_home_pad_m": _nullable({"type": "number"}),
    })})


def _pair_context() -> dict:
    return _object({"derived": _object({
        "nearest_aircraft_distance_m": _nullable({"type": "number"}),
    })})


def _structure_context() -> dict:
    return _object({"derived": _object({
        "nearest_building_distance_m": _nullable({"type": "number"}),
    })})


def _corridor_context() -> dict:
    return _object({"scene": _object({
        "maximum_corridor_occupancy_count": _nullable({"type": "integer"}),
        "maximum_corridor_capacity": {"type": "integer"},
    })})


def _communication_context() -> dict:
    return _object({"direct_predicate_value": _truth_value()})


def _gnss_context() -> dict:
    return _object({"domain": _object({"gnss_navigation": _object({
        "quality_level": _nullable({"enum": [
            "nominal", "degraded", "spoofing_suspect", "unknown"]}),
    })})})


def _charger_facility() -> dict:
    return _object({
        "availability": _nullable({"enum": ["available", "unavailable"]}),
        "requester_count": {"type": "integer"},
        "capacity": {"type": "integer"},
        "requester_ids": _strings(),
        "fault_active": {"oneOf": [{"type": "boolean"}, {"type": "null"},
                                   {"const": "unknown"}]},
        "missing": _strings(),
    })


def _charger_context() -> dict:
    return _object({"domain": _object({"pad_facility": _charger_facility()})})


def _pad_context() -> dict:
    return _object({"domain": _object({"pad_facility": _object({
        "pair_contention": _nullable({"type": "boolean"}),
        "requester_ids": _nullable({"type": "array",
                                    "items": {"type": "string"}}),
        "capacity": {"type": "integer"},
    })})})


def _signal_context() -> dict:
    return _object({"domain": _object({"signal_queue": _object({
        "controller_health": _nullable({"enum": ["all_red_fault", "nominal"]}),
    })})})


_CONTEXT_FACTORIES = {
    "altitude": _altitude_context,
    "aircraft_ground": _ground_context,
    "landing": _ground_context,
    "home": _home_context,
    "pair": _pair_context,
    "structure": _structure_context,
    "corridor": _corridor_context,
    "communication": _communication_context,
    "gnss": _gnss_context,
    "charger": _charger_context,
    "pad": _pad_context,
    "signal": _signal_context,
}


def _tick_branch(predicate_id: str) -> dict:
    kind, role_classes = _VARIANTS[predicate_id]
    roles = tuple(role_classes)
    return _object({
        "tick": _tick(),
        "predicate_id": {"const": predicate_id},
        "bindings": _object({role: {"type": "string"} for role in roles}),
        "binding_ontology_classes": _object({
            role: {"const": "world:" + ontology_class}
            for role, ontology_class in role_classes.items()}),
        "value": _truth_value(),
        "context": _CONTEXT_FACTORIES[kind](),
        "source_refs": _source_refs(),
        "missing_source_record": _strings(),
    })


def arm_predicate_truth_ticks_schema() -> dict:
    """Return the closed L2 dense-predicate tick schema for one jsonl row."""
    return {
        "$schema": _DRAFT,
        "title": "L2 ARM predicate truth tick record",
        "oneOf": [_tick_branch(predicate_id) for predicate_id in _VARIANTS],
    }


__all__ = ["arm_predicate_truth_ticks_schema"]
