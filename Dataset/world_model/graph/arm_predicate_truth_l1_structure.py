"""arm_predicate_truth（L1 窗口实验）冻结记录的封闭整记录语法。

仅本文件可写。导出 :func:`arm_predicate_truth_schema`：返回 L1 家族
``arm_predicate_truth``（``objective/predicate_truth.jsonl``）冻结 3.0.0
记录的 Draft 2020-12 封闭 JSON Schema；不做 I/O，import 无副作用，
编译与校验由调用方（G1 结构闭包）执行。

生产链（当前会话实读证据）：
``scripts/arms/L1-*/.../arm_*.py`` → ``Dataset/tools/l1_window.py:execute_config``
（窗口过滤 l1_window.py:448-452，写出 l1_window.py:465
``objective/predicate_truth.jsonl``）→ ``l1_window.semantic_window`` 调
``build_episode_objective_artifacts`` → ``Dataset/tools/l1_semantics.py:project``
（证据上下文 l1_semantics.py:96-127，采样/断言 l1_semantics.py:131-137）→
``Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_assertions``
（807-865 行，逐键拼装）。

冻结契约（21 个顶层键，固定顺序无关）：
schema_name/schema_version/annotation_layer/source_layer/materialization_policy/
truth_id/episode_id/tick/authoritative_tick/predicate_id/tuple_id/bindings/
value/evidence/source_class/mechanism_id/template_parameters/rule_digest/
parameter_digest/input_digest；``evidence = {source_refs, observations,
missing_requirements}``，``observations`` 恰一个元素，
``observations[].value`` 按谓词分支携带唯一证据节。

分支与实例依据（本会话逐行读取的记录，均为 3.0.0、21 键、
authoritative_tick=true、parameter_digest=sha256:27092370…a68ca0）：
- airspace.aircraft_inside_restricted_airspace：
  arms/L1-1_v1/seed00/grp_remove_interrupting_hover/arm_factual/objective/
  predicate_truth.jsonl 第 1 行（tick=255，value=false）。
- airspace.aircraft_outside_assigned_altitude_band：
  arms/L1-2_v1/seed00/grp_prevent_altitude_excursion/arm_factual/objective/
  predicate_truth.jsonl 第 1 行（tick=215，value=false）。
- safety.aircraft_pair_below_separation：
  arms/L1-3_v1/seed00/grp_prevent_target_pair_convergence/arm_factual/objective/
  predicate_truth.jsonl 第 1 行（tick=215，value=true）。
- airspace.airspace_corridor_over_capacity：
  arms/L1-4_v1/seed00/grp_staggered_admission/arm_factual/objective/
  predicate_truth.jsonl 第 1 行（tick=360，value=false，occupancy=0）。
  占用数>0 的记录未在本会话复核，scene 节字段集以
  l1_semantics.py:111-116 为准（pair_positions/occupied_ids/corridor_active
  只进稠密行，不进 observations[].value）。

声明的未实例路径（尚无实例验证）：``value``/``world_truth_value`` 的
``"out_of_scope"`` 来自 l1_semantics.py:127 的缺位路径，本会话抽样的
记录只出现 "true"/"false"。

未声明路径（有生产代码分支、无实例，不猜测声明）：lifecycle 谓词
AIRBORNE/HOME/TOUCHDOWN/LANDING 的分支（l1_semantics.py:81-85、118-120
有完整 geometry 证据形状）。谓词分支由事件族主谓词经 l1_semantics.py:70-86
决定；本会话行级抽样覆盖 L1-1_v1/L1-2_v1/L1-3_v1/L1-4_v1 四个 profile
（factual 与 intervention 行），只出现四个主谓词，未见 lifecycle 行。
此类行如出现将被校验拒绝，按契约偏差上报，而不是静默放行。

适配器漂移（不在本语法内建模）：当前
``minimal_semantics_adapter._adapt_assertions``（833-836 行）产出
3.1.0 并要求 ``truth_state_update_tick``/``evidence_update_tick``，而
``l1_semantics.project``（141 行）传入的行无这两个键——当前代码无法
再生本家族的冻结 3.0.0 文件。本语法按冻结记录建模；重生成属于
生产链修复，不是本模块范围。

元数据排除：``rule_digest``/``parameter_digest``/``input_digest`` 已在
``structure_schema._PUBLICATION_EXCLUDED_PATHS``（26-57 行），本家族
无需新增排除路径。
"""
from __future__ import annotations

import copy
from typing import Any

_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_ID = "https://aeroworld.local/schema/arm_predicate_truth_l1.record.schema.json"

_DIGEST = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_TRUTH_ENUM = {"enum": ["true", "false", "out_of_scope"]}
_REFS_PATTERN = r"^[^#]+#(?:tick=[0-9]+&entity=[^#]+|entity=[^#]+)$"


def _object(properties: dict[str, Any], required: list[str] | None = None,
            description: str | None = None) -> dict[str, Any]:
    """Closed object schema: every declared property is required."""
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else list(required),
        "additionalProperties": False,
    }
    if description is not None:
        schema["description"] = description
    return schema


def _bindings(roles: tuple[str, ...]) -> dict[str, Any]:
    """Closed role→entity-id map; role set is fixed per predicate branch."""
    return _object({role: {"type": "string", "minLength": 1} for role in roles})


def _binding_classes(classes: dict[str, str]) -> dict[str, Any]:
    """Closed role→ontology-class map with registry-observed const values."""
    return _object({role: {"const": value} for role, value in classes.items()})


def _source_refs() -> dict[str, Any]:
    """Non-empty unique refs: `<path>#tick=<n>&entity=<id>` or `<path>#entity=<id>`.

    实证：l1_semantics.py:96（轨迹 ref）与 l1_semantics.py:116（走廊场景
    ref，Dataset/scenarios/L1_airspace/.../scene_setup.json#entity=<corridor>）。
    """
    return {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "pattern": _REFS_PATTERN},
    }


def _observation(path: str, roles: tuple[str, ...],
                 classes: dict[str, str], section: dict[str, Any]) -> dict[str, Any]:
    """The single closed observation: const path plus the branch evidence node."""
    value_properties: dict[str, Any] = {
        "bindings": _bindings(roles),
        "binding_ontology_classes": _binding_classes(classes),
        "world_truth_value": dict(_TRUTH_ENUM),
    }
    value_properties.update(section)
    return _object({
        "path": {"const": path},
        "value": _object(value_properties),
    })


def _branch(predicate_id: str, observation_path: str, roles: tuple[str, ...],
            classes: dict[str, str], section: dict[str, Any],
            description: str) -> dict[str, Any]:
    """One self-contained closed record alternative, keyed by predicate_id const."""
    return _object(
        {
            "schema_name": {"const": "predicate_truth"},
            "schema_version": {"const": "3.0.0"},
            "annotation_layer": {"const": "L1"},
            "source_layer": {"const": "L0"},
            "materialization_policy": {"const": "on_demand_epi_predicate"},
            "truth_id": {"type": "string", "pattern": r"^truth:[0-9a-f]{24}$"},
            "episode_id": {"type": "string", "minLength": 1},
            "tick": {"type": "integer", "minimum": 0},
            "authoritative_tick": {"const": True},
            "predicate_id": {"const": predicate_id},
            "tuple_id": {"type": "string", "pattern": r"^l1_tuple:[0-9a-f]{24}$"},
            "bindings": _bindings(roles),
            "value": dict(_TRUTH_ENUM),
            "evidence": _object({
                "source_refs": _source_refs(),
                "observations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": _observation(observation_path, roles, classes, section),
                },
                "missing_requirements": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string"},
                },
            }),
            "source_class": {"const": "deterministic_derived"},
            "mechanism_id": {"const": None},
            "template_parameters": _object({}),
            "rule_digest": dict(_DIGEST),
            "parameter_digest": dict(_DIGEST),
            "input_digest": dict(_DIGEST),
        },
        description=description,
    )


_GEO_NUMBER = {"type": "number"}

_RESTRICTED_SECTION = {
    "geometry": _object({
        "restricted_region_active": {"type": "boolean"},
        "minimum_restricted_boundary_distance_m": dict(_GEO_NUMBER),
    }),
}
_ALTITUDE_SECTION = {
    "geometry": _object({
        "position_z_m": dict(_GEO_NUMBER),
        "assigned_altitude_m": dict(_GEO_NUMBER),
    }),
}
_PAIR_SECTION = {
    "derived": _object({
        "nearest_aircraft_distance_m": dict(_GEO_NUMBER),
    }),
}
_CAPACITY_SECTION = {
    "scene": _object({
        "maximum_corridor_occupancy_count": {"type": "integer", "minimum": 0},
        "maximum_corridor_capacity": {"type": "integer", "minimum": 0},
    }),
}

_UAV = "world:UnmannedAircraft"

_SCHEMA: dict[str, Any] = {
    "$schema": _DRAFT,
    "$id": _SCHEMA_ID,
    "title": "ARM L1 predicate truth record (frozen 3.0.0 contract)",
    "description": (
        "arm_predicate_truth L1 窗口记录（objective/predicate_truth.jsonl）的封闭"
        "整记录语法：21 个顶层键全部声明且必填，对象全闭合，数组定界。"
        "四个分支按 predicate_id const 互斥，实例依据见模块 docstring。"
        "value 的 \"out_of_scope\" 声明自 l1_semantics.py:127 缺位路径，尚无实例验证；"
        "lifecycle 谓词分支未声明（无实例、角色类未复核）。"
    ),
    "oneOf": [
        _branch(
            "airspace.aircraft_inside_restricted_airspace",
            "normalized.grounded_roles__actor__restricted_region",
            ("actor", "restricted_region"),
            {"actor": _UAV, "restricted_region": "world:RestrictedAirspaceRegion"},
            _RESTRICTED_SECTION,
            "实例依据：arms/L1-1_v1/seed00/grp_remove_interrupting_hover/"
            "arm_factual/objective/predicate_truth.jsonl 第 1 行。",
        ),
        _branch(
            "airspace.aircraft_outside_assigned_altitude_band",
            "normalized.grounded_roles__actor",
            ("actor",),
            {"actor": _UAV},
            _ALTITUDE_SECTION,
            "实例依据：arms/L1-2_v1/seed00/grp_prevent_altitude_excursion/"
            "arm_factual/objective/predicate_truth.jsonl 第 1 行。",
        ),
        _branch(
            "safety.aircraft_pair_below_separation",
            "normalized.grounded_roles__first_aircraft__second_aircraft",
            ("first_aircraft", "second_aircraft"),
            {"first_aircraft": _UAV, "second_aircraft": _UAV},
            _PAIR_SECTION,
            "实例依据：arms/L1-3_v1/seed00/grp_prevent_target_pair_convergence/"
            "arm_factual/objective/predicate_truth.jsonl 第 1 行。",
        ),
        _branch(
            "airspace.airspace_corridor_over_capacity",
            "normalized.grounded_roles__corridor",
            ("corridor",),
            {"corridor": "world:AirspaceCorridor"},
            _CAPACITY_SECTION,
            "实例依据：arms/L1-4_v1/seed00/grp_staggered_admission/"
            "arm_factual/objective/predicate_truth.jsonl 第 1 行（occupancy=0）；"
            "占用>0 行未复核，字段集以 l1_semantics.py:111-116 为准。",
        ),
    ],
}


def arm_predicate_truth_schema() -> dict[str, Any]:
    """Return a detached copy of the closed L1 predicate-truth record schema."""
    return copy.deepcopy(_SCHEMA)
