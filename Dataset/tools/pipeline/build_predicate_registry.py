#!/usr/bin/env python3
"""Generate the current world-truth registry from ontology IDs and rule contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataset.tools.runtime_state_contract import (  # noqa: E402
    RUNTIME_STATE_CONSUMED_FIELDS,
    RUNTIME_STATE_METADATA_FIELDS,
)

DOMAIN_CATALOG = ROOT / "Dataset" / "knowledge_graph" / "domain" / "catalog.yaml"
CONTRACTS_PATH = (
    ROOT
    / "Dataset"
    / "semantic_rules"
    / "predicates"
    / "world_truth_predicate_contracts.yaml"
)
OUTPUT = (
    ROOT
    / "Dataset"
    / "semantic_rules"
    / "predicates"
    / "core_semantic_predicate_templates.json"
)
EVENT_FAMILY_MAPPING = (
    ROOT
    / "Dataset"
    / "semantic_rules"
    / "profiles"
    / "event_family_authority_mapping.json"
)
EVENT_CATALOG = ROOT / "Dataset" / "knowledge_graph" / "events" / "catalog.yaml"
EVENT_ONTOLOGY_ROOT = ROOT / "Dataset" / "knowledge_graph" / "events"
WORLD_CORE_CATALOG = ROOT / "Dataset" / "knowledge_graph" / "world_core_catalog.yaml"
RUNTIME_STATE_SCHEDULE_ROOT = (
    ROOT / "Dataset" / "semantic_rules" / "profiles" / "epi_runtime_state_schedules"
)
EVENT_TYPE_IDENTIFIER = re.compile(r'aw:eventTypeIdentifier\s+"([^"]+)"')
EXPECTED_DECLARED_PREDICATE_COUNT = 79
EXPECTED_EXECUTABLE_PREDICATE_COUNT = 72
EXPECTED_NON_EXECUTABLE_PREDICATE_COUNT = 7
GROUNDING_AUTHORITY_KINDS = frozenset(
    {
        "scope_entity",
        "scope_relation",
        "domain_observation",
        "domain_nested_relation",
        "direct_predicate_row",
        "utm_record",
        "episode_operational_region",
    }
)
GROUNDING_ROLE_SOURCE_KINDS = frozenset(
    {"scope_entity", "context", "record", "record_binding", "manifest"}
)
GROUNDING_ROLE_KINDS_BY_AUTHORITY = {
    "scope_entity": frozenset({"scope_entity"}),
    "scope_relation": frozenset({"scope_entity", "context"}),
    "domain_observation": frozenset({"record"}),
    "domain_nested_relation": frozenset({"record"}),
    "direct_predicate_row": frozenset({"record_binding"}),
    "utm_record": frozenset({"record"}),
    "episode_operational_region": frozenset({"manifest"}),
}
EVENT_ROLE_SOURCE_KINDS = frozenset({"trigger_binding", "support_binding"})
TRUTH_FRAME_REQUIRED_FIELD_PATHS = frozenset(
    {
        "truth.controlling_signal_id",
        "truth.controlling_signal_state",
        "truth.crossed_stop_line",
        "truth.following_distance_m",
        "truth.in_crosswalk",
        "truth.lane_id",
        "truth.posted_speed_limit_mps",
        "truth.speed_mps",
    }
)
TRUTH_FRAME_FRAGMENT_FIELDS = {
    "sumo_vehicle": frozenset({"following_distance_m"}),
}


GOVERNED_DEFAULTS: dict[str, int | float] = {
    "airborne_threshold_m": 1.5,
    "aircraft_stationary_speed_threshold_mps": 0.3,
    "altitude_tolerance_m": 5.0,
    "battery_low_ratio": 0.2,
    "boundary_margin_m": 14.0,
    "boundary_response_distance_m": 10.0,
    "fog_density_threshold": 0.1,
    "following_distance_threshold_m": 5.0,
    "formal_step_ticks": 5,
    "gnss_error_tolerance_m": 8.0,
    "home_pad_radius_m": 4.0,
    "landing_zone_radius_m": 8.0,
    "low_illumination_threshold_lux": 400.0,
    "queue_length_threshold_vehicles": 5,
    "rain_active_threshold": 0.0,
    "recovery_hold_samples": 3,
    "safety_near_pedestrian_m": 15.0,
    "safety_near_structure_m": 15.0,
    "safety_near_vehicle_m": 15.0,
    "separation_margin_m": 15.0,
    "temperature_max_c": 45.0,
    "temperature_min_c": 0.0,
    "terminal_hold_samples": 2,
    "touchdown_threshold_m": 0.5,
    "vehicle_pedestrian_conflict_m": 5.0,
    "vehicle_stopped_speed_threshold_mps": 0.3,
    "visibility_threshold_m": 250.0,
    "wind_speed_threshold_mps": 10.0,
}


PARAMETER_MODELS: dict[str, dict[str, Any]] = {
    "airborne_threshold_m": {
        "range": [0.5, 3.0],
        "model": "ground-clearance discrimination above touchdown noise",
    },
    "aircraft_stationary_speed_threshold_mps": {
        "range": [0.0, 1.0],
        "model": "speed-resolution deadband",
    },
    "altitude_tolerance_m": {
        "range": [0.0, 15.0],
        "model": "assigned-band half-width plus pose error",
    },
    "battery_low_ratio": {
        "range": [0.0, 1.0],
        "model": "usable state-of-charge reserve fraction",
    },
    "boundary_margin_m": {
        "range": [0.0, 30.0],
        "model": "scenario conflict envelope; in-band horizontal bound d_margin=v*t_reaction+v^2/(2*a_deceleration), with Euclidean vertical clearance outside the restricted prism",
    },
    "boundary_response_distance_m": {
        "range": [0.0, 14.0],
        "model": "L1-1 boundary-conflict response trigger radius: the authored reaction point stays inside the governed conflict envelope, so it may not exceed boundary_margin_m; the episode's boundary crossing and response horizon are measured against this radius",
    },
    "fog_density_threshold": {
        "range": [0.0, 1.0],
        "model": "normalized simulated fog-density onset",
    },
    "following_distance_threshold_m": {
        "range": [0.0, 30.0],
        "model": "vehicle stopping headway lower bound",
    },
    "formal_step_ticks": {"range": [5, 5], "model": "formal corpus sampling contract"},
    "gnss_error_tolerance_m": {
        "range": [0.0, 30.0],
        "model": "navigation error envelope",
    },
    "home_pad_radius_m": {
        "range": [0.0, 10.0],
        "model": "pad footprint plus pose error",
    },
    "landing_zone_radius_m": {
        "range": [0.0, 20.0],
        "model": "landing approach containment radius",
    },
    "low_illumination_threshold_lux": {
        "range": [0.0, 2000.0],
        "model": (
            "maximum log-margin separator between the formal low-light profile "
            "(80 lux) and the nearest non-low-light profile (1800 lux); "
            "sqrt(80*1800)=379.47 lux, rounded to the governed 400-lux boundary"
        ),
    },
    "queue_length_threshold_vehicles": {
        "range": [1, 100],
        "model": "observable road-lane storage capacity",
    },
    "rain_active_threshold": {
        "range": [0.0, 1.0],
        "model": "normalized nonzero precipitation onset",
    },
    "recovery_hold_samples": {
        "range": [1, 36],
        "model": "debounce horizon on five-tick samples",
    },
    "safety_near_pedestrian_m": {
        "range": [0.0, 50.0],
        "model": "UAV-pedestrian horizontal risk envelope",
    },
    "safety_near_structure_m": {
        "range": [0.0, 50.0],
        "model": "UAV-structure clearance envelope",
    },
    "safety_near_vehicle_m": {
        "range": [0.0, 50.0],
        "model": "UAV-vehicle clearance envelope",
    },
    "separation_margin_m": {
        "range": [0.0, 100.0],
        "model": "aircraft pair separation lower bound",
    },
    "temperature_max_c": {
        "range": [-50.0, 100.0],
        "model": "platform operating-temperature upper bound",
    },
    "temperature_min_c": {
        "range": [-50.0, 100.0],
        "model": "platform operating-temperature lower bound",
    },
    "terminal_hold_samples": {
        "range": [1, 36],
        "model": "terminal-state debounce horizon",
    },
    "touchdown_threshold_m": {
        "range": [0.0, 2.0],
        "model": "ground-contact height tolerance",
    },
    "vehicle_pedestrian_conflict_m": {
        "range": [0.0, 20.0],
        "model": "vehicle-pedestrian collision envelope",
    },
    "vehicle_stopped_speed_threshold_mps": {
        "range": [0.0, 1.0],
        "model": "SUMO speed-resolution deadband",
    },
    "visibility_threshold_m": {
        "range": [0.0, 20000.0],
        "model": "minimum operational visual range",
    },
    "wind_speed_threshold_mps": {
        "range": [0.0, 50.0],
        "model": "small-UAV wind operating-envelope bound",
    },
}


SOURCE_CLASSES = [
    "engine_truth",
    "deterministic_derived",
    "simulated_derived",
    "visual_derived",
]
FORBIDDEN_TRUTH_INPUTS = [
    "active_event_ids",
    "confirmedAtTick",
    "dynamic_labels",
    "event_realization",
    "event_trace",
    "expected_event",
    "expected_event_id",
    "level",
    "scenario_plan",
    "scenario_title",
    "semantic_role",
    "state_facets",
    "task_id",
    "variant",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _authored_runtime_state_fields() -> frozenset[str]:
    fields: set[str] = set()

    def collect_patch(patch: Any) -> None:
        if not isinstance(patch, Mapping):
            return
        for namespace, values in patch.items():
            if not isinstance(namespace, str) or not isinstance(values, Mapping):
                continue
            allowed = RUNTIME_STATE_CONSUMED_FIELDS.get(namespace)
            if allowed is None:
                raise ValueError(
                    f"runtime-state schedule authors unsupported namespace {namespace!r}"
                )
            for field in values:
                if not isinstance(field, str) or (
                    field not in allowed and field not in RUNTIME_STATE_METADATA_FIELDS
                ):
                    raise ValueError(
                        f"runtime-state schedule authors unconsumed field "
                        f"{namespace}.{field}"
                    )
                fields.add(f"{namespace}.{field}")

    paths = sorted(RUNTIME_STATE_SCHEDULE_ROOT.glob("*.json"))
    if not paths:
        raise ValueError("runtime-state schedule authority is missing")
    for path in paths:
        profile = _load_json(path)
        scenarios = profile.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError(f"{path}: runtime-state profile lacks scenarios")
        for scenario in scenarios:
            if not isinstance(scenario, Mapping):
                raise ValueError(f"{path}: runtime-state scenario must be an object")
            initial = scenario.get("initial_entity_states")
            if not isinstance(initial, Mapping):
                raise ValueError(f"{path}: initial_entity_states must be an object")
            for patch in initial.values():
                collect_patch(patch)
            transitions = scenario.get("event_state_transitions")
            if not isinstance(transitions, Mapping):
                raise ValueError(f"{path}: event_state_transitions must be an object")
            for records in transitions.values():
                if not isinstance(records, list):
                    raise ValueError(f"{path}: event transition list is invalid")
                for record in records:
                    if not isinstance(record, Mapping):
                        raise ValueError(f"{path}: event transition must be an object")
                    collect_patch(record.get("state_patch"))
    return frozenset(fields)


def _validate_runtime_state_source_reachability(
    *,
    predicate_id: str,
    fields: Any,
    authored_runtime_fields: frozenset[str],
) -> None:
    if not isinstance(fields, list):
        raise ValueError(f"{predicate_id}: fields must be an array")
    truth_frames_source = "truth/truth_frames.jsonl"
    prefix = "truth/truth_frames.jsonl#"
    for item in fields:
        if not isinstance(item, list) or len(item) != 3:
            continue
        field_name, source, _unit = item
        if not isinstance(source, str) or not source.startswith(truth_frames_source):
            continue
        if field_name not in TRUTH_FRAME_REQUIRED_FIELD_PATHS:
            raise ValueError(
                f"{predicate_id}: truth-frame field {field_name!r} is outside the "
                "explicit runtime projection contract"
            )
        if source == truth_frames_source:
            continue
        if not source.startswith(prefix):
            raise ValueError(f"{predicate_id}: malformed truth-frame source {source!r}")
        fragment = source.removeprefix(prefix)
        namespace, separator, source_field = fragment.partition(".")
        if not separator or not source_field:
            raise ValueError(
                f"{predicate_id}: runtime-state source must name an exact field, "
                f"not container {source!r}"
            )
        if (
            not isinstance(field_name, str)
            or source_field != field_name.rsplit(".", 1)[-1]
        ):
            raise ValueError(
                f"{predicate_id}: runtime-state source field differs from required field"
            )
        if namespace in TRUTH_FRAME_FRAGMENT_FIELDS:
            if source_field not in TRUTH_FRAME_FRAGMENT_FIELDS[namespace]:
                raise ValueError(
                    f"{predicate_id}: truth-frame fragment {fragment!r} is outside "
                    "the explicit physical projection contract"
                )
            continue
        if namespace not in RUNTIME_STATE_CONSUMED_FIELDS:
            raise ValueError(
                f"{predicate_id}: truth-frame fragment namespace {namespace!r} "
                "has no declared producer/consumer contract"
            )
        runtime_field = f"{namespace}.{source_field}"
        if source_field not in RUNTIME_STATE_CONSUMED_FIELDS.get(
            namespace, frozenset()
        ):
            raise ValueError(
                f"{predicate_id}: executable runtime-state field {runtime_field} "
                "has no formal runtime consumer"
            )
        if runtime_field not in authored_runtime_fields:
            raise ValueError(
                f"{predicate_id}: executable runtime-state field {runtime_field} "
                "has no authored scenario producer"
            )


def _event_catalog_roles(
    event_catalog: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    modules = event_catalog.get("modules")
    if not isinstance(modules, Mapping):
        raise ValueError("event catalog lacks modules")
    identifiers: list[str] = []
    roles_by_event: dict[str, list[dict[str, str]]] = {}
    for module_id, module in modules.items():
        if not isinstance(module, Mapping):
            raise ValueError(f"event catalog module {module_id} must be an object")
        events = module.get("events")
        if not isinstance(events, list):
            raise ValueError(f"event catalog module {module_id} lacks events")
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError(f"event catalog module {module_id} has invalid event")
            identifier = event.get("identifier")
            raw_roles = event.get("roles")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(
                    f"event catalog module {module_id} has invalid identifier"
                )
            if identifier in roles_by_event:
                raise ValueError(f"duplicate event identifier: {identifier}")
            if not isinstance(raw_roles, list) or not raw_roles:
                raise ValueError(f"{identifier}: event roles must be a non-empty array")
            normalized_roles: list[dict[str, str]] = []
            for raw_role in raw_roles:
                if not isinstance(raw_role, Mapping):
                    raise ValueError(f"{identifier}: invalid event role")
                key = raw_role.get("key")
                ontology_class = raw_role.get("class")
                if (
                    not isinstance(key, str)
                    or not key
                    or not isinstance(ontology_class, str)
                    or not ontology_class
                ):
                    raise ValueError(f"{identifier}: event role requires key/class")
                normalized_roles.append({"key": key, "ontology_class": ontology_class})
            role_keys = [item["key"] for item in normalized_roles]
            if len(role_keys) != len(set(role_keys)):
                raise ValueError(f"{identifier}: duplicate event participant role")
            identifiers.append(identifier)
            roles_by_event[identifier] = normalized_roles
    if not identifiers:
        raise ValueError("event catalog authority is empty")
    return identifiers, roles_by_event


def _event_ttl_ids() -> list[str]:
    identifiers = [
        identifier
        for path in sorted(EVENT_ONTOLOGY_ROOT.glob("*_events.ttl"))
        for identifier in EVENT_TYPE_IDENTIFIER.findall(
            path.read_text(encoding="utf-8")
        )
    ]
    if not identifiers:
        raise ValueError("event ontology TTL authority is empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("event ontology TTL files contain duplicate identifiers")
    return identifiers


def _class_parent_map(
    world_catalog: Mapping[str, Any], domain_catalog: Mapping[str, Any]
) -> dict[str, str]:
    rows: list[Mapping[str, Any]] = []
    world_classes = world_catalog.get("classes")
    if not isinstance(world_classes, list):
        raise ValueError("world-core catalog lacks classes")
    rows.extend(item for item in world_classes if isinstance(item, Mapping))
    modules = domain_catalog.get("modules")
    if not isinstance(modules, Mapping):
        raise ValueError("domain catalog lacks modules")
    for module in modules.values():
        if not isinstance(module, Mapping):
            continue
        concepts = module.get("concepts", [])
        if not isinstance(concepts, list):
            raise ValueError("domain catalog concepts must be an array")
        rows.extend(item for item in concepts if isinstance(item, Mapping))
    result: dict[str, str] = {}
    for row in rows:
        ontology_class = row.get("id")
        parent = row.get("parent")
        if not isinstance(ontology_class, str) or not isinstance(parent, str):
            raise ValueError("ontology class declarations require id/parent")
        previous = result.get(ontology_class)
        if previous is not None and previous != parent:
            raise ValueError(f"ontology class {ontology_class} has conflicting parents")
        result[ontology_class] = parent
    return result


def _class_is_a(
    source_class: str, event_class: str, class_parents: Mapping[str, str]
) -> bool:
    current = source_class
    visited: set[str] = set()
    while current not in visited:
        if current == event_class:
            return True
        visited.add(current)
        parent = class_parents.get(current)
        if parent is None:
            return False
        current = parent
    raise ValueError(f"ontology class hierarchy contains a cycle at {current}")


def _validated_event_role_sources(
    *,
    family_id: str,
    event_type: str,
    trigger: Mapping[str, Any],
    support_conditions: Sequence[Mapping[str, Any]],
    event_roles: Sequence[Mapping[str, str]],
    predicate_role_classes: Mapping[str, Mapping[str, str]],
    class_parents: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    raw_sources = trigger.get("event_role_sources")
    expected_roles = [str(role["key"]) for role in event_roles]
    if not isinstance(raw_sources, Mapping) or set(raw_sources) != set(expected_roles):
        actual_roles = sorted(raw_sources) if isinstance(raw_sources, Mapping) else []
        raise ValueError(
            f"{family_id}:{event_type}: event_role_sources must exactly cover event "
            f"roles; expected={expected_roles}, actual={actual_roles}"
        )
    trigger_predicate_id = str(trigger.get("predicate_id") or "")
    support_predicate_ids = {
        str(condition.get("predicate_id") or "") for condition in support_conditions
    }
    event_class_by_role = {
        str(role["key"]): str(role["ontology_class"]) for role in event_roles
    }
    for event_role in expected_roles:
        raw_source = raw_sources[event_role]
        if not isinstance(raw_source, Mapping):
            raise ValueError(
                f"{family_id}:{event_role}: event role source must be an object"
            )
        kind = raw_source.get("kind")
        source_class = raw_source.get("source_ontology_class")
        background_allowed = raw_source.get("background_allowed")
        if kind not in EVENT_ROLE_SOURCE_KINDS:
            raise ValueError(
                f"{family_id}:{event_role}: unsupported event role source kind"
            )
        if not isinstance(source_class, str) or not source_class:
            raise ValueError(
                f"{family_id}:{event_role}: source_ontology_class is required"
            )
        if not isinstance(background_allowed, bool):
            raise ValueError(
                f"{family_id}:{event_role}: background_allowed must be boolean"
            )
        event_class = event_class_by_role[event_role]
        if not _class_is_a(source_class, event_class, class_parents):
            raise ValueError(
                f"{family_id}:{event_role}: source class {source_class} is not "
                f"compatible with event class {event_class}"
            )

    normalized: dict[str, dict[str, Any]] = {}
    for event_role in expected_roles:
        raw_source = raw_sources[event_role]
        assert isinstance(raw_source, Mapping)
        kind = str(raw_source["kind"])
        common_keys = {"kind", "source_ontology_class", "background_allowed"}
        expected_keys = common_keys | {"predicate_id", "predicate_role"}
        if set(raw_source) != expected_keys:
            raise ValueError(
                f"{family_id}:{event_role}: {kind} requires exactly "
                f"{sorted(expected_keys)}"
            )
        predicate_id = raw_source.get("predicate_id")
        predicate_role = raw_source.get("predicate_role")
        if not isinstance(predicate_id, str) or not isinstance(predicate_role, str):
            raise ValueError(
                f"{family_id}:{event_role}: predicate_id/predicate_role are required"
            )
        if kind == "trigger_binding" and predicate_id != trigger_predicate_id:
            raise ValueError(
                f"{family_id}:{event_role}: trigger source must reference "
                f"{trigger_predicate_id}"
            )
        if kind == "support_binding" and predicate_id not in support_predicate_ids:
            raise ValueError(
                f"{family_id}:{event_role}: support source references undeclared "
                f"support predicate {predicate_id}"
            )
        role_classes = predicate_role_classes.get(predicate_id)
        if role_classes is None or predicate_role not in role_classes:
            raise ValueError(
                f"{family_id}:{event_role}: predicate role does not exist: "
                f"{predicate_id}:{predicate_role}"
            )
        authoritative_class = role_classes[predicate_role]
        if raw_source["source_ontology_class"] != authoritative_class:
            raise ValueError(
                f"{family_id}:{event_role}: declared source class "
                f"{raw_source['source_ontology_class']} differs from predicate "
                f"role class {authoritative_class}"
            )
        normalized[event_role] = {
            "event_role": event_role,
            "event_ontology_class": event_class_by_role[event_role],
            **dict(raw_source),
        }
    return normalized


def _trigger_options(family: Mapping[str, Any]) -> list[dict[str, Any]]:
    trigger = family.get("trigger")
    if not isinstance(trigger, Mapping):
        return []
    any_of = trigger.get("any_of")
    if isinstance(any_of, list):
        return [dict(item) for item in any_of if isinstance(item, Mapping)]
    return [dict(trigger)] if isinstance(trigger.get("predicate_id"), str) else []


def _terminal_for_trigger(
    family: Mapping[str, Any], trigger: Mapping[str, Any]
) -> dict[str, Any]:
    terminal = family.get("terminal")
    if not isinstance(terminal, Mapping):
        raise ValueError("executable event family lacks terminal expression")
    any_of = terminal.get("any_of")
    if isinstance(any_of, list):
        options = [dict(item) for item in any_of if isinstance(item, Mapping)]
        if not options:
            raise ValueError("terminal.any_of must contain expressions")
        matching = [
            option
            for option in options
            if option.get("predicate_id") == trigger.get("predicate_id")
        ]
        if len(matching) != 1:
            raise ValueError(
                "terminal.any_of must contain exactly one terminal for each trigger "
                f"predicate: {trigger.get('predicate_id')}"
            )
        return matching[0]
    return dict(terminal)


def _build_event_rules(
    mapping: Mapping[str, Any],
    predicate_role_classes: Mapping[str, Mapping[str, str]],
    event_roles_by_type: Mapping[str, Sequence[Mapping[str, str]]],
    class_parents: Mapping[str, str],
) -> list[dict[str, Any]]:
    families = mapping.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("event-family mapping lacks families")
    declared_family_count = mapping.get("family_count")
    if declared_family_count != len(families):
        raise ValueError(
            "event-family mapping family_count differs from declared families"
        )
    predicate_ids = set(predicate_role_classes)
    ontology_event_ids = set(event_roles_by_type)
    rules: list[dict[str, Any]] = []
    for family_id, family in families.items():
        if not isinstance(family, Mapping) or not family.get("executable"):
            continue
        declared_event_ids = family.get("ontology_event_type_ids")
        if not isinstance(declared_event_ids, list) or not declared_event_ids:
            raise ValueError(f"{family_id}: executable family lacks ontology events")
        trigger_options = _trigger_options(family)
        if not trigger_options:
            raise ValueError(f"{family_id}: executable family lacks trigger options")
        selected_event_types: list[str] = []
        for trigger_index, trigger in enumerate(trigger_options):
            predicate_id = trigger.get("predicate_id")
            event_type = (
                trigger.get("event_type_id")
                or declared_event_ids[min(trigger_index, len(declared_event_ids) - 1)]
            )
            if predicate_id not in predicate_ids:
                raise ValueError(
                    f"{family_id}: non-authoritative trigger {predicate_id}"
                )
            if event_type not in ontology_event_ids:
                raise ValueError(f"{family_id}: non-authoritative event {event_type}")
            if trigger.get("direction") not in {"rising", "falling"}:
                raise ValueError(f"{family_id}: invalid trigger direction")
            selected_event_types.append(str(event_type))
            support = family.get("support_conditions", [])
            if not isinstance(support, list):
                raise ValueError(f"{family_id}: support_conditions must be an array")
            trigger_roles = set(predicate_role_classes[str(predicate_id)])
            support_conditions: list[dict[str, Any]] = []
            referenced_predicates: set[str] = set()
            for item in support:
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"{family_id}: support condition must be an object"
                    )
                condition = dict(item)
                support_predicate_id = str(condition.get("predicate_id") or "")
                referenced_predicates.add(support_predicate_id)
                if support_predicate_id not in predicate_role_classes:
                    raise ValueError(
                        f"{family_id}: non-authoritative support predicate "
                        f"{support_predicate_id}"
                    )
                declared_support_join_roles = condition.get("join_roles")
                if (
                    not isinstance(declared_support_join_roles, list)
                    or not declared_support_join_roles
                    or not all(
                        isinstance(role, str) and role
                        for role in declared_support_join_roles
                    )
                    or len(declared_support_join_roles)
                    != len(set(declared_support_join_roles))
                ):
                    raise ValueError(
                        f"{family_id}: support predicate {support_predicate_id} "
                        "requires explicit unique join_roles"
                    )
                support_role_keys = set(predicate_role_classes[support_predicate_id])
                if not set(declared_support_join_roles) <= (
                    trigger_roles & support_role_keys
                ):
                    raise ValueError(
                        f"{family_id}: support predicate {support_predicate_id} "
                        "join_roles are not present in both exact predicate tuples"
                    )
                condition["join_roles"] = list(declared_support_join_roles)
                support_conditions.append(condition)
            terminal = _terminal_for_trigger(family, trigger)
            referenced_predicates.add(str(terminal.get("predicate_id")))
            foreign = sorted(referenced_predicates - predicate_ids)
            if foreign:
                raise ValueError(
                    f"{family_id}: non-authoritative support/terminal predicates {foreign}"
                )
            terminal = dict(terminal)
            terminal_predicate_id = str(terminal.get("predicate_id") or "")
            declared_terminal_join_roles = terminal.get("join_roles")
            if isinstance(declared_terminal_join_roles, list) and all(
                isinstance(role, str) for role in declared_terminal_join_roles
            ):
                terminal_join_roles = list(declared_terminal_join_roles)
            else:
                raise ValueError(f"{family_id}: terminal requires explicit join_roles")
            if len(terminal_join_roles) != len(set(terminal_join_roles)):
                raise ValueError(f"{family_id}: terminal join_roles must be unique")
            terminal_role_keys = set(
                predicate_role_classes.get(terminal_predicate_id, {})
            )
            if terminal_join_roles and not set(terminal_join_roles) <= (
                trigger_roles & terminal_role_keys
            ):
                raise ValueError(
                    f"{family_id}: terminal join_roles are not present in both "
                    "exact predicate tuples"
                )
            if not terminal_join_roles:
                raise ValueError(
                    f"{family_id}: terminal predicate {terminal_predicate_id} "
                    "requires at least one exact join role"
                )
            terminal["join_roles"] = terminal_join_roles
            event_roles = event_roles_by_type[str(event_type)]
            event_role_sources = _validated_event_role_sources(
                family_id=str(family_id),
                event_type=str(event_type),
                trigger=trigger,
                support_conditions=support_conditions,
                event_roles=event_roles,
                predicate_role_classes=predicate_role_classes,
                class_parents=class_parents,
            )
            rules.append(
                {
                    "rule_id": f"event_family.{family_id}.{trigger_index}",
                    "event_family_id": str(family_id),
                    "event_type": str(event_type),
                    "mapping_class": str(family["mapping_class"]),
                    "trigger_predicate_id": str(predicate_id),
                    "trigger_direction": str(trigger["direction"]),
                    "event_tick": str(trigger.get("event_tick", "after")),
                    "support_conditions": support_conditions,
                    "terminal": terminal,
                    "event_role_sources": event_role_sources,
                    "participant_roles": [
                        {
                            "role": str(item["key"]),
                            "ontology_class": str(item["ontology_class"]),
                            "source_kind": str(
                                event_role_sources[str(item["key"])]["kind"]
                            ),
                            "background_allowed": bool(
                                event_role_sources[str(item["key"])][
                                    "background_allowed"
                                ]
                            ),
                        }
                        for item in event_roles
                    ],
                }
            )
        if set(selected_event_types) != set(declared_event_ids):
            raise ValueError(
                f"{family_id}: executable ontology events and trigger event types "
                "must match exactly"
            )
    return rules


def _world_class(value: str) -> str:
    return value if ":" in value else f"world:{value}"


def _ontology_class_path(identifier_path: str) -> str:
    prefix, separator, leaf = identifier_path.rpartition(".")
    if not leaf.endswith("_id"):
        raise ValueError(
            f"grounding identifier path must end in _id: {identifier_path}"
        )
    class_leaf = f"{leaf[:-3]}_ontology_class_id"
    return f"{prefix}.{class_leaf}" if separator else class_leaf


def _validated_execution_partition(
    raw: Any,
    *,
    ontology_roles_by_predicate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not isinstance(raw, Mapping):
        raise ValueError("world-truth contracts lack predicate_execution")
    if set(raw) != {
        "policy",
        "executable_l1_predicate_ids",
        "non_executable_predicates",
    }:
        raise ValueError("predicate_execution has unsupported or missing fields")
    if raw.get("policy") != "explicit_partition_no_default":
        raise ValueError("predicate_execution must forbid implicit status defaults")
    executable = raw.get("executable_l1_predicate_ids")
    non_executable = raw.get("non_executable_predicates")
    if (
        not isinstance(executable, list)
        or not all(isinstance(item, str) and item for item in executable)
        or len(executable) != len(set(executable))
    ):
        raise ValueError(
            "predicate_execution.executable_l1_predicate_ids must be unique IDs"
        )
    if not isinstance(non_executable, Mapping):
        raise ValueError(
            "predicate_execution.non_executable_predicates must be an object"
        )
    executable_set = set(executable)
    non_executable_set = set(non_executable)
    ontology_ids = set(ontology_roles_by_predicate)
    if (
        len(ontology_ids) != EXPECTED_DECLARED_PREDICATE_COUNT
        or len(executable_set) != EXPECTED_EXECUTABLE_PREDICATE_COUNT
        or len(non_executable_set) != EXPECTED_NON_EXECUTABLE_PREDICATE_COUNT
    ):
        raise ValueError(
            "predicate execution partition must contain exactly "
            f"{EXPECTED_DECLARED_PREDICATE_COUNT} declared, "
            f"{EXPECTED_EXECUTABLE_PREDICATE_COUNT} executable, and "
            f"{EXPECTED_NON_EXECUTABLE_PREDICATE_COUNT} non-executable predicates"
        )
    if executable_set & non_executable_set:
        raise ValueError("predicate execution partitions overlap")
    if executable_set | non_executable_set != ontology_ids:
        raise ValueError(
            "predicate execution partition differs from ontology vocabulary: "
            f"unclassified={sorted(ontology_ids - executable_set - non_executable_set)}, "
            f"foreign={sorted((executable_set | non_executable_set) - ontology_ids)}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    required_keys = {"reason_code", "missing_authorities", "closure_statement"}
    for predicate_id, contract in non_executable.items():
        if not isinstance(contract, Mapping) or set(contract) != required_keys:
            raise ValueError(
                f"{predicate_id}: non-executable contract must declare exactly "
                "reason_code, missing_authorities, and closure_statement"
            )
        reason_code = contract.get("reason_code")
        missing_authorities = contract.get("missing_authorities")
        closure = contract.get("closure_statement")
        if not isinstance(reason_code, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", reason_code
        ):
            raise ValueError(f"{predicate_id}: invalid non-executable reason_code")
        if (
            not isinstance(missing_authorities, list)
            or not missing_authorities
            or not all(isinstance(item, Mapping) for item in missing_authorities)
        ):
            raise ValueError(
                f"{predicate_id}: missing_authorities must be a non-empty object array"
            )
        declared_role_classes = {
            str(role.get("class") or "")
            for role in ontology_roles_by_predicate[str(predicate_id)]
            if isinstance(role, Mapping)
        }
        normalized_authorities: list[dict[str, str]] = []
        authority_keys: set[tuple[str, str]] = set()
        for authority in missing_authorities:
            kind = authority.get("kind")
            if kind in {"ontology_role_individual", "supporting_individual"}:
                if set(authority) != {"kind", "ontology_class"}:
                    raise ValueError(
                        f"{predicate_id}: {kind} authority must declare ontology_class"
                    )
                value = authority.get("ontology_class")
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{predicate_id}: missing authority ontology_class is invalid"
                    )
                if (
                    kind == "ontology_role_individual"
                    and value not in declared_role_classes
                ):
                    raise ValueError(
                        f"{predicate_id}: missing role class is not an ontology argument"
                    )
                normalized_authority = {"kind": str(kind), "ontology_class": value}
            elif kind == "runtime_state_field":
                if set(authority) != {"kind", "field"}:
                    raise ValueError(
                        f"{predicate_id}: runtime_state_field authority must declare field"
                    )
                value = authority.get("field")
                if not isinstance(value, str) or not re.fullmatch(
                    r"[a-z][a-z0-9_]*_state\.[a-z][a-z0-9_]*", value
                ):
                    raise ValueError(
                        f"{predicate_id}: missing runtime-state authority field is invalid"
                    )
                normalized_authority = {"kind": str(kind), "field": value}
            else:
                raise ValueError(
                    f"{predicate_id}: unsupported missing authority kind {kind!r}"
                )
            authority_key = (str(kind), str(value))
            if authority_key in authority_keys:
                raise ValueError(f"{predicate_id}: duplicate missing authority")
            authority_keys.add(authority_key)
            normalized_authorities.append(normalized_authority)
        if not isinstance(closure, str) or not closure.strip():
            raise ValueError(f"{predicate_id}: closure_statement is required")
        normalized[str(predicate_id)] = {
            "reason_code": reason_code,
            "missing_authorities": normalized_authorities,
            "closure_statement": closure.strip(),
        }
    return list(executable), normalized


def _validated_grounding_spec(
    *,
    predicate_id: str,
    raw: Any,
    ontology_roles: list[Mapping[str, Any]],
    required_field_names: list[str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{predicate_id}: grounding_spec must be an object")
    authority = raw.get("candidate_authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("kind") not in GROUNDING_AUTHORITY_KINDS
        or not isinstance(authority.get("source"), str)
        or not authority["source"]
    ):
        raise ValueError(
            f"{predicate_id}: candidate_authority requires a supported kind/source"
        )
    expected_roles = [str(role["key"]) for role in ontology_roles]
    role_sources = raw.get("role_sources")
    if not isinstance(role_sources, Mapping) or set(role_sources) != set(
        expected_roles
    ):
        raise ValueError(
            f"{predicate_id}: grounding role set differs from ontology roles; "
            f"expected={expected_roles}"
        )
    authority_kind = str(authority["kind"])
    allowed_role_kinds = GROUNDING_ROLE_KINDS_BY_AUTHORITY[authority_kind]
    normalized_roles: dict[str, dict[str, Any]] = {}
    role_class_by_key = {
        str(role["key"]): str(role["class"]) for role in ontology_roles
    }
    for role in expected_roles:
        value = role_sources[role]
        if (
            not isinstance(value, Mapping)
            or value.get("kind") not in GROUNDING_ROLE_SOURCE_KINDS
        ):
            raise ValueError(f"{predicate_id}:{role}: invalid role source")
        path = value.get("path")
        if value["kind"] != "scope_entity" and (not isinstance(path, str) or not path):
            raise ValueError(f"{predicate_id}:{role}: role source requires path")
        role_kind = str(value["kind"])
        if role_kind not in allowed_role_kinds:
            raise ValueError(
                f"{predicate_id}:{role}: {role_kind} is incompatible with "
                f"candidate authority {authority_kind}"
            )
        if role_kind == "scope_entity":
            identifier_source = "typed_scope_entity.entity_id"
            ontology_class_sources = ["typed_scope_entity.ontology_class_id"]
        elif role_kind == "record_binding":
            assert isinstance(path, str)
            identifier_source = f"bindings.{path}"
            ontology_class_sources = [f"binding_ontology_classes.{path}"]
        elif role_kind == "manifest":
            assert isinstance(path, str)
            if role_class_by_key[role] != "OperationalRegion":
                raise ValueError(
                    f"{predicate_id}:{role}: manifest identity requires OperationalRegion"
                )
            identifier_source = path
            ontology_class_sources = ["constant:world:OperationalRegion"]
        else:
            assert isinstance(path, str)
            identifier_source = path
            ontology_class_sources = [
                _ontology_class_path(path),
                "typed_instance_catalog[identifier]",
            ]
        normalized_roles[role] = {
            "kind": role_kind,
            **({"path": str(path)} if isinstance(path, str) and path else {}),
            "identity_provenance": {
                "identifier_source": identifier_source,
                "ontology_class_sources": ontology_class_sources,
                "required_ontology_class_id": _world_class(role_class_by_key[role]),
                "validation": "runtime_exact_is_a_fail_closed",
            },
        }
    if authority_kind in {"scope_entity", "scope_relation"}:
        scope_role_count = sum(
            source["kind"] == "scope_entity" for source in normalized_roles.values()
        )
        if scope_role_count != 1:
            raise ValueError(
                f"{predicate_id}: scope authority requires exactly one scope_entity role"
            )
    field_sources = raw.get("field_sources")
    if not isinstance(field_sources, Mapping) or set(field_sources) != set(
        required_field_names
    ):
        raise ValueError(
            f"{predicate_id}: field_sources must exactly cover required fields"
        )
    if any(not isinstance(value, str) or not value for value in field_sources.values()):
        raise ValueError(
            f"{predicate_id}: field source paths must be non-empty strings"
        )
    distinct = raw.get("distinct_role_sets", [])
    unordered = raw.get("unordered_role_sets", [])
    for label, constraints in (("distinct", distinct), ("unordered", unordered)):
        if not isinstance(constraints, list):
            raise ValueError(f"{predicate_id}: {label}_role_sets must be an array")
        for pair in constraints:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or pair[0] == pair[1]
                or any(role not in expected_roles for role in pair)
            ):
                raise ValueError(f"{predicate_id}: invalid {label} role pair {pair!r}")
    for first, second in unordered:
        if role_class_by_key[first] != role_class_by_key[second]:
            raise ValueError(
                f"{predicate_id}: unordered roles must have the same ontology class"
            )
    return {
        "candidate_authority": dict(authority),
        "role_sources": normalized_roles,
        "field_sources": {
            str(field): str(source) for field, source in field_sources.items()
        },
        "distinct_role_sets": [list(pair) for pair in distinct],
        "unordered_role_sets": [list(pair) for pair in unordered],
        "role_class_validation": "exact_ontology_class_per_binding",
    }


def build_registry() -> dict[str, Any]:
    catalog = _load_yaml(DOMAIN_CATALOG)
    event_catalog = _load_yaml(EVENT_CATALOG)
    world_catalog = _load_yaml(WORLD_CORE_CATALOG)
    contract_document = _load_yaml(CONTRACTS_PATH)
    event_family_mapping = _load_json(EVENT_FAMILY_MAPPING)
    event_catalog_ids, event_roles_by_type = _event_catalog_roles(event_catalog)
    event_ontology_ids = _event_ttl_ids()
    if set(event_catalog_ids) != set(event_ontology_ids):
        raise ValueError(
            "event catalog and TTL authority differ: "
            f"catalog_only={sorted(set(event_catalog_ids) - set(event_ontology_ids))}, "
            f"ttl_only={sorted(set(event_ontology_ids) - set(event_catalog_ids))}"
        )
    class_parents = _class_parent_map(world_catalog, catalog)
    contracts = contract_document.get("predicates")
    grounding_specs = contract_document.get("grounding_specs")
    scope_types = contract_document.get("scope_types")
    if (
        contract_document.get("schema")
        != "aeroworld.world-truth-predicate-contracts.v3"
    ):
        raise ValueError("unsupported world-truth contract schema")
    if (
        not isinstance(contracts, Mapping)
        or not isinstance(grounding_specs, Mapping)
        or not isinstance(scope_types, list)
    ):
        raise ValueError(
            "world-truth contracts lack predicates, grounding_specs, or scope_types"
        )

    ontology_rows: list[tuple[str, Mapping[str, Any], str]] = []
    for module_id, module in catalog["modules"].items():
        for predicate in module.get("predicates", []):
            ontology_rows.append((module_id, predicate, str(predicate["identifier"])))
    identifiers = [identifier for _, _, identifier in ontology_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("domain catalog contains duplicate predicate identifiers")
    if set(contracts) != set(identifiers):
        raise ValueError(
            "world-truth contracts and ontology vocabulary differ: "
            f"ontology_only={sorted(set(identifiers) - set(contracts))}, "
            f"contracts_only={sorted(set(contracts) - set(identifiers))}"
        )
    ontology_roles_by_predicate = {
        identifier: list(predicate["roles"])
        for _module_id, predicate, identifier in ontology_rows
    }
    executable_ids, non_executable_contracts = _validated_execution_partition(
        contract_document.get("predicate_execution"),
        ontology_roles_by_predicate=ontology_roles_by_predicate,
    )
    executable_id_set = set(executable_ids)
    if set(grounding_specs) != executable_id_set:
        raise ValueError(
            "world-truth grounding specs must exactly cover executable predicates: "
            f"executable_only={sorted(executable_id_set - set(grounding_specs))}, "
            f"grounding_only={sorted(set(grounding_specs) - executable_id_set)}"
        )
    if set(scope_types) != {
        "uav",
        "vehicle",
        "pedestrian",
        "facility",
        "compute_node",
        "scene",
    }:
        raise ValueError("world-truth contracts must declare exactly six scope types")

    authored_runtime_fields = _authored_runtime_state_fields()
    for predicate_id in executable_ids:
        predicate_contract = contracts[predicate_id]
        if not isinstance(predicate_contract, Mapping):
            raise ValueError(f"{predicate_id}: predicate contract must be an object")
        _validate_runtime_state_source_reachability(
            predicate_id=predicate_id,
            fields=predicate_contract.get("fields"),
            authored_runtime_fields=authored_runtime_fields,
        )

    templates: list[dict[str, Any]] = []
    for module_id, predicate, identifier in ontology_rows:
        contract = contracts[identifier]
        predicate_scopes = contract.get("scope_types")
        if not isinstance(predicate_scopes, list) or not predicate_scopes:
            raise ValueError(f"{identifier}: scope_types must be non-empty")
        if not set(predicate_scopes) <= set(scope_types):
            raise ValueError(f"{identifier}: invalid scope_types {predicate_scopes}")
        common_template = {
            "id": identifier,
            "ontology_module": module_id,
            "argument_roles": predicate["roles"],
            "tuple_type": str(contract["tuple_type"]),
            "scope_types": list(predicate_scopes),
            "truth_value_logic": "four_valued",
            "expression": str(contract["expression"]),
        }
        if identifier not in executable_id_set:
            templates.append(
                {
                    **common_template,
                    "implementation_status": "non_executable",
                    "non_executable_contract": non_executable_contracts[identifier],
                }
            )
            continue
        fields = contract.get("fields")
        thresholds = contract.get("thresholds", [])
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"{identifier}: fields must be a non-empty array")
        required_fields: list[dict[str, str]] = []
        for item in fields:
            if (
                not isinstance(item, list)
                or len(item) != 3
                or not all(isinstance(value, str) and value for value in item)
            ):
                raise ValueError(
                    f"{identifier}: every field must be [field, source, unit]"
                )
            required_fields.append(
                {"field": item[0], "source": item[1], "unit": item[2]}
            )
        unknown_thresholds = sorted(set(thresholds) - set(GOVERNED_DEFAULTS))
        if unknown_thresholds:
            raise ValueError(f"{identifier}: unknown thresholds {unknown_thresholds}")
        evaluation = contract.get("evaluation")
        if not isinstance(evaluation, Mapping) or not evaluation.get("op"):
            raise ValueError(f"{identifier}: evaluation must declare an operator")
        templates.append(
            {
                **common_template,
                "implementation_status": "executable_l1",
                "state_contract": {
                    "required_fields": required_fields,
                    "unknown_if_missing": [field["field"] for field in required_fields],
                    "threshold_parameters": list(thresholds),
                    "missing_record_field": "missing_source_record",
                },
                "grounding_spec": _validated_grounding_spec(
                    predicate_id=identifier,
                    raw=grounding_specs[identifier],
                    ontology_roles=list(predicate["roles"]),
                    required_field_names=[field["field"] for field in required_fields],
                ),
                "simulation_log_projection": {
                    "status": "not_projected",
                    "no_projection_reason": (
                        "current L1 predicates are grounded from authoritative "
                        "candidate bindings and world_truth base/delta artifacts; "
                        "simulation_logs remain raw producer evidence, not a "
                        "predicate projection authority."
                    ),
                },
                "evaluation_spec": dict(evaluation),
            }
        )

    event_rules = _build_event_rules(
        event_family_mapping,
        {
            str(template["id"]): {
                str(role["key"]): str(role["class"])
                for role in template.get("argument_roles", [])
                if isinstance(role, Mapping)
            }
            for template in templates
            if template["implementation_status"] == "executable_l1"
        },
        event_roles_by_type,
        class_parents,
    )
    mapped_families = event_family_mapping.get("families")
    assert isinstance(mapped_families, Mapping)
    executable_families = {
        str(family_id)
        for family_id, family in mapped_families.items()
        if isinstance(family, Mapping) and family.get("executable") is True
    }
    registered_families = {rule["event_family_id"] for rule in event_rules}
    if registered_families != executable_families:
        raise ValueError(
            "executable event-family mapping and event rules differ: "
            f"mapping_only={sorted(executable_families - registered_families)}, "
            f"rules_only={sorted(registered_families - executable_families)}"
        )
    executable_event_types = sorted({rule["event_type"] for rule in event_rules})

    registry: dict[str, Any] = {
        "schema_name": "aeroworld_core_semantic_predicate_registry",
        "schema_version": "3.0.0",
        "registry_id": "aeroworld.world_truth_predicates.v3",
        "authority": "Dataset/knowledge_graph/domain/*.ttl",
        "scope": "Ontology-authoritative 79-predicate world-truth evaluator over six scope classes.",
        "runtime_policy": {
            "official_predicate_vocabulary": "domain_ontology",
            "implementation_authority": "Dataset.semantic_truth.world_truth",
            "authored_event_outcomes_as_truth_inputs": "forbidden",
            "plan_windows_as_state_inputs": "allowed",
            "l1_materialization": (
                "tick0_grounded_candidate_base_plus_add_set_remove_delta"
            ),
            "l2_materialization": "roi_or_event_entities_only",
            "non_executable_predicate_policy": (
                "declared_vocabulary_only; forbidden_from_l1_candidates_and_deltas"
            ),
        },
        "truth_values": ["true", "false", "unknown", "out_of_scope"],
        "source_classes": SOURCE_CLASSES,
        "forbidden_truth_inputs": FORBIDDEN_TRUTH_INPUTS,
        "world_scope_types": list(scope_types),
        "transition_rule": "Emit delta only when the four-valued state changes at adjacent five-tick authoritative samples; boolean rising/falling transitions remain the L2 event trigger surface.",
        "parameter_governance": {
            "status": "reviewed_for_controlled_simulation",
            "note": "Every threshold is bounded by an explicit simulation-domain model; values are not real-world certification claims.",
            "governed_defaults": GOVERNED_DEFAULTS,
            "scenario_override_policy": {
                "boundary_margin_m": {
                    "source": "parameter_governance.governed_defaults.boundary_margin_m",
                    "materialized_by": "runtime_region_state_computer.region_activity",
                    "resolution": "single_global_authority; any scenario declaration must equal the governed value",
                },
                "boundary_response_distance_m": {
                    "source": "parameter_governance.governed_defaults.boundary_response_distance_m",
                    "materialized_by": "plan_window_computer.boundary_conflict_response",
                    "resolution": "single_global_authority; a scenario that declares parameters.boundary_response_distance_m must equal the governed value and its boundary_conflict trigger distance, and must carry basis and scope",
                }
            },
            "theoretical_performance_bounds": PARAMETER_MODELS,
        },
        "implementation": {
            "engine_module": "Dataset.semantic_truth.world_truth",
            "engine_entrypoint": "evaluate_world_truth",
            "event_engine_module": "Dataset.semantic_truth.minimal_semantics",
            "event_engine_entrypoint": "build_events",
            "event_adapter_module": "Dataset.semantic_truth.minimal_semantics_adapter",
            "event_adapter_entrypoint": "run_minimal_semantic_engine",
            "declared_predicate_vocabulary_ids": identifiers,
            "executable_predicate_ids": executable_ids,
            "non_executable_predicate_ids": list(non_executable_contracts),
            "non_executable_predicates": non_executable_contracts,
            "declared_predicate_vocabulary_count": len(templates),
            "executable_predicate_count": len(executable_ids),
            "non_executable_predicate_count": len(non_executable_contracts),
            "implemented_event_types": executable_event_types,
        },
        "template_count": len(templates),
        "templates": templates,
        "event_counts": {
            "ontology_event_type_count": len(event_ontology_ids),
            "registered_event_rule_count": len(event_rules),
            "executable_event_type_count": len(executable_event_types),
            "event_family_count": len(mapped_families),
            "executable_event_family_count": len(executable_families),
        },
        "event_ontology_type_ids": event_ontology_ids,
        "event_occurrence_type_count": len(event_rules),
        "event_occurrence_types": event_rules,
    }
    return registry


def render() -> str:
    return (
        json.dumps(build_registry(), ensure_ascii=False, sort_keys=False, indent=2)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"stale current predicate registry: {OUTPUT}")
        print(f"current predicate registry is current: {OUTPUT}")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
