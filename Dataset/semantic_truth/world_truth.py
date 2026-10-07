"""V3 L1 grounded world-truth evaluator and replayable graph representation.

The evaluator interprets the ontology-aligned contracts in
``world_truth_predicate_contracts.yaml`` through the generated core registry.
It materializes exact predicate candidates with ontology-role bindings at every
formal tick. Tick 0 stores the grounded base; later rows store add/remove/set
deltas only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_predicate_templates,
    get_governed_parameter_defaults,
    load_core_semantic_registry,
)
from Dataset.semantic_truth.provenance import (
    canonical_json,
    digest_file,
    digest_object,
    read_jsonl,
    stable_identifier,
    without_integrity_metadata,
)
from Dataset.semantic_truth.facility_scope import (
    FacilityScopeError,
    ontology_class_is_a,
    validate_roster_facility_scope,
)


SCHEMA_VERSION = "3.1.0"
BASE_SCHEMA_NAME = "world_truth_graph_base"
DELTA_SCHEMA_NAME = "world_truth_graph_delta"
REPLAY_SCHEMA_NAME = "world_truth_graph_replay_state"
FORMAL_TICKS = tuple(range(0, 901, 5))
SCOPE_TYPES = ("uav", "vehicle", "pedestrian", "facility", "compute_node", "scene")
TRUTH_VALUES = frozenset({"true", "false", "unknown", "out_of_scope"})
MISSING = object()


class WorldTruthError(ValueError):
    """Raised when a world-truth input or replay artifact is inconsistent."""


@dataclass(frozen=True)
class WorldTruthResult:
    base_graph: Mapping[str, Any]
    deltas: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class _ScopeEntity:
    scope_type: str
    entity_id: str
    source_entity_id: str | None = None
    scope_subtype: str | None = None
    ontology_class_id: str | None = None


def evaluate_world_truth(
    episode_root: Path,
    *,
    source_availability: Mapping[str, Any],
    domain_rows: Sequence[Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    compute_predicate_rows: Sequence[Mapping[str, Any]],
    utm_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> WorldTruthResult:
    """Evaluate exact grounded predicate candidates and emit replayable deltas."""

    return _evaluate_grounded_world_truth(
        episode_root,
        source_availability=source_availability,
        domain_rows=domain_rows,
        compute_rows=compute_rows,
        communication_rows=communication_rows,
        compute_predicate_rows=compute_predicate_rows,
        utm_records=utm_records,
    )


def replay_world_truth(
    base_graph: Mapping[str, Any],
    deltas: Sequence[Mapping[str, Any]],
    tick: int,
) -> dict[str, Any]:
    """Reconstruct the exact grounded L1 candidate set at a formal tick."""

    return _replay_grounded_world_truth(base_graph, deltas, tick)


def serialize_world_truth_deltas(
    deltas: Iterable[Mapping[str, Any]],
) -> str:
    rows = sorted(
        (dict(row) for row in deltas),
        key=lambda row: (int(row.get("tick", -1)), str(row.get("delta_id", ""))),
    )
    return "".join(f"{canonical_json(without_integrity_metadata(row))}\n" for row in rows)


def _index_source_unavailability(
    value: Mapping[str, Any],
    episode_id: str,
    predicate_ids: frozenset[str],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    context = "L0 predicate source availability"
    if (value.get("schema_name") != "l0_predicate_source_availability"
            or value.get("schema_version") != "1.0.0"):
        raise WorldTruthError(f"{context}: unexpected schema")
    if value.get("episode_id") != episode_id:
        raise WorldTruthError(f"{context}: episode_id mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise WorldTruthError(f"{context}: entries must be an array")
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise WorldTruthError(
                f"{context}: entry must be an object"
            )
        predicate_id = entry.get("predicate_id")
        scope_type = entry.get("scope_type")
        entity_id = entry.get("scope_entity_id")
        if (
            predicate_id not in predicate_ids
            or scope_type not in SCOPE_TYPES
            or not isinstance(entity_id, str)
            or not entity_id
            or entry.get("executable") is not False
            or not isinstance(entry.get("contract_id"), str)
            or not isinstance(entry.get("non_executable_reason"), str)
            or not isinstance(entry.get("closure_statement"), str)
            or not isinstance(entry.get("source_ref"), str)
        ):
            raise WorldTruthError(f"{context}: incomplete entry")
        key = (str(scope_type), entity_id, str(predicate_id))
        if key in result:
            raise WorldTruthError(f"{context}: duplicate key {key}")
        result[key] = entry
    return result


def _evaluate_expression(
    spec: Mapping[str, Any],
    context: Mapping[str, Any],
    previous_context: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> bool | str | None:
    op = str(spec.get("op") or "")
    if op == "all":
        values = [
            _evaluate_expression(arg, context, previous_context, defaults)
            for arg in spec.get("args", ())
            if isinstance(arg, Mapping)
        ]
        if "out_of_scope" in values:
            return "out_of_scope"
        if any(value is False for value in values):
            return False
        return True if values and all(value is True for value in values) else None
    field = str(spec.get("field") or "")
    value = _value_at_path(context, field) if field else MISSING
    if op == "direct_truth":
        if _missing_value(value):
            return None
        if value == "true":
            return True
        if value == "false":
            return False
        if value == "out_of_scope":
            return "out_of_scope"
        raise WorldTruthError(f"invalid direct truth value at {field}: {value!r}")
    if op == "changed":
        previous = _value_at_path(previous_context, field)
        if previous is MISSING:
            return False
        return value != previous
    if op == "present":
        return not _missing_value(value) and not (
            isinstance(value, str) and value.strip().lower() == "none"
        )
    if op == "truthy":
        return value if isinstance(value, bool) else None
    if op == "falsy":
        return not value if isinstance(value, bool) else None
    if op == "equals":
        return value == spec.get("value")
    if op == "in":
        values = spec.get("values")
        return value in values if isinstance(values, list) else None
    if op in {"gt", "gte", "lt", "lte"}:
        number = _number(value)
        threshold = _number(defaults.get(str(spec.get("threshold") or "")))
        if number is None or threshold is None:
            return None
        return {
            "gt": number > threshold,
            "gte": number >= threshold,
            "lt": number < threshold,
            "lte": number <= threshold,
        }[op]
    if op == "gt_constant":
        number = _number(value)
        constant = _number(spec.get("value"))
        return (
            number > constant if number is not None and constant is not None else None
        )
    if op == "gt_fields":
        left = _number(_value_at_path(context, str(spec.get("left") or "")))
        right = _number(_value_at_path(context, str(spec.get("right") or "")))
        return left > right if left is not None and right is not None else None
    if op == "absolute_difference_gt":
        left = _number(_value_at_path(context, str(spec.get("left") or "")))
        right = _number(_value_at_path(context, str(spec.get("right") or "")))
        threshold = _number(defaults.get(str(spec.get("threshold") or "")))
        if left is None or right is None or threshold is None:
            return None
        return abs(left - right) > threshold
    if op == "outside_range":
        number = _number(value)
        minimum = _number(defaults.get(str(spec.get("minimum") or "")))
        maximum = _number(defaults.get(str(spec.get("maximum") or "")))
        if number is None or minimum is None or maximum is None:
            return None
        return number < minimum or number > maximum
    raise WorldTruthError(f"unsupported declarative evaluation operator: {op!r}")


def _expression_truth_value(value: bool | str | None) -> str:
    """Serialize an expression result without collapsing applicability into truth."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "unknown"
    if isinstance(value, str) and value in TRUTH_VALUES:
        return value
    raise WorldTruthError(f"invalid expression result: {value!r}")


def _build_tick_contexts(
    *,
    episode_id: str,
    tick: int,
    scopes: Sequence[_ScopeEntity],
    roster_entities: Mapping[str, Mapping[str, Any]],
    entities_at_tick: Mapping[str, Mapping[str, Any]],
    weather_row: Mapping[str, Any],
    domain_index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    domain_by_tick_family: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    compute_by_node: Mapping[tuple[int, str], Mapping[str, Any]],
    utm_index: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    region_runtime_rows = domain_by_tick_family.get(
        (tick, "predicate_contract_region_runtime_state"), ()
    )
    region_active = _aggregate_boolean(
        [
            _value_at_path(_values(row), "restricted_region_active")
            for row in region_runtime_rows
        ]
    )
    lockdown_rows = domain_by_tick_family.get((tick, "lockdown_region_state"), ())
    lockdown_active = _aggregate_boolean(
        [_values(row).get("temporary_lockdown_active") for row in lockdown_rows]
    )
    effective_restricted_region_active = _aggregate_boolean(
        [region_active, lockdown_active]
    )
    global_domain = _global_domain_values(tick, domain_by_tick_family)
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for scope in scopes:
        source_id = scope.source_entity_id or scope.entity_id
        entity = entities_at_tick.get(source_id)
        context: dict[str, Any] = {
            "l0": {
                "scope_active": entity is not None,
                "scope_activity_authority": "truth_frames.entities",
            },
            "truth": _truth_projection(entity),
            "domain": _domain_projection(
                tick, source_id, domain_index, global_domain, scope.scope_type
            ),
            "geometry": {},
            "plan": {},
            "control": {},
            "derived": {},
            "scene": {},
            "weather": dict(weather_row),
            "compute": dict(compute_by_node.get((tick, scope.entity_id), {})),
            "utm": _utm_projection(tick, source_id, utm_index),
            "_observation_sources": {},
        }
        frame_ref = f"truth_frames.jsonl#tick={tick}&entity={source_id}"
        context["_observation_sources"]["l0.scope_active"] = frame_ref
        for field in context["truth"]:
            context["_observation_sources"][f"truth.{field}"] = frame_ref
        weather_ref = f"weather_meta.jsonl#tick={tick}"
        for field in context["weather"]:
            context["_observation_sources"][f"weather.{field}"] = weather_ref

        def remember_observation(namespace: str, row: Mapping[str, Any]) -> None:
            observation_id = row.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise WorldTruthError(
                    f"{episode_id}@{tick}: {namespace} row lacks observation_id"
                )
            source_ref = f"l0_predicate_state.jsonl#observation_id={observation_id}"
            for field in _values(row):
                context["_observation_sources"][f"{namespace}.{field}"] = source_ref

        if scope.scope_type in {"uav", "vehicle", "pedestrian"}:
            proximity = domain_index.get(
                (tick, "predicate_contract_agent_proximity_geometry", source_id)
            )
            if proximity is not None:
                context["derived"].update(_values(proximity))
                remember_observation("derived", proximity)
        aircraft_geometry = domain_index.get(
            (tick, "predicate_contract_aircraft_geometry", source_id)
        )
        if aircraft_geometry is not None:
            context["geometry"].update(_values(aircraft_geometry))
            remember_observation("geometry", aircraft_geometry)
            position = context["geometry"].get("position_enu_m")
            if isinstance(position, list) and len(position) >= 3:
                context["geometry"]["position_z_m"] = position[2]
                context["_observation_sources"]["geometry.position_z_m"] = (
                    context["_observation_sources"]["geometry.position_enu_m"]
                )
        aircraft_plan = domain_index.get(
            (tick, "predicate_contract_aircraft_plan", source_id)
        )
        if aircraft_plan is not None:
            context["plan"].update(_values(aircraft_plan))
            remember_observation("plan", aircraft_plan)
        restricted_airspace_state = domain_index.get(
            (tick, "predicate_contract_restricted_airspace_state", source_id)
        )
        if restricted_airspace_state is not None:
            context["geometry"].update(_values(restricted_airspace_state))
            remember_observation("geometry", restricted_airspace_state)
        control = domain_index.get((tick, "control_response_state", source_id))
        if control is not None:
            _project_control_response_values(context, _values(control))
        if scope.scope_type == "scene":
            corridor_rows = domain_by_tick_family.get(
                (tick, "predicate_contract_corridor_geometry"), ()
            )
            occupancies = [
                number
                for row in corridor_rows
                if (number := _number(_values(row).get("corridor_occupancy_count")))
                is not None
            ]
            capacities = [
                number
                for row in corridor_rows
                if (number := _number(_values(row).get("corridor_capacity")))
                is not None
            ]
            if occupancies:
                context["scene"]["maximum_corridor_occupancy_count"] = max(occupancies)
            if capacities:
                context["scene"]["maximum_corridor_capacity"] = max(capacities)
            context["scene"]["restricted_region_active"] = (
                effective_restricted_region_active
            )
            context["scene"]["emergency_isolation_active"] = lockdown_active
            security_rows = domain_by_tick_family.get((tick, "security_command"), ())
            jamming_values = [
                _values(row).get("jamming_indicator") for row in security_rows
            ]
            context["scene"]["security_jamming_active"] = _aggregate_boolean(
                jamming_values
            )
        contexts[(scope.scope_type, scope.entity_id)] = context
    return contexts


def _project_control_response_values(
    context: MutableMapping[str, Any],
    values: Mapping[str, Any],
) -> None:
    context["control"].update(values)
    api_mapping = values.get("supported_api_value_keys")
    if not isinstance(api_mapping, Mapping) or not api_mapping:
        raise WorldTruthError("control_response_state lacks supported_api_value_keys")
    for api_path, value_key in sorted(api_mapping.items()):
        parts = str(api_path).split(".")
        if len(parts) != 2 or not all(parts):
            raise WorldTruthError(
                f"control_response_state has invalid API path {api_path!r}"
            )
        if not isinstance(value_key, str) or value_key not in values:
            raise WorldTruthError(
                f"control_response_state API path {api_path!r} lacks value key "
                f"{value_key!r}"
            )
        namespace, field = parts
        target = context.setdefault(namespace, {})
        if not isinstance(target, MutableMapping):
            raise WorldTruthError(
                f"control_response_state API namespace {namespace!r} is not an object"
            )
        target[field] = values[value_key]


def _build_scope_inventory(
    episode_id: str,
    roster_entities: Mapping[str, Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]],
) -> list[_ScopeEntity]:
    result: list[_ScopeEntity] = []
    for entity_id, entity in sorted(roster_entities.items()):
        category = _entity_category(entity)
        scope_type = "facility" if category == "ground_station" else category
        if scope_type in {"uav", "vehicle", "pedestrian", "facility"}:
            if scope_type == "facility":
                try:
                    semantic_scope = validate_roster_facility_scope(entity)
                except FacilityScopeError as exc:
                    raise WorldTruthError(str(exc)) from exc
                result.append(
                    _ScopeEntity(
                        scope_type,
                        entity_id,
                        entity_id,
                        str(semantic_scope["scope_subtype"]),
                        str(semantic_scope["ontology_class_id"]),
                    )
                )
            else:
                result.append(_ScopeEntity(scope_type, entity_id, entity_id))
    node_sources: dict[str, str | None] = {}
    for row in compute_rows:
        node_id = row.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        entity_id = row.get("entity_id")
        source_id = entity_id if isinstance(entity_id, str) and entity_id else None
        previous = node_sources.setdefault(node_id, source_id)
        if previous != source_id:
            raise WorldTruthError(f"compute node changes source entity: {node_id}")
    result.extend(
        _ScopeEntity("compute_node", node_id, source_id)
        for node_id, source_id in sorted(node_sources.items())
    )
    result.append(_ScopeEntity("scene", f"scene:{episode_id}", None))
    result.sort(
        key=lambda scope: (SCOPE_TYPES.index(scope.scope_type), scope.entity_id)
    )
    return result


def _index_frames(
    path: Path,
    episode_id: str,
) -> dict[int, dict[str, Mapping[str, Any]]]:
    frames: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in read_jsonl(path):
        tick = row.get("tick")
        if tick not in FORMAL_TICKS:
            continue
        if tick in frames:
            raise WorldTruthError(f"{path}: duplicate formal tick {tick}")
        entities: dict[str, Mapping[str, Any]] = {}
        for entity in row.get("entities", ()):
            if not isinstance(entity, Mapping):
                continue
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id:
                raise WorldTruthError(f"{path}: tick {tick} entity lacks entity_id")
            if entity_id in entities:
                raise WorldTruthError(
                    f"{path}: tick {tick} duplicate entity {entity_id}"
                )
            entities[entity_id] = entity
        frames[int(tick)] = entities
    missing = sorted(set(FORMAL_TICKS) - set(frames))
    if missing:
        raise WorldTruthError(f"{path}: missing formal ticks {missing}")
    return frames


def _index_tick_rows(
    rows: Iterable[Mapping[str, Any]],
    label: str,
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        tick = row.get("tick")
        if tick not in FORMAL_TICKS:
            continue
        if int(tick) in result:
            raise WorldTruthError(f"duplicate {label} row at tick {tick}")
        result[int(tick)] = row
    missing = sorted(set(FORMAL_TICKS) - set(result))
    if missing:
        raise WorldTruthError(f"{label}: missing formal ticks {missing}")
    return result


def _index_domain_rows(
    rows: Sequence[Mapping[str, Any]],
    episode_id: str,
) -> tuple[
    dict[tuple[int, str, str], Mapping[str, Any]],
    dict[tuple[int, str], list[Mapping[str, Any]]],
]:
    by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    by_family: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("episode_id")) != episode_id:
            continue
        tick = row.get("tick")
        family = row.get("observation_family")
        subject = row.get("subject_id")
        if (
            tick not in FORMAL_TICKS
            or not isinstance(family, str)
            or not isinstance(subject, str)
        ):
            continue
        key = (int(tick), family, subject)
        if key in by_key:
            if dict(by_key[key]) != dict(row):
                raise WorldTruthError(f"conflicting domain state rows: {key}")
            continue
        by_key[key] = row
        by_family[(int(tick), family)].append(row)
    for items in by_family.values():
        items.sort(key=lambda row: str(row.get("subject_id", "")))
    return by_key, by_family


def _index_compute_rows(
    rows: Sequence[Mapping[str, Any]],
    episode_id: str,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        if (
            str(row.get("episode_id")) != episode_id
            or row.get("tick") not in FORMAL_TICKS
        ):
            continue
        node_id = row.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        key = (int(row["tick"]), node_id)
        if key in result:
            raise WorldTruthError(f"duplicate compute state row: {key}")
        result[key] = row
    return result


def _index_utm_records(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_id: str,
) -> dict[tuple[str, int, str], list[Mapping[str, Any]]]:
    result: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for name, rows in records.items():
        for row in rows:
            if (
                str(row.get("episode_id")) != episode_id
                or row.get("tick") not in FORMAL_TICKS
            ):
                continue
            uav_ids: list[str] = []
            if isinstance(row.get("uav_id"), str):
                uav_ids.append(str(row["uav_id"]))
            pair = row.get("pair_uav_ids")
            if isinstance(pair, list):
                uav_ids.extend(str(item) for item in pair if isinstance(item, str))
            for key in ("first_uav_id", "second_uav_id"):
                value = row.get(key)
                if isinstance(value, str):
                    uav_ids.append(value)
            for uav_id in sorted(set(uav_ids)):
                result[(name, int(row["tick"]), uav_id)].append(row)
    return result


def _utm_projection(
    tick: int,
    uav_id: str,
    index: Mapping[tuple[str, int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    mapping = {
        "airspace_authorization_log.jsonl": "authorization",
        "operational_intent_log.jsonl": "operational_intent",
        "flight_plan_log.jsonl": "flight_plan",
    }
    for file_name, key in mapping.items():
        rows = index.get((file_name, tick, uav_id), ())
        if len(rows) == 1:
            result[key] = dict(rows[0])
        elif len(rows) > 1:
            raise WorldTruthError(
                f"multiple UTM {key} rows for {uav_id} at tick {tick}"
            )
    deconfliction = index.get(("deconfliction_log.jsonl", tick, uav_id), ())
    if deconfliction:
        statuses = [str(row.get("resolution_status")) for row in deconfliction]
        result["deconfliction"] = {
            "resolution_status": "active" if "active" in statuses else "not_required"
        }
    return result


def _global_domain_values(
    tick: int,
    by_family: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for family in (
        "ambulance_priority",
        "av_safe_stop",
        "crowd_evacuation",
        "road_closure",
        "signal_queue",
    ):
        rows = by_family.get((tick, family), ())
        if len(rows) == 1:
            result[family] = dict(_values(rows[0]))
    return result


def _resolve_episode_governed_parameters(
    defaults: Mapping[str, Any],
    by_family: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    governed = _number(defaults["boundary_margin_m"])
    if governed is None:
        raise WorldTruthError("governed boundary_margin_m must be finite and numeric")
    for tick in FORMAL_TICKS:
        for row in by_family.get((tick, "predicate_contract_region_runtime_state"), ()):
            observed = _number(_values(row).get("boundary_margin_m"))
            if observed != governed:
                raise WorldTruthError(
                    f"L0 restricted-region state at tick {tick} declares "
                    f"boundary_margin_m={observed!r}; the global authority requires "
                    f"{governed} and forbids an episode override"
                )
    return dict(defaults)


def _domain_projection(
    tick: int,
    entity_id: str,
    index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    global_domain: Mapping[str, Mapping[str, Any]],
    scope_type: str,
) -> dict[str, Any]:
    result = {key: dict(value) for key, value in global_domain.items()}
    for family in (
        "forced_landing_state",
        "gnss_navigation",
        "medical_response",
        "pad_facility",
        "payload_energy",
        "security_command",
    ):
        row = index.get((tick, family, entity_id))
        if row is not None:
            result[family] = dict(_values(row))
    pad = result.get("pad_facility")
    if isinstance(pad, MutableMapping):
        requesters = pad.get("requester_ids")
        occupiers = pad.get("occupier_ids")
        if isinstance(requesters, list):
            pad["requester_count"] = len(requesters)
        if isinstance(occupiers, list):
            pad["occupier_count"] = len(occupiers)
    if scope_type == "facility" and "security_command" not in result:
        result.pop("security_command", None)
    return result


def _truth_projection(entity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(entity, Mapping):
        return {}
    result: dict[str, Any] = {}
    position = _position(entity)
    if position is not None:
        result["position_enu_m"] = list(position)
    speed = _speed(entity)
    if speed is not None:
        result["speed_mps"] = speed
    sumo = entity.get("sumo_vehicle")
    if isinstance(sumo, Mapping):
        for source_key, target_key in (
            ("lane_id", "lane_id"),
            ("lane_ontology_class_id", "lane_ontology_class_id"),
            ("accel_mps2", "acceleration_mps2"),
            ("allowed_speed_mps", "posted_speed_limit_mps"),
            ("speed_limit_regulation_id", "speed_limit_regulation_id"),
            (
                "speed_limit_regulation_ontology_class_id",
                "speed_limit_regulation_ontology_class_id",
            ),
            ("controlling_signal_id", "controlling_signal_id"),
            (
                "controlling_signal_ontology_class_id",
                "controlling_signal_ontology_class_id",
            ),
            ("controlling_signal_state", "controlling_signal_state"),
            ("stop_line_id", "stop_line_id"),
            ("stop_line_ontology_class_id", "stop_line_ontology_class_id"),
            ("right_of_way_id", "right_of_way_id"),
            (
                "right_of_way_ontology_class_id",
                "right_of_way_ontology_class_id",
            ),
            ("crossed_stop_line", "crossed_stop_line"),
            ("leading_vehicle_id", "leading_vehicle_id"),
            (
                "leading_vehicle_ontology_class_id",
                "leading_vehicle_ontology_class_id",
            ),
            ("following_distance_m", "following_distance_m"),
            ("following_distance_rule_id", "following_distance_rule_id"),
            (
                "following_distance_rule_ontology_class_id",
                "following_distance_rule_ontology_class_id",
            ),
        ):
            if source_key in sumo:
                if source_key.endswith("_id") and not _valid_grounding_identifier(
                    sumo[source_key]
                ):
                    continue
                result[target_key] = sumo[source_key]
        if "lane_id" not in result and "sumo_lane_id" in sumo:
            result["lane_id"] = sumo["sumo_lane_id"]
    pedestrian = entity.get("pedestrian_state")
    if isinstance(pedestrian, Mapping):
        for field in (
            "in_crosswalk",
            "crosswalk_id",
            "crosswalk_ontology_class_id",
        ):
            if field in pedestrian:
                result[field] = pedestrian[field]
    return result


def _value_at_path(value: Mapping[str, Any], path: str) -> Any:
    if not path:
        return MISSING
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def _values(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("values")
    return value if isinstance(value, Mapping) else {}


def _aggregate_boolean(values: Sequence[Any]) -> bool | object:
    if any(value is True for value in values):
        return True
    known = [value for value in values if isinstance(value, bool)]
    return False if known and all(value is False for value in known) else MISSING


def _missing_value(value: Any) -> bool:
    return (
        value is MISSING
        or value is None
        or (isinstance(value, str) and value == "unknown")
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _position(entity: Mapping[str, Any]) -> tuple[float, float, float] | None:
    pose = entity.get("truth_pose")
    value = pose.get("position_enu_m") if isinstance(pose, Mapping) else None
    if (
        isinstance(value, list)
        and len(value) >= 3
        and all(_number(item) is not None for item in value[:3])
    ):
        return float(value[0]), float(value[1]), float(value[2])
    return None


def _speed(entity: Mapping[str, Any]) -> float | None:
    sumo = entity.get("sumo_vehicle")
    if isinstance(sumo, Mapping):
        value = _number(sumo.get("speed_mps"))
        if value is not None:
            return value
    annotations = entity.get("annotations")
    if isinstance(annotations, Mapping):
        value = _number(annotations.get("speed_mps"))
        if value is not None:
            return value
    pose = entity.get("truth_pose")
    velocity = pose.get("velocity_enu_mps") if isinstance(pose, Mapping) else None
    if isinstance(velocity, list) and len(velocity) >= 3:
        numbers = [_number(item) for item in velocity[:3]]
        if all(item is not None for item in numbers):
            return math.sqrt(
                sum(float(item) ** 2 for item in numbers if item is not None)
            )
    return None


def _entity_category(entity: Mapping[str, Any]) -> str:
    value = entity.get("entity_category") or entity.get("category")
    return str(value).lower() if isinstance(value, str) else ""


def _index_roster(values: Any, path: Path) -> dict[str, Mapping[str, Any]]:
    if not isinstance(values, list):
        raise WorldTruthError(f"{path}: entities must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for entity in values:
        if not isinstance(entity, Mapping):
            raise WorldTruthError(f"{path}: every entity must be an object")
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise WorldTruthError(f"{path}: entity lacks entity_id")
        if entity_id in result:
            raise WorldTruthError(f"{path}: duplicate entity_id {entity_id}")
        result[entity_id] = entity
    return result


def _evaluate_grounded_world_truth(
    episode_root: Path,
    *,
    source_availability: Mapping[str, Any],
    domain_rows: Sequence[Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]],
    communication_rows: Sequence[Mapping[str, Any]],
    compute_predicate_rows: Sequence[Mapping[str, Any]],
    utm_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> WorldTruthResult:
    episode_root = episode_root.resolve()
    paths = {
        "manifest": episode_root / "episode_manifest.json",
        "roster": episode_root / "global_entity_roster.json",
        "truth": episode_root / "truth_frames.jsonl",
        "weather": episode_root / "weather_meta.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise WorldTruthError(f"required L0 input is missing: {path}")

    manifest = _load_object(paths["manifest"])
    episode_id = str(manifest.get("episode_id") or episode_root.name)
    if episode_id != episode_root.name:
        raise WorldTruthError(
            f"manifest episode_id {episode_id!r} differs from directory {episode_root.name!r}"
        )
    registry = load_core_semantic_registry()
    templates = get_core_predicate_templates()
    if len(templates) != 79 or tuple(registry["world_scope_types"]) != SCOPE_TYPES:
        raise WorldTruthError("world-truth registry must contain exactly 79 predicates")
    all_template_by_id = {str(template["id"]): template for template in templates}
    if len(all_template_by_id) != len(templates):
        raise WorldTruthError("world-truth registry contains duplicate predicate ids")
    template_by_id = {
        predicate_id: template
        for predicate_id, template in all_template_by_id.items()
        if template.get("implementation_status") == "executable_l1"
    }
    declared_executable_ids = set(
        registry["implementation"]["executable_predicate_ids"]
    )
    if set(template_by_id) != declared_executable_ids:
        raise WorldTruthError(
            "world-truth executable templates differ from implementation metadata"
        )
    non_executable_contracts = dict(
        registry["implementation"]["non_executable_predicates"]
    )

    roster = _load_object(paths["roster"])
    roster_entities = _index_roster(roster.get("entities"), paths["roster"])
    frames = _index_frames(paths["truth"], episode_id)
    weather = _index_tick_rows(read_jsonl(paths["weather"]), "weather")
    domain_index, domain_by_tick_family = _index_domain_rows(domain_rows, episode_id)
    defaults = _resolve_episode_governed_parameters(
        get_governed_parameter_defaults(), domain_by_tick_family
    )
    scopes = _build_scope_inventory(episode_id, roster_entities, compute_rows)
    scope_by_entity = {
        scope.entity_id: scope for scope in scopes if scope.scope_type != "scene"
    }
    source_unavailability = _index_source_unavailability(
        source_availability, episode_id, frozenset(all_template_by_id)
    )
    compute_by_node = _index_compute_rows(compute_rows, episode_id)
    utm_index = _index_utm_records(utm_records, episode_id)
    direct_by_tick_predicate = _grounded_direct_rows(
        compute_predicate_rows, episode_id, all_template_by_id
    )
    utm_by_file_tick = _grounded_utm_rows(utm_records, episode_id)
    entity_classes = _build_grounding_class_catalog(
        scopes=scopes,
        frames=frames,
        domain_rows=domain_rows,
        direct_rows=compute_predicate_rows,
        utm_records=utm_records,
    )

    input_digest = digest_object(
        {key: digest_file(path) for key, path in sorted(paths.items())}
        | {
            "domain_rows": digest_object(domain_rows),
            "compute_rows": digest_object(compute_rows),
            "communication_rows": digest_object(communication_rows),
            "compute_predicate_rows": digest_object(compute_predicate_rows),
            "utm_records": digest_object(utm_records),
            "registry_templates": digest_object(templates),
            "source_availability": digest_object(source_availability),
        }
    )
    parameter_digest = digest_object(defaults)
    rule_digest_by_id = {
        predicate_id: digest_object(template)
        for predicate_id, template in template_by_id.items()
    }

    previous_states: dict[tuple[str, str], dict[str, Any]] = {}
    previous_contexts: dict[tuple[str, str], dict[str, Any]] = {}
    base_assertions: list[dict[str, Any]] = []
    base_count_by_predicate: Counter[str] = Counter()
    evaluated_count_by_predicate: Counter[str] = Counter()
    value_counts: Counter[str] = Counter()
    value_counts_by_predicate: dict[str, Counter[str]] = defaultdict(Counter)
    operation_counts: Counter[str] = Counter()
    deltas: list[dict[str, Any]] = []
    prior_delta_id: str | None = None

    for tick in FORMAL_TICKS:
        contexts = _build_tick_contexts(
            episode_id=episode_id,
            tick=tick,
            scopes=scopes,
            roster_entities=roster_entities,
            entities_at_tick=frames[tick],
            weather_row=weather[tick],
            domain_index=domain_index,
            domain_by_tick_family=domain_by_tick_family,
            compute_by_node=compute_by_node,
            utm_index=utm_index,
        )
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        for predicate_id, template in template_by_id.items():
            for candidate in _ground_candidates_for_template(
                episode_id=episode_id,
                tick=tick,
                manifest=manifest,
                template=template,
                scopes=scopes,
                contexts=contexts,
                domain_by_tick_family=domain_by_tick_family,
                direct_by_tick_predicate=direct_by_tick_predicate,
                utm_by_file_tick=utm_by_file_tick,
                entity_classes=entity_classes,
            ):
                tuple_id = candidate.get("tuple_id")
                if not isinstance(tuple_id, str) or not tuple_id:
                    raise WorldTruthError(
                        f"{predicate_id}@{tick}: grounded candidate lacks string tuple_id"
                    )
                key = (predicate_id, tuple_id)
                existing = candidates.get(key)
                if existing is None:
                    candidates[key] = candidate
                    continue
                _merge_duplicate_candidate(existing, candidate)

        current_states: dict[tuple[str, str], dict[str, Any]] = {}
        current_contexts: dict[tuple[str, str], dict[str, Any]] = {}
        for key, candidate in sorted(candidates.items()):
            predicate_id, _tuple_id = key
            template = template_by_id[predicate_id]
            context = _candidate_evaluation_context(
                tick=tick,
                candidate=candidate,
                template=template,
                domain_by_tick_family=domain_by_tick_family,
            )
            state = _evaluate_grounded_state(
                episode_id=episode_id,
                tick=tick,
                template=template,
                candidate=candidate,
                context=context,
                previous_context=previous_contexts.get(key, {}),
                defaults=defaults,
                rule_digest=rule_digest_by_id[predicate_id],
                parameter_digest=parameter_digest,
                input_digest=input_digest,
                scope_by_entity=scope_by_entity,
                source_unavailability=source_unavailability,
            )
            current_states[key] = state
            current_contexts[key] = context
            evaluated_count_by_predicate[predicate_id] += 1
            value_counts[str(state["value"])] += 1
            value_counts_by_predicate[predicate_id][str(state["value"])] += 1

        operations: list[dict[str, Any]] = []
        if tick == FORMAL_TICKS[0]:
            base_assertions = list(current_states.values())
            base_count_by_predicate.update(
                predicate_id for predicate_id, _tuple_id in current_states
            )
        else:
            for key in sorted(previous_states.keys() - current_states.keys()):
                previous = previous_states[key]
                operations.append(
                    {
                        "operation": "remove_predicate_assertion",
                        "assertion_id": previous["assertion_id"],
                        "predicate_id": previous["predicate_id"],
                        "tuple_id": previous["tuple_id"],
                        "bindings": previous["bindings"],
                        "binding_ontology_classes": previous[
                            "binding_ontology_classes"
                        ],
                        "from_value": previous["value"],
                    }
                )
                operation_counts["remove_predicate_assertion"] += 1
            for key in sorted(current_states.keys() - previous_states.keys()):
                operations.append(
                    {
                        "operation": "add_predicate_assertion",
                        "assertion": current_states[key],
                    }
                )
                operation_counts["add_predicate_assertion"] += 1
            for key in sorted(current_states.keys() & previous_states.keys()):
                current = current_states[key]
                previous = previous_states[key]
                if (
                    current["bindings"] != previous["bindings"]
                    or current["binding_ontology_classes"]
                    != previous["binding_ontology_classes"]
                ):
                    raise WorldTruthError(
                        f"grounded tuple identity changed in place: {key}"
                    )
                if current["value"] != previous["value"]:
                    operations.append(
                        {
                            "operation": "set_predicate_value",
                            "assertion_id": current["assertion_id"],
                            "predicate_id": current["predicate_id"],
                            "tuple_id": current["tuple_id"],
                            "bindings": current["bindings"],
                            "binding_ontology_classes": current[
                                "binding_ontology_classes"
                            ],
                            "from_value": previous["value"],
                            "to_value": current["value"],
                            "missing_source_record": current["missing_source_record"],
                            "observations": current["observations"],
                            "source_refs": current["source_refs"],
                            "rule_digest": current["rule_digest"],
                        }
                    )
                    operation_counts["set_predicate_value"] += 1
                else:
                    current["truth_state_update_tick"] = previous[
                        "truth_state_update_tick"
                    ]
                    if any(
                        current[field] != previous[field]
                        for field in (
                            "missing_source_record",
                            "observations",
                            "source_refs",
                        )
                    ):
                        operations.append(
                            {
                                "operation": "refresh_predicate_evidence",
                                "assertion_id": current["assertion_id"],
                                "predicate_id": current["predicate_id"],
                                "tuple_id": current["tuple_id"],
                                "bindings": current["bindings"],
                                "binding_ontology_classes": current[
                                    "binding_ontology_classes"
                                ],
                                "value": current["value"],
                                "missing_source_record": current[
                                    "missing_source_record"
                                ],
                                "observations": current["observations"],
                                "source_refs": current["source_refs"],
                            }
                        )
                        operation_counts["refresh_predicate_evidence"] += 1
                    else:
                        current["evidence_update_tick"] = previous[
                            "evidence_update_tick"
                        ]
        if operations:
            delta_id = stable_identifier(
                "world_truth_graph_delta", episode_id, tick, operations
            )
            delta = {
                "schema_name": DELTA_SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "annotation_layer": "L1",
                "source_layer": "L0",
                "episode_id": episode_id,
                "tick": tick,
                "delta_id": delta_id,
                "previous_delta_id": prior_delta_id,
                "operations": operations,
                "operation_count": len(operations),
            }
            deltas.append(delta)
            prior_delta_id = delta_id
        previous_states = current_states
        previous_contexts = current_contexts

    base_assertions.sort(key=lambda row: (row["predicate_id"], row["tuple_id"]))
    scope_nodes = [_scope_node(episode_id, scope) for scope in scopes]
    base_graph: dict[str, Any] = {
        "schema_name": BASE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "annotation_layer": "L1",
        "source_layer": "L0",
        "episode_id": episode_id,
        "tick": FORMAL_TICKS[0],
        "representation": "grounded_predicate_candidate_base",
        "candidate_set_complete": True,
        "candidate_set_scope": "executable_l1_predicates_only",
        "predicate_vocabulary": sorted(all_template_by_id),
        "executable_predicate_ids": sorted(template_by_id),
        "non_executable_predicates": non_executable_contracts,
        "scope_entities": scope_nodes,
        "initial_assertions": base_assertions,
        "summary": {
            "predicate_count": len(all_template_by_id),
            "executable_predicate_count": len(template_by_id),
            "non_executable_predicate_count": len(non_executable_contracts),
            "initial_assertion_count": len(base_assertions),
            "initial_candidate_counts_by_predicate": {
                predicate_id: base_count_by_predicate[predicate_id]
                for predicate_id in sorted(template_by_id)
            },
        },
        "input_digest": input_digest,
        "parameter_digest": parameter_digest,
        "governed_parameters": defaults,
    }
    base_graph["base_graph_digest"] = digest_object(base_graph)
    for delta in deltas:
        delta["base_graph_digest"] = base_graph["base_graph_digest"]
        delta["delta_digest"] = digest_object(delta)

    summary = {
        "schema_name": "world_truth_evaluation_summary",
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "representation": "grounded_predicate_candidates",
        "candidate_set_complete": True,
        "candidate_set_scope": "executable_l1_predicates_only",
        "formal_tick_count": len(FORMAL_TICKS),
        "predicate_count": len(all_template_by_id),
        "executable_predicate_count": len(template_by_id),
        "non_executable_predicate_count": len(non_executable_contracts),
        "non_executable_predicates": non_executable_contracts,
        "evaluated_candidate_count": sum(evaluated_count_by_predicate.values()),
        "evaluated_candidate_counts_by_predicate": {
            predicate_id: evaluated_count_by_predicate[predicate_id]
            for predicate_id in sorted(template_by_id)
        },
        "value_counts": dict(sorted(value_counts.items())),
        "value_counts_by_predicate": {
            predicate_id: dict(sorted(value_counts_by_predicate[predicate_id].items()))
            for predicate_id in sorted(template_by_id)
        },
        "base_assertion_count": len(base_assertions),
        "delta_batch_count": len(deltas),
        "delta_operation_count": sum(operation_counts.values()),
        "delta_operation_counts": dict(sorted(operation_counts.items())),
        "input_digest": input_digest,
        "parameter_digest": parameter_digest,
        "governed_parameters": defaults,
    }
    return WorldTruthResult(
        base_graph=base_graph,
        deltas=tuple(deltas),
        summary=summary,
    )


def _grounded_identity(
    value: Mapping[str, Any],
    context: str,
) -> tuple[str, str]:
    predicate_id = value.get("predicate_id")
    tuple_id = value.get("tuple_id")
    if not isinstance(predicate_id, str) or not predicate_id:
        raise WorldTruthError(f"{context} lacks non-empty string predicate_id")
    if not isinstance(tuple_id, str) or not tuple_id:
        raise WorldTruthError(f"{context} lacks non-empty string tuple_id")
    return predicate_id, tuple_id


def _replay_grounded_world_truth(
    base_graph: Mapping[str, Any],
    deltas: Sequence[Mapping[str, Any]],
    tick: int,
) -> dict[str, Any]:
    if base_graph.get("schema_name") != BASE_SCHEMA_NAME:
        raise WorldTruthError("unexpected world-truth base schema")
    if base_graph.get("schema_version") != SCHEMA_VERSION:
        raise WorldTruthError("world-truth base version mismatch")
    if base_graph.get("representation") != "grounded_predicate_candidate_base":
        raise WorldTruthError("world-truth base is not the grounded representation")
    if tick not in FORMAL_TICKS:
        raise WorldTruthError("world-truth replay tick must be a formal tick")
    registry = load_core_semantic_registry()
    implementation = registry["implementation"]
    vocabulary_ids = set(implementation["declared_predicate_vocabulary_ids"])
    executable_ids = set(implementation["executable_predicate_ids"])
    non_executable_contracts = dict(implementation["non_executable_predicates"])
    if set(base_graph.get("predicate_vocabulary", ())) != vocabulary_ids:
        raise WorldTruthError("world-truth base predicate vocabulary is stale")
    if set(base_graph.get("executable_predicate_ids", ())) != executable_ids:
        raise WorldTruthError("world-truth base executable predicate set is stale")
    if base_graph.get("non_executable_predicates") != non_executable_contracts:
        raise WorldTruthError("world-truth base non-executable contracts are stale")
    if base_graph.get("candidate_set_scope") != "executable_l1_predicates_only":
        raise WorldTruthError("world-truth base does not declare executable-only scope")
    claimed_base_digest = base_graph.get("base_graph_digest")
    episode_id = str(base_graph["episode_id"])
    state: dict[tuple[str, str], dict[str, Any]] = {}
    for assertion in base_graph.get("initial_assertions", ()):
        if not isinstance(assertion, Mapping):
            raise WorldTruthError("world-truth base assertion must be an object")
        key = _grounded_identity(assertion, "world-truth base assertion")
        if key[0] not in executable_ids:
            raise WorldTruthError(
                "world-truth base contains a non-executable predicate assertion"
            )
        if key in state:
            raise WorldTruthError(f"duplicate grounded base assertion: {key}")
        state[key] = dict(assertion)

    previous_delta_id: str | None = None
    for delta in sorted(
        deltas, key=lambda row: (int(row.get("tick", -1)), str(row.get("delta_id", "")))
    ):
        if str(delta.get("episode_id")) != episode_id:
            raise WorldTruthError("world-truth delta episode mismatch")
        if delta.get("schema_version") != SCHEMA_VERSION:
            raise WorldTruthError("world-truth delta version mismatch")
        if delta.get("previous_delta_id") != previous_delta_id:
            raise WorldTruthError("world-truth delta chain is not contiguous")
        previous_delta_id = str(delta["delta_id"])
        if int(delta["tick"]) > tick:
            continue
        for operation in delta.get("operations", ()):
            operation_name = str(operation.get("operation") or "")
            if operation_name == "add_predicate_assertion":
                assertion = operation.get("assertion")
                if not isinstance(assertion, Mapping):
                    raise WorldTruthError("grounded add operation lacks assertion")
                if assertion.get("predicate_id") not in executable_ids:
                    raise WorldTruthError(
                        "grounded add targets a non-executable predicate"
                    )
                key = _grounded_identity(assertion, "grounded add assertion")
                if key in state:
                    raise WorldTruthError(f"grounded add duplicates assertion: {key}")
                state[key] = dict(assertion)
            elif operation_name == "remove_predicate_assertion":
                if operation.get("predicate_id") not in executable_ids:
                    raise WorldTruthError(
                        "grounded remove targets a non-executable predicate"
                    )
                key = _grounded_identity(operation, "grounded remove operation")
                current = state.get(key)
                if current is None:
                    raise WorldTruthError(
                        f"grounded remove references absent tuple: {key}"
                    )
                if (
                    current["value"] != operation["from_value"]
                    or current["bindings"] != operation["bindings"]
                    or current["binding_ontology_classes"]
                    != operation["binding_ontology_classes"]
                ):
                    raise WorldTruthError(
                        f"grounded remove precondition mismatch: {key}"
                    )
                del state[key]
            elif operation_name == "set_predicate_value":
                if operation.get("predicate_id") not in executable_ids:
                    raise WorldTruthError(
                        "grounded set targets a non-executable predicate"
                    )
                key = _grounded_identity(operation, "grounded set operation")
                current = state.get(key)
                if current is None:
                    raise WorldTruthError(
                        f"grounded set references absent tuple: {key}"
                    )
                if (
                    current["value"] != operation["from_value"]
                    or current["bindings"] != operation["bindings"]
                    or current["binding_ontology_classes"]
                    != operation["binding_ontology_classes"]
                ):
                    raise WorldTruthError(f"grounded set precondition mismatch: {key}")
                current.update(
                    value=str(operation["to_value"]),
                    missing_source_record=list(operation["missing_source_record"]),
                    observations=list(operation["observations"]),
                    source_refs=list(operation["source_refs"]),
                    truth_state_update_tick=int(delta["tick"]),
                    evidence_update_tick=int(delta["tick"]),
                )
            elif operation_name == "refresh_predicate_evidence":
                key = _grounded_identity(operation, "grounded evidence refresh")
                current = state.get(key)
                if current is None or current["value"] != operation["value"]:
                    raise WorldTruthError(
                        f"grounded evidence refresh precondition mismatch: {key}"
                    )
                if (
                    current["bindings"] != operation["bindings"]
                    or current["binding_ontology_classes"]
                    != operation["binding_ontology_classes"]
                ):
                    raise WorldTruthError(
                        f"grounded evidence refresh binding mismatch: {key}"
                    )
                current.update(
                    missing_source_record=list(operation["missing_source_record"]),
                    observations=list(operation["observations"]),
                    source_refs=list(operation["source_refs"]),
                    evidence_update_tick=int(delta["tick"]),
                )
            else:
                raise WorldTruthError(
                    f"unknown grounded world-truth delta operation: {operation_name}"
                )
    assertions = sorted(
        state.values(),
        key=lambda row: (row["predicate_id"], row["tuple_id"]),
    )
    result = {
        "schema_name": REPLAY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "annotation_layer": "L1",
        "source_layer": "L0",
        "episode_id": episode_id,
        "tick": tick,
        "representation": "grounded_predicate_candidates",
        "candidate_set_complete": True,
        "base_graph_digest": claimed_base_digest,
        "assertions": assertions,
        "assertion_count": len(assertions),
    }
    result["state_digest"] = digest_object(assertions)
    return result


def _scope_node(episode_id: str, scope: _ScopeEntity) -> dict[str, Any]:
    return {
        "node_id": stable_identifier(
            "world_scope_entity", episode_id, scope.scope_type, scope.entity_id
        ),
        "kind": "world_scope_entity",
        "scope_type": scope.scope_type,
        "entity_id": scope.entity_id,
        "source_entity_id": scope.source_entity_id,
        "scope_subtype": scope.scope_subtype,
        "ontology_class_id": _scope_ontology_class(scope),
    }


def _scope_ontology_class(scope: _ScopeEntity) -> str | None:
    if scope.ontology_class_id:
        return str(scope.ontology_class_id)
    return {
        "uav": "world:UnmannedAircraft",
        "vehicle": "world:GroundVehicle",
        "pedestrian": "world:Pedestrian",
        "compute_node": "world:ComputeNode",
    }.get(scope.scope_type)


def _grounded_direct_rows(
    rows: Sequence[Mapping[str, Any]],
    episode_id: str,
    templates: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[int, str], list[Mapping[str, Any]]]:
    result: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str, tuple[tuple[str, str], ...]]] = set()
    for row in rows:
        if str(row.get("episode_id")) != episode_id:
            raise WorldTruthError("compute predicate row episode mismatch")
        tick = row.get("tick")
        predicate_id = str(row.get("predicate_id") or "")
        if tick not in FORMAL_TICKS or predicate_id not in templates:
            raise WorldTruthError("compute predicate row has invalid tick or predicate")
        if templates[predicate_id].get("implementation_status") != "executable_l1":
            raise WorldTruthError(
                f"direct predicate row targets non-executable predicate: {predicate_id}"
            )
        if not predicate_id.startswith(("compute.", "communication.")):
            raise WorldTruthError(
                f"direct predicate row is not compute/communication: {predicate_id}"
            )
        if row.get("binding_status") != "complete":
            raise WorldTruthError(
                f"direct predicate row has incomplete grounding: {predicate_id}@{tick}"
            )
        expected_roles = [
            str(role["key"]) for role in templates[predicate_id]["argument_roles"]
        ]
        bindings = row.get("bindings")
        classes = row.get("binding_ontology_classes")
        if (
            not isinstance(bindings, Mapping)
            or list(bindings) != expected_roles
            or not isinstance(classes, Mapping)
            or set(classes) != set(expected_roles)
        ):
            raise WorldTruthError(
                f"direct predicate row role set/order differs from registry: "
                f"{predicate_id}@{tick}"
            )
        if any(
            not _valid_grounding_identifier(bindings[role]) for role in expected_roles
        ):
            raise WorldTruthError(
                f"direct predicate row has invalid binding id: {predicate_id}@{tick}"
            )
        duplicate_key = (
            int(tick),
            predicate_id,
            tuple((role, str(bindings[role])) for role in expected_roles),
        )
        if duplicate_key in seen:
            raise WorldTruthError(
                f"duplicate direct grounded predicate row: {duplicate_key}"
            )
        seen.add(duplicate_key)
        result[(int(tick), predicate_id)].append(row)
    return result


def _grounded_utm_rows(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_id: str,
) -> dict[tuple[str, int], list[Mapping[str, Any]]]:
    result: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for file_name, rows in records.items():
        for row in rows:
            if str(row.get("episode_id")) != episode_id:
                raise WorldTruthError(f"{file_name}: UTM episode mismatch")
            tick = row.get("tick")
            if tick not in FORMAL_TICKS:
                raise WorldTruthError(f"{file_name}: invalid formal tick {tick!r}")
            result[(str(file_name), int(tick))].append(row)
    return result


def _build_grounding_class_catalog(
    *,
    scopes: Sequence[_ScopeEntity],
    frames: Mapping[int, Mapping[str, Mapping[str, Any]]],
    domain_rows: Sequence[Mapping[str, Any]],
    direct_rows: Sequence[Mapping[str, Any]],
    utm_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, set[str]]:
    catalog: dict[str, set[str]] = defaultdict(set)
    for scope in scopes:
        ontology_class = _scope_ontology_class(scope)
        if ontology_class:
            catalog[scope.entity_id].add(ontology_class)
            if scope.source_entity_id:
                catalog[scope.source_entity_id].add(ontology_class)
    for entities in frames.values():
        for entity in entities.values():
            _collect_explicit_identifier_classes(entity, catalog)
    for row in domain_rows:
        _collect_explicit_identifier_classes(row, catalog)
    for row in direct_rows:
        _collect_explicit_identifier_classes(row, catalog)
    for rows in utm_records.values():
        for row in rows:
            _collect_explicit_identifier_classes(row, catalog)
    return catalog


def _collect_explicit_identifier_classes(
    value: Mapping[str, Any], catalog: MutableMapping[str, set[str]]
) -> None:
    bindings = value.get("bindings")
    binding_classes = value.get("binding_ontology_classes")
    if isinstance(bindings, Mapping) and isinstance(binding_classes, Mapping):
        for role, entity_id in bindings.items():
            ontology_class = binding_classes.get(role)
            if _valid_grounding_identifier(entity_id) and isinstance(
                ontology_class, str
            ):
                catalog[str(entity_id)].add(_world_class(ontology_class))
    for key, entity_id in value.items():
        if not key.endswith("_id") or not _valid_grounding_identifier(entity_id):
            continue
        class_key = f"{key[:-3]}_ontology_class_id"
        ontology_class = value.get(class_key)
        if isinstance(ontology_class, str) and ontology_class:
            catalog[str(entity_id)].add(_world_class(ontology_class))
    for nested in value.values():
        if isinstance(nested, Mapping):
            _collect_explicit_identifier_classes(nested, catalog)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, Mapping):
                    _collect_explicit_identifier_classes(item, catalog)


def _ground_candidates_for_template(
    *,
    episode_id: str,
    tick: int,
    manifest: Mapping[str, Any],
    template: Mapping[str, Any],
    scopes: Sequence[_ScopeEntity],
    contexts: Mapping[tuple[str, str], Mapping[str, Any]],
    domain_by_tick_family: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    direct_by_tick_predicate: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    utm_by_file_tick: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    entity_classes: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    predicate_id = str(template["id"])
    if template.get("implementation_status") != "executable_l1":
        raise WorldTruthError(
            f"candidate grounding requested for non-executable predicate: {predicate_id}"
        )
    grounding = template["grounding_spec"]
    authority = grounding["candidate_authority"]
    kind = str(authority["kind"])
    records: list[tuple[Mapping[str, Any] | None, _ScopeEntity | None]] = []
    if kind in {"scope_entity", "scope_relation"}:
        scope_roles = [
            role
            for role, source in grounding["role_sources"].items()
            if source["kind"] == "scope_entity"
        ]
        if len(scope_roles) != 1:
            raise WorldTruthError(
                f"{predicate_id}: scope authority must declare one scope_entity role"
            )
        expected = _role_class(template, scope_roles[0])
        for scope in scopes:
            actual = _scope_ontology_class(scope)
            if actual and ontology_class_is_a(actual, expected):
                records.append((contexts[(scope.scope_type, scope.entity_id)], scope))
    elif kind == "domain_observation":
        records.extend(
            (row, None)
            for row in domain_by_tick_family.get((tick, str(authority["source"])), ())
        )
    elif kind == "domain_nested_relation":
        source = str(authority["source"])
        family, separator, nested_path = source.partition(":")
        if not separator:
            raise WorldTruthError(f"{predicate_id}: invalid nested authority {source}")
        for parent in domain_by_tick_family.get((tick, family), ()):
            nested = _value_at_path(parent, nested_path)
            if nested is MISSING or nested is None:
                continue
            if not isinstance(nested, list):
                raise WorldTruthError(
                    f"{predicate_id}: nested authority is not an array"
                )
            parent_refs = [
                str(ref)
                for ref in parent.get("source_refs", ())
                if isinstance(ref, str) and ref
            ]
            if not parent_refs:
                raise WorldTruthError(
                    f"{predicate_id}: nested authority parent lacks source_refs"
                )
            for index, item in enumerate(nested):
                if not isinstance(item, Mapping):
                    continue
                child = dict(item)
                child_refs = [
                    str(ref)
                    for ref in child.get("source_refs", ())
                    if isinstance(ref, str) and ref
                ]
                child["source_refs"] = sorted(
                    set(child_refs)
                    | {f"{ref}:{nested_path}[{index}]" for ref in parent_refs}
                )
                records.append((child, None))
    elif kind == "direct_predicate_row":
        records.extend(
            (row, None)
            for row in direct_by_tick_predicate.get((tick, predicate_id), ())
        )
    elif kind == "utm_record":
        records.extend(
            (row, None)
            for row in utm_by_file_tick.get((str(authority["source"]), tick), ())
        )
    elif kind == "episode_operational_region":
        scene_context = next(
            context
            for (scope_type, _entity_id), context in contexts.items()
            if scope_type == "scene"
        )
        operational_region_record = dict(manifest)
        operational_region_record["weather"] = dict(scene_context["weather"])
        operational_region_record["source_refs"] = [
            str(authority["source"]),
            f"weather_meta.jsonl#tick={tick}",
        ]
        records.append((operational_region_record, None))
    else:
        raise WorldTruthError(f"{predicate_id}: unsupported grounding authority {kind}")

    candidates: list[dict[str, Any]] = []
    for record, scope in records:
        candidate = _candidate_from_authority_record(
            episode_id=episode_id,
            tick=tick,
            template=template,
            authority_kind=kind,
            authority_source=str(authority["source"]),
            record=record or {},
            scope=scope,
            entity_classes=entity_classes,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidate_from_authority_record(
    *,
    episode_id: str,
    tick: int,
    template: Mapping[str, Any],
    authority_kind: str,
    authority_source: str,
    record: Mapping[str, Any],
    scope: _ScopeEntity | None,
    entity_classes: Mapping[str, set[str]],
) -> dict[str, Any] | None:
    predicate_id = str(template["id"])
    if template.get("implementation_status") != "executable_l1":
        raise WorldTruthError(
            f"candidate projection requested for non-executable predicate: {predicate_id}"
        )
    grounding = template["grounding_spec"]
    roles = [str(role["key"]) for role in template["argument_roles"]]
    bindings: dict[str, str] = {}
    classes: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for role in roles:
        role_source = grounding["role_sources"][role]
        source_kind = str(role_source["kind"])
        path = str(role_source.get("path") or "")
        if source_kind == "scope_entity":
            if scope is None:
                raise WorldTruthError(f"{predicate_id}: scope role lacks scope entity")
            raw_id: Any = scope.entity_id
            explicit_class = _scope_ontology_class(scope)
        elif source_kind == "record_binding":
            raw_id = _value_at_path(record, f"bindings.{path}")
            explicit_class = _value_at_path(record, f"binding_ontology_classes.{path}")
        else:
            raw_id = _value_at_path(record, path)
            explicit_class = _explicit_class_for_identifier_path(record, path)
            if source_kind == "manifest":
                explicit_class = "world:OperationalRegion"
        if raw_id is MISSING or raw_id is None or raw_id == "":
            return None
        if not _valid_grounding_identifier(raw_id):
            raise WorldTruthError(
                f"{predicate_id}@{tick}: invalid {role} grounding id {raw_id!r}"
            )
        entity_id = str(raw_id)
        expected_class = _role_class(template, role)
        actual_class = _resolve_actual_class(
            entity_id,
            expected_class,
            explicit_class,
            entity_classes,
            predicate_id,
            role,
        )
        bindings[role] = entity_id
        classes[role] = actual_class
        provenance[role] = {
            "entity_id": entity_id,
            "ontology_class_id": actual_class,
            "candidate_authority_kind": authority_kind,
            "candidate_authority_source": authority_source,
            "binding_source": (
                "scope_entity" if source_kind == "scope_entity" else path
            ),
        }

    _validate_binding_constraints(predicate_id, bindings, grounding)
    tuple_id = stable_identifier(
        "world_predicate_tuple",
        predicate_id,
        [(role, bindings[role]) for role in roles],
    )
    source_refs = {
        str(ref)
        for ref in record.get("source_refs", ())
        if isinstance(ref, str) and ref
    }
    if scope is not None:
        source_refs.add(f"global_entity_roster.json#entity={scope.entity_id}")
    return {
        "predicate_id": predicate_id,
        "tuple_id": tuple_id,
        "bindings": bindings,
        "binding_ontology_classes": classes,
        "binding_provenance": provenance,
        "authority_kind": authority_kind,
        "authority_source": authority_source,
        "record": dict(record),
        "scope": scope,
        "source_refs": source_refs,
    }


def _role_class(template: Mapping[str, Any], role_key: str) -> str:
    for role in template["argument_roles"]:
        if role["key"] == role_key:
            return _world_class(str(role["class"]))
    raise WorldTruthError(f"{template['id']}: unknown ontology role {role_key}")


def _world_class(value: str) -> str:
    return value if ":" in value else f"world:{value}"


def _explicit_class_for_identifier_path(record: Mapping[str, Any], path: str) -> Any:
    prefix, separator, leaf = path.rpartition(".")
    if not leaf.endswith("_id"):
        return MISSING
    class_leaf = f"{leaf[:-3]}_ontology_class_id"
    class_path = f"{prefix}.{class_leaf}" if separator else class_leaf
    return _value_at_path(record, class_path)


def _resolve_actual_class(
    entity_id: str,
    expected_class: str,
    explicit_class: Any,
    entity_classes: Mapping[str, set[str]],
    predicate_id: str,
    role: str,
) -> str:
    if isinstance(explicit_class, str) and explicit_class:
        actual = _world_class(explicit_class)
    else:
        compatible = sorted(
            ontology_class
            for ontology_class in entity_classes.get(entity_id, ())
            if ontology_class_is_a(ontology_class, expected_class)
        )
        if not compatible:
            raise WorldTruthError(
                f"{predicate_id}: {role}={entity_id!r} lacks typed instance provenance "
                f"compatible with {expected_class}"
            )
        most_specific = [
            candidate
            for candidate in compatible
            if not any(
                other != candidate and ontology_class_is_a(other, candidate)
                for other in compatible
            )
        ]
        if len(most_specific) != 1:
            raise WorldTruthError(
                f"{predicate_id}: {role}={entity_id!r} has ambiguous classes "
                f"{most_specific}"
            )
        actual = most_specific[0]
    if not ontology_class_is_a(actual, expected_class):
        raise WorldTruthError(
            f"{predicate_id}: role {role} requires {expected_class}, got {actual}"
        )
    return actual


def _validate_binding_constraints(
    predicate_id: str,
    bindings: Mapping[str, str],
    grounding: Mapping[str, Any],
) -> None:
    for role_set in grounding.get("distinct_role_sets", ()):
        values = [bindings[str(role)] for role in role_set]
        if len(values) != len(set(values)):
            raise WorldTruthError(
                f"{predicate_id}: distinct roles bind the same instance: {role_set}"
            )
    for role_set in grounding.get("unordered_role_sets", ()):
        values = [bindings[str(role)] for role in role_set]
        if values != sorted(values):
            raise WorldTruthError(
                f"{predicate_id}: unordered role binding is not canonical: {values}"
            )


def _valid_grounding_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.strip().lower()
        not in {
            "unknown",
            "none",
            "null",
            "n/a",
        }
    )


def _merge_duplicate_candidate(
    existing: MutableMapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    for field in (
        "bindings",
        "binding_ontology_classes",
        "binding_provenance",
        "authority_kind",
        "authority_source",
    ):
        if existing[field] != candidate[field]:
            raise WorldTruthError(
                f"conflicting duplicate grounded candidate {existing['predicate_id']}:"
                f"{existing['tuple_id']}"
            )
    existing["source_refs"].update(candidate["source_refs"])
    records = existing.setdefault("duplicate_records", [existing["record"]])
    records.append(candidate["record"])


def _candidate_evaluation_context(
    *,
    tick: int,
    candidate: Mapping[str, Any],
    template: Mapping[str, Any],
    domain_by_tick_family: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    observation_sources: dict[str, str] = {}
    record = candidate["record"]
    records = candidate.get("duplicate_records", [record])
    for target_path, source in template["grounding_spec"]["field_sources"].items():
        source = str(source)
        if source.startswith("context:"):
            record_path = source.split(":", 1)[1]
            value = _value_at_path(record, record_path)
            origin = record.get("_observation_sources", {}).get(record_path)
            if origin is not None:
                observation_sources[str(target_path)] = str(origin)
        elif source.startswith("record:"):
            record_path = source.split(":", 1)[1]
            values = [_value_at_path(item, record_path) for item in records]
            if any(item != values[0] for item in values[1:]):
                raise WorldTruthError(
                    f"{template['id']}: duplicate authority records disagree for "
                    f"{candidate['tuple_id']} field {record_path}"
                )
            value = values[0]
        elif source.startswith("join:"):
            value = _joined_domain_value(
                tick,
                candidate,
                source,
                domain_by_tick_family,
            )
        else:
            raise WorldTruthError(
                f"{template['id']}: unsupported field source {source!r}"
            )
        _set_value_at_path(context, str(target_path), value)
    context["_observation_sources"] = observation_sources
    return context


def _joined_domain_value(
    tick: int,
    candidate: Mapping[str, Any],
    source: str,
    domain_by_tick_family: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> Any:
    parts = source.split(":", 3)
    if len(parts) != 4:
        raise WorldTruthError(f"invalid grounded join source: {source}")
    _join, family, binding_role, value_path = parts
    binding_id = candidate["bindings"].get(binding_role)
    if binding_id is None:
        raise WorldTruthError(f"grounded join references absent role: {binding_role}")
    matches = [
        row
        for row in domain_by_tick_family.get((tick, family), ())
        if str(row.get("subject_id")) == binding_id
    ]
    if len(matches) > 1:
        raise WorldTruthError(
            f"grounded join {family}:{binding_id}@{tick} is not unique"
        )
    return _value_at_path(matches[0], value_path) if matches else MISSING


def _set_value_at_path(target: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise WorldTruthError(f"grounded context path collides at {path}")
        current = child
    current[parts[-1]] = value


def _evaluate_grounded_state(
    *,
    episode_id: str,
    tick: int,
    template: Mapping[str, Any],
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    previous_context: Mapping[str, Any],
    defaults: Mapping[str, Any],
    rule_digest: str,
    parameter_digest: str,
    input_digest: str,
    scope_by_entity: Mapping[str, _ScopeEntity],
    source_unavailability: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    predicate_id = str(template["id"])
    tuple_id = candidate.get("tuple_id")
    if not isinstance(tuple_id, str) or not tuple_id:
        raise WorldTruthError(
            f"{predicate_id}@{tick}: candidate lacks non-empty string tuple_id"
        )
    observations: list[dict[str, Any]] = []
    missing: list[str] = []
    source_refs = set(candidate["source_refs"])
    anchor = next(
        (
            scope_by_entity[entity_id]
            for entity_id in candidate["bindings"].values()
            if entity_id in scope_by_entity
        ),
        None,
    )
    unavailable = None
    if anchor is not None:
        availability_key = (anchor.scope_type, anchor.entity_id, predicate_id)
        unavailable = source_unavailability.get(availability_key)
    if isinstance(unavailable, Mapping):
        value = "out_of_scope"
        observations.append(
            {"path": "l0.predicate_source_unavailability", "value": dict(unavailable)}
        )
        source_refs.add(str(unavailable["source_ref"]))
    elif (
        anchor is not None
        and anchor.scope_type in {"uav", "vehicle"}
        and candidate.get("scope") is not None
        and _value_at_path(candidate["record"], "l0.scope_active") is False
    ):
        value = "out_of_scope"
        origin = candidate["record"].get("_observation_sources", {}).get(
            "l0.scope_active"
        )
        observation = {"path": "l0.scope_active", "value": False}
        if origin is not None:
            observation["source_ref"] = origin
            source_refs.add(origin)
        observations.append(observation)
    else:
        for field in template["state_contract"]["required_fields"]:
            path = str(field["field"])
            observed = _value_at_path(context, path)
            origin = context.get("_observation_sources", {}).get(path)
            if _missing_value(observed):
                missing.append(
                    f"missing_source_record:{candidate['authority_source']}"
                    f"#tick={tick}&tuple={tuple_id}&field={path}"
                )
            else:
                observation = {"path": path, "value": observed}
                if origin is not None:
                    observation["source_ref"] = origin
                    source_refs.add(origin)
                observations.append(observation)
        if missing:
            value = "unknown"
        else:
            evaluated = _evaluate_expression(
                template["evaluation_spec"], context, previous_context, defaults
            )
            value = _expression_truth_value(evaluated)
    if value not in TRUTH_VALUES:
        raise WorldTruthError(f"{predicate_id}: invalid truth value {value!r}")
    return {
        "assertion_id": stable_identifier(
            "world_predicate_assertion", episode_id, tuple_id
        ),
        "predicate_id": predicate_id,
        "ontology_argument_roles": [
            str(role["key"]) for role in template["argument_roles"]
        ],
        "tuple_id": tuple_id,
        "bindings": dict(candidate["bindings"]),
        "binding_ontology_classes": dict(candidate["binding_ontology_classes"]),
        "binding_provenance": copy.deepcopy(candidate["binding_provenance"]),
        "value": value,
        "missing_source_record": sorted(missing),
        "observations": observations,
        "source_refs": sorted(source_refs),
        "candidate_authority": {
            "kind": candidate["authority_kind"],
            "source": candidate["authority_source"],
        },
        "rule_digest": rule_digest,
        "parameter_digest": parameter_digest,
        "input_digest": input_digest,
        "truth_state_update_tick": tick,
        "evidence_update_tick": tick,
    }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise WorldTruthError(f"{path}: root must be an object")
    return value


__all__ = [
    "BASE_SCHEMA_NAME",
    "DELTA_SCHEMA_NAME",
    "FORMAL_TICKS",
    "SCOPE_TYPES",
    "WorldTruthError",
    "WorldTruthResult",
    "evaluate_world_truth",
    "replay_world_truth",
    "serialize_world_truth_deltas",
]
