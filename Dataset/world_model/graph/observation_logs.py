"""Replay local monitoring logs from recorded sample states, never from plans.

The observer has synchronous access to the instrumented local recorders. This
is a simulation observation condition, not a claim of successful C2 delivery.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from Dataset.world_model.graph.contract import REPO, CONTRACT_VERSION, iter_jsonl, read_json
from Dataset.world_model.graph.records import DEFAULT_ROOT, load_scope, partition_for
from Dataset.semantic_truth.input_adapter import resolve_scenario_scene_setup, scene_setup_geometry

LOG_VERSION = "receiver-log-3"
OUTPUT = REPO / "Dataset/world_model/runs/belief_observation_logs"
CHANNELS = ("weather", "fault", "request", "communication", "context")
KINDS = ("weather_report", "self_diagnostic", "facility_request_register",
         "priority_controller_register", "local_link_status", "gateway_diagnostic",
         "compute_request_register", "posture_report", "public_registry", "traffic_manager_status",
         "facility_manager_register")
RECEIVERS = ("public_weather", "local_self_monitor", "local_facility_manager",
             "local_priority_controller", "local_link_monitor", "local_gateway_monitor",
             "local_compute_manager", "local_roadside_monitor", "public_registry", "local_traffic_manager",
             "local_service_manager")
REFERENCE_ROLES = ("subject", "station", "requester", "session", "flow",
                   "retransmission_monitor", "handover_monitor", "compute_node",
                   "task", "queue", "resource", "offload_destination", "member")
PERSISTENT_KINDS = ("public_registry",)
POLICY = (
    "Synchronous access to sampled local monitor records; sample_tick equals "
    "availability_tick. Numeric communication estimates are simulated-derived. "
    "Local registers do not establish remote receipt, ACK, or command completion. "
    "No plans, dispatches, future states, event/predicate labels, covert attack "
    "truth, or frozen-parent ARM feedback are observations. Initial public map "
    "geometry is persistent context, without any current operational-state claim."
)

WEATHER_FIELDS = {
    "rain": "ratio", "wetness": "ratio", "fog_density": "ratio",
    "visibility_m": "m", "wind_speed": "m/s", "wind_direction_deg": "deg",
    "temperature_c": "degC", "illumination_lux": "lux", "dust": "ratio",
    "hazard_concentration_ppm": "ppm",
}
FAULT_FIELDS = {
    "vehicle_state.sensor_fault": "bool", "sensor_state.sensor_fault": "bool",
    "incident_state.physical_fault_active": "bool", "incident_state.motor_fault": "bool",
    "incident_state.propulsion_fault": "bool", "incident_state.emergency_fault": "bool",
    "incident_state.fault_active": "bool", "incident_state.uav_fault": "bool",
    "facility_state.fault": "bool", "facility_state.availability": "availability_code",
    "navigation_state.multipath_warning": "bool", "navigation_state.degraded": "bool",
    "navigation_state.route_degraded": "bool",
    "navigation_state.route_uncertain": "bool",
    "navigation_state.visual_navigation_degraded": "bool",
    "navigation_state.visual_navigation_failed": "bool",
    "sensor_state.image_underexposed": "bool", "sensor_state.underexposed": "bool",
}
REQUEST_FIELDS = {
    "facility_state.capacity": "count", "facility_state.request_count": "count",
    "facility_state.allocation_failed": "bool", "facility_state.allocation_stale": "bool",
    "facility_state.arbitration_active": "bool", "facility_state.priority_granted": "bool",
    "facility_state.reserved": "bool", "facility_state.backup_charger_accepted": "bool",
}
PRIORITY_FIELDS = {"vehicle_state.priority_request_acknowledged": "bool"}
LOCAL_LINK_FIELDS = {
    "communication_state.availability": "ratio",
    "communication_state.link_availability": "ratio",
    "communication_state.communication_unavailable": "bool",
    "communication_state.station_unavailable": "bool",
    "communication_state.link_available": "bool",
    "communication_state.station_available": "bool",
    "communication_state.backup_link_active": "bool",
    "communication_state.handover_active": "bool",
}
DIAGNOSTIC_FIELDS = {
    "link_quality.latency_ms": "ms", "link_quality.packet_loss_ratio": "ratio",
    "link_quality.availability": "ratio", "heartbeat_age_ms": "ms",
    "channel.capacity_mbps": "Mbit/s", "channel.queued_bandwidth_mbps": "Mbit/s",
    "bandwidth.requested_mbps": "Mbit/s", "bandwidth.allocated_mbps": "Mbit/s",
    "bandwidth.dropped_mbps": "Mbit/s", "route.distance_m": "m",
    "retransmission.count": "count", "handover.active": "bool",
}
COMPUTE_FIELDS = {
    **{f"compute.{group}.{field}": unit
       for group in ("capacity", "resource_request", "allocation")
       for field, unit in (("cpu_cores", "cores"), ("gpu_units", "gpu_units"), ("memory_mb", "MB"))},
    "compute.capacity.max_queue_depth": "count", "compute.queue_depth": "count",
    "compute.cpu_utilization_pct": "percent", "compute.gpu_utilization_pct": "percent",
    "compute.memory_used_mb": "MB", "compute.node_availability.available": "bool",
    "compute.deadline.queue_started_tick": "tick", "compute.deadline.deadline_tick": "tick",
    "compute.offload.requested_cpu_cores": "cores", "compute.offload.accepted_cpu_cores": "cores",
}
POSTURE_FIELDS = {"pedestrian_state.posture": "posture_code", "posture": "posture_code"}
TRAFFIC_FIELDS = {"road.road_closed": "bool", "signal.controller_health": "signal_health_code"}
SERVICE_FIELDS = {"service.availability": "availability_code", "service.capacity": "count",
                  "service.queue_depth": "count", "service.reserved": "bool"}
REGISTRY_FIELDS = {
    "map.kind": "map_kind_code", "map.vertex_index": "index", "map.vertex_count": "count",
    "map.e_m": "m", "map.n_m": "m", "map.u_m": "m", "map.base_z_m": "m",
    "map.height_m": "m", "map.normal_e": "ratio", "map.normal_n": "ratio",
    "map.normal_u": "ratio", "map.service_capacity": "count",
}
FIELD_GROUPS = (WEATHER_FIELDS, FAULT_FIELDS, REQUEST_FIELDS, PRIORITY_FIELDS,
                LOCAL_LINK_FIELDS, DIAGNOSTIC_FIELDS, COMPUTE_FIELDS, POSTURE_FIELDS, REGISTRY_FIELDS,
                TRAFFIC_FIELDS, SERVICE_FIELDS)
FEATURE_UNITS = {key: unit for group in FIELD_GROUPS for key, unit in group.items()}
FEATURES = tuple(FEATURE_UNITS)
ENUM_VALUES = {"availability_code": ["available", "unavailable", "reserved", "fault", "failed"],
               "posture_code": ["standing", "fallen", "lying", "prone"],
               "signal_health_code": ["nominal", "all_red_fault"],
               "map_kind_code": ["polygon_prism", "box_volume", "facade_anchor", "facility_point", "gnss_service", "cohort"]}
MISSING_REASONS = ("not_recorded", "source_unknown", "diagnostic_source_absent", "source_required")
SCHEMA = pa.schema([
    ("observation_id", pa.string()), ("channel", pa.string()), ("kind", pa.string()),
    ("receiver_kind", pa.string()), ("sample_tick", pa.int32()), ("availability_tick", pa.int32()),
    ("references_json", pa.large_string()), ("values_json", pa.large_string()),
    ("missing_json", pa.large_string()), ("source_refs_json", pa.large_string()),
])


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _field(raw, path, unit):
    value = raw
    for key in path.split("."):
        if not isinstance(value, dict):
            raise ValueError(f"monitor field parent is not an object: {path}")
        if key not in value:
            return None, "not_recorded"
        value = value[key]
    if value is None or value == "unknown":
        return None, "source_unknown"
    if unit in ENUM_VALUES:
        if value not in ENUM_VALUES[unit]:
            raise ValueError(f"monitor enum differs: {path}={value!r}")
        return float(ENUM_VALUES[unit].index(value)), None
    if unit == "bool":
        if type(value) is not bool:
            raise ValueError(f"monitor boolean differs: {path}={value!r}")
        return float(value), None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"monitor numeric field differs: {path}={value!r}")
    return float(value), None


def _record(kind, tick, references, raw, fields, source_refs, *, missing_reason=None, part=None):
    if any(not isinstance(entity, str) or not entity for entities in references.values() for entity in entities):
        raise ValueError(f"monitor reference has invalid identity: {kind}:{tick}")
    index = KINDS.index(kind)
    channel = ("weather", "fault", "request", "request", "communication", "communication",
               "request", "fault", "context", "fault", "request")[index]
    values, missing = {}, {}
    for key, unit in fields.items():
        value, reason = _field(raw, key, unit)
        if missing_reason is not None:
            value, reason = None, missing_reason
        values[key] = value
        if reason is not None:
            missing[key] = reason
    return {"observation_id": _json([kind, tick, references, part]), "channel": channel,
            "kind": kind, "receiver_kind": RECEIVERS[index], "sample_tick": tick,
            "availability_tick": tick, "references": references,
            "values": values, "missing": missing, "source_refs": source_refs}


def _sources(partition, scope):
    partition = partition.resolve()
    result = {"nodes": partition / "nodes.parquet", "scope": partition / "scope.parquet",
              "producer": Path(__file__),
              "geometry_producer": REPO / "Dataset/semantic_truth/input_adapter.py",
              "domain_profile": REPO / "Dataset/semantic_rules/profiles/domain_state_supplement_profile.json"}
    if scope["source_kind"] == "formal":
        capture = REPO / "aw_data/render_ready_episodes_capture_filtered" / scope["episode_id"]
        result.update(weather=capture / "weather_meta.jsonl", manifest=capture / "episode_manifest.json",
                      communication=REPO / scope["source_root"] / "communication_state.jsonl",
                      compute=REPO / scope["source_root"] / "compute_state.jsonl",
                      traffic=REPO / scope["source_root"] / "domain_state_observations.jsonl")
        result["scene"] = resolve_scenario_scene_setup(read_json(result["manifest"]))
    else:
        branch = REPO / scope["source_root"]
        result.update(weather=branch / "raw/weather.jsonl", manifest=branch / "manifest.json",
                      scene=branch / "ue/scene_setup.json",
                      initial_cohort=REPO / "aw_data/objective_semantic_truth" / scope["episode_id"] / "domain_state_observations.jsonl")
        if scope["profile"].startswith("L5-"):
            result["weather_binding"] = branch / "raw/binding_catalog.json"
        if scope["profile"].startswith("L6-"):
            result["communication"] = branch / "raw/communication_state.jsonl"
    for path in result.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def artifact_inputs(partition):
    scope = load_scope(partition)
    return {"log_version": LOG_VERSION, "graph_version": CONTRACT_VERSION,
            "sampling_step_ticks": 5, "feature_units": FEATURE_UNITS,
            "enum_values": ENUM_VALUES,
            "sources": [{"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size,
                         "modified_ns": path.stat().st_mtime_ns}
                        for path in _sources(partition, scope).values()]}


def _entity_rows(partition, scope):
    expression = ((ds.field("node_type") == "EntityState")
                  & (ds.field("tick") >= scope["start_tick"])
                  & (ds.field("tick") <= scope["end_tick"]))
    rows = {}
    for batch in ds.dataset(partition / "nodes.parquet").to_batches(filter=expression):
        for node in batch.to_pylist():
            tick = node["tick"]
            if tick % 5:
                continue
            attrs = json.loads(node["attrs_json"])
            if attrs.get("observed") is False or attrs.get("state_layer") in ("frozen_parent_reference", "history"):
                continue
            if attrs["identity_status"] != "resolved":
                raise ValueError(f"monitor subject identity is unresolved: {node['node_id']}")
            raw = attrs["raw"]
            # L5/L6 sampled wrappers project runtime state into a different
            # shape or omit families. Their own trajectories are the full
            # producer authority; never merge the two projections.
            if scope["source_kind"] == "arm" and scope["profile"].startswith(("L5-", "L6-")) and attrs.get("state_layer") == "state":
                continue
            key = tick, attrs["entity_id"]
            selected = {family: raw[family] for family in
                        ("vehicle_state", "sensor_state", "incident_state", "facility_state",
                         "navigation_state", "communication_state", "pedestrian_state", "posture") if family in raw}
            entry = {"tick": tick, "entity": attrs["entity_id"], "category": attrs["entity_category"],
                     "raw": selected, "source_refs": [{"path": node["source_file"],
                         "line": node["source_line"], "graph_node": node["node_id"]}]}
            if key in rows:
                previous = rows[key]
                if previous["raw"] != selected or previous["category"] != entry["category"]:
                    raise ValueError(f"conflicting monitor state copies: {key}")
                previous["source_refs"].extend(entry["source_refs"])
            else:
                rows[key] = entry
    return [rows[key] for key in sorted(rows)]


def _public_registry(sources, scope):
    """Publish initial public geometry, without future visibility or actions.

    Geometry describes a known map object; it does not assert that an excluded
    parent trajectory or that object's operational state was re-simulated.
    """
    scene = read_json(sources["scene"])
    source = str(sources["scene"].relative_to(REPO))
    geometry = scene_setup_geometry(scene, source)
    tick = scope["start_tick"] + (-scope["start_tick"]) % 5
    if tick > scope["end_tick"]:
        return []
    rows = []
    for index, entity in enumerate(scene["entities"]):
        category = entity["category"]
        if category not in ("facility", "airspace_corridor", "airspace_constraint", "facade_anchor"):
            continue
        identity = entity["entity_id"]
        if entity["activation_tick"] > scope["start_tick"]:
            continue
        refs = {"subject": [identity]}
        source_refs = [{"path": source, "json_path": f"entities[{index}].placement"}]
        if identity in geometry:
            shape = geometry[identity]["geometry"]
            mode = entity["placement_mode"]
            if mode in ("polygon_prism", "box_volume"):
                points = shape["polygon_enu_m"]
                for vertex, point in enumerate(points):
                    values = {"kind": mode, "vertex_index": vertex, "vertex_count": len(points),
                              "e_m": point[0], "n_m": point[1],
                              "base_z_m": shape["base_z_m"], "height_m": shape["height_m"]}
                    rows.append(_record("public_registry", tick, refs, {"map": values}, REGISTRY_FIELDS,
                                        source_refs, part=vertex))
            else:
                point, normal = shape["point_enu_m"], shape["outward_normal_enu"]
                values = {"kind": "facade_anchor", "vertex_index": 0, "vertex_count": 1,
                          "e_m": point[0], "n_m": point[1], "u_m": point[2],
                          "normal_e": normal[0], "normal_n": normal[1], "normal_u": normal[2]}
                rows.append(_record("public_registry", tick, refs, {"map": values}, REGISTRY_FIELDS, source_refs))
        elif category == "facility":
            semantic = entity["semantic_scope"]
            if semantic["scope_subtype"] not in ("charging_station", "communication_base_station",
                    "ground_control_station", "landing_pad", "no_fly_zone", "radio_tower"):
                continue
            values = {"kind": "facility_point", "vertex_index": 0, "vertex_count": 1}
            placement = entity["placement"]
            if "resolved_position_enu_m" in placement:
                point = placement["resolved_position_enu_m"]
                if not isinstance(point, list) or len(point) != 3:
                    raise ValueError(f"public registry position differs: {identity}")
                values.update(e_m=point[0], n_m=point[1], u_m=point[2])
            if "service_capacity" in semantic:
                values["service_capacity"] = semantic["service_capacity"]
            rows.append(_record("public_registry", tick, refs, {"map": values}, REGISTRY_FIELDS, source_refs))
    profile = read_json(sources["domain_profile"])
    service = profile["domain_models"]["gnss"]["service_instance"]
    rows.append(_record("public_registry", tick, {"subject": [service["service_id"]]},
                        {"map": {"kind": "gnss_service"}}, REGISTRY_FIELDS,
                        [{"path": str(sources["domain_profile"].relative_to(REPO)),
                          "json_path": "domain_models.gnss.service_instance"}]))
    return rows


def _initial_cohort_registry(sources, entities):
    """Reuse only the pre-fork initial roster, once its members are observed locally."""
    path = sources["initial_cohort"]
    for line, raw in iter_jsonl(path):
        if raw["tick"] != 0:
            raise ValueError(f"initial cohort record missing at tick zero: {path}")
        if raw["observation_family"] != "crowd_evacuation":
            continue
        values = raw["values"]
        members, identity = values["cohort_member_ids"], values["crowd_id"]
        if not isinstance(members, list) or values["cohort_member_count"] != len(members):
            raise ValueError(f"initial cohort membership differs: {path}")
        if identity is None and not members:
            return []
        if not isinstance(identity, str) or not identity or not members or len(set(members)) != len(members):
            raise ValueError(f"initial cohort identity differs: {path}")
        observed = {member: set() for member in members}
        for entity in entities:
            if entity["entity"] in observed:
                observed[entity["entity"]].add(entity["tick"])
        common_ticks = set.intersection(*observed.values())
        if not common_ticks:
            return []  # Declared members have no jointly observed branch state.
        return [_record("public_registry", min(common_ticks), {"subject": [identity], "member": members},
                        {"map": {"kind": "cohort"}}, REGISTRY_FIELDS,
                        [{"path": str(path.relative_to(REPO)), "line": line,
                          "fields": ["values.crowd_id", "values.cohort_member_ids", "values.cohort_member_count"]}])]
    raise ValueError(f"initial cohort source is empty: {path}")


def generate(partition):
    scope = load_scope(partition)
    if scope["contract_version"] != CONTRACT_VERSION:
        raise ValueError("monitoring requires the current source graph")
    sources = _sources(partition, scope)
    manifest = read_json(sources["manifest"])
    if scope["source_kind"] == "formal":
        weather_subject = manifest["capture_boundary_id"]
    elif scope["profile"].startswith("X"):
        weather_subject = scope["profile"] + ".authored_weather_domain"
    elif scope["profile"].startswith("L5-"):
        weather_subject = read_json(sources["weather_binding"])["operational_region_id"]
    else:
        # A monitor's local weather domain need not be an ontology target.
        weather_subject = "local_weather_domain"
    rows = _public_registry(sources, scope)
    for line, raw in iter_jsonl(sources["weather"]):
        tick = raw["tick"]
        if tick % 5 or not scope["start_tick"] <= tick <= scope["end_tick"]:
            continue
        rows.append(_record("weather_report", tick, {"subject": [weather_subject]}, raw,
                            WEATHER_FIELDS, [{"path": str(sources["weather"].relative_to(REPO)), "line": line}]))
    entities = _entity_rows(partition, scope)
    if scope["source_kind"] == "arm":
        rows.extend(_initial_cohort_registry(sources, entities))
    for entity in entities:
        tick, raw, category = entity["tick"], entity["raw"], entity["category"]
        references = {"subject": [entity["entity"]]}
        if category in ("uav", "vehicle", "facility", "traffic_light"):
            rows.append(_record("self_diagnostic", tick, references, raw, FAULT_FIELDS, entity["source_refs"]))
        if category == "pedestrian":
            rows.append(_record("posture_report", tick, references, raw, POSTURE_FIELDS, entity["source_refs"]))
        if category in ("facility", "traffic_light"):
            facility = raw.get("facility_state", {})
            if "requester_ids" in facility:
                requesters = facility["requester_ids"]
                if not isinstance(requesters, list) or any(not isinstance(e, str) or not e for e in requesters):
                    raise ValueError("facility request register has invalid requester identities")
                if len(requesters) != len(set(requesters)):
                    raise ValueError("facility request register duplicates a requester")
                if "request_count" in facility and facility["request_count"] != len(requesters):
                    raise ValueError("facility request count differs from its register")
                references = {**references, "requester": sorted(requesters)}
            rows.append(_record("facility_request_register", tick, references, raw, REQUEST_FIELDS, entity["source_refs"]))
        if category == "vehicle":
            rows.append(_record("priority_controller_register", tick, references, raw, PRIORITY_FIELDS, entity["source_refs"]))
        if category in ("uav", "facility"):
            rows.append(_record("local_link_status", tick, references, raw, LOCAL_LINK_FIELDS, entity["source_refs"]))
        if category == "uav" and "communication" not in sources:
            rows.append(_record("gateway_diagnostic", tick, references, {}, DIAGNOSTIC_FIELDS,
                                entity["source_refs"], missing_reason="diagnostic_source_absent"))
    if "communication" in sources:
        for line, raw in iter_jsonl(sources["communication"]):
            tick = raw["tick"]
            if tick % 5 or not scope["start_tick"] <= tick <= scope["end_tick"]:
                continue
            refs = {"subject": [raw["entity_id"]]}
            for field, role in (("station_id", "station"), ("session_id", "session"), ("flow_id", "flow"),
                                ("retransmission_id", "retransmission_monitor"), ("handover_id", "handover_monitor")):
                identity = raw[field]
                if identity not in (None, "unknown", "none"):
                    refs[role] = [identity]
            requirement = raw["requirement"]["status"]
            if requirement not in ("source_required", "not_required"):
                raise ValueError(f"unknown communication source requirement: {requirement}")
            rows.append(_record("gateway_diagnostic", tick, refs, raw, DIAGNOSTIC_FIELDS,
                                [{"path": str(sources["communication"].relative_to(REPO)), "line": line}],
                                missing_reason="source_required" if requirement == "source_required" else None))
    if "compute" in sources:
        for line, raw in iter_jsonl(sources["compute"]):
            tick = raw["tick"]
            if tick % 5 or not scope["start_tick"] <= tick <= scope["end_tick"]:
                continue
            refs = {"subject": [raw["entity_id"]], "compute_node": [raw["node_id"]],
                    "task": [raw["task_id"]], "queue": [raw["queue_id"]], "resource": [raw["resource_id"]]}
            if raw["offload"]["target_node_id"] not in (None, "none", "unknown"):
                refs["offload_destination"] = [raw["offload"]["target_node_id"]]
            requirement = raw["failure"]["requirement_status"]
            if requirement not in ("source_required", "not_required"):
                raise ValueError(f"unknown compute source requirement: {requirement}")
            rows.append(_record("compute_request_register", tick, refs, {"compute": raw}, COMPUTE_FIELDS,
                                [{"path": str(sources["compute"].relative_to(REPO)), "line": line}],
                                missing_reason="source_required" if requirement == "source_required" else None))
    if "traffic" in sources:
        for line, raw in iter_jsonl(sources["traffic"]):
            tick, family = raw["tick"], raw["observation_family"]
            if tick % 5 or not scope["start_tick"] <= tick <= scope["end_tick"]:
                continue
            if family == "pad_facility" and raw["values"]["facility_subtype"] == "charging_station":
                requesters = raw["values"]["requester_ids"]
                if not isinstance(requesters, list) or len(set(requesters)) != len(requesters):
                    raise ValueError("service manager requester register differs")
                rows.append(_record("facility_manager_register", tick,
                                    {"subject": [raw["subject_id"]], "requester": requesters},
                                    {"service": raw["values"]}, SERVICE_FIELDS,
                                    [{"path": str(sources["traffic"].relative_to(REPO)), "line": line}]))
                continue
            if family not in ("road_segment_state", "traffic_signal_state"):
                continue
            group = "road" if family == "road_segment_state" else "signal"
            rows.append(_record("traffic_manager_status", tick, {"subject": [raw["subject_id"]]},
                                {group: raw["values"]}, TRAFFIC_FIELDS,
                                [{"path": str(sources["traffic"].relative_to(REPO)), "line": line}]))
    rows.sort(key=lambda row: (row["availability_tick"], row["kind"], row["observation_id"]))
    if len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError("monitoring replay produced duplicate records")
    return rows


def _stored(row):
    return {**{key: row[key] for key in ("observation_id", "channel", "kind", "receiver_kind", "sample_tick", "availability_tick")},
            **{key + "_json": _json(row[key]) for key in ("references", "values", "missing", "source_refs")}}


def build(partition, output=OUTPUT):
    partition = partition.resolve()
    scope = load_scope(partition)
    destination = partition_for(scope, output)
    inputs = artifact_inputs(partition)
    marker = destination / "metadata.json"
    if marker.is_file():
        metadata = read_json(marker)
        if metadata["artifact_inputs"] == inputs and (destination / "observations.parquet").is_file():
            return metadata
    rows = generate(partition)
    counts, fields, missing = Counter(), Counter(), Counter()
    for row in rows:
        counts[row["channel"] + ".records"] += 1
        counts[row["channel"] + ".records_with_values"] += int(any(v is not None for v in row["values"].values()))
        fields.update(key for key, value in row["values"].items() if value is not None)
        missing.update(row["missing"].values())
    destination.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([_stored(row) for row in rows], schema=SCHEMA),
                   destination / "observations.parquet", compression="zstd")
    stored = pq.read_table(destination / "observations.parquet").to_pylist()
    if stored != [_stored(row) for row in rows]:
        raise ValueError("published monitoring records differ after readback")
    metadata = {"sample_key": scope["sample_key"], "log_version": LOG_VERSION,
                "graph_partition": str(partition.relative_to(REPO)), "policy": POLICY,
                "start_tick": scope["start_tick"], "end_tick": scope["end_tick"],
                "record_count": len(rows), "counts": dict(counts), "field_values": dict(fields),
                "missing_values": dict(missing), "artifact_inputs": inputs,
                "channels": {channel: {"records": counts[channel + ".records"],
                    "records_with_values": counts[channel + ".records_with_values"],
                    "status": "available" if counts[channel + ".records_with_values"] else "source_not_recorded"}
                             for channel in CHANNELS}}
    marker.write_text(_json(metadata) + "\n")
    return metadata


def read_nodes(partition, *, start=None, end=None, output=OUTPUT):
    destination = partition_for(load_scope(partition), output)
    metadata = read_json(destination / "metadata.json")
    if metadata["artifact_inputs"] != artifact_inputs(partition):
        raise ValueError(f"monitor logs require rebuilding: {partition}")
    expression = None
    if start is not None:
        expression = ((ds.field("availability_tick") >= start) | ds.field("kind").isin(PERSISTENT_KINDS))
        expression &= ds.field("availability_tick") <= end
    rows = ds.dataset(destination / "observations.parquet").to_table(filter=expression).to_pylist()
    nodes = []
    for row in rows:
        payload = {key: row[key] for key in ("observation_id", "channel", "kind", "receiver_kind", "sample_tick", "availability_tick")}
        payload.update({key: json.loads(row[key + "_json"]) for key in ("references", "values", "missing", "source_refs")})
        nodes.append({"node_type": "ReceiverObservation", "tick": row["availability_tick"], "attrs": payload})
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample", action="append", default=[])
    args = parser.parse_args()
    reports, counts = [], Counter()
    for path in sorted(args.graph_root.rglob("scope.parquet")):
        scope = load_scope(path.parent)
        if args.sample and scope["sample_key"] not in args.sample:
            continue
        report = build(path.parent, args.output)
        reports.append(report)
        counts.update(report["counts"])
        print(_json({"monitoring_completed": len(reports), "sample": scope["sample_key"], "records": report["record_count"]}), flush=True)
    if args.sample and set(args.sample) != {row["sample_key"] for row in reports}:
        raise ValueError("requested monitoring sample was not found")
    if not args.sample:
        summary = {"log_version": LOG_VERSION, "policy": POLICY, "feature_units": FEATURE_UNITS,
                   "enum_values": ENUM_VALUES,
                   "channels": list(CHANNELS), "kinds": list(KINDS), "receivers": list(RECEIVERS),
                   "partitions": reports, "counts": dict(counts)}
        (args.output / "scope.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
