"""Closed published rows of the current control, geometry and observable writers.

The objective pipeline removes integrity metadata before publication. These
rules describe that published surface. The original supplement writer remains
in domain_state_l1_structure; its rows have a different envelope.
"""

from __future__ import annotations

from copy import deepcopy

from Dataset.semantic_simulation import observable_state_completion as observable
from Dataset.semantic_simulation import control_response_state as control
from Dataset.world_model.graph.history_structure import _control_values


def _object(properties: dict, *, optional: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties,
            "required": [k for k in properties if k not in optional],
            "additionalProperties": False}


def _array(item: dict) -> dict:
    return {"type": "array", "items": item}


def _vector(size: int) -> dict:
    return {"type": "array", "prefixItems": [{"type": "number"} for _ in range(size)],
            "minItems": size, "maxItems": size, "items": False}


def _nullable(schema: dict) -> dict:
    return {"oneOf": [schema, {"type": "null"}]}


def _unknown(schema: dict) -> dict:
    return {"oneOf": [schema, {"const": "unknown"}]}


_TYPES = {"s": {"type": "string"}, "b": {"type": "boolean"},
          "n": {"type": "number"}, "i": {"type": "integer"},
          "l": _array({"type": "string"}), "v": _vector(3)}
_TYPES.update({k.upper(): _nullable(v) for k, v in tuple(_TYPES.items())})
_TYPES.update({"ub": _unknown(_TYPES["b"]), "un": _unknown(_TYPES["n"]),
              "ui": _unknown(_TYPES["i"]), "ul": _unknown(_TYPES["l"]),
              "uv": _unknown(_TYPES["v"])})


def _values(*, extra: dict | None = None, optional: tuple[str, ...] = (), **groups) -> dict:
    fields = {}
    for kind, names in groups.items():
        for name in names.split():
            if name in fields:
                raise ValueError(f"duplicate current domain value: {name}")
            fields[name] = deepcopy(_TYPES[kind])
    for name, schema in (extra or {}).items():
        if name in fields:
            raise ValueError(f"duplicate current domain value: {name}")
        fields[name] = deepcopy(schema)
    return _object(fields, optional=optional)


def _envelope(observation_family: str, values: dict, schema_version: str, rule_version: str,
              rule_id: str, **extra) -> dict:
    return _object({"schema_name": {"const": "domain_state_observation"},
        "schema_version": {"const": schema_version}, "observation_id": _TYPES["s"],
        "episode_id": _TYPES["s"], "tick": {"type": "integer", "minimum": 0},
        "observation_family": {"const": observation_family}, "subject_id": _TYPES["s"],
        "source_class": {"const": "derived_from_observed"},
        "rule_id": {"const": rule_id}, "rule_version": {"const": rule_version},
        "source_refs": _TYPES["l"], "values": values, **extra})


def _geometry_rows() -> list[dict]:
    specs = {
        "aircraft_geometry": ("aircraft_state", "uav", _values(
            v="position_enu_m", V="velocity_enu_mps home_pad_pose_enu_m assigned_landing_zone_pose_enu_m",
            N="speed_mps z_agl_m local_ground_reference_z_m takeoff_ground_z_m xy_distance_to_home_pad_m xy_distance_to_assigned_landing_zone_m assigned_altitude_m altitude_deviation_m",
            S="home_pad_id home_pad_ontology_class_id assigned_landing_zone_id assigned_landing_zone_ontology_class_id",
            b="aircraft_inside_corridor", l="containing_corridor_ids", s="entity_role")),
        "aircraft_region_geometry": ("restricted_region_distance", "uav_region", _values(
            s="aircraft_id aircraft_ontology_class_id restricted_region_id restricted_region_ontology_class_id",
            b="restricted_region_active inside_restricted_region",
            n="distance_to_boundary_m horizontal_distance_to_boundary_m signed_distance_to_boundary_m",
            extra={"boundary_polygon_enu_m": _array(_vector(2))})),
        "aircraft_protected_region_geometry": ("protected_region_containment", "uav_protected_region", _values(
            s="aircraft_id aircraft_ontology_class_id protected_region_id protected_region_ontology_class_id",
            b="inside_protected_airspace", n="distance_to_protected_airspace_m")),
        "aircraft_corridor_geometry": ("aircraft_corridor_membership", "uav_corridor", _values(
            s="aircraft_id aircraft_ontology_class_id corridor_id corridor_ontology_class_id",
            b="aircraft_inside_corridor", N="speed_mps")),
        "corridor_geometry": ("corridor_occupancy_and_capacity", "airspace_corridor", _values(
            s="corridor_id corridor_ontology_class_id", i="corridor_occupancy_count corridor_capacity",
            l="occupying_aircraft_ids", v="corridor_center_enu_m corridor_extent_m",
            n="corridor_yaw_deg minimum_center_separation_m",
            extra={"capacity_model": {"const": "independent_cross_section_lanes"},
                   "corridor_cross_section_size_m": _vector(2)})),
    }
    pair_common = dict(s="first_aircraft_id first_aircraft_ontology_class_id second_aircraft_id second_aircraft_ontology_class_id",
                       n="pair_distance_m pair_euclidean_distance_m pair_horizontal_distance_m pair_vertical_distance_m")
    pair_ids = {"type": "array", "items": _TYPES["s"], "minItems": 2, "maxItems": 2}
    pair_variants = []
    for metric in ("euclidean_3d", "anisotropic_safety_volume_equivalent_distance"):
        extras = {"pair_entity_ids": pair_ids, "pair_distance_metric": {"const": metric}}
        if metric != "euclidean_3d":
            extras.update({k: _TYPES["n"] for k in ("pair_horizontal_limit_m", "pair_vertical_limit_m")})
        pair_variants.append(_values(**pair_common, extra=extras))
    specs["aircraft_pair_geometry"] = ("aircraft_pair_distance", "uav_pair", {"oneOf": pair_variants})
    nearest = "nearest_aircraft nearest_ground_vehicle nearest_vehicle nearest_pedestrian nearest_population nearest_building".split()
    specs["agent_proximity_geometry"] = ("agent_proximity", ("uav", "vehicle", "pedestrian"), _values(
        N=" ".join(k + "_distance_m" for k in nearest), S=" ".join(k + "_id" for k in nearest)))
    for name, kind, first, second in (
            ("aircraft_vehicle_geometry", "uav_vehicle", "aircraft", "vehicle"),
            ("aircraft_pedestrian_geometry", "uav_pedestrian", "aircraft", "pedestrian"),
            ("aircraft_structure_geometry", "uav_structure", "aircraft", "structure"),
            ("vehicle_pedestrian_geometry", "vehicle_pedestrian", "vehicle", "pedestrian")):
        specs[name] = ("full_grounded_pair_distance", kind, _values(n="distance_m",
            s=f"{first}_id {first}_ontology_class_id {second}_id {second}_ontology_class_id"))
    specs.update({
        "region_runtime_state": ("runtime_region_state_computer.region_activity", "restricted_region", _values(
            s="no_fly_zone_id no_fly_zone_ontology_class_id", b="restricted_region_active", n="boundary_margin_m")),
        "restricted_airspace_state": ("plan_window_computer.restricted_airspace_effective_range", "uav", _values(
            b="restricted_region_active", N="minimum_restricted_boundary_distance_m", n="boundary_margin_m", l="active_restricted_region_ids")),
        "aircraft_plan": ("plan_window_computer.aircraft_control_and_compliance", "uav", _values(
            b="rth_mode_active cooperative_flag", s="entity_role", S="restricted_region_id")),
    })
    result = []
    for suffix, (rule, kinds, values) in specs.items():
        if "." not in rule:
            rule = "geometric_computer." + rule
        category = {"enum": list(kinds)} if isinstance(kinds, tuple) else {"const": kinds}
        result.append(_envelope("predicate_contract_" + suffix, values, "3.0.0", "3.0.0", rule,
            annotation_layer={"const": "L0"}, subject_kind=category, subject_category=category,
            model_id={"const": rule.split(".", 1)[0]}, model_version={"const": "3.0.0"},
            missing_source_record=_TYPES["l"], quality={"enum": ["complete", "unknown_missing_source"]}))
    return result


def _observable_row(family: str, values: dict, roles: str,
                    source_class: str = "derived_from_observed") -> dict:
    return _envelope(family, values, observable.SCHEMA_VERSION, observable.RULE_VERSION,
        "observable_state_completion." + family,
        family={"const": family}, bindings=_object({r: _TYPES["s"] for r in roles.split()}),
        source_class={"const": source_class}, missing_inputs=_TYPES["l"])


def _observable_rows() -> list[dict]:
    rows = [
        _observable_row("observable_environment", _values(
            ub="gust_active heavy_rain_active visibility_low hazard_source_active",
            un="rain_rate visibility_m wind_speed_mps wind_direction_deg temperature_c illumination_lux hazard_concentration_ppm hazard_radius_m",
            i="gust_window_ticks"), "weather_region"),
        _observable_row("observable_ground_surface", _values(ub="wet_road_active", un="surface_wetness"), "surface"),
        _observable_row("observable_sensor_state", _values(
            ub="underexposed infrared exposure_recovered sensor_fault intruder_detected image_underexposed infrared_mode_active",
            s="exposure_status imaging_mode"), "uav"),
        _observable_row("observable_vehicle_response", _values(
            ub="braking yielding emergency_stop collision_contact near_miss_classified minimal_risk_maneuver_active stopped_safe warning_active safe_stop_failed sensor_fault ambulance_passage_completed vehicle_braking vehicle_emergency_stop_active",
            un="speed_mps accel_mps2"), "vehicle"),
        _observable_row("observable_pedestrian_response", _values(s="posture health risk_vehicle_id",
            ub="fallen injured retreating jaywalking evacuation_active safe_zone_reached pedestrian_retreating",
            un="risk_distance_m"), "pedestrian vehicle"),
        _observable_row("observable_pad_facility", _values(
            ub="simultaneous_pad_requests simultaneous_pad_approaches pad_priority_granted contention allocation_failed allocation_stale arbitration_active priority_granted backup_charger_accepted reserved",
            s="priority_granted_to availability", b="eta_inputs_complete control_modes_complete",
            ui="capacity request_count", i="eta_window_ticks", ul="requester_ids approaching_uav_ids holding_or_diverting_uav_ids",
            extra={"inference_class": {"const": "physical_approach_inference"},
                   "fault": {"oneOf": [{"type": t} for t in ("boolean", "number", "string")]}}),
            "pad", "mixed_observed_and_physical_inference"),
        _observable_row("hazmat_response_state", _values(
            ub="hazard_source_active hazard_active isolation_active all_targets_safe ambulance_closing_after_isolation arrival_condition responder_arrived handoff_complete hazmat_resolved",
            b="isolation_observed ambulance_present dispatch_observed",
            un="hazard_concentration_ppm hazard_radius_m ambulance_distance_m ambulance_previous_distance_m",
            uv="hazard_center_enu_m", ul="target_pedestrian_ids missing_target_pedestrian_ids", s="ambulance_id",
            i="arrival_dwell_ticks arrival_required_dwell_ticks handoff_dwell_ticks handoff_required_dwell_ticks safe_required_dwell_ticks",
            n="safe_radius_margin_m ambulance_perimeter_tolerance_m",
            extra={"target_pedestrian_distances_m": {"type": "object", "patternProperties": {"^.+$": _TYPES["n"]}, "additionalProperties": False},
                   "target_pedestrian_safe_dwell_ticks": {"type": "object", "patternProperties": {"^.+$": _TYPES["i"]}, "additionalProperties": False}}), "weather_region"),
        _observable_row("lockdown_control", _values(ub="clear requires_reroute"), "entity"),
    ]
    for family, fields in (
            ("mission", observable._MISSION_BOOL_FIELDS), ("navigation", observable._NAVIGATION_BOOL_FIELDS),
            ("communication", observable._COMMUNICATION_BOOL_FIELDS), ("incident", observable._INCIDENT_BOOL_FIELDS),
            ("control", observable._CONTROL_BOOL_FIELDS), ("security", observable._SECURITY_BOOL_FIELDS)):
        rows.append(_observable_row(f"observable_{family}_state", _values(ub=" ".join(fields)), "entity"))
    prism = _object({"geometry_kind": {"const": "polygon_prism"},
        "polygon_enu_m": {"type": "array", "items": _vector(2), "minItems": 4, "maxItems": 4},
        "base_z_m": _TYPES["n"], "height_m": _TYPES["n"],
        "construction": {"const": "governed_buffer_around_observed_isolation_control"}})
    common = _values(s="emergency_isolation_zone_id region_scope_id associated_patient_id",
        ub="temporary_lockdown_active lockdown_clear", l="controller_ids deployed_controller_ids",
        i="deployed_controller_count", un="associated_patient_distance_m lockdown_radius_m",
        n="patient_association_max_distance_m",
        extra={"emergency_isolation_zone_ontology_class_id": {"const": "world:EmergencyIsolationZone"}, "geometry": _unknown(prism)})
    rows.append(_observable_row("lockdown_region_state", common, "region", "simulated_derived"))
    static_values = deepcopy(common)
    static_values["properties"]["region_state"] = {"enum": ["active", "inactive", "unknown"]}
    static_values["required"].append("region_state")
    # Current frozen scenes supply no geometry member. Unknown stays explicit;
    # an added physical geometry must be declared before this branch accepts it.
    static_values["properties"]["geometry"] = {"const": "unknown"}
    for key in ("associated_patient_id", "associated_patient_distance_m", "lockdown_radius_m"):
        static_values["properties"][key] = {"const": "unknown"}
    rows.append(_observable_row("lockdown_region_state", static_values, "region"))
    return rows


def current_domain_variants() -> list[dict]:
    control_row = _envelope(control.CONTROL_FAMILY, _control_values(), control.SCHEMA_VERSION,
        control.RULE_VERSION, "control_response_state.structured_truth_route_communication_mapping",
        subject_kind=_TYPES["s"], model_id={"const": "control_response_state.structured_truth_route_comm_kinematic_rth_v5"},
        model_version={"const": control.RULE_VERSION})
    return [control_row, *_geometry_rows(), *_observable_rows()]


__all__ = ["current_domain_variants"]
