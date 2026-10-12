"""Closed whole-JSON grammar for L4 ARM window manifests.

The current window writer supplies the base manifest. UE delivery adds optional
capture inputs. Two frozen manifest roots retain an earlier path-object source
shape and its code/file inventory; that shape has its own closed branch.
"""

from __future__ import annotations

from copy import deepcopy

from Dataset.world_model.graph.arm_manifest_l1_structure import arm_manifest_schema as _l1_schema

__all__ = ["arm_manifest_schema"]

_BUSINESS_PROFILES = r"L4-(?:4_v[12]|5_v[123]|9_v[12]|1[01]_v[12])"
_OTHER_PROFILES = r"L4-(?:[12]_v[12]|3_v[123]|[678]_v[12])"
_EVENT_FAMILIES = (
    "crowd_evacuation", "uav_separation_conflict", "pedestrian_medical",
    "l4_uav_vehicle_contact_observed", "l4_intersection_braking_observed",
    "l4_uav_pedestrian_clearance_observed",
    "l4_ambulance_priority_passage_observed", "forced_landing",
    "l4_av_safe_stop_observed", "vehicle_pedestrian_conflict",
    "structure_near_miss",
)
_PARAMETER_SOURCE_KEYSETS = (
    ("anchor", "event_script", "event_trace", "factual_root"),
    ("anchor", "event_script", "factual_root", "note", "trace"),
    ("anchor", "factual_root", "measured_facts"),
    ("anchor", "factual_root", "source_script", "trace"),
    ("anchor", "factual_root", "trace"),
    ("business_rules", "completion_state", "event_script", "factual_root"),
    ("business_rules", "factual_root", "runtime_state_schedule", "window"),
    ("canonical_event_trace", "canonical_trajectories", "design_facts",
     "event_script", "factual_root"),
    ("canonical_event_trace", "canonical_trajectories", "event_family_authority",
     "event_script", "factual_root", "predicate_scope", "runtime_state_schedule"),
    ("crawl_move", "factual_root"),
    ("event_script", "event_trace", "factual_root", "kept_actions",
     "standoff_basis", "standoff_waypoint_enu_m"),
    ("event_script", "event_trace", "factual_root", "preempt_action"),
    ("factual_rebuild", "factual_root", "physical_model"),
    ("factual_root", "intervention_waypoints", "omitted_action_ids"),
    ("factual_root", "intervention_waypoints", "omitted_action_ids", "subject_note"),
    ("factual_root", "omitted_action_id", "separated_route"),
    ("factual_root", "omitted_action_ids", "source_script"),
    ("factual_root", "omitted_action_ids", "trigger_rewrites"),
    ("factual_root", "trigger_rewrite"),
)
_REWRITE_MECHANISMS = (
    "tick_shift", "min_true_ticks_shift", "fired_after_delay_shift",
)
_REWRITE_SEMANTICS = (
    "delay_ticks is added to the source trigger threshold by the explicit dedicated plan"
)
_BUSINESS_SCOPE = "L4 ARM observed business response; not a formal ontology event family"
_UE_SCOPE = (
    "all authored engine entities, actual weather, interpreter events and executed actions"
)


def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _string() -> dict:
    return {"type": "string", "minLength": 1}


def _tick() -> dict:
    return {"type": "integer", "minimum": 0}


def _array(item: dict, *, minimum: int | None = None, maximum: int | None = None) -> dict:
    rule = {"type": "array", "items": item}
    if minimum is not None:
        rule["minItems"] = minimum
    if maximum is not None:
        rule["maxItems"] = maximum
    return rule


def _vector3() -> dict:
    return {
        "type": "array", "prefixItems": [{"type": "number"} for _ in range(3)],
        "items": False, "minItems": 3, "maxItems": 3,
    }


def _parameter_sources() -> dict:
    branches = []
    for keys in _PARAMETER_SOURCE_KEYSETS:
        properties = {
            key: (
                {"const": "aw_data"} if key == "factual_root"
                else _vector3() if key == "standoff_waypoint_enu_m"
                else _string()
            )
            for key in keys
        }
        branches.append(_object(properties))
    return {"oneOf": branches}


def _provenance() -> dict:
    rewrite = _object({
        "trigger_id": _string(),
        "delay_ticks": {"type": "integer", "minimum": 1, "maximum": 60},
        "mechanism": {"enum": list(_REWRITE_MECHANISMS)},
    })
    return _object({
        "parameter_sources": _parameter_sources(),
        "applied_trigger_rewrites": _array(rewrite),
        "trigger_rewrite_semantics": {"const": _REWRITE_SEMANTICS},
    })


def _factual_source_audit() -> dict:
    presence = _object({
        "tick": _tick(), "entity_id": _string(),
        "field": {"const": "sample_presence"}, "note": _string(),
    })
    position = _object({
        "tick": _tick(), "entity_id": _string(),
        "field": {"const": "pos_enu"},
        "capture": _vector3(), "engine": _vector3(),
    })
    return _object({
        "factual_source": {"const": "current_engine_factual_source_execution"},
        "generation_method": {
            "const": "execute_source_script_from_tick_zero_then_apply_exact_config_actions_no_snapshot_restore"
        },
        "historical_capture_role": {
            "const": "comparison_reference_only_never_the_product"
        },
        "capture_vs_engine_disagreement_count": _tick(),
        "capture_vs_engine_disagreement_sample": _array(
            {"oneOf": [presence, position]}, maximum=20
        ),
        "prefix_position_mismatch_ticks": _array(_tick()),
        "prefix_state_mismatch_ticks": _array(_tick()),
        "role": {"enum": [
            "factual_source", "intervention_prefix_must_match_factual",
        ]},
    })


def _base_schema() -> dict:
    schema = _l1_schema()
    for key in ("$schema", "$id", "title"):
        schema.pop(key)
    properties = schema["properties"]
    properties["event_family"] = {"enum": list(_EVENT_FAMILIES)}
    properties["window_status"] = {"enum": [
        "blocked_non_executable_family", "target_event_absent_in_window",
        "right_censored", "pending", "observed_response_confirmed",
        "events_recovered",
    ]}
    properties["provenance"] = _provenance()
    properties["factual_source_audit"] = _factual_source_audit()
    properties["ue_execution"] = _object({
        "method": {"const": "current_engine_execution_scoped_to_declared_window"},
        "scope": {"const": _UE_SCOPE},
        "source_scene": _string(),
        "source_script": _string(),
    })
    properties["ue_entry_module"] = _object({
        "path": {"const": "Dataset/tools/l4_ue_delivery.py"},
    })
    schema["then"]["properties"]["factual_source_audit"]["properties"]["role"] = {
        "const": "factual_source",
    }
    schema["else"]["properties"]["factual_source_audit"]["properties"]["role"] = {
        "const": "intervention_prefix_must_match_factual",
    }
    schema["required"].append("business_semantic_authority")
    return schema


def _profile_branch(base: dict, *, business: bool, frozen_paths: bool) -> dict:
    schema = deepcopy(base)
    properties = schema["properties"]
    profile = _BUSINESS_PROFILES if business else _OTHER_PROFILES
    properties["group_key"]["pattern"] = (
        "^" + profile + r"/seed[0-9]+/grp_[a-z0-9_]+$"
    )
    properties["episode_id"]["pattern"] = "^" + profile + r"__seed[0-9]+$"
    properties["business_semantic_authority"] = (
        _object({
            "path": {
                "const": "Dataset/semantic_rules/profiles/l4_arm_business_semantics.json"
            },
            "scope": {"const": _BUSINESS_SCOPE},
        }) if business else {"type": "null"}
    )
    if frozen_paths:
        keys = tuple(properties["sources"]["properties"])
        properties["sources"] = _object({
            key: _object({"path": _string()}) for key in keys
        })
        properties["snapshot_contract"]["properties"]["reconstruction"] = {
            "const": "execution from tick 0 with hashed sources and config"
        }
        properties["code"] = _array(_string(), minimum=1)
        properties["files"] = _array(_string(), minimum=1)
        properties["ue_execution"]["properties"]["code"] = _array(
            _string(), minimum=1
        )
    return schema


def arm_manifest_schema() -> dict:
    """Return the complete L4 ARM manifest-root schema."""
    base = _base_schema()
    branches = [
        _profile_branch(base, business=business, frozen_paths=frozen_paths)
        for business in (True, False)
        for frozen_paths in (False, True)
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_manifest_l4.record.schema.json",
        "title": "AERO_WORLD arm_manifest L4 record schema",
        "oneOf": branches,
    }
