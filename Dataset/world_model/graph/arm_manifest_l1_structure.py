"""Closed whole-JSON grammar for L1 ARM window manifests.

``l1_window.execute_config`` writes the manifest root from a validated ARM
configuration and measured window status. ``l1_ue_delivery._bind`` adds the
optional ``ue_entry_module`` block after UE export.  The current 84 L1
configurations supply only ``factual_root`` under
``provenance.parameter_sources``.  This constructor reads no files.
"""

from __future__ import annotations

__all__ = ["arm_manifest_schema"]

_PROFILE_PATTERN = r"L1-(?:1_v[12]|2_v1|3_v[12]|4_v[12])"
_EVENT_FAMILIES = (
    "uav_takeoff", "uav_landing", "restricted_region_intrusion",
    "altitude_excursion", "uav_separation_conflict", "corridor_congestion",
)
_SOURCE_KEYS = (
    "scene", "script", "trajectories", "weather", "event_realization",
    "predicate_transitions", "event_outcomes", "roster",
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


def _array(item: dict, *, minimum: int | None = None) -> dict:
    rule = {"type": "array", "items": item}
    if minimum is not None:
        rule["minItems"] = minimum
    return rule


def _fixed_array(values: tuple[str, ...]) -> dict:
    return {
        "type": "array",
        "prefixItems": [{"const": value} for value in values],
        "items": False,
        "minItems": len(values),
        "maxItems": len(values),
    }


def _window() -> dict:
    return _object({
        "start_tick": _tick(),
        "fork_tick": _tick(),
        "end_tick": _tick(),
        "phase": {"const": "after_state_sample_before_actions"},
        "intervention_order": {"const": "source_actions_then_intervention"},
    })


def _scope() -> dict:
    entity_ids = _array(_string(), minimum=1)
    entity_ids["uniqueItems"] = True
    predicate_ids = _array(_string(), minimum=1)
    predicate_ids["uniqueItems"] = True
    return _object({
        "entity_ids": entity_ids,
        "predicate_ids": predicate_ids,
        "background_policy": {
            "const": "frozen_parent_reference_no_feedback_claim"
        },
        "not_claimed": _fixed_array((
            "all_ontology_predicates", "whole_episode",
            "acceleration_feasible_flight", "UE_capture",
        )),
    })


def _factual_source_audit() -> dict:
    return _object({
        "compared_fields": _fixed_array((
            "pos_enu", "vel_mps", "state", "activity_type",
        )),
        "prefix_position_mismatch_ticks": _array(_tick()),
        "prefix_state_mismatch_ticks": _array(_tick()),
        "role": {"enum": ["baseline_drift", "prefix_shared_with_factual"]},
    })


def arm_manifest_schema() -> dict:
    """Return the complete manifest-root schema for the seven L1 profiles."""
    properties = {
        "schema": {"const": "aeroworld_window_arm"},
        "version": {"const": 1},
        "arm_id": {"type": "string", "pattern": r"^arm_[a-z0-9_]+$"},
        "group_key": {
            "type": "string",
            "pattern": "^" + _PROFILE_PATTERN + r"/seed[0-9]+/grp_[a-z0-9_]+$",
        },
        "episode_id": {
            "type": "string",
            "pattern": "^" + _PROFILE_PATTERN + r"__seed[0-9]+$",
        },
        "mode": {"enum": ["factual_observation", "intervention"]},
        "event_family": {"enum": list(_EVENT_FAMILIES)},
        "window": _window(),
        "scope": _scope(),
        "sources": _object({key: _string() for key in _SOURCE_KEYS}),
        "config": _object({"path": _string()}),
        "window_status": {"enum": [
            "target_event_absent_in_window", "right_censored", "pending",
            "events_recovered",
        ]},
        "provenance": _object({
            "parameter_sources": _object({"factual_root": {"const": "aw_data"}}),
        }),
        "factual_source_audit": _factual_source_audit(),
        "snapshot_contract": _object({
            "sampled_states": {"const": "actual_current_engine_rows_before_actions"},
            "entity_definitions": {
                "const": "static_engine_definitions_not_current_positions"
            },
            "standalone_restore_supported": {"const": False},
            "reconstruction": {
                "const": "execution from tick 0 with the referenced sources and config"
            },
        }),
        "release_status": {"const": "research_candidate_not_release_certified"},
        "ue_entry_module": _object({
            "path": {"const": "Dataset/tools/l1_ue_delivery.py"},
        }),
    }
    schema = _object(properties, [
        key for key in properties if key != "ue_entry_module"
    ])
    schema.update({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://aeroworld.local/schema/arm_manifest_l1.record.schema.json",
        "title": "AERO_WORLD arm_manifest L1 record schema",
        "if": {"properties": {"mode": {"const": "factual_observation"}}},
        "then": {"properties": {
            "factual_source_audit": {"properties": {
                "role": {"const": "baseline_drift"},
            }},
        }},
        "else": {"properties": {
            "factual_source_audit": {"properties": {
                "role": {"const": "prefix_shared_with_factual"},
            }},
        }},
    })
    return schema
