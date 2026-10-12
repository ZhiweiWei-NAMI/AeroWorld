"""Closed whole-row grammar for L3 ARM ``objective/predicate_truth.jsonl``.

``Dataset/tools/l3_v2/semantics.py`` runs the objective pipeline's world-truth
and minimal-semantic producers.  ``arm_runtime.py`` window-crops their
``predicate_truth`` rows and writes one JSON object per line.  The companion
``predicate_truth_ticks.jsonl`` is a separate six-key projection.

The current adapter writes 3.1.0 rows with three root digests and two update
ticks at both the root and observation value.  Frozen L3 rows are 3.0.0 and
have no update ticks: one observed envelope has no digest fields, while
another carries all three root digests.  The three envelopes are separate
closed branches.  Registry-bound predicate roles, ontology classes, evidence
field paths and value units are taken from the current formal producer schema;
that schema returns a detached copy, so this module never mutates its cache.
For ``airspace.aircraft_in_landing_zone`` the observed frozen 3.0.0 L3
records bind a ``world:LandingPad``.  The current world-truth resolver emits
the actual class, which may be ``world:LandingArea`` or its ontology subclass
``world:LandingPad``; only this predicate's class leaf differs from the
registry role's base class.

Only L3-1_v1, L3-2_v1, L3-2_v2, L3-3_v1 and L3-3_v2 episode identities are
accepted.  The 3.1.0 envelope is producer-backed but has no frozen L3 instance
checked here.  Its predicate selector has 29 reachable ids: the L3 stage and
roster projections (``objective_pipeline.py:2223-2285``) are composed by
``l3_v2/semantics.py:300-307`` with event trigger/support/terminal ids
(``minimal_semantics_adapter.py:267-280``), and the adapter outputs only required assertions
(``minimal_semantics_adapter.py:296-313,321-327``).  Frozen 3.0.0 envelopes
retain the full formal predicate set.  Predicate-to-episode causal and cross-field equality checks
remain with the producer and are not new grammar constraints.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from Dataset.world_model.graph.predicate_structure import formal_predicate_truth_schema


_DRAFT = "https://json-schema.org/draft/2020-12/schema"
_DIGEST_FIELDS = ("rule_digest", "parameter_digest", "input_digest")
_UPDATE_FIELDS = ("truth_state_update_tick", "evidence_update_tick")
_CURRENT_PREDICATE_IDS = frozenset({
    "airspace.aircraft_airborne",
    "airspace.aircraft_at_home_pad",
    "airspace.aircraft_in_landing_zone",
    "airspace.aircraft_inside_restricted_airspace",
    "airspace.aircraft_outside_assigned_altitude_band",
    "airspace.aircraft_touchdown",
    "airspace.airspace_corridor_over_capacity",
    "communication.link_degraded",
    "communication.link_unavailable",
    "environment.fog_active",
    "environment.low_illumination_active",
    "environment.rain_active",
    "environment.temperature_outside_operating_range",
    "facility.charging_queue_over_capacity",
    "facility.charging_station_unavailable",
    "facility.landing_pad_contention",
    "ground.road_segment_closed",
    "ground.traffic_signal_malfunction_active",
    "positioning.aircraft_deviates_from_planned_route",
    "positioning.gnss_quality_degraded",
    "positioning.gnss_spoofing_active",
    "safety.aircraft_near_building_structure",
    "safety.aircraft_near_ground_vehicle",
    "safety.aircraft_near_pedestrian",
    "safety.aircraft_pair_below_separation",
    "safety.crowd_evacuation_active",
    "safety.vehicle_pedestrian_conflict",
    "security.communication_jamming_impact_active",
    "security.ground_control_station_compromised",
})
_EPISODE_ID_PATTERN = (
    r"^(?:L3-1_v1|L3-2_v[12]|L3-3_v[12])__seed(?:00|01|02)$"
)


def _branch(formal: dict[str, Any], *, version: str,
            digests: bool) -> dict[str, Any]:
    """Detach one formal branch and adjust only producer-proven envelope keys."""
    schema = deepcopy(formal)
    for annotation in ("$schema", "$id", "title"):
        schema.pop(annotation)
    properties = schema["properties"]
    required = schema["required"]
    properties["schema_version"] = {"const": version}
    properties["episode_id"] = {
        "type": "string", "pattern": _EPISODE_ID_PATTERN,
    }
    properties["truth_id"] = {
        "type": "string", "pattern": r"^truth:[0-9a-f]{24}$",
    }
    properties["tuple_id"] = {
        "type": "string", "pattern": r"^world_predicate_tuple:[0-9a-f]{24}$",
    }
    properties["tick"] = {
        "type": "integer", "minimum": 0, "maximum": 900, "multipleOf": 5,
    }
    # The L3 world projection produces only deterministic assertions and
    # supplies no mechanism id; _adapt_assertions writes that id as null.
    properties["source_class"] = {"const": "deterministic_derived"}
    properties["mechanism_id"] = {"const": None}
    if "mechanism_id" not in required:
        required.append("mechanism_id")
    if not digests:
        for field in _DIGEST_FIELDS:
            properties.pop(field)
            required.remove(field)
    if version == "3.0.0":
        for field in _UPDATE_FIELDS:
            properties.pop(field)
            required.remove(field)
    else:
        for field in _UPDATE_FIELDS:
            properties[field] = {
                "type": "integer", "minimum": 0, "maximum": 900,
                "multipleOf": 5,
            }

    predicate_branches = schema["allOf"][0]["oneOf"]
    if version == "3.1.0":
        predicate_branches = [
            predicate for predicate in predicate_branches
            if predicate["properties"]["predicate_id"]["const"]
            in _CURRENT_PREDICATE_IDS
        ]
        if len(predicate_branches) != len(_CURRENT_PREDICATE_IDS):
            raise ValueError("current L3 predicate ids diverged from the formal registry")
        schema["allOf"][0]["oneOf"] = predicate_branches

    for predicate in predicate_branches:
        observations = predicate["properties"]["evidence"]["properties"]["observations"]
        # The adapter emits exactly one normalized observation for each
        # assertion, regardless of world_observations length.
        observations["minItems"] = 1
        observations["maxItems"] = 1
        value = observations["items"]["oneOf"][0]["properties"]["value"]
        if predicate["properties"]["predicate_id"]["const"] == "airspace.aircraft_in_landing_zone":
            classes = value["properties"]["binding_ontology_classes"]["properties"]
            if classes["landing_zone"] != {"const": "world:LandingArea"}:
                raise ValueError("landing-zone ontology role changed; review L3 grammar")
            # All observed frozen L3 rows use a physical landing pad. The
            # current resolver emits actual compatible classes, not the role
            # base class (world_truth.py::_resolve_actual_class); LandingPad
            # is the sole LandingArea subclass in world_core.ttl.
            classes["landing_zone"] = (
                {"const": "world:LandingPad"}
                if version == "3.0.0"
                else {"enum": ["world:LandingArea", "world:LandingPad"]}
            )
        if version == "3.0.0":
            for field in _UPDATE_FIELDS:
                value["properties"].pop(field)
                value["required"].remove(field)
        else:
            for field in _UPDATE_FIELDS:
                value["properties"][field] = {
                    "type": "integer", "minimum": 0, "maximum": 900,
                    "multipleOf": 5,
                }
    return schema


def arm_predicate_truth_schema() -> dict[str, Any]:
    """Return the L3 closed whole-record schema; import performs no I/O."""
    formal = formal_predicate_truth_schema()
    return {
        "$schema": _DRAFT,
        "$id": "https://aeroworld.local/schema/arm_predicate_truth_l3.record.schema.json",
        "title": "L3 ARM predicate truth JSONL record",
        "oneOf": [
            _branch(formal, version="3.0.0", digests=False),
            _branch(formal, version="3.0.0", digests=True),
            _branch(formal, version="3.1.0", digests=True),
        ],
    }
