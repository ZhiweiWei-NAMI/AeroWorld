"""Producer-owned record structures for ARM initial event snapshots.

L3 projects inherited lifecycles in ``window_events``. L6 projects inherited
events and predicate transitions in ``window_semantics``. Predicate and event
binding roles come from the current core semantic registry, as consumed by the
minimal semantics adapter. Two integrity fields are declared explicitly because
the current L6 producer emits them and they are real source data here: the
content-address ``source.sha256`` on the pre-window evidence reference and
``rule_digest`` on an inherited predicate transition. Both are optional in the
grammar (L6 records that do not carry them remain valid) but, when present,
must match the sha256 pattern. No other integrity path is declared; the
publication projection still excludes these exact paths.
"""

from __future__ import annotations

from Dataset.semantic_truth.core_semantic_registry import (
    get_core_event_occurrence_types,
    get_core_predicate_templates,
)


def _object(properties: dict, *, required: tuple[str, ...] | None = None) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": properties,
    }


def _string_list(*, min_items: int = 0, max_items: int | None = None) -> dict:
    result = {"type": "array", "items": {"type": "string"}, "minItems": min_items}
    if max_items is not None:
        result["maxItems"] = max_items
    return result


def _binding_roles(roles: tuple[str, ...]) -> dict:
    return _object({role: {"type": "string"} for role in roles})


def _sha256() -> dict:
    """The same content-address shape the producer and predicate grammar use."""
    return {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}


def _event_variants() -> list[dict]:
    """Bind each event family to the roles emitted by the occurrence adapter."""
    variants = {
        (rule["event_family_id"], tuple(sorted(role["role"] for role in rule["participant_roles"])))
        for rule in get_core_event_occurrence_types()
    }
    return [
        {
            "properties": {
                "event_family_id": {"const": family},
                "bindings": _binding_roles(roles),
            },
            "required": ["event_family_id", "bindings"],
        }
        for family, roles in sorted(variants)
    ]


def _event(fields: dict) -> dict:
    schema = _object(fields)
    schema["oneOf"] = _event_variants()
    return schema


def _transition() -> dict:
    predicate_variants = [
        {
            "properties": {
                "predicate_id": {"const": template["id"]},
                "bindings": _binding_roles(
                    tuple(role["key"] for role in template["argument_roles"])
                ),
            },
            "required": ["predicate_id", "bindings"],
        }
        for template in get_core_predicate_templates()
    ]
    schema = _object({
        "schema_name": {"const": "predicate_transition"},
        "schema_version": {"const": "3.0.0"},
        "annotation_layer": {"const": "L1"},
        "source_layer": {"const": "L0"},
        "materialization_policy": {"const": "predicate_delta"},
        "transition_id": {"type": "string"},
        "episode_id": {"type": "string"},
        "predicate_id": {"type": "string"},
        "tuple_id": {"type": "string"},
        "bindings": {"type": "object"},
        "from_tick": {"type": "integer", "minimum": 0},
        "to_tick": {"type": "integer", "minimum": 0},
        "from_value": {"enum": ["true", "false"]},
        "to_value": {"enum": ["true", "false"]},
        "direction": {"enum": ["rising", "falling"]},
        "evidence_truth_ids": _string_list(min_items=2, max_items=2),
        "source_refs": _string_list(min_items=1),
        "template_parameters": _object({}),
        "parent_episode_id": {"type": "string"},
        "arm_key": {"type": "string"},
        "rule_digest": _sha256(),
    }, required=(
        "schema_name", "schema_version", "annotation_layer", "source_layer",
        "materialization_policy", "transition_id", "episode_id", "predicate_id",
        "tuple_id", "bindings", "from_tick", "to_tick", "from_value",
        "to_value", "direction", "evidence_truth_ids", "source_refs",
        "template_parameters", "parent_episode_id", "arm_key",
    ))
    schema["allOf"] = [
        {"oneOf": predicate_variants},
        {"oneOf": [
            {"properties": {
                "direction": {"const": "rising"},
                "from_value": {"const": "false"},
                "to_value": {"const": "true"},
            }},
            {"properties": {
                "direction": {"const": "falling"},
                "from_value": {"const": "true"},
                "to_value": {"const": "false"},
            }},
        ]},
    ]
    return schema


def initial_events_schema(producer: str) -> dict:
    """Return the complete Draft 2020-12 schema for an L3 or L6 producer."""
    tick = {"type": "integer", "minimum": 0}
    event_identity = {
        "event_id": {"type": "string"},
        "event_family_id": {"type": "string"},
        "bindings": {"type": "object"},
        "onset_tick": tick,
    }

    if producer == "L3":
        event = _event({
            **event_identity,
            "status_at_start": {
                "enum": ["ongoing", "recovery_confirmed_at_start"]
            },
            "supporting_truth_ids": _string_list(),
            "supporting_transition_ids": _string_list(),
        })
        properties = {
            "tick": tick,
            "policy": {"const": "inherited_source_lifecycle_not_a_new_window_trigger"},
            "source": {"type": "string"},
            "events": {"type": "array", "items": event},
        }
    elif producer == "L6":
        event = _event(event_identity)
        properties = {
            "tick": tick,
            "policy": {"const": "inherit_original_onset_and_producer_memory"},
            "source": _object({
                "path": {"type": "string"},
                "format": {"const": "gzip_json"},
                "scope": {"const": "pre_window_measured_evidence"},
                "sha256": _sha256(),
            }, required=("path", "format", "scope")),
            "events": {"type": "array", "items": event},
            "transitions": {"type": "array", "items": _transition()},
        }
    else:
        raise ValueError(f"unknown initial-events producer: {producer!r}")

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://aeroworld.local/schema/arm_initial_events.{producer}.schema.json",
        "title": f"{producer} ARM initial events record",
        **_object(properties),
    }
