"""Compile closed Draft 2020-12 record structures without assigning field meaning.

The authoring-input literal compiler fixes object keys and array positions from
one source document. Scalar values remain typed rather than fixed to that sample.
Runtime products must use a producer-owned schema instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

from jsonschema import Draft202012Validator, ValidationError, validators

from Dataset.tools.runtime_state_source_guard import (
    RUNTIME_STATE_SOURCE_GUARD_POLICIES,
    forbidden_runtime_state_paths,
)


_REPO = Path(__file__).resolve().parents[3]
_SCENARIOS = _REPO / "Dataset" / "scenarios"
_DRAFT = "https://json-schema.org/draft/2020-12/schema"
# Full field paths omitted only by the explicit publication projection below.
# Raw validation never excludes them: every declared field is required and
# pattern-validated, and any other integrity-keyed field is an undeclared
# property.
_PUBLICATION_EXCLUDED_PATHS = frozenset({
    ("rule_digest",),
    ("parameter_digest",),
    ("input_digest",),
    ("base_graph_digest",),
    ("delta_digest",),
    ("initial_assertions", "rule_digest"),
    ("initial_assertions", "parameter_digest"),
    ("initial_assertions", "input_digest"),
    ("operations", "rule_digest"),
    ("operations", "assertion", "rule_digest"),
    ("operations", "assertion", "parameter_digest"),
    ("operations", "assertion", "input_digest"),
    ("source", "sha256"),
    ("source_event_script_digest",),
    ("generation", "source_contract_hash"),
    ("raw", "communication_state", "input_digest"),
    ("raw", "communication_state", "parameter_digest"),
    ("raw", "communication_state", "seed_digest"),
    ("raw", "communication_ticks", "input_digest"),
    ("raw", "communication_ticks", "parameter_digest"),
    ("raw", "communication_ticks", "seed_digest"),
    ("raw", "domain_observations", "input_digest"),
    ("raw", "domain_observations", "parameter_digest"),
    ("raw", "domain_observations", "seed_digest"),
    ("objective", "predicate_truth", "rule_digest"),
    ("objective", "predicate_truth", "parameter_digest"),
    ("objective", "predicate_truth", "input_digest"),
    ("objective", "predicate_transitions", "rule_digest"),
    ("transitions", "rule_digest"),
    ("arm_script", "digest"),
    ("config", "digest"),
    ("semantic_model", "seed_digest"),
    ("sources", "roster", "digest"),
    ("sources", "scene", "digest"),
    ("sources", "script", "digest"),
    ("sources", "trajectories", "digest"),
    ("sources", "weather", "digest"),
    ("record_digest",),
    ("provenance", "event_script_digest"),
    ("provenance", "event_trace_digest"),
    ("provenance", "event_realization_digest"),
    ("parameters", "roi_contract", "minimal_change_evidence", "roadway_mesh_hash"),
    ("domain_profile", "sha256"),
    ("compute_profile", "sha256"),
    ("contract_profile", "sha256"),
    ("stage_acceptance_profile", "sha256"),
    ("domain_input_digest",),
    ("context_input_digest",),
    ("static_geometry_authority", "resolved_source_sha256"),
    ("projection", "input_digest"),
    ("projection", "parameter_digest"),
    ("global_detection_record_digests", "semantic_state_observations"),
    ("global_detection_record_digests", "predicate_truth"),
    ("global_detection_record_digests", "predicate_transitions"),
    ("global_detection_record_digests", "predicate_continuity_breaks"),
    ("global_detection_record_digests", "event_occurrences"),
    ("global_detection_record_digests", "event_outcomes"),
    ("world_truth_summary", "input_digest"),
    ("world_truth_summary", "parameter_digest"),
    ("utm_supplement_summary", "input_digest"),
    ("utm_supplement_summary", "parameter_digest"),
    ("trajectory_union", "world_source", "sha256"),
    ("trajectory_union", "scenario_source", "sha256"),
    ("simulation_logs", "compute_log.jsonl", "sha256"),
    ("simulation_logs", "communication_log.jsonl", "sha256"),
    ("simulation_logs", "charging_log.jsonl", "sha256"),
    ("simulation_logs", "battery_log.jsonl", "sha256"),
    ("simulation_logs", "utm_log.jsonl", "sha256"),
    ("simulation_logs", "gnss_log.jsonl", "sha256"),
    ("simulation_logs", "weather_log.jsonl", "sha256"),
    ("simulation_logs", "facility_log.jsonl", "sha256"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "compute_log.jsonl", "multiset_digest"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "communication_log.jsonl", "multiset_digest"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "charging_log.jsonl", "multiset_digest"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "battery_log.jsonl", "multiset_digest"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "utm_log.jsonl", "multiset_digest"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "gnss_log.jsonl", "multiset_digest"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "weather_log.jsonl", "multiset_digest"),
    ("simulation_log_reconciliation", "source_digest_multiset_by_log", "facility_log.jsonl", "multiset_digest"),
    ("l0_contract_artifacts", "l0_predicate_source_availability.json", "sha256"),
    ("event_family_mapping", "sha256"),
    ("l0_state_materialization", "sumo_roster_never_active_digest"),
    ("l0_state_materialization", "render_vehicle_authority_conflict_digest"),
    ("l0_state_materialization", "l0_state_profile_sha256"),
    ("l0_state_materialization", "sumo_authority_digest"),
    ("l0_state_materialization", "state_digest"),
    ("runtime_state_materialization", "runtime_schedule_sha256"),
    ("runtime_state_materialization", "l0_state_profile_sha256"),
    ("runtime_state_materialization", "event_fire_tick_digest"),
})
_L2_FROZEN_CODE_KEYS = frozenset({
    "Dataset/tools/batch_generate.py", "Dataset/tools/l2_arm_semantics.py",
    "Dataset/tools/l2_window.py", "rules/l2_window.schema.json",
})
_L2_FROZEN_FILE_KEY = re.compile(
    r"^(?:raw|objective|chain)/[A-Za-z0-9_./-]+\.(?:json|jsonl)$")
_L3_FROZEN_IMPLEMENTATION_KEYS = frozenset({
    "Dataset/tools/l3_v2/engine.py", "Dataset/tools/l3_v2/semantics.py",
    "Dataset/tools/l3_v2/arm_runtime.py", "Dataset/tools/batch_generate.py",
    "Plugins/SumoImporter/Scripts/donghu_core/event_script_interpreter.py",
    "Dataset/semantic_truth/objective_pipeline.py",
    "Dataset/semantic_simulation/predicate_state_computers.py",
    "Dataset/semantic_rules/predicates/core_semantic_predicate_templates.json",
})


# Matcher over original source traversal tokens: string object keys plus int
# array indices and DynamicMapKey pattern tokens.  Array positions are
# transparent to key-path matching.  The L2 frozen files map has
# producer-declared file-key syntax; its values and the fixed L2 code and L3
# implementation values are integrity metadata.  This same projection is used
# by record and source-value walks.
def is_publication_excluded_path(tokens: tuple[str | int | ArrayItem | DynamicMapKey, ...]) -> bool:
    path = tuple(token for token in tokens if type(token) is str)
    if path in _PUBLICATION_EXCLUDED_PATHS:
        return True
    # The L2 frozen manifest files map is keyed by a dynamic producer-declared
    # pattern and every value is an integrity digest.  A DynamicMapKey whose rule
    # is exactly that frozen-file-key pattern denotes those digest leaves, so the
    # whole dynamic leaf is integrity metadata -- match the precise rule, never a
    # name/hash heuristic.  Concrete file keys are still matched by fullmatch.
    if (path == ("files",) and
            any(isinstance(token, DynamicMapKey) and
                token.rule == _L2_FROZEN_FILE_KEY.pattern for token in tokens)):
        return True
    if len(path) != 2:
        return False
    parent, leaf = path
    return ((parent == "code" and leaf in _L2_FROZEN_CODE_KEYS) or
            (parent == "files" and _L2_FROZEN_FILE_KEY.fullmatch(leaf) is not None) or
            (parent == "implementation" and
             leaf in _L3_FROZEN_IMPLEMENTATION_KEYS))


_SCHEMA_KEYS = frozenset({
    "$schema", "$id", "$defs", "$ref", "$comment", "title", "description",
    "default", "examples", "deprecated", "readOnly", "writeOnly",
    "type", "const", "enum", "allOf", "anyOf", "oneOf", "if", "then", "else",
    "properties", "patternProperties", "propertyNames", "required",
    "additionalProperties", "unevaluatedProperties", "minProperties", "maxProperties",
    "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "format", "x-source-value-types",
    "x-runtime-state-source-guard",
})
_SCHEMA_MAPS = ("$defs", "properties", "patternProperties")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_SINGLES = ("if", "then", "else", "propertyNames", "additionalProperties",
                   "unevaluatedProperties", "items")


@dataclass(frozen=True)
class ArrayItem:
    """An unbounded typed array position in a field path."""


@dataclass(frozen=True)
class DynamicMapKey:
    """A map key constrained by an anchored pattern or a finite value rule."""

    rule: str


@dataclass(frozen=True)
class VariantSelector:
    schema_path: str
    mode: str
    branch: int | str


@dataclass(frozen=True)
class GrammarField:
    path: tuple[str | int | ArrayItem | DynamicMapKey, ...]
    schema_path: str
    kind: str
    value_type: str | None
    source_value_types: tuple[str, ...] | None
    selectors: tuple[VariantSelector, ...]


@dataclass(frozen=True)
class _Term:
    schema: Mapping[str, Any]
    pointer: str


@dataclass(frozen=True)
class _Alternative:
    terms: tuple[_Term, ...]
    selectors: tuple[VariantSelector, ...] = ()


def _pointer(parent: str, *parts: str | int) -> str:
    return parent + "".join("/" + str(part).replace("~", "~0").replace("/", "~1")
                             for part in parts)


def _copy_record(value: Any, path: str = "$") -> Any:
    """Copy a JSON value unchanged; reject anything outside JSON."""
    if type(value) is dict:
        result = {}
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path}: object key is not a string: {key!r}")
            result[key] = _copy_record(child, f"{path}.{key}")
        return result
    if type(value) is list:
        return [_copy_record(child, f"{path}[{index}]")
                for index, child in enumerate(value)]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not (float("-inf") < value < float("inf")):
            raise ValueError(f"{path}: non-finite JSON number")
        return value
    raise TypeError(f"{path}: unsupported JSON value type {type(value).__name__}")


def _resolve_ref(root: Mapping[str, Any], ref: str) -> tuple[Mapping[str, Any], str]:
    if ref == "#":
        return root, "#"
    if not ref.startswith("#/") or "%" in ref:
        raise ValueError(f"external or non-pointer $ref is unsupported: {ref!r}")
    value: Any = root
    for escaped in ref[2:].split("/"):
        part = escaped.replace("~1", "/").replace("~0", "~")
        if type(value) is dict and part in value:
            value = value[part]
        elif type(value) is list and part.isascii() and part.isdecimal() and (
                part == "0" or not part.startswith("0")) and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValueError(f"unresolved local $ref: {ref}")
    if type(value) is not dict:
        raise ValueError(f"$ref does not address an object schema: {ref}")
    return value, ref


def _scan_schema(root: Mapping[str, Any], node: Any, pointer: str = "#") -> None:
    if type(node) is not dict:
        raise ValueError(f"{pointer}: expected an object schema")
    unknown = set(node) - _SCHEMA_KEYS
    if unknown:
        raise ValueError(f"{pointer}: unsupported schema keywords {sorted(unknown)}")
    if "x-runtime-state-source-guard" in node:
        declaration = node["x-runtime-state-source-guard"]
        if (node.get("type") != "object" or type(declaration) is not dict
                or set(declaration) != {"policy"}
                or type(declaration["policy"]) is not str
                or declaration["policy"] not in RUNTIME_STATE_SOURCE_GUARD_POLICIES):
            raise ValueError(
                f"{pointer}/x-runtime-state-source-guard: expected an object "
                "schema with exactly one policy: engine_whole or regenerate_split"
            )
    if pointer != "#" and "$id" in node:
        raise ValueError(f"{pointer}: nested $id is unsupported")
    if "additionalProperties" in node and node["additionalProperties"] is True:
        raise ValueError(f"{pointer}/additionalProperties: open object")
    if "unevaluatedProperties" in node and node["unevaluatedProperties"] is True:
        raise ValueError(f"{pointer}/unevaluatedProperties: open object")
    if "items" in node and node["items"] is True:
        raise ValueError(f"{pointer}/items: untyped array tail")
    if "$ref" in node:
        _resolve_ref(root, node["$ref"])
    for keyword in _SCHEMA_MAPS:
        for name, child in node.get(keyword, {}).items():
            if keyword == "patternProperties":
                _anchored_pattern(name, _pointer(pointer, keyword, name))
            _scan_schema(root, child, _pointer(pointer, keyword, name))
    for keyword in _SCHEMA_LISTS:
        for index, child in enumerate(node.get(keyword, [])):
            _scan_schema(root, child, _pointer(pointer, keyword, index))
    for keyword in _SCHEMA_SINGLES:
        child = node.get(keyword)
        if type(child) is dict:
            _scan_schema(root, child, _pointer(pointer, keyword))
        elif child is not None and child is not False:
            raise ValueError(f"{pointer}/{keyword}: unsupported boolean schema")


def _anchored_pattern(pattern: str, pointer: str) -> None:
    if not pattern.startswith("^") or not pattern.endswith("$"):
        raise ValueError(f"{pointer}: dynamic map key pattern must be anchored")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"{pointer}: invalid key pattern: {exc}") from exc


def _combine(left: list[_Alternative], right: list[_Alternative]) -> list[_Alternative]:
    return [_Alternative(a.terms + b.terms, a.selectors + b.selectors)
            for a in left for b in right]


def _expand_term(root: Mapping[str, Any], term: _Term,
                 ref_stack: tuple[str, ...] = ()) -> list[_Alternative]:
    schema = term.schema
    result = [_Alternative((term,))]
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in ref_stack:
            raise ValueError(f"{term.pointer}/$ref: recursive local reference {ref}")
        target, pointer = _resolve_ref(root, ref)
        result = _combine(result, _expand_term(root, _Term(target, pointer),
                                               ref_stack + (ref,)))
    for index, child in enumerate(schema.get("allOf", [])):
        result = _combine(result, _expand_term(
            root, _Term(child, _pointer(term.pointer, "allOf", index)), ref_stack))
    for mode in ("oneOf", "anyOf"):
        if mode not in schema:
            continue
        branches = []
        for index, child in enumerate(schema[mode]):
            selector = VariantSelector(_pointer(term.pointer, mode), mode, index)
            for option in _expand_term(root, _Term(
                    child, _pointer(term.pointer, mode, index)), ref_stack):
                branches.append(_Alternative(option.terms,
                                             (selector,) + option.selectors))
        result = _combine(result, branches)
    if "if" in schema:
        pointer = _pointer(term.pointer, "if")
        branches = []
        for branch, keyword in (("true", "then"), ("false", "else")):
            selector = VariantSelector(pointer, "if", branch)
            if keyword in schema:
                options = _expand_term(root, _Term(
                    schema[keyword], _pointer(term.pointer, keyword)), ref_stack)
            else:
                options = [_Alternative(())]
            branches.extend(_Alternative(option.terms,
                                         (selector,) + option.selectors)
                            for option in options)
        result = _combine(result, branches)
    return result


def _expand(root: Mapping[str, Any], terms: tuple[_Term, ...]) -> list[_Alternative]:
    result = [_Alternative(())]
    for term in terms:
        result = _combine(result, _expand_term(root, term))
    return result


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _source_type(value: Any) -> str:
    return {"null": "null", "boolean": "bool", "integer": "int",
            "number": "float", "string": "string", "array": "array",
            "object": "object"}[_json_type(value)]


def _strict_const(validator: Any, expected: Any, instance: Any,
                  schema: Mapping[str, Any]) -> Iterator[ValidationError]:
    if type(instance) is not type(expected) or instance != expected:
        yield ValidationError(f"source value differs from const {expected!r}")


def _strict_enum(validator: Any, choices: list[Any], instance: Any,
                 schema: Mapping[str, Any]) -> Iterator[ValidationError]:
    if not any(type(instance) is type(choice) and instance == choice
               for choice in choices):
        yield ValidationError(f"source value is outside enum {choices!r}")


def _source_value_types(validator: Any, allowed: list[str], instance: Any,
                        schema: Mapping[str, Any]) -> Iterator[ValidationError]:
    if _source_type(instance) not in allowed:
        yield ValidationError(f"source value type {_source_type(instance)}"
                              f" is outside {allowed!r}")


def _runtime_state_source_guard(
    validator: Any, declaration: Mapping[str, str], instance: Any,
    schema: Mapping[str, Any]
) -> Iterator[ValidationError]:
    if not isinstance(instance, Mapping):
        return
    forbidden = forbidden_runtime_state_paths(instance, policy=declaration["policy"])
    if forbidden:
        yield ValidationError(
            f"runtime state subtree contains forbidden semantic fields or paths: {forbidden}"
        )


_TYPE_CHECKER = (Draft202012Validator.TYPE_CHECKER
                 .redefine("integer", lambda checker, value: type(value) is int)
                 .redefine("number", lambda checker, value:
                           type(value) in (int, float)))
# This project validator enforces x-runtime-state-source-guard. A plain
# external Draft202012Validator accepts the declaration but ignores the check.
_RecordValidator = validators.extend(
    Draft202012Validator,
    validators={"const": _strict_const, "enum": _strict_enum,
                "x-source-value-types": _source_value_types,
                "x-runtime-state-source-guard": _runtime_state_source_guard},
    type_checker=_TYPE_CHECKER,
)


def _effective_type(terms: tuple[_Term, ...], path: tuple[Any, ...]) -> str:
    types = []
    for term in terms:
        schema = term.schema
        if "type" in schema:
            declared = schema["type"]
            if type(declared) is not str:
                raise ValueError(f"{term.pointer}/type: use explicit branches for type alternatives")
            types.append(declared)
        if "const" in schema:
            types.append(_json_type(schema["const"]))
        if "enum" in schema:
            choices = {_json_type(value) for value in schema["enum"]}
            if len(choices) != 1:
                raise ValueError(f"{term.pointer}/enum: use explicit branches for mixed types")
            types.extend(choices)
    if not types:
        raise ValueError(f"{path!r}: accepted value has no explicit JSON type")
    effective = types[0]
    for declared in types[1:]:
        if {effective, declared} == {"integer", "number"}:
            effective = "integer"
        elif effective != declared:
            raise ValueError(f"{path!r}: incompatible type constraints {types}")
    if effective in ("object", "array") and any(
            "const" in term.schema or "enum" in term.schema for term in terms):
        raise ValueError(f"{path!r}: container const/enum requires an explicit structure")
    return effective


def _effective_source_types(terms: tuple[_Term, ...], kind: str,
                            path: tuple[Any, ...]) -> tuple[str, ...]:
    defaults = {
        "null": ("null",), "boolean": ("bool",), "integer": ("int",),
        "number": ("int", "float"), "string": ("string",),
        "object": ("object",), "array": ("array",),
    }[kind]
    allowed = set(defaults)
    for term in terms:
        declared = term.schema.get("x-source-value-types")
        if declared is not None:
            if kind in ("object", "array"):
                raise ValueError(f"{term.pointer}/x-source-value-types: use on leaves only")
            if (type(declared) is not list or not declared or
                    any(type(name) is not str or name not in defaults for name in declared) or
                    len(declared) != len(set(declared))):
                raise ValueError(f"{term.pointer}/x-source-value-types: invalid raw type list")
            allowed.intersection_update(declared)
    if not allowed:
        raise ValueError(f"{path!r}: source value types contradict JSON type")
    return tuple(name for name in defaults if name in allowed)


def _key_rule(terms: tuple[_Term, ...]) -> str | None:
    for term in terms:
        names = term.schema.get("propertyNames")
        if type(names) is not dict:
            continue
        if "pattern" in names:
            pattern = names["pattern"]
            _anchored_pattern(pattern, _pointer(term.pointer, "propertyNames", "pattern"))
            return pattern
        if "const" in names and type(names["const"]) is str:
            return f"const:{names['const']}"
        if "enum" in names and all(type(key) is str for key in names["enum"]):
            return "enum:" + json.dumps(names["enum"], ensure_ascii=False)
    return None


def dynamic_map_key_matches(rule: str, key: str) -> bool:
    """Whether a ``DynamicMapKey`` rule admits ``key`` as one of its members.

    ``DynamicMapKey.rule`` is the anchored pattern that named a map's open
    keys, or the ``const:``/``enum:`` encoding ``_key_rule`` returns for a
    ``propertyNames`` rule.  A named business key is only reachable through
    that token when the rule actually admits it: ``_visual_state`` names
    ``mode``/``lights_on``/the runtime families as properties and then excludes
    them from its pattern with a negative lookahead, so they are not pattern
    members even though the same map holds them.  ``re.search`` is the same
    admission test ``_object_children`` uses to fold a key into a pattern's
    terms.
    """
    if rule.startswith("const:"):
        return key == rule[len("const:"):]
    if rule.startswith("enum:"):
        return key in json.loads(rule[len("enum:"):])
    return re.search(rule, key) is not None


def _object_children(terms: tuple[_Term, ...]) -> Iterator[tuple[str, tuple[_Term, ...]]]:
    keys = dict.fromkeys(name for term in terms
                         for name in term.schema.get("properties", {}))
    for key in keys:
        children = []
        for term in terms:
            schema = term.schema
            properties = schema.get("properties", {})
            if key in properties:
                children.append(_Term(properties[key], _pointer(term.pointer, "properties", key)))
            matched = False
            for pattern, child in schema.get("patternProperties", {}).items():
                if re.search(pattern, key):
                    matched = True
                    children.append(_Term(child, _pointer(term.pointer, "patternProperties", pattern)))
            if key not in properties and not matched and type(schema.get("additionalProperties")) is dict:
                children.append(_Term(schema["additionalProperties"],
                                      _pointer(term.pointer, "additionalProperties")))
        yield key, tuple(children)


# A patternProperties/propertyNames rule that is a pure alternation of literal
# identifiers (^a$|^b$ style as ^(a|b)$) names a finite business enum, not an
# open dynamic map.  Those keys are fixed business fields and must keep their
# literal names in the projection (measurements.communication_state, weather.rain,
# …); only genuinely open patterns collapse into a dimension template.
_ENUM_RULE = re.compile(r"^\^\(([A-Za-z_][A-Za-z0-9_]*(?:\|[A-Za-z_][A-Za-z0-9_]*)*)\)\$$")


def _finite_enum_names(rule: Any) -> tuple[str, ...] | None:
    if type(rule) is not str:
        return None
    match = _ENUM_RULE.fullmatch(rule)
    if match is None:
        return None
    return tuple(match.group(1).split("|"))


def _walk(root: Mapping[str, Any], terms: tuple[_Term, ...],
          path: tuple[str | int | ArrayItem | DynamicMapKey, ...],
          selectors: tuple[VariantSelector, ...],
          output: list[GrammarField]) -> None:
    for alternative in _expand(root, terms):
        current_selectors = selectors + alternative.selectors
        for selector in alternative.selectors:
            output.append(GrammarField(path, selector.schema_path, "variant", None, None,
                                       current_selectors))
        active = alternative.terms
        kind = _effective_type(active, path)
        raw_types = _effective_source_types(active, kind, path)
        pointer = active[0].pointer
        output.append(GrammarField(path, pointer,
                                   "primitive" if kind not in ("object", "array") else kind,
                                   kind, raw_types, current_selectors))
        if kind == "object":
            has_closed_boundary = any(
                term.schema.get("additionalProperties") is False or
                term.schema.get("unevaluatedProperties") is False
                for term in active)
            typed_extras = [term for term in active
                            if type(term.schema.get("additionalProperties")) is dict]
            if not has_closed_boundary and not typed_extras:
                raise ValueError(f"{path!r} ({pointer}): object has open additional properties")
            key_rule = _key_rule(active)
            if typed_extras and key_rule is None:
                raise ValueError(f"{path!r} ({pointer}): typed dynamic map lacks a key rule")
            for name, child_terms in _object_children(active):
                _walk(root, child_terms, path + (name,), current_selectors, output)
            patterns = dict.fromkeys(pattern for term in active
                                     for pattern in term.schema.get("patternProperties", {}))
            for pattern in patterns:
                child_terms = tuple(_Term(term.schema["patternProperties"][pattern],
                                          _pointer(term.pointer, "patternProperties", pattern))
                                    for term in active if pattern in term.schema.get("patternProperties", {}))
                enum_names = _finite_enum_names(pattern)
                if enum_names is not None:
                    for enum_name in enum_names:
                        _walk(root, child_terms, path + (enum_name,),
                              current_selectors, output)
                else:
                    _walk(root, child_terms, path + (DynamicMapKey(pattern),),
                          current_selectors, output)
            if typed_extras:
                child_terms = tuple(_Term(term.schema["additionalProperties"],
                                          _pointer(term.pointer, "additionalProperties"))
                                    for term in typed_extras)
                enum_names = _finite_enum_names(key_rule)
                if enum_names is not None:
                    for enum_name in enum_names:
                        _walk(root, child_terms, path + (enum_name,),
                              current_selectors, output)
                else:
                    _walk(root, child_terms, path + (DynamicMapKey(key_rule),),
                          current_selectors, output)
        elif kind == "array":
            if not any("items" in term.schema for term in active):
                raise ValueError(f"{path!r} ({pointer}): array has no items rule")
            prefix_count = max((len(term.schema.get("prefixItems", [])) for term in active),
                               default=0)
            for index in range(prefix_count):
                child_terms = []
                impossible = False
                for term in active:
                    schema = term.schema
                    prefix = schema.get("prefixItems", [])
                    if index < len(prefix):
                        child_terms.append(_Term(prefix[index],
                                                 _pointer(term.pointer, "prefixItems", index)))
                    elif schema.get("items") is False:
                        impossible = True
                    elif type(schema.get("items")) is dict:
                        child_terms.append(_Term(schema["items"],
                                                 _pointer(term.pointer, "items")))
                if not impossible:
                    _walk(root, tuple(child_terms), path + (index,), current_selectors, output)
            if not any(term.schema.get("items") is False for term in active):
                child_terms = tuple(_Term(term.schema["items"],
                                          _pointer(term.pointer, "items"))
                                    for term in active if type(term.schema.get("items")) is dict)
                if not child_terms:
                    raise ValueError(f"{path!r} ({pointer}): array tail is untyped")
                _walk(root, child_terms, path + (ArrayItem(),), current_selectors, output)


@dataclass(frozen=True, init=False)
class ClosedRecordGrammar:
    schema_ref: str
    authority: str
    _schema_json: str = field(repr=False)
    _fields: tuple[GrammarField, ...] = field(repr=False)

    def __init__(self, schema_ref: str, authority: str, schema: Mapping[str, Any]) -> None:
        if not isinstance(schema_ref, str) or not schema_ref:
            raise ValueError("schema_ref must be a nonempty source reference")
        if not isinstance(authority, str) or not authority:
            raise ValueError("authority must be a nonempty declaration")
        if type(schema) is not dict:
            raise TypeError("record schema must be a JSON object")
        encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False, sort_keys=True)
        root = json.loads(encoded)
        if root.get("$schema", _DRAFT) != _DRAFT:
            raise ValueError(f"{schema_ref}: expected Draft 2020-12 JSON Schema")
        Draft202012Validator.check_schema(root)
        _scan_schema(root, root)
        declarations: list[GrammarField] = []
        _walk(root, (_Term(root, "#"),), (), (), declarations)
        unique = tuple(dict.fromkeys(declarations))
        object.__setattr__(self, "schema_ref", schema_ref)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "_schema_json", encoded)
        object.__setattr__(self, "_fields", unique)

    @property
    def schema(self) -> dict[str, Any]:
        """Return a detached native JSON Schema document."""
        return json.loads(self._schema_json)

    def iter_fields(self) -> Iterator[GrammarField]:
        """Keep tuple positions and each accepted branch separate."""
        return iter(self._fields)


def compile_record_grammar(schema: Mapping[str, Any], *, schema_ref: str,
                           authority: str) -> ClosedRecordGrammar:
    return ClosedRecordGrammar(schema_ref, authority, schema)


def _literal_schema(value: Any) -> dict[str, Any]:
    if type(value) is dict:
        return {"type": "object", "properties": {
            key: _literal_schema(child) for key, child in value.items()},
            "required": list(value), "additionalProperties": False}
    if type(value) is list:
        length = len(value)
        schema = {"type": "array", "items": False,
                  "minItems": length, "maxItems": length}
        if value:
            schema["prefixItems"] = [_literal_schema(child) for child in value]
        return schema
    return {"type": _json_type(value),
            "x-source-value-types": [_source_type(value)]}


def compile_authoring_input(path: str | Path) -> ClosedRecordGrammar:
    """Compile one exact scenario authoring document, never a derived product."""
    source = Path(path).resolve(strict=True)
    try:
        relative = source.relative_to(_SCENARIOS)
    except ValueError as exc:
        raise ValueError(f"not a scenario authoring input: {source}") from exc
    if (len(relative.parts) < 2 or
            relative.name not in ("scene_setup.json", "event_script.json") or
            not source.is_file()):
        raise ValueError(f"not a scenario authoring input: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{source}: non-finite JSON constant {value}")))
    schema = {"$schema": _DRAFT, **_literal_schema(raw)}
    return compile_record_grammar(schema, schema_ref=str(source.relative_to(_REPO)),
                                  authority="authoring_input_literal")


def project_record_for_publication(value: Any) -> Any:
    """Return the publication projection: the record minus exactly the
    integrity field paths listed in ``_PUBLICATION_EXCLUDED_PATHS``.  This is
    the only integrity-key exclusion in this module, is never applied by
    ``validate_record``, and matches nothing else."""
    if type(value) is not dict:
        raise TypeError(f"publication projection expects an object root: {type(value).__name__}")

    def _project(node: Any, tokens: tuple[str | int, ...]) -> Any:
        if type(node) is dict:
            result = {}
            for key, child in node.items():
                if type(key) is not str:
                    shown = "".join(
                        f"[{token}]" if type(token) is int else f".{token}"
                        for token in tokens)
                    raise TypeError(f"${shown}: object key is not a string: {key!r}")
                child_tokens = tokens + (key,)
                if is_publication_excluded_path(child_tokens):
                    continue
                result[key] = _project(child, child_tokens)
            return result
        if type(node) is list:
            return [_project(child, tokens + (index,))
                    for index, child in enumerate(node)]
        return node

    return _project(value, ())


def validate_record(grammar: ClosedRecordGrammar, raw: Any, *,
                    source_path: str | Path | None = None) -> None:
    """Validate the unchanged caller value against the unchanged full schema."""
    if not isinstance(grammar, ClosedRecordGrammar):
        raise TypeError("grammar must be a ClosedRecordGrammar")
    source = str(source_path) if source_path is not None else grammar.schema_ref
    record = _copy_record(raw)
    validator = _RecordValidator(grammar.schema)
    error = next(validator.iter_errors(record), None)
    if error is not None:
        value_path = "$" + "".join(
            f"[{token}]" if type(token) is int else f".{token}"
            for token in error.absolute_path)
        schema_path = _pointer("#", *error.absolute_schema_path)
        raise ValueError(f"{source}: {value_path} violates {grammar.schema_ref}"
                         f"{schema_path}: {error.message}")
