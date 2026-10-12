"""External graph contract checker used by materialization and the editor."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from Dataset.world_model.graph.contract import (
    CONTRACT_VERSION, EDGE_ENDPOINTS, ENTITY_CATEGORIES, NODE_TYPES, REPO, TRUTH, iter_jsonl, read_json,
)
from Dataset.world_model.graph.field_contract import DEFAULT_DEFINITIONS, coordinate_context_for_value
from Dataset.world_model.graph.field_catalog import validate_normalizations
from Dataset.world_model.graph.field_paths import (
    FieldValue, source_field_id, source_value_at_tokens,
)
from Dataset.world_model.graph.source_projection import (
    ProjectedField, _source_family, project_node_values, projected_catalog_binding,
    projected_source_rows,
)
from Dataset.world_model.graph.sample_identity import node_source_scope
from Dataset.world_model.graph.entity_identity import (
    BUSINESS_SOURCE_OBJECTS, action_target_scope, business_object_binding_allowed, identity_node_id,
)
from Dataset.semantic_truth.facility_scope import ontology_class_is_a
from Dataset.semantic_truth.entity_scope import entity_scope_contract


class ContractError(ValueError):
    pass


def _same_typed(actual: Any, expected: Any) -> bool:
    """Compare serialized source values without numeric or token coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return (actual.keys() == expected.keys() and
                all(_same_typed(actual[key], expected[key]) for key in actual))
    if isinstance(actual, (list, tuple)):
        return (len(actual) == len(expected) and
                all(_same_typed(a, b) for a, b in zip(actual, expected)))
    return actual == expected


@lru_cache(maxsize=512)
def _class_compatible(actual: str, expected: str) -> bool:
    if expected == "world:Entity":
        return True
    qualified = expected if expected.startswith("world:") else "world:" + expected
    return ontology_class_is_a(actual, qualified)


@lru_cache(maxsize=210)
def _capture_source_documents(episode_id: str) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    manifest = read_json(REPO / "aw_data/render_ready_episodes_capture_filtered" / episode_id / "episode_manifest.json")
    script_ref, scene_ref = manifest["source_event_script_path"], manifest["source_scene_setup_path"]
    return script_ref, scene_ref, read_json(REPO / script_ref), read_json(REPO / scene_ref)


@lru_cache(maxsize=690)
def _branch_execution_script(source_file: str) -> dict[str, Any]:
    return read_json(REPO / source_file)


class GraphValidator:
    def __init__(self, catalog_path: Path | None = None):
        path = catalog_path or REPO / "design/belief/contract.jsonl"
        rows = [row for _, row in iter_jsonl(path)]
        schema = [row for row in rows if row["kind"] == "schema"]
        if len(schema) != 1 or schema[0]["version"] != CONTRACT_VERSION:
            raise ContractError("catalog version mismatch")
        vocabulary_path = REPO / "design/belief/category_index.jsonl"
        vocabulary = [row for _, row in iter_jsonl(vocabulary_path)]
        if not vocabulary or vocabulary[0].get("contract_version") != CONTRACT_VERSION:
            raise ContractError("category index version mismatch")
        self.category_index = {(row["node_type"], row["category_id"])
                               for row in vocabulary if row["kind"] == "node_category"}
        self.predicates = {row["id"]: row for row in rows if row["kind"] == "predicate"}
        self.events = {row["id"]: row for row in rows if row["kind"] == "event"}
        self.raw_predicates = {row["id"]: row for row in rows if row["kind"] == "raw_predicate_mapping"}
        self.script_intents = {row["id"] for row in rows if row["kind"] == "script_event_category"}
        self.action_categories = {row["id"] for row in rows if row["kind"] == "command_category"}
        self.context_categories = {row["id"]: row for row in rows if row["kind"] == "context_category"}
        self.context_anchor_kinds = {row["id"] for row in rows if row["kind"] == "context_anchor_kind"}
        self.phase_categories = {row["id"] for row in rows if row["kind"] == "event_phase_category"}
        self.predicate_outcome_categories = {row["id"] for row in rows if row["kind"] == "predicate_episode_outcome_category"}
        self.time_categories = {row["id"] for row in rows if row["kind"] == "time_category"}
        field_path = REPO / "design/belief/field_catalog.jsonl"
        if not field_path.is_file():
            raise ContractError(f"field catalog missing: {field_path}")
        field_rows = [row for _, row in iter_jsonl(field_path)]
        self.fields = {row["id"]: row for row in field_rows}
        if not self.fields or len(self.fields) != len(field_rows):
            raise ContractError("field catalog is empty")
        if (any(row.get("contract_version") != CONTRACT_VERSION for row in field_rows)
                or {row["index"] for row in field_rows} != set(range(len(field_rows)))):
            raise ContractError("field catalog version or indices differ")
        try:
            self.normalized_codes = validate_normalizations(field_rows)
        except ValueError as error:
            raise ContractError("field catalog structural indices differ") from error
        definitions = [row for _, row in iter_jsonl(DEFAULT_DEFINITIONS)]
        self.field_definitions = {row["id"]: row for row in definitions}
        if (not definitions or len(self.field_definitions) != len(definitions) or
                any(row.get("contract_version") != CONTRACT_VERSION for row in definitions) or
                {row["semantic_id"] for row in field_rows} - set(self.field_definitions)):
            raise ContractError("field definitions and catalog differ")

    def validate_node(self, node: dict[str, Any], *, draft: bool = False) -> None:
        required = ("node_id", "sample_key", "node_type", "category_id", "tick", "attrs", "source_file", "source_replay_status")
        missing = [key for key in required if key not in node]
        if missing and not draft:
            raise ContractError(f"node missing {missing}")
        if node.get("node_type") not in NODE_TYPES:
            raise ContractError(f"unknown node type {node.get('node_type')}")
        node_type = node["node_type"]
        source_scope = None
        if not draft:
            source_family, _ = _source_family(node)
            source_scope = node_source_scope(node, source_family, require_stored=True)
        category = node.get("category_id")
        if (category is not None or not draft) and (node_type, category) not in self.category_index:
            raise ContractError(f"category outside frozen index: {node_type}:{category}")
        if draft and category is None:
            return
        if node.get("node_type") == "PredicateAssertion":
            if category not in self.predicates and not (draft and category is None):
                raise ContractError(f"unknown predicate {category}")
            if category == "x.geometry.entity_proximity":
                raw_id = node.get("attrs", {}).get("raw_predicate_id")
                if raw_id not in self.raw_predicates and not draft:
                    raise ContractError(f"geometry assertion lacks registered trigger ID: {raw_id}")
            if node.get("attrs", {}).get("truth_status") not in TRUTH and not draft:
                raise ContractError("assertion requires four-state truth_status")
        if node.get("node_type") == "EventOccurrence" and category not in self.events and not (draft and category is None):
            raise ContractError(f"unknown event {node.get('category_id')}")
        if (node_type == "EventOccurrence" and node["source_file"].startswith("aw_data/objective_semantic_truth/")
                and node["source_file"].endswith("/event_occurrences.jsonl") and not draft):
            attrs = node["attrs"]
            if (attrs.get("trigger_tick") != attrs.get("start_tick") or
                    attrs.get("detection_tick") != attrs.get("end_tick") or
                    node.get("tick") != attrs.get("detection_tick")):
                raise ContractError("formal event trigger and detection times differ from source roles")
        if node_type in {"PredicateTransition"} and category not in self.predicates:
            raise ContractError(f"unknown predicate transition {category}")
        if node_type in {"EventOutcome"} and category not in self.events:
            raise ContractError(f"unknown event outcome {category}")
        if node_type == "ScriptEvent":
            if category is None:
                if not draft and node.get("attrs", {}).get("category_status") != "unresolved_source_intent":
                    raise ContractError("ScriptEvent needs registered intent or explicit unresolved source intent")
            elif category not in self.script_intents:
                raise ContractError(f"unknown script intent {category}")
        if node_type in {"ActionPlan", "CommandDispatch"} and category not in self.action_categories:
            raise ContractError(f"unknown action category {category}")
        if node_type == "ContextAnchor" and category not in self.context_categories:
            raise ContractError(f"unknown context category {category}")
        if node_type == "ContextAnchor":
            anchor_kind = node.get("attrs", {}).get("anchor_kind")
            if anchor_kind not in self.context_anchor_kinds:
                raise ContractError(f"unknown context anchor kind {anchor_kind}")
            if anchor_kind not in self.context_categories[category]["allowed_anchor_kinds"]:
                raise ContractError(f"context category {category} cannot make {anchor_kind}")
            if node.get("attrs", {}).get("source_family") != category:
                raise ContractError("context source family differs from category")
            if category == "formal_weather_meta" and node["attrs"].get("raw", {}).get("tick") != node.get("tick"):
                raise ContractError("weather state tick differs from source row")
            if category == "sumo_controller_state" and (
                    node["attrs"].get("raw", {}).get("tick") != node.get("tick") or
                    sorted(node["attrs"].get("raw", {}).get("sumo_traffic_light_states", {})) !=
                    node["attrs"].get("controller_ids")):
                raise ContractError("SUMO controller state keys or tick differ from source frame")
            if category == "arm_action_audit" and (
                    node["attrs"].get("raw", {}).get("result", {}).get("status") !=
                    node["attrs"].get("dispatch_status") or
                    node["attrs"].get("raw", {}).get("origin") != node["attrs"].get("origin") or
                    node["attrs"].get("raw", {}).get("tick") != node.get("tick")):
                raise ContractError("ARM command audit differs from source row")
            if category == "arm_action_audit" and (
                    node["attrs"].get("origin") not in {"source_script", "intervention"} or
                    node["attrs"].get("dispatch_status") not in {
                        "ok", "omitted_by_explicit_intervention", "omitted_by_configuration"}):
                raise ContractError("ARM command audit has unknown origin or status")
            if category == "arm_action_audit" and (
                    node["attrs"].get("native_action_id") !=
                    node["attrs"].get("raw", {}).get("action", {}).get("action_id") or
                    node["attrs"].get("raw", {}).get("action", {}).get("type") not in self.action_categories):
                raise ContractError("ARM command audit action identity differs from source")
            if (category == "arm_action_audit" and node["attrs"].get("dispatch_status") != "ok"
                    and node["attrs"].get("origin") != "source_script"):
                raise ContractError("omitted command audit requires script origin")
        if node_type == "EventPhase" and category not in self.phase_categories:
            raise ContractError(f"unknown event phase category {category}")
        if node_type == "PredicateEpisodeOutcome" and category not in self.predicate_outcome_categories:
            raise ContractError(f"unknown predicate outcome category {category}")
        if node_type == "TimePoint" and category not in self.time_categories:
            raise ContractError(f"unknown time category {category}")
        if node.get("node_type") == "EntityIdentity" and node.get("tick") is not None:
            raise ContractError("stable identity must not have state tick")
        if node_type == "EntityIdentity" and not draft:
            attrs = node["attrs"]
            namespace, entity_id = attrs.get("identity_namespace"), attrs.get("entity_id")
            if (namespace not in {"sample", "compute_node", "sumo_tls_controller"}
                    or not isinstance(entity_id, str) or not entity_id
                    or node["node_id"] != identity_node_id(node["sample_key"], namespace, entity_id)):
                raise ContractError("identity node ID and declared namespace differ")
        if node.get("node_type") == "EntityIdentity" and node.get("category_id") not in (*ENTITY_CATEGORIES, None):
            raise ContractError("identity category must be a roster category or null")
        if node.get("node_type") == "EntityState" and node.get("category_id") not in (*ENTITY_CATEGORIES, None):
            raise ContractError("state category must be a roster category or null")
        if node.get("node_type") in {"EntityIdentity", "EntityState"} and not draft:
            if node.get("attrs", {}).get("identity_status") not in {"resolved", "unresolved", "conflict"}:
                raise ContractError("entity identity status missing")
            if (node["node_type"] == "EntityState" and
                    (node["attrs"]["identity_status"] == "resolved") !=
                    (isinstance(node["attrs"].get("identity_ref"), str))):
                raise ContractError("state identity reference and status differ")
        if node.get("node_type") == "TimePoint" and not isinstance(node.get("tick"), int):
            raise ContractError("TimePoint requires integer tick")
        if node_type in {"ActionPlan", "CommandDispatch"} and not draft:
            action = (node["attrs"].get("raw") if node_type == "ActionPlan"
                      else node["attrs"].get("source_action")) or {}
            action_type = action.get("type") if node_type == "ActionPlan" or node["source_file"].startswith("arms/") else action.get("action_type")
            if (node["attrs"].get("native_action_id") != action.get("action_id") or
                    action_type != category):
                raise ContractError("command identity or type differs from source action")
            if node["attrs"].get("target_scope") != action_target_scope(action):
                raise ContractError("action target scope differs from source action")
            if node["attrs"].get("target_entity_id") != action.get("entity_id"):
                raise ContractError("command target differs from source action")
        if node_type == "ActionPlan" and node["attrs"].get("audit_status") not in {
                "ok", "omitted_by_explicit_intervention", "omitted_by_configuration", "not_recorded"}:
            raise ContractError("action plan audit status is unknown")
        if node_type == "ActionPlan" and not draft:
            status = node["attrs"]["audit_status"]
            source_file = node["attrs"].get("audit_source_file")
            source_line = node["attrs"].get("audit_source_line")
            if (status == "not_recorded" and (source_file is not None or source_line is not None) or
                    status != "not_recorded" and
                    (not isinstance(source_file, str) or not source_file or
                     type(source_line) is not int or source_line < 1)):
                raise ContractError("action plan audit source differs from recorded status")
        if node_type == "ActionPlan" and node["source_file"].startswith("arms/") and not draft:
            path = REPO / node["source_file"]
            if path != (REPO / "arms" / node["sample_key"].removeprefix("arm:") / "ue/event_script.json"):
                raise ContractError("ARM action plan is not from the saved branch execution script")
            event_index, action_index = (node["attrs"].get("source_event_index"),
                                         node["attrs"].get("source_action_index"))
            script = _branch_execution_script(node["source_file"])
            if (type(event_index) is not int or type(action_index) is not int or
                    event_index < 0 or event_index >= len(script["events"])):
                raise ContractError("ARM action plan event index is outside execution script")
            event = script["events"][event_index]
            if (action_index < 0 or action_index >= len(event["actions"]) or
                    event["event_id"] != node["attrs"].get("native_event_id") or
                    event["actions"][action_index] != node["attrs"].get("raw")):
                raise ContractError("ARM action plan differs from execution script")
        if node_type == "CommandDispatch" and not draft:
            if node["attrs"].get("dispatch_status") != "ok":
                raise ContractError("command dispatch requires source status ok")
            expected_use = "capture_request" if category == "capture_screenshot" else "simulation_control"
            if node["attrs"].get("semantic_use") != expected_use:
                raise ContractError("command semantic use differs from category")
            if category == "capture_screenshot" and node["source_file"].startswith("aw_data/"):
                request = node["attrs"].get("source_script_request") or {}
                attrs = node["attrs"]
                episode_id = Path(node["source_file"]).parent.name
                script_ref, scene_ref, script, scene = _capture_source_documents(episode_id)
                event_index, action_index, camera_index = (
                    attrs.get("script_event_index"), attrs.get("script_action_index"), attrs.get("camera_index"))
                if (any(type(index) is not int or index < 0 for index in
                        (event_index, action_index, camera_index)) or
                        attrs.get("source_script_file") != script_ref or
                        attrs.get("source_scene_file") != scene_ref or
                        event_index >= len(script["events"]) or
                        camera_index >= len(scene["cameras"])):
                    raise ContractError("formal capture source references differ from episode manifest")
                event = script["events"][event_index]
                if action_index >= len(event["actions"]):
                    raise ContractError("formal capture source action index is outside script event")
                if (request.get("type") != category or
                        request != event["actions"][action_index] or
                        script["scenario_id"] != attrs.get("scenario_id") or
                        event["event_id"] != attrs.get("native_event_id") or
                        request.get("action_id") != attrs.get("native_action_id") or
                        request.get("camera_id") != attrs.get("camera_id") or
                        scene["cameras"][camera_index]["camera_id"] != attrs.get("camera_id")):
                    raise ContractError("formal capture request lacks script and camera evidence")
            if node["source_file"].startswith("arms/") and (
                    node["attrs"].get("origin") not in {"source_script", "intervention"} or
                    node["attrs"].get("source_result", {}).get("status") != "ok"):
                raise ContractError("ARM command dispatch lacks executed source audit")
        if node.get("node_type") in {"EntityState", "PredicateAssertion", "PredicateTransition", "EventPhase", "CommandDispatch"} and not isinstance(node.get("tick"), int) and not draft:
            raise ContractError(f"{node['node_type']} requires integer tick")
        fields = node.get("attrs", {}).get("fields")
        if fields is not None:
            if not isinstance(fields, list):
                raise ContractError("fields must be a list")
            raw = node.get("attrs", {}).get("raw")
            family, _ = _source_family(node)
            if not isinstance(raw, dict) or family is None:
                raise ContractError("structured fields lack a source raw record")
            if draft:
                source_scope = node_source_scope(node, family)
            expected_fields = {
                projected.sources[0].field_value.input_tokens:
                    (projected.sources[0].field_value, projected)
                for projected in project_node_values(node, catalog=self.fields,
                                                     scope=source_scope,
                                                     require_stored_scope=not draft)}
            seen_fields: set[tuple[str | int, ...]] = set()
            for item in fields:
                if not isinstance(item, dict):
                    raise ContractError("field is not an object")
                input_tokens = item.get("input_tokens")
                if (not isinstance(input_tokens, list) or
                        any(type(token) not in {str, int} for token in input_tokens)):
                    raise ContractError("field input location is missing")
                expected = expected_fields.get(tuple(input_tokens))
                if expected is None:
                    raise ContractError(f"field input location is absent: {input_tokens}")
                if tuple(input_tokens) in seen_fields:
                    raise ContractError(f"field input location repeats: {input_tokens}")
                seen_fields.add(tuple(input_tokens))
                self._validate_field(item, node, *expected, draft=draft)
            if not draft and seen_fields != set(expected_fields):
                raise ContractError(f"source fields are incomplete: {node['node_id']}")

    def _validate_field(self, item: dict[str, Any], node: dict[str, Any],
                        expected: FieldValue, projected: ProjectedField,
                        *, draft: bool = False) -> None:
        for key in ("path", "value", "value_tick", "source_file", "source_line", "unit", "unit_status"):
            if key not in item:
                raise ContractError(f"field missing {key}")
        if item["value_tick"] is not None and not isinstance(item["value_tick"], int):
            raise ContractError("field value_tick must be integer or null")
        if item["unit_status"] not in {"declared", "unresolved", "dimensionless", "not_applicable"}:
            raise ContractError("unknown unit_status")
        if item["unit_status"] == "declared" and not item["unit"]:
            raise ContractError("declared unit is empty")
        if node["node_type"] == "EntityState" and item["value_tick"] != node["tick"]:
            raise ContractError("state field value tick differs from state tick")
        if node["node_type"] == "EntityState" and not draft:
            reference = item.get("entity_ref")
            if not isinstance(reference, dict) or reference.get("raw_entity_id") != node["attrs"].get("entity_id"):
                raise ContractError("state field entity reference differs from source entity")
            if reference.get("sample_key") != node["sample_key"]:
                raise ContractError("state field entity sample differs")
            if reference.get("status") == "resolved":
                if reference.get("identity_node_id") != node["attrs"].get("identity_ref"):
                    raise ContractError("state field resolved identity differs")
                if (reference.get("basis") != node["attrs"].get("identity_basis") or
                        reference.get("basis_path") != node["attrs"].get("identity_basis_path") or
                        not reference.get("basis") or not reference.get("basis_path")):
                    raise ContractError("state field identity authority differs")
            elif reference.get("identity_node_id") is not None:
                raise ContractError("unresolved state field has a resolved identity node")
        if item["source_file"] != node.get("source_file") or item["source_line"] != node.get("source_line"):
            raise ContractError("state field provenance differs from source row")
        if isinstance(item["value"], float) and (item["value"] != item["value"]):
            raise ContractError("NaN numeric field")
        catalog_id = item.get("catalog_id")
        if catalog_id not in self.fields:
            raise ContractError(f"state field absent from catalog: {catalog_id}")
        declared = self.fields[catalog_id]
        if declared.get("owner") != node["node_type"]:
            raise ContractError(f"field owner differs from node type: {catalog_id}")
        if item["path"] != declared["field_path"]:
            raise ContractError(f"state field path differs from catalog: {catalog_id}")
        raw = node.get("attrs", {}).get("raw")
        source_tokens = item.get("source_tokens")
        input_tokens = item.get("input_tokens")
        if (not isinstance(raw, dict) or not isinstance(source_tokens, list) or
                not isinstance(input_tokens, list) or
                any(type(token) not in {str, int} for token in source_tokens + input_tokens)):
            raise ContractError(f"state field source location is missing: {catalog_id}")
        family, _ = _source_family(node)
        if family is None or declared["source_family"] != family:
            raise ContractError(f"state field source family differs from record: {catalog_id}")
        if (item["path"] != expected.field_path or
                catalog_id != source_field_id(family, expected.field_path,
                                              expected.source_branch, expected.source_subtype) or
                not _same_typed(source_tokens, list(expected.source_tokens)) or
                not _same_typed(input_tokens, list(expected.input_tokens)) or
                not _same_typed(item.get("element_positions"), list(expected.element_positions)) or
                item.get("element_path") != expected.raw_path or
                item.get("predicate_id") != expected.predicate_id or
                item.get("source_path_key") != expected.source_path_key):
            raise ContractError(f"state field source location differs from record: {catalog_id}")
        try:
            source_value = source_value_at_tokens(raw, tuple(input_tokens))
        except (KeyError, IndexError, TypeError) as error:
            raise ContractError(f"state field input location is invalid: {catalog_id}") from error
        if (not _same_typed(source_value, item["value"]) or
                not _same_typed(expected.value, item["value"])):
            raise ContractError(f"state field input value differs: {catalog_id}")
        if item.get("value_type") not in declared["value_types"]:
            raise ContractError(f"state field type differs from catalog: {catalog_id}")
        actual_type = ("null" if item["value"] is None else "bool" if type(item["value"]) is bool else
                       "int" if type(item["value"]) is int else "float" if type(item["value"]) is float else
                       "string" if type(item["value"]) is str else "empty_list" if item["value"] == [] else
                       "empty_object" if item["value"] == {} else None)
        if item.get("value_type") != actual_type:
            raise ContractError(f"state field value differs from declared type: {catalog_id}")
        try:
            normalized_index = projected_catalog_binding(projected, self.fields)
        except ValueError as error:
            raise ContractError(f"normalized source binding differs: {catalog_id}") from error
        if (item.get("field_id") != projected.normalized_id or
                item.get("field_index") != normalized_index or
                item.get("normalized_index") != normalized_index or
                item.get("normalized_id") != projected.normalized_id or
                not _same_typed(item.get("normalized_value"), projected.value) or
                item.get("normalized_value_type") != projected.value_type or
                item.get("value_status") != projected.value_status or
                item.get("structure_status") != projected.structure_status or
                item.get("binding_slot") != projected.binding_slot or
                not _same_typed(item.get("declared_shape"),
                                list(projected.declared_shape) if projected.declared_shape is not None else None) or
                not _same_typed(item.get("observed_shape"),
                                list(projected.observed_shape) if projected.observed_shape is not None else None) or
                not _same_typed(item.get("normalized_shape"),
                                list(projected.normalized_shape) if projected.normalized_shape is not None else None) or
                not _same_typed(item.get("catalog_ids"),
                                [source.source_field_id for source in projected.sources]) or
                item.get("record_observed") != projected.record_observed or
                not _same_typed(item.get("envelope_tokens"), list(projected.envelope_tokens)) or
                not _same_typed(item.get("payload_tokens"), list(projected.payload_tokens)) or
                not _same_typed(item.get("payload_positions"), list(projected.payload_positions)) or
                not _same_typed(item.get("payload_shape"), list(projected.payload_shape)) or
                not _same_typed(item.get("sources"),
                                projected_source_rows(projected, catalog=self.fields))):
            raise ContractError(f"normalized field projection differs from source: {catalog_id}")
        for source in projected.sources:
            source_id = source.source_field_id
            source_declared = self.fields.get(source_id)
            if source_declared is None or source_declared["owner"] != node["node_type"]:
                raise ContractError(f"normalized source lacks its catalog owner: {source_id}")
            original = source_value_at_tokens(raw, source.field_value.input_tokens)
            if not _same_typed(original, source.field_value.value):
                raise ContractError(f"normalized source leaf differs from record: {source_id}")
        if item["unit_status"] != declared["unit_status"] or item["unit"] != declared.get("unit"):
            raise ContractError(f"unit status or value differs from catalog: {catalog_id}")
        if item.get("unit_basis") != declared.get("unit_basis"):
            raise ContractError(f"unit authority differs from catalog: {catalog_id}")
        semantic_rows = [self.fields[source.source_field_id] for source in projected.sources]
        semantic_pairs = {(source_row["semantic_id"], source_row["meaning_status"])
                          for source_row in semantic_rows}
        semantic_pair = next(iter(semantic_pairs)) if len(semantic_pairs) == 1 else None
        semantic_id = (semantic_pair[0] if semantic_pair is not None and
                       semantic_pair[1] == "declared" and
                       projected.structure_status != "structure_conflict" else None)
        if (item.get("semantic_id") != semantic_id or
                item.get("semantic_binding_status") !=
                    ("declared" if semantic_id is not None else "unbound") or
                not _same_typed(item.get("source_semantic_ids"),
                                [source_row["semantic_id"] for source_row in semantic_rows]) or
                item.get("source_unit") != declared["source_unit"] or
                item.get("canonical_unit") != declared["canonical_unit"]):
            raise ContractError(f"semantic or unit mapping differs from catalog: {catalog_id}")
        context = coordinate_context_for_value(declared["source_family"], item["path"],
                                               item["source_file"], node.get("attrs", {}).get("raw") or {})
        if item.get("coordinate_context") != context:
            raise ContractError(f"coordinate context differs from source: {catalog_id}")
        if context is None and declared.get("coordinate_status") == "declared":
            raise ContractError(f"declared coordinate field lacks source context: {catalog_id}")
        if context is not None and (
                declared.get("coordinate_status") != "declared" or
                declared.get("coordinate_contract_id") != context["coordinate_contract_id"] or
                declared.get("coordinate_frame") != context["coordinate_frame"] or
                declared.get("coordinate_axes") != context["axis_order"]):
            raise ContractError(f"coordinate definition differs from source: {catalog_id}")

    def validate_edge(self, edge: dict[str, Any], source: dict[str, Any], target: dict[str, Any], *, draft: bool = False) -> None:
        required = ("edge_id", "sample_key", "edge_type", "source", "target", "role", "source_file", "source_replay_status")
        missing = [key for key in required if key not in edge]
        if missing and not draft:
            raise ContractError(f"edge missing {missing}")
        kind = edge.get("edge_type")
        if not draft and any(
            item.get("node_type") in {"EntityIdentity", "EntityState"}
            and item.get("attrs", {}).get("identity_status") != "resolved"
            for item in (source, target)
        ) and edge.get("source_replay_status") != "unresolved_identity":
            raise ContractError("edge with unresolved entity must be marked unresolved_identity")
        if kind not in EDGE_ENDPOINTS:
            raise ContractError(f"unknown edge type {kind}")
        allowed_sources, allowed_targets = EDGE_ENDPOINTS[kind]
        if source.get("node_type") not in allowed_sources or target.get("node_type") not in allowed_targets:
            raise ContractError(f"{kind} cannot connect {source.get('node_type')} -> {target.get('node_type')}")
        if edge.get("source") != source.get("node_id") or edge.get("target") != target.get("node_id"):
            raise ContractError("edge endpoint ID mismatch")
        if len({edge.get("sample_key"), source.get("sample_key"), target.get("sample_key")}) != 1:
            raise ContractError("cross-sample edge")
        s_tick, t_tick = source.get("tick"), target.get("tick")
        if kind == "at_time" and s_tick != t_tick:
            raise ContractError("time anchor tick differs")
        if kind == "at_time" and edge.get("role") not in {"target_tick", "detection_tick", "phase_tick",
                                                           "terminal_tick", "trigger_tick", "dispatch_tick",
                                                           "scheduled_tick"}:
            raise ContractError("invalid time role")
        if kind in {"same_entity_next", "same_predicate_next", "candidate_scope_change", "RelativeState"}:
            if not isinstance(s_tick, int) or not isinstance(t_tick, int):
                raise ContractError("temporal edge needs two ticks")
        if kind in {"same_entity_next", "same_predicate_next"} and not s_tick < t_tick:
            raise ContractError("successor tick must increase")
        if kind == "candidate_scope_change" and s_tick > t_tick:
            raise ContractError("candidate scope change goes backward")
        if kind == "same_entity_next" and (
            source.get("attrs", {}).get("identity_status") != "resolved" or
            target.get("attrs", {}).get("identity_status") != "resolved" or
            source.get("attrs", {}).get("identity_ref") != target.get("attrs", {}).get("identity_ref")
        ):
            raise ContractError("entity successor identity differs or is unresolved")
        if kind == "same_predicate_next" and (
            source.get("category_id") != target.get("category_id") or
            source.get("attrs", {}).get("bindings") != target.get("attrs", {}).get("bindings")
        ):
            raise ContractError("predicate successor binding differs")
        if kind == "identity_state" and (
            source.get("attrs", {}).get("identity_status") != "resolved" or
            target.get("attrs", {}).get("identity_status") != "resolved" or
            target.get("attrs", {}).get("identity_ref") != source.get("node_id")
        ):
            raise ContractError("state identity differs or is unresolved")
        if kind == "compute_node_host":
            attrs = source.get("attrs", {})
            if (source.get("category_id") != "compute_node" or
                    attrs.get("identity_namespace") != "compute_node" or
                    attrs.get("source_host_entity_id") != target.get("attrs", {}).get("entity_id") or
                    target.get("category_id") not in {"uav", "vehicle"} or
                    target.get("attrs", {}).get("identity_namespace") != "sample"):
                raise ContractError("compute-node host relation differs from world scope")
        if kind == "matrix_scope_entity":
            index = edge.get("attrs", {}).get("source_scope_index")
            source_attrs = source.get("attrs", {})
            values = source_attrs.get("_scope_values")
            if values is None:
                values = [(item["scope_type"], item["scope_entity_id"])
                          for item in source_attrs.get("raw", {}).get("scope_values", [])]
            if type(index) is not int or index < 0 or index >= len(values):
                raise ContractError("matrix scope relation lacks source index")
            scope_type, scope_id = values[index]
            if (scope_type not in {"uav", "compute_node"} or
                    scope_id != target.get("attrs", {}).get("entity_id") or
                    scope_type != edge.get("role") or
                    target.get("category_id") != ("compute_node" if scope_type == "compute_node" else "uav") or
                    target.get("attrs", {}).get("identity_namespace") !=
                    ("compute_node" if scope_type == "compute_node" else "sample")):
                raise ContractError("matrix scope relation differs from source value")
        if kind == "compute_state_node":
            source_attrs = source.get("attrs", {})
            compute_ref = source_attrs.get("_compute_ref")
            if compute_ref is None:
                raw = source_attrs.get("raw", {})
                compute_ref = (raw.get("node_id"), raw.get("entity_id"))
            attrs = target.get("attrs", {})
            if (compute_ref[0] != attrs.get("entity_id") or
                    compute_ref[1] != attrs.get("source_host_entity_id") or
                    attrs.get("identity_namespace") != "compute_node"):
                raise ContractError("compute state node relation differs from source row")
        if kind == "controller_state":
            controller = target.get("attrs", {}).get("entity_id")
            if (source.get("category_id") != "sumo_controller_state" or
                    controller not in source.get("attrs", {}).get("controller_ids", ()) or
                    target.get("attrs", {}).get("identity_namespace") != "sumo_tls_controller" or
                    edge.get("role") != "sumo_tls_controller" or
                    edge.get("attrs", {}).get("source_path") != f"sumo_traffic_light_states.{controller}"):
                raise ContractError("SUMO controller relation differs from source state")
        if kind == "action_target":
            if (source.get("attrs", {}).get("target_entity_id") != target.get("attrs", {}).get("entity_id") or
                    target.get("attrs", {}).get("identity_namespace") != "sample" or
                    edge.get("role") != "target"):
                raise ContractError("command target differs from source ID")
        if kind == "action_plan_dispatch":
            if (source.get("attrs", {}).get("native_action_id") != target.get("attrs", {}).get("native_action_id") or
                    source.get("category_id") != target.get("category_id") or
                    source.get("attrs", {}).get("audit_status") != "ok" or
                    target.get("attrs", {}).get("origin") != "source_script" or
                    source.get("attrs", {}).get("_action_payload") != target.get("attrs", {}).get("_action_payload")):
                raise ContractError("plan and dispatched command differ")
        if kind == "action_plan_audit":
            if (source.get("attrs", {}).get("native_action_id") != target.get("attrs", {}).get("native_action_id") or
                    source.get("attrs", {}).get("audit_status") != target.get("attrs", {}).get("dispatch_status") or
                    source.get("attrs", {}).get("audit_source_file") != target.get("source_file") or
                    source.get("attrs", {}).get("audit_source_line") != target.get("source_line") or
                    target.get("attrs", {}).get("origin") != "source_script" or
                    target.get("category_id") != "arm_action_audit" or
                    source.get("attrs", {}).get("_action_payload") != target.get("attrs", {}).get("_action_payload")):
                raise ContractError("plan and command audit differ")
        if kind == "observed_dispatch" and source.get("source_file", "").endswith("/event_realization.jsonl"):
            index = target.get("attrs", {}).get("source_action_index")
            actions = source.get("attrs", {}).get("_action_refs")
            if (source.get("attrs", {}).get("native_event_id") != target.get("attrs", {}).get("native_event_id") or
                    type(index) is not int or actions is None or not 0 <= index < len(actions) or
                    actions[index] != target.get("attrs", {}).get("_action_payload") or
                    actions[index].get("dispatch_tick") != target.get("tick")):
                raise ContractError("observed dispatch differs from source event action")
        if kind == "transition_from" and (
                source.get("attrs", {}).get("truth_status") != target.get("attrs", {}).get("from_value")
                or source.get("tick") != target.get("attrs", {}).get("from_tick")):
            raise ContractError("transition before value or tick differs")
        if kind == "transition_to" and (
                source.get("attrs", {}).get("to_value") != target.get("attrs", {}).get("truth_status")
                or target.get("tick") != source.get("attrs", {}).get("to_tick")):
            raise ContractError("transition after value or tick differs")
        if kind == "predicate_argument":
            predicate = self.predicates[source["category_id"]]
            role = edge.get("role")
            roles = {item["key"]: item for item in predicate["roles"]}
            if role not in roles:
                raise ContractError(f"predicate role {role} invalid for {source['category_id']}")
            bound = source.get("attrs", {}).get("bindings", {}).get(role)
            target_id = target.get("attrs", {}).get("entity_id")
            if bound is not None and bound != target_id:
                raise ContractError("predicate argument entity differs from binding")
            bindings = source.get("attrs", {}).get("bindings") or {}
            for group in predicate.get("distinct_role_sets", []):
                if role in group and any(bindings.get(other) == target_id for other in group
                                         if other != role and bindings.get(other) is not None):
                    raise ContractError("distinct predicate roles share an entity")
            for group in predicate.get("unordered_role_sets", []):
                if role in group:
                    values = [target_id if item == role else bindings.get(item) for item in group]
                    if all(value is not None for value in values) and values != sorted(values):
                        raise ContractError("unordered role set is not canonical")
            if source["category_id"] == "x.geometry.entity_proximity":
                raw_id = source.get("attrs", {}).get("raw_predicate_id")
                mapping = self.raw_predicates.get(raw_id)
                if mapping is None:
                    raise ContractError("geometry trigger ID is not registered")
                trigger_role = "entity_a" if role == "first" else "entity_b"
                if target_id != mapping["rule"][trigger_role]:
                    raise ContractError(f"geometry {role} must bind {mapping['rule'][trigger_role]}")
            declared_class = roles[role]["class"]
            declared_ontology = declared_class if ":" in declared_class else "world:" + declared_class
            if (declared_ontology in BUSINESS_SOURCE_OBJECTS and target["node_type"] == "ContextAnchor"
                    and not business_object_binding_allowed(target, bound, declared_class,
                        edge.get("attrs", {}).get("source_object_evidence"), s_tick)):
                raise ContractError("business predicate argument lacks exact same-tick state evidence")
            actual_class = (source.get("attrs", {}).get("binding_ontology_classes") or {}).get(role)
            target_class = target.get("attrs", {}).get("ontology_class")
            if actual_class and target_class and not (
                _class_compatible(actual_class, target_class) or _class_compatible(target_class, actual_class)
            ) and (draft or edge.get("attrs", {}).get("ontology_claim_status") != "source_conflict"):
                raise ContractError(f"binding class conflicts with target entity class: {actual_class} vs {target_class}")
            if not actual_class:
                actual_class = target_class
            if not actual_class:
                if draft or (edge.get("source_replay_status") != "unresolved_role_class" and
                             edge.get("attrs", {}).get("ontology_role_status") != "unresolved"):
                    raise ContractError(f"unresolved ontology class for predicate role {role}")
            elif not _class_compatible(actual_class, declared_class):
                raise ContractError(f"role class mismatch: {declared_class} vs {actual_class}")
            allowed_classes = roles[role].get("allowed_classes")
            if allowed_classes and actual_class and not any(_class_compatible(actual_class, item) for item in allowed_classes):
                raise ContractError(f"role class {actual_class} outside allowed union {allowed_classes}")
            if s_tick is not None and t_tick is not None and s_tick != t_tick:
                raise ContractError("predicate argument state tick differs")
        if kind == "event_participant" and source.get("node_type") == "EventOccurrence":
            event = self.events[source["category_id"]]
            roles = {item["key"]: item["class"] for item in event["roles"]}
            if roles and edge.get("role") not in roles:
                raise ContractError("event participant role invalid")
            role = edge.get("role")
            policy_roles = entity_scope_contract()["role_role_kinds"]
            target_attrs = target.get("attrs", {})
            if business_object_binding_allowed(target,
                    source.get("attrs", {}).get("bindings", {}).get(role),
                    roles.get(role, ""), edge.get("attrs", {}).get("source_object_evidence"), s_tick):
                policy_allowed = True
            elif roles.get(role, "").removeprefix("world:") == "ComputeNode":
                policy_allowed = (target.get("category_id") == "compute_node" and
                                  target_attrs.get("identity_namespace") == "compute_node")
                if target_attrs.get("identity_status") != "resolved":
                    policy_allowed = None
            elif event.get("role_policy", {}).get(role) == "actor" or role in policy_roles["actor_roles"]:
                policy_allowed = target.get("attrs", {}).get("event_actor_allowed")
            elif role in policy_roles["non_actor_roles"]:
                policy_allowed = target.get("attrs", {}).get("event_non_actor_allowed")
            else:
                policy_allowed = None
            policy_status = edge.get("attrs", {}).get("role_policy_status")
            if policy_allowed is None and (draft or policy_status != "unresolved"):
                raise ContractError(f"event role {role} has unresolved binding policy")
            if policy_allowed is False and (draft or policy_status != "source_conflict"):
                raise ContractError(f"event role {role} conflicts with roster binding policy")
            if role in roles:
                bound = source.get("attrs", {}).get("bindings", {}).get(role)
                target_id = target.get("attrs", {}).get("entity_id")
                if bound is not None and bound != target_id:
                    raise ContractError("event participant entity differs from binding")
                actual = (source.get("attrs", {}).get("binding_ontology_classes") or {}).get(role) or target.get("attrs", {}).get("ontology_class")
                bound_class = (source.get("attrs", {}).get("binding_ontology_classes") or {}).get(role)
                target_class = target.get("attrs", {}).get("ontology_class")
                if bound_class and target_class and not (
                    _class_compatible(bound_class, target_class) or _class_compatible(target_class, bound_class)
                ) and (draft or edge.get("attrs", {}).get("ontology_claim_status") != "source_conflict"):
                    raise ContractError(f"event binding class conflicts with target entity class: {bound_class} vs {target_class}")
                if not actual:
                    if draft or (edge.get("source_replay_status") != "unresolved_role_class" and
                                 edge.get("attrs", {}).get("ontology_role_status") != "unresolved"):
                        raise ContractError(f"unresolved ontology class for event role {role}")
                elif not _class_compatible(actual, roles[role]):
                    raise ContractError(f"event role class mismatch: {roles[role]} vs {actual}")
        if kind == "RelativeState":
            if s_tick != t_tick:
                raise ContractError("relative value endpoints need same tick")
            if not edge.get("attrs", {}).get("field") or edge["attrs"].get("unit_status") not in {"declared", "unresolved", "dimensionless"}:
                raise ContractError("relative value needs field and unit status")
            if edge["attrs"].get("unit_status") == "declared" and not edge["attrs"].get("unit_basis"):
                raise ContractError("declared relative unit needs source or derivation authority")

    def validate_graph(self, nodes: Iterable[dict[str, Any]], edges: Iterable[dict[str, Any]], *,
                       draft: bool = False, complete_node_ids: set[str] | None = None,
                       nodes_prevalidated: bool = False) -> dict[str, int]:
        node_index: dict[str, dict[str, Any]] = {}
        for node in nodes:
            if not nodes_prevalidated:
                self.validate_node(node, draft=draft)
            if node["node_id"] in node_index:
                raise ContractError(f"duplicate node {node['node_id']}")
            node_index[node["node_id"]] = node
        edge_ids: set[str] = set()
        argument_roles: dict[str, Counter[str]] = {}
        event_roles: dict[str, Counter[str]] = {}
        controller_targets: dict[str, Counter[str]] = {}
        plan_links: dict[str, Counter[str]] = {}
        observed_dispatches: dict[str, Counter[int]] = {}
        observed_targets: Counter[str] = Counter()
        action_targets: Counter[str] = Counter()
        time_anchors: Counter[str] = Counter()
        for edge in edges:
            if edge["edge_id"] in edge_ids:
                raise ContractError(f"duplicate edge {edge['edge_id']}")
            edge_ids.add(edge["edge_id"])
            try:
                source, target = node_index[edge["source"]], node_index[edge["target"]]
            except KeyError as exc:
                raise ContractError(f"dangling edge {edge['edge_id']}") from exc
            self.validate_edge(edge, source, target, draft=draft)
            if edge["edge_type"] == "predicate_argument":
                argument_roles.setdefault(edge["source"], Counter())[edge["role"]] += 1
            if edge["edge_type"] == "event_participant" and source["node_type"] == "EventOccurrence":
                event_roles.setdefault(edge["source"], Counter())[edge["role"]] += 1
            if edge["edge_type"] == "controller_state":
                controller_targets.setdefault(edge["source"], Counter())[edge["target"]] += 1
            if edge["edge_type"] in {"action_plan_audit", "action_plan_dispatch"}:
                plan_links.setdefault(edge["source"], Counter())[edge["edge_type"]] += 1
            if edge["edge_type"] == "observed_dispatch" and source["source_file"].endswith(
                    "/event_realization.jsonl"):
                observed_dispatches.setdefault(edge["source"], Counter())[
                    target["attrs"]["source_action_index"]] += 1
                observed_targets[edge["target"]] += 1
            if edge["edge_type"] == "action_target" and source["node_type"] == "CommandDispatch":
                action_targets[edge["source"]] += 1
            if edge["edge_type"] == "at_time":
                time_anchors[edge["source"]] += 1
        if not draft:
            for node in node_index.values():
                if complete_node_ids is not None and node["node_id"] not in complete_node_ids:
                    continue
                if (node["node_type"] in {"EntityState", "PredicateAssertion", "EventPhase", "CommandDispatch"}
                        or node["node_type"] == "ContextAnchor" and node["category_id"] in
                        {"formal_weather_meta", "sumo_controller_state", "arm_action_audit"}):
                    if time_anchors[node["node_id"]] != 1:
                        raise ContractError(f"node time anchor count differs: {node['node_id']}")
                if node["node_type"] == "ContextAnchor" and node["category_id"] == "sumo_controller_state":
                    expected = Counter({identity_node_id(node["sample_key"], "sumo_tls_controller", item): 1
                                        for item in node["attrs"]["controller_ids"]})
                    if controller_targets.get(node["node_id"], Counter()) != expected:
                        raise ContractError(f"SUMO controller state edges differ: {node['node_id']}")
                if node["node_type"] == "ActionPlan":
                    status = node["attrs"]["audit_status"]
                    expected = Counter({"action_plan_audit": 1, "action_plan_dispatch": 1}) if status == "ok" else (
                        Counter({"action_plan_audit": 1}) if status != "not_recorded" else Counter())
                    if plan_links.get(node["node_id"], Counter()) != expected:
                        raise ContractError(f"action plan audit/dispatch edges differ: {node['node_id']}")
                if node["node_type"] == "ScriptEvent" and node["source_file"].endswith(
                        "/event_realization.jsonl"):
                    expected = Counter(range(len(node["attrs"]["_action_refs"])))
                    if observed_dispatches.get(node["node_id"], Counter()) != expected:
                        raise ContractError(f"formal observed dispatch edges differ: {node['node_id']}")
                if node["node_type"] == "CommandDispatch":
                    expected = 1 if node["attrs"].get("target_entity_id") else 0
                    if action_targets[node["node_id"]] != expected:
                        raise ContractError(f"command target edges differ: {node['node_id']}")
                    if (node["source_file"].endswith("/event_realization.jsonl") and
                            observed_targets[node["node_id"]] != 1):
                        raise ContractError(f"formal command source edge differs: {node['node_id']}")
                if node["node_type"] == "PredicateAssertion" and (complete_node_ids is None or node["node_id"] in complete_node_ids):
                    signature = self.predicates[node["category_id"]]
                    actual = argument_roles.get(node["node_id"], Counter())
                    expected = Counter({item["key"]: 1 for item in signature["roles"]})
                    if actual != expected:
                        raise ContractError(f"predicate arity/roles differ for {node['node_id']}: {actual} != {expected}")
                    bindings = node["attrs"].get("bindings", {})
                    for group in signature.get("distinct_role_sets", []):
                        if len({bindings.get(role) for role in group}) != len(group):
                            raise ContractError("distinct predicate roles share an entity")
                    for group in signature.get("unordered_role_sets", []):
                        if [bindings.get(role) for role in group] != sorted(bindings.get(role) for role in group):
                            raise ContractError("unordered role set is not canonical")
                if node["node_type"] == "EventOccurrence":
                    signature = self.events[node["category_id"]]
                    if signature["roles"]:
                        actual = event_roles.get(node["node_id"], Counter())
                        expected = Counter(item["key"] for item in signature["roles"])
                        if actual != expected:
                            raise ContractError(f"event roles differ for {node['node_id']}: {actual} != {expected}")
        return {"nodes": len(node_index), "edges": len(edge_ids)}

    def allowed_targets(self, source: dict[str, Any], edge_type: str, role: str | None,
                        candidates: Iterable[dict[str, Any]],
                        existing_edges: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
        single_role = edge_type in {"predicate_argument", "event_participant"}
        single_successor = edge_type in {"at_time", "same_entity_next", "same_predicate_next"}
        occupied = any(edge.get("source") == source.get("node_id") and
                       edge.get("edge_type") == edge_type and
                       (single_successor or (single_role and edge.get("role") == role))
                       for edge in existing_edges)
        result = []
        for target in candidates:
            edge = {"edge_id": "draft", "sample_key": source.get("sample_key"), "edge_type": edge_type,
                    "source": source.get("node_id"), "target": target.get("node_id"), "role": role,
                    "source_file": "draft", "source_replay_status": "draft", "attrs": {}}
            try:
                if occupied:
                    raise ContractError(f"{edge_type} {role or ''} already assigned")
                self.validate_edge(edge, source, target, draft=True)
            except (ContractError, KeyError) as exc:
                result.append({"node_id": target.get("node_id"), "allowed": False, "reason": str(exc)})
            else:
                result.append({"node_id": target.get("node_id"), "allowed": True, "reason": None})
        return result


def validate_node(node: dict[str, Any], *, draft: bool = False) -> None:
    GraphValidator().validate_node(node, draft=draft)


def validate_edge(edge: dict[str, Any], source: dict[str, Any], target: dict[str, Any], *, draft: bool = False) -> None:
    GraphValidator().validate_edge(edge, source, target, draft=draft)


def validate_graph(nodes: Iterable[dict[str, Any]], edges: Iterable[dict[str, Any]], *, draft: bool = False) -> dict[str, int]:
    return GraphValidator().validate_graph(nodes, edges, draft=draft)


def allowed_targets(source: dict[str, Any], edge_type: str, role: str | None,
                    candidates: Iterable[dict[str, Any]],
                    existing_edges: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    return GraphValidator().allowed_targets(source, edge_type, role, candidates, existing_edges)
