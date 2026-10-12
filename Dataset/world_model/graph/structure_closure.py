"""Resolve complete record grammars by source, effective type, and profile.

Declaration queries reject unreviewed combinations. The source reader checks
registered complete records before projecting their fields; selected-subtree
and source-projection roles remain separate.
"""

from functools import cache
from importlib import import_module
import json
from pathlib import Path
import re
from typing import Any, Callable

from Dataset.world_model.graph.published_structure import published_record_schema

from Dataset.world_model.graph.authoring_structure import (
    AUTHORING_SOURCE_FAMILIES, authoring_source_grammar,
    validate_authoring_source_record,
)
from Dataset.world_model.graph.arm_actions_x_structure import (
    arm_actions_schema as arm_actions_x_schema,
)
from Dataset.world_model.graph.arm_actions_l2_structure import (
    arm_actions_schema as arm_actions_l2_schema,
)
from Dataset.world_model.graph.arm_actions_l5_structure import (
    arm_actions_schema as arm_actions_l5_schema,
)
from Dataset.world_model.graph.arm_states_l1_structure import (
    arm_states_schema as arm_states_l1_schema,
)
from Dataset.world_model.graph.arm_states_l5_structure import (
    arm_states_schema as arm_states_l5_schema,
)
from Dataset.world_model.graph.arm_states_l6_structure import (
    arm_states_schema as arm_states_l6_schema,
)
from Dataset.world_model.graph.arm_states_x_structure import (
    arm_states_schema as arm_states_x_schema,
)
from Dataset.world_model.graph.arm_trajectories_l1_structure import (
    arm_trajectories_schema as arm_trajectories_l1_schema,
)
from Dataset.world_model.graph.arm_trajectories_l2_structure import (
    arm_trajectories_schema as arm_trajectories_l2_schema,
)
from Dataset.world_model.graph.arm_trajectories_l5_structure import (
    arm_trajectories_schema as arm_trajectories_l5_schema,
)
from Dataset.world_model.graph.arm_trajectories_l6_structure import (
    arm_trajectories_schema as arm_trajectories_l6_schema,
)
from Dataset.world_model.graph.arm_predicate_truth_l2_structure import (
    arm_predicate_truth_schema as arm_predicate_truth_l2_schema,
)
from Dataset.world_model.graph.arm_predicate_truth_l3_structure import (
    arm_predicate_truth_schema as arm_predicate_truth_l3_schema,
)
from Dataset.world_model.graph.arm_predicate_truth_l4_structure import (
    arm_predicate_truth_schema as arm_predicate_truth_l4_schema,
)
from Dataset.world_model.graph.arm_predicate_truth_l5_structure import (
    arm_predicate_truth_schema as arm_predicate_truth_l5_schema,
)
from Dataset.world_model.graph.arm_predicate_truth_x_structure import (
    arm_predicate_truth_schema as arm_predicate_truth_x_schema,
)
from Dataset.world_model.graph.arm_predicate_ticks_l1_structure import (
    arm_predicate_truth_ticks_schema as arm_predicate_ticks_l1_schema,
)
from Dataset.world_model.graph.arm_predicate_ticks_l2_structure import (
    arm_predicate_truth_ticks_schema as arm_predicate_ticks_l2_schema,
)
from Dataset.world_model.graph.arm_predicate_ticks_x_structure import (
    arm_predicate_truth_ticks_schema as arm_predicate_ticks_x_schema,
)
from Dataset.world_model.graph.arm_predicate_transition_l1_structure import (
    arm_predicate_transitions_schema as arm_predicate_transition_l1_schema,
)
from Dataset.world_model.graph.arm_predicate_transition_l2_structure import (
    arm_predicate_transitions_schema as arm_predicate_transition_l2_schema,
)
from Dataset.world_model.graph.contract import REPO
from Dataset.world_model.graph.initial_events_structure import initial_events_schema
from Dataset.world_model.graph.compute_structure import compute_state_record_schema
from Dataset.world_model.graph.compute_comm_structure import (
    communication_state_record_schema, compute_comm_event_record_schema,
)
from Dataset.world_model.graph.event_structure import (
    formal_event_occurrence_schema, formal_event_outcome_schema,
)
from Dataset.world_model.graph.field_catalog import (
    SourceDeclaration, _arm_source_applicable, declared_source_inventory,
)
from Dataset.world_model.graph.sample_identity import manifest_reference
from Dataset.world_model.graph.formal_roster_l4_structure import formal_roster_schema
from Dataset.world_model.graph.formal_roster_l3_structure import (
    formal_roster_schema as formal_roster_l3_schema,
)
from Dataset.world_model.graph.formal_roster_l6_structure import (
    formal_roster_schema as formal_roster_l6_schema,
)
from Dataset.world_model.graph.formal_weather_l1_structure import (
    formal_weather_meta_schema,
)
from Dataset.world_model.graph.formal_weather_l3_structure import (
    formal_weather_meta_schema as formal_weather_l3_schema,
)
from Dataset.world_model.graph.formal_weather_l5_structure import (
    formal_weather_meta_schema as formal_weather_l5_schema,
)
from Dataset.world_model.graph.formal_weather_l6_structure import (
    formal_weather_meta_schema as formal_weather_l6_schema,
)
from Dataset.world_model.graph.formal_weather_x_structure import (
    formal_weather_meta_schema as formal_weather_x_schema,
)
from Dataset.world_model.graph.predicate_structure import (
    formal_predicate_truth_schema,
    predicate_continuity_break_schema,
    predicate_transition_schema,
)
from Dataset.world_model.graph.history_structure import (
    _L6_PROFILES as HISTORY_L6_PROFILES,
    arm_semantic_history_schema,
)
from Dataset.world_model.graph.supplement_predicate_structure import (
    predicate_truth_record_schema,
    predicate_truth_matrix_record_schema,
)
from Dataset.world_model.graph.world_truth_structure import (
    world_truth_base_schema, world_truth_delta_schema,
)
from Dataset.world_model.graph.structure_schema import (
    ClosedRecordGrammar, compile_record_grammar, validate_record,
)


# Explicit producer modules for the family/level work.  A callable is resolved
# only for a declared source and is never selected from a source row's shape.
_CANDIDATE_MODULES = {
    "arm_actions": ("arm_actions", "arm_actions_schema"),
    "arm_states": ("arm_states", "arm_states_schema"),
    "arm_trajectories": ("arm_trajectories", "arm_trajectories_schema"),
    "formal_roster": ("formal_roster", "formal_roster_schema"),
    "formal_truth_frame": ("formal_truth_frame", "formal_truth_frame_schema"),
    "formal_weather_meta": ("formal_weather", "formal_weather_meta_schema"),
    "arm_predicate_truth": ("arm_predicate_truth", "arm_predicate_truth_schema"),
    "arm_predicate_truth_ticks": ("arm_predicate_ticks", "arm_predicate_truth_ticks_schema"),
    "arm_predicate_transitions": ("arm_predicate_transition", "arm_predicate_transitions_schema"),
    "arm_event_occurrences": ("arm_event_occurrence", "arm_event_occurrences_schema"),
    "arm_event_outcomes": ("arm_event_outcome", "arm_event_outcomes_schema"),
    "arm_local_business_truth": ("arm_local_business", "arm_local_business_truth_schema"),
    "formal_event_realization": ("formal_realization", "formal_event_realization_schema"),
    "arm_chain_events": ("arm_chain_event", "arm_chain_events_schema"),
    "arm_chain_nodes": ("arm_chain_node", "arm_chain_nodes_schema"),
    "arm_chain_relations": ("arm_chain_relation", "arm_chain_relations_schema"),
    "formal_event_chain_relations": ("formal_chain", "formal_event_chain_relations_schema"),
    "arm_branch_roster": ("arm_branch_roster", "arm_branch_roster_schema"),
    "arm_manifest": ("arm_manifest", "arm_manifest_schema"),
    "arm_script_plan": ("arm_script_plan", "arm_script_plan_schema"),
    "arm_scene_setup": ("arm_scene_setup", "arm_scene_setup_schema"),
    "formal_episode_manifest": ("formal_episode_manifest", "formal_episode_manifest_schema"),
    "formal_objective_manifest": ("formal_objective_manifest", "formal_objective_manifest_schema"),
    "domain_state": ("domain_state", "domain_state_schema"),
}
_CANDIDATE_ARM_SUFFIXES = {
    "arm_chain_events": "chain/events.jsonl",
    "arm_chain_nodes": "chain/nodes.jsonl",
    "arm_chain_relations": "chain/relations.jsonl",
    "arm_local_business_truth": "objective/local_business_truth.jsonl",
    "arm_event_occurrences": "objective/event_occurrences.jsonl",
    "arm_event_outcomes": "objective/event_outcomes.jsonl",
    "arm_branch_roster": "ue/global_entity_roster.json",
    "arm_manifest": "manifest.json",
    "arm_script_plan": "ue/event_script.json",
}
_CANDIDATE_FORMAL_PATHS = {
    "formal_truth_frame": ("render_ready_episodes_capture_filtered", "truth_frames.jsonl"),
    "formal_event_realization": ("render_ready_episodes_capture_filtered", "event_realization.jsonl"),
    "formal_episode_manifest": ("render_ready_episodes_capture_filtered", "episode_manifest.json"),
    "formal_objective_manifest": ("objective_semantic_truth", "manifest.json"),
    "formal_event_chain_relations": ("event_chains", "event_chain_relations.jsonl"),
    "domain_state": ("objective_semantic_truth", "domain_state_observations.jsonl"),
}
_MECHANICAL_ARM_FAMILIES = frozenset({
    "arm_local_business_truth", "arm_chain_events", "arm_chain_nodes",
})


def declaration_not_applicable(declaration: SourceDeclaration) -> bool:
    """Classify only producer-excluded mechanical ARM path candidates."""
    if declaration.source_family not in _MECHANICAL_ARM_FAMILIES:
        return False
    applicable = _arm_source_applicable(declaration)
    if not applicable and declaration.presence != "missing":
        raise ValueError(f"producer-excluded source unexpectedly exists: {declaration.path}")
    return not applicable


def _candidate_profile_for_declaration(declaration: SourceDeclaration) -> str | None:
    family = declaration.source_family
    if family == "arm_scene_setup":
        if not declaration.reference_samples:
            raise ValueError(f"ARM scene has no manifest references: {declaration.path}")
        for sample in declaration.reference_samples:
            if not sample.startswith("arm:") or len(sample.split("/")) != 4:
                raise ValueError(f"ARM scene has an invalid reference sample: {sample}")
            branch = REPO / "arms" / sample.removeprefix("arm:")
            manifest = json.loads((branch / "manifest.json").read_text(encoding="utf-8"))
            if manifest_reference(manifest, "scene") != declaration.path.relative_to(REPO).as_posix():
                raise ValueError(f"ARM scene differs from manifest: {declaration.path}")
        return None
    if family in _CANDIDATE_FORMAL_PATHS or family in {"formal_roster", "formal_weather_meta"}:
        profile, separator, seed = declaration.sample.rpartition("__")
        if not separator or re.fullmatch(r"seed[0-9]{2}", seed) is None:
            raise ValueError(f"formal source has no episode profile: {declaration.sample}")
        folder, name = (_CANDIDATE_FORMAL_PATHS[family]
                        if family in _CANDIDATE_FORMAL_PATHS else
                        ("render_ready_episodes_capture_filtered",
                         "global_entity_roster.json" if family == "formal_roster"
                         else "weather_meta.jsonl"))
        expected = REPO / "aw_data" / folder / declaration.sample / name
        if declaration.path != expected:
            raise ValueError(f"formal source path differs: {declaration.path}")
        return profile
    suffix = {**_ARM_DECLARATION_SUFFIXES, **_CANDIDATE_ARM_SUFFIXES}.get(family)
    if suffix is None:
        raise ValueError(f"source family has no candidate path binding: {family}")
    parts = declaration.sample.split("/")
    if len(parts) != 5 or parts[0] != "arms" or not all(parts[1:]):
        raise ValueError(f"ARM source has no branch sample: {declaration.sample}")
    if declaration.path != REPO / declaration.sample / suffix:
        raise ValueError(f"ARM source path differs: {declaration.path}")
    return parts[1]


def _candidate_level(family: str, source: str, profile: str | None) -> str:
    if family == "arm_scene_setup":
        relative = (REPO / source).resolve().relative_to(REPO)
        parts = relative.parts
        if parts[:2] == ("Dataset", "scenarios"):
            level = parts[2].split("_", 1)[0]
        elif parts[0] == "design":
            level = "L5" if parts[1].startswith("L5-") else ""
        elif parts[0] == "interventions":
            level = "X" if parts[1].startswith("X") else ""
        else:
            level = ""
        if level not in _PROFILES_BY_PRODUCER:
            raise ValueError(f"ARM scene has no declared producer level: {source}")
        return level
    if profile is None:
        relative = (REPO / source).resolve().relative_to(REPO)
        if relative.parts[0] == "arms":
            profile = relative.parts[1]
        elif len(relative.parts) > 2 and relative.parts[0] == "aw_data":
            profile = relative.parts[2].rpartition("__")[0]
        else:
            raise ValueError(f"source has no declared profile: {source}")
    if family in _CANDIDATE_FORMAL_PATHS or family in {
            "formal_roster", "formal_weather_meta"}:
        match = re.fullmatch(r"L([1-6])-[0-9]+_v[0-9]+", profile)
        if match is not None:
            return "L" + match.group(1)
        if re.fullmatch(r"X[1-6]_[a-z0-9_]+", profile):
            return "X"
        raise ValueError(f"formal source has no declared producer level: {source}")
    return _producer_for_profile(profile, source)


def _candidate_module(family: str, level: str):
    prefix, function = _CANDIDATE_MODULES[family]
    if family in {"arm_chain_events", "arm_chain_nodes", "arm_chain_relations"}:
        name = prefix + "_structure"
    elif family in {"formal_event_realization", "formal_event_chain_relations",
                    "arm_branch_roster", "formal_objective_manifest", "domain_state"}:
        name = prefix + "_l1_structure"
    elif family == "arm_script_plan" and level in {"L1", "L2", "L3", "L4", "L5"}:
        name = prefix + "_l1_structure"
    elif family == "formal_weather_meta" and level == "L4":
        name = prefix + "_l1_structure"
    else:
        name = f"{prefix}_{level.lower()}_structure"
    module = import_module("Dataset.world_model.graph." + name)
    return getattr(module, function), name


def _branch_manifest(source: str) -> tuple[Path, dict[str, Any]]:
    path = (REPO / source).resolve()
    relative = path.relative_to(REPO)
    if relative.parts[0] != "arms" or len(relative.parts) < 6:
        raise ValueError(f"ARM source is not under a branch: {source}")
    manifest_path = REPO.joinpath(*relative.parts[:5]) / "manifest.json"
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _candidate_effective_families(family: str, level: str) -> tuple[str, ...]:
    if family == "arm_chain_events":
        if level == "L2":
            return ("arm_chain_events:l2_event_occurrence",)
        if level not in {"L1", "L4", "L5"}:
            return ()
        module = import_module("Dataset.world_model.graph.arm_chain_event_structure")
        return tuple("arm_chain_events:" + kind for kind in sorted(module._KINDS[level]))
    if family == "arm_chain_nodes":
        if level not in {"L3", "L6", "X"}:
            return ()
        module = import_module("Dataset.world_model.graph.arm_chain_node_structure")
        return tuple("arm_chain_nodes:" + kind for kind in sorted(module._KINDS[level]))
    if family == "arm_predicate_truth_ticks" and level == "L4":
        return ("arm_domain_state_ticks", "arm_predicate_truth_ticks")
    return (family,)


# Immutable compiled grammars keyed by their full schema_ref
# (authority:builder(args,kwargs)).  Distinct source declarations often share one
# builder/args/kwargs, so the expensive compile_record_grammar runs once per
# distinct schema_ref rather than once per source.  In-process only: source-derived
# args/kwargs are still computed per call, so L6 retime / X entrypoint / L5
# pedestrian branches stay distinct because their arguments differ.
_COMPILED_BY_REF: dict[str, ClosedRecordGrammar] = {}


@cache
def _candidate_grammar(family: str, effective: str, source: str,
                       level: str, profile: str | None) -> ClosedRecordGrammar:
    if effective not in _candidate_effective_families(family, level):
        raise ValueError(f"unregistered effective source type: {family!r} -> {effective!r} at {level}")
    builder, module_name = _candidate_module(family, level)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}
    authority = f"Dataset/world_model/graph/{module_name}.py"
    if family in {"arm_chain_events", "arm_chain_nodes"}:
        args = (level, effective)
    elif family in {"arm_chain_relations", "formal_event_chain_relations",
                    "formal_event_realization", "domain_state", "arm_branch_roster"}:
        args = (level,)
    elif family == "arm_local_business_truth":
        if profile is None:
            raise ValueError(f"local business source has no profile: {source}")
        args = (profile,)
    elif family == "arm_scene_setup":
        # Normalize both absolute (declaration compile) and repo-relative
        # (actual validation) inputs to one repo-relative argument so the
        # scene schema_ref is identical for either call form.
        args = ((REPO / source).resolve().relative_to(REPO).as_posix(),)
    elif family == "arm_script_plan":
        manifest_path, manifest = _branch_manifest(source)
        script = manifest_reference(manifest, "script")
        if script is None:
            raise ValueError(f"ARM manifest has no script reference: {manifest_path}")
        if level in {"L1", "L2", "L3", "L4", "L5"}:
            args = (level, script)
        elif level == "L6":
            configuration = manifest_reference(manifest, "configuration")
            if configuration is None:
                raise ValueError(f"L6 manifest has no configuration reference: {manifest_path}")
            spec = json.loads((REPO / configuration).read_text(encoding="utf-8"))["spec"]
            mechanism = manifest["declared_mechanism"]
            if (tuple(spec["cancel_event_ids"]) != tuple(mechanism["cancel_event_ids"]) or
                    spec["retime"] != mechanism["retime"] or
                    manifest["provenance"]["parameter_sources"]["script"] != script):
                raise ValueError(f"L6 script declarations differ: {manifest_path}")
            args = (script,)
            kwargs = {"cancel_event_ids": tuple(spec["cancel_event_ids"]),
                      "retime": spec["retime"]}
        elif level == "X":
            import yaml
            config = REPO / manifest_path.parent.relative_to(REPO).as_posix().replace("arms/", "interventions/", 1)
            config = config.parent / (config.name + ".yaml")
            declaration = yaml.safe_load(config.read_text(encoding="utf-8"))
            if declaration["sources"]["script"] != script:
                raise ValueError(f"X script declarations differ: {manifest_path}")
            args = (script, declaration["definition"]["entrypoint"])
        else:
            raise ValueError(f"ARM script has no producer: {level}")
    elif family in {"arm_event_occurrences", "arm_event_outcomes"} and level == "L5":
        from Dataset.tools.l5_arm_common import load_config
        branch = (REPO / source).parent.parent
        config = REPO / "interventions" / branch.relative_to(REPO / "arms")
        config = config.parent / (config.name + ".yaml")
        kwargs = {"has_semantic_pedestrians": bool(load_config(config)["semantic_pedestrian_ids"])}
    elif family == "arm_predicate_truth_ticks":
        if effective == "arm_domain_state_ticks":
            builder = getattr(import_module("Dataset.world_model.graph.arm_predicate_ticks_l4_structure"),
                              "arm_domain_state_ticks_schema")
        elif level in {"L1", "L5", "L6", "X"}:
            args = (level,)
    arguments = ",".join(
        [*(repr(value) for value in args),
         *(f"{key}={value!r}" for key, value in sorted(kwargs.items()))]
    )
    schema_ref = f"{authority}:{builder.__name__}({arguments})"
    grammar = _COMPILED_BY_REF.get(schema_ref)
    if grammar is not None:
        return grammar
    schema = builder(*args, **kwargs)
    grammar = compile_record_grammar(schema, schema_ref=schema_ref, authority=authority)
    _COMPILED_BY_REF[schema_ref] = grammar
    return grammar


@cache
def _history_grammar(profile: str) -> ClosedRecordGrammar:
    """Compile the complete frozen L6 gzip-JSON root for one exact profile.

    The schema_ref carries the profile so the in-process cache cannot reuse a
    digest-cohort archive grammar for a non-digest profile or the reverse.
    """
    if not isinstance(profile, str) or profile not in HISTORY_L6_PROFILES:
        raise ValueError(f"arm_semantic_history has no reviewed L6 profile: {profile!r}")
    authority = ("Dataset/tools/l6_v2/semantics.py:producer_observations;"
                 "Dataset/semantic_simulation/domain_state.py:_observation_row;"
                 "Dataset/semantic_simulation/control_response_state.py:_observation_row")
    schema_ref = ("Dataset/world_model/graph/history_structure.py:"
                  f"arm_semantic_history_schema({profile!r})")
    grammar = _COMPILED_BY_REF.get(schema_ref)
    if grammar is not None:
        return grammar
    grammar = compile_record_grammar(arm_semantic_history_schema(profile),
                                     schema_ref=schema_ref, authority=authority)
    _COMPILED_BY_REF[schema_ref] = grammar
    return grammar


@cache
def initial_grammar(producer: str):
    return compile_record_grammar(
        initial_events_schema(producer),
        schema_ref=f"Dataset/world_model/graph/initial_events_structure.py:initial_events_schema({producer})",
        authority="Dataset/tools/l3_v2/semantics.py;Dataset/tools/l6_v2/observations.py",
    )


@cache
def compute_grammar():
    return compile_record_grammar(
        compute_state_record_schema(),
        schema_ref="Dataset/world_model/graph/compute_structure.py:compute_state_record_schema",
        authority="Dataset/semantic_simulation/compute_comm.py:_build_compute_rows;_artifact_rows",
    )


@cache
def formal_predicate_truth_grammar():
    return compile_record_grammar(
        formal_predicate_truth_schema(),
        schema_ref="Dataset/world_model/graph/predicate_structure.py:formal_predicate_truth_schema",
        authority="Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_assertions",
    )


@cache
def formal_predicate_transition_grammar():
    return compile_record_grammar(
        predicate_transition_schema(),
        schema_ref="Dataset/world_model/graph/predicate_structure.py:predicate_transition_schema",
        authority="Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_transitions",
    )


@cache
def formal_predicate_continuity_break_grammar():
    return compile_record_grammar(
        predicate_continuity_break_schema(),
        schema_ref="Dataset/world_model/graph/predicate_structure.py:predicate_continuity_break_schema",
        authority="Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_transitions",
    )


@cache
def supplement_predicate_truth_grammar():
    return compile_record_grammar(
        predicate_truth_record_schema(),
        schema_ref="Dataset/world_model/graph/supplement_predicate_structure.py:predicate_truth_record_schema",
        authority="Dataset/semantic_simulation/compute_comm.py:_predicate_row;_artifact_rows",
    )


@cache
def supplement_predicate_matrix_grammar():
    return compile_record_grammar(
        predicate_truth_matrix_record_schema(),
        schema_ref="Dataset/world_model/graph/supplement_predicate_structure.py:predicate_truth_matrix_record_schema",
        authority="Dataset/semantic_simulation/compute_comm.py:_predicate_matrix_tick_row;_artifact_rows",
    )


@cache
def communication_state_grammar():
    return compile_record_grammar(
        communication_state_record_schema(),
        schema_ref="Dataset/world_model/graph/compute_comm_structure.py:communication_state_record_schema",
        authority="Dataset/semantic_simulation/compute_comm.py:_build_communication_rows;_artifact_rows",
    )


@cache
def supplement_event_grammar():
    return compile_record_grammar(
        compute_comm_event_record_schema(),
        schema_ref="Dataset/world_model/graph/compute_comm_structure.py:compute_comm_event_record_schema",
        authority="Dataset/semantic_simulation/compute_comm.py:_build_event_rows;_artifact_rows",
    )


@cache
def formal_event_occurrence_grammar():
    return compile_record_grammar(
        formal_event_occurrence_schema(),
        schema_ref="Dataset/world_model/graph/event_structure.py:formal_event_occurrence_schema",
        authority="Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_occurrences",
    )


@cache
def formal_event_outcome_grammar():
    return compile_record_grammar(
        formal_event_outcome_schema(),
        schema_ref="Dataset/world_model/graph/event_structure.py:formal_event_outcome_schema",
        authority="Dataset/semantic_truth/minimal_semantics_adapter.py:_adapt_outcomes",
    )


@cache
def world_truth_base_grammar():
    return compile_record_grammar(
        world_truth_base_schema(),
        schema_ref="Dataset/world_model/graph/world_truth_structure.py:world_truth_base_schema",
        authority="Dataset/semantic_truth/world_truth.py:_evaluate_grounded_world_truth",
    )


@cache
def world_truth_delta_grammar():
    return compile_record_grammar(
        world_truth_delta_schema(),
        schema_ref="Dataset/world_model/graph/world_truth_structure.py:world_truth_delta_schema",
        authority="Dataset/semantic_truth/world_truth.py:_evaluate_grounded_world_truth;serialize_world_truth_deltas",
    )


CLOSED_RECORD_FAMILIES = frozenset({
    "arm_initial_events", "compute_state",
    "formal_predicate_truth", "formal_predicate_transitions",
    "formal_predicate_continuity_breaks", "predicate_truth", "predicate_truth_matrix",
    "formal_event_occurrences", "formal_event_outcomes",
    "formal_world_truth_graph_base", "formal_world_truth_graph_deltas",
    "communication_state", "events",
}) | AUTHORING_SOURCE_FAMILIES
GRAMMAR_SOURCE_FAMILIES = (CLOSED_RECORD_FAMILIES |
                           frozenset(_CANDIDATE_MODULES) |
                           {"arm_semantic_history"})

# These producers have reviewed grammars for the listed source roles only.
# The family-wide coverage flags above stay unchanged until every profile and
# effective record type in the declared inventory has a reviewed grammar.
_PROFILE_GRAMMAR_BUILDERS: dict[
    tuple[str, str, str], tuple[Callable[[], dict], str]
] = {
    ("arm_actions", "arm_actions", "X"): (
        arm_actions_x_schema, "Dataset/tools/x_arm_pipeline.py:run_arm"),
    ("arm_actions", "arm_actions", "L2"): (
        arm_actions_l2_schema, "Dataset/tools/l2_window.py:_execute"),
    ("arm_actions", "arm_actions", "L5"): (
        arm_actions_l5_schema, "Dataset/tools/l5_arm_pipeline.py:execute_config"),
    ("arm_states", "arm_states", "L1"): (
        arm_states_l1_schema, "Dataset/tools/l1_window.py:execute_config"),
    ("arm_states", "arm_states", "L5"): (
        arm_states_l5_schema, "Dataset/tools/l5_arm_common.py:states_from_rows"),
    ("arm_states", "arm_states", "L6"): (
        arm_states_l6_schema, "Dataset/tools/l6_v2/arm_runtime.py"),
    ("arm_states", "arm_states", "X"): (
        arm_states_x_schema, "Dataset/tools/x_arm_pipeline.py:run_arm"),
    ("arm_trajectories", "arm_trajectories", "L1"): (
        arm_trajectories_l1_schema, "Dataset/tools/l1_window.py:execute_config"),
    ("arm_trajectories", "arm_trajectories", "L2"): (
        arm_trajectories_l2_schema, "Dataset/tools/l2_window.py:_execute"),
    ("arm_trajectories", "arm_trajectories", "L5"): (
        arm_trajectories_l5_schema, "Dataset/tools/l5_arm_pipeline.py:execute_config"),
    ("arm_trajectories", "arm_trajectories", "L6"): (
        arm_trajectories_l6_schema, "Dataset/tools/l6_v2/arm_runtime.py:run_arm"),
    ("arm_predicate_truth", "arm_predicate_truth", "L2"): (
        arm_predicate_truth_l2_schema, "Dataset/tools/l2_arm_semantics.py:recompute"),
    ("arm_predicate_truth", "arm_predicate_truth", "L3"): (
        arm_predicate_truth_l3_schema, "Dataset/tools/l3_v2/semantics.py"),
    ("arm_predicate_truth", "arm_predicate_truth", "L4"): (
        arm_predicate_truth_l4_schema,
        "Dataset/tools/l4_window.py:execute_config;"
        "Dataset/semantic_truth/objective_pipeline.py:build_episode_objective_artifacts;"
        "Dataset/tools/l4_business_semantics.py:evaluate"),
    ("arm_predicate_truth", "arm_predicate_truth", "L5"): (
        arm_predicate_truth_l5_schema, "Dataset/tools/l5_arm_semantics.py:calculate"),
    ("arm_predicate_truth", "arm_predicate_truth", "X"): (
        arm_predicate_truth_x_schema, "Dataset/tools/x_arm_pipeline.py:run_arm"),
    ("arm_predicate_truth_ticks", "arm_predicate_truth_ticks", "L1"): (
        arm_predicate_ticks_l1_schema, "Dataset/tools/l1_semantics.py:project"),
    ("arm_predicate_truth_ticks", "arm_predicate_truth_ticks", "L2"): (
        arm_predicate_ticks_l2_schema, "Dataset/tools/l2_arm_semantics.py:recompute"),
    ("arm_predicate_truth_ticks", "arm_predicate_truth_ticks", "X"): (
        arm_predicate_ticks_x_schema, "Dataset/tools/x_arm_pipeline.py:run_arm"),
    ("arm_predicate_transitions", "arm_predicate_transitions", "L1"): (
        arm_predicate_transition_l1_schema, "Dataset/tools/l1_semantics.py:project"),
    ("arm_predicate_transitions", "arm_predicate_transitions", "L2"): (
        arm_predicate_transition_l2_schema, "Dataset/tools/l2_arm_semantics.py:recompute"),
}
_ARM_PROFILE_GRAMMAR_SOURCE_FAMILIES = frozenset(
    family for family, _, _ in _PROFILE_GRAMMAR_BUILDERS
)
_FORMAL_PROFILE_GRAMMAR_BUILDERS: dict[
    tuple[str, str, str], tuple[Callable[[], dict], str, frozenset[str]]
] = {
    ("formal_roster", "formal_roster", "L3"): (
        formal_roster_l3_schema,
        "Dataset/tools/convert_to_render_ready.py;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py",
        frozenset({
            "L3-1_v1", "L3-2_v1", "L3-2_v2", "L3-3_v1", "L3-3_v2",
        }),
    ),
    ("formal_weather_meta", "formal_weather_meta", "L1"): (
        formal_weather_meta_schema,
        "Dataset/tools/convert_to_render_ready.py:normalize_weather_row;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py:copy_static_episode_files",
        frozenset({
            "L1-1_v1", "L1-1_v2", "L1-2_v1", "L1-3_v1", "L1-3_v2",
            "L1-4_v1", "L1-4_v2",
        }),
    ),
    ("formal_weather_meta", "formal_weather_meta", "L3"): (
        formal_weather_l3_schema,
        "Dataset/tools/convert_to_render_ready.py:normalize_weather_row;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py:copy_static_episode_files",
        frozenset({
            "L3-1_v1", "L3-2_v1", "L3-2_v2", "L3-3_v1", "L3-3_v2",
        }),
    ),
    ("formal_weather_meta", "formal_weather_meta", "L4"): (
        formal_weather_meta_schema,
        "Dataset/tools/convert_to_render_ready.py:normalize_weather_row;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py:copy_static_episode_files",
        frozenset({
            "L4-1_v1", "L4-1_v2", "L4-2_v1", "L4-2_v2", "L4-3_v1",
            "L4-3_v2", "L4-3_v3", "L4-4_v1", "L4-4_v2", "L4-5_v1",
            "L4-5_v2", "L4-5_v3", "L4-6_v1", "L4-6_v2", "L4-7_v1",
            "L4-7_v2", "L4-8_v1", "L4-8_v2", "L4-9_v1", "L4-9_v2",
            "L4-10_v1", "L4-10_v2", "L4-11_v1", "L4-11_v2",
        }),
    ),
    ("formal_roster", "formal_roster", "L4"): (
        formal_roster_schema,
        "Dataset/tools/convert_to_render_ready.py;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py",
        frozenset({
            "L4-1_v1", "L4-1_v2", "L4-2_v1", "L4-2_v2", "L4-3_v1",
            "L4-3_v2", "L4-3_v3", "L4-4_v1", "L4-4_v2", "L4-5_v1",
            "L4-5_v2", "L4-5_v3", "L4-6_v1", "L4-6_v2", "L4-7_v1",
            "L4-7_v2", "L4-8_v1", "L4-8_v2", "L4-9_v1", "L4-9_v2",
            "L4-10_v1", "L4-10_v2", "L4-11_v1", "L4-11_v2",
        }),
    ),
    ("formal_roster", "formal_roster", "L6"): (
        formal_roster_l6_schema,
        "Dataset/tools/convert_to_render_ready.py;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py",
        frozenset({
            "L6-1_v1", "L6-1_v2", "L6-2_v1", "L6-2_v2", "L6-3_v1",
            "L6-3_v2", "L6-4_v1", "L6-4_v2", "L6-5_v1", "L6-5_v2",
        }),
    ),
    ("formal_weather_meta", "formal_weather_meta", "L5"): (
        formal_weather_l5_schema,
        "Dataset/tools/convert_to_render_ready.py:normalize_weather_row;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py:copy_static_episode_files",
        frozenset({
            "L5-1_v1", "L5-1_v2", "L5-1_v3", "L5-2_v1", "L5-2_v2",
            "L5-3_v1", "L5-3_v2", "L5-4_v1", "L5-5_v1",
        }),
    ),
    ("formal_weather_meta", "formal_weather_meta", "L6"): (
        formal_weather_l6_schema,
        "Dataset/tools/convert_to_render_ready.py:normalize_weather_row;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py:copy_static_episode_files",
        frozenset({
            "L6-1_v1", "L6-1_v2", "L6-2_v1", "L6-2_v2", "L6-3_v1",
            "L6-3_v2", "L6-4_v1", "L6-4_v2", "L6-5_v1", "L6-5_v2",
        }),
    ),
    ("formal_weather_meta", "formal_weather_meta", "X"): (
        formal_weather_x_schema,
        "Dataset/tools/convert_to_render_ready.py:normalize_weather_row;"
        "Dataset/tools/filter_render_ready_truth_for_capture.py:copy_static_episode_files",
        frozenset({
            "X1_rain_to_c2loss_to_forced_landing",
            "X2_gnss_spoof_to_geofence_violation",
            "X3_pedestrian_fall_to_emergency_response",
            "X4_fog_to_uav_conflict",
            "X5_comm_failure_to_pad_contention",
            "X6_crowd_evacuation_to_airspace_lockdown",
        }),
    ),
}
_FORMAL_PROFILE_GRAMMAR_SOURCE_FAMILIES = frozenset(
    family for family, _, _ in _FORMAL_PROFILE_GRAMMAR_BUILDERS
)
PROFILE_GRAMMAR_SOURCE_FAMILIES = (
    _ARM_PROFILE_GRAMMAR_SOURCE_FAMILIES |
    _FORMAL_PROFILE_GRAMMAR_SOURCE_FAMILIES
)
_PROFILES_BY_PRODUCER = {
    "L1": frozenset({
        "L1-1_v1", "L1-1_v2", "L1-2_v1", "L1-3_v1", "L1-3_v2",
        "L1-4_v1", "L1-4_v2",
    }),
    "L2": frozenset({
        "L2-1_v1", "L2-1_v2", "L2-2_v1", "L2-2_v2", "L2-3_v1",
        "L2-3_v2", "L2-4_v1", "L2-4_v2", "L2-5_v1",
    }),
    "L3": frozenset({
        "L3-1_v1", "L3-2_v1", "L3-2_v2", "L3-3_v1", "L3-3_v2",
    }),
    "L4": frozenset({
        "L4-1_v1", "L4-1_v2", "L4-2_v1", "L4-2_v2", "L4-3_v1",
        "L4-3_v2", "L4-3_v3", "L4-4_v1", "L4-4_v2", "L4-5_v1",
        "L4-5_v2", "L4-5_v3", "L4-6_v1", "L4-6_v2", "L4-7_v1",
        "L4-7_v2", "L4-8_v1", "L4-8_v2", "L4-9_v1", "L4-9_v2",
        "L4-10_v1", "L4-10_v2", "L4-11_v1", "L4-11_v2",
    }),
    "L5": frozenset({
        "L5-1_v1", "L5-1_v2", "L5-1_v3", "L5-2_v1", "L5-2_v2",
        "L5-3_v1", "L5-3_v2",
    }),
    "L6": frozenset({
        "L6-1_v1", "L6-1_v2", "L6-2_v1", "L6-2_v2", "L6-3_v1",
        "L6-3_v2", "L6-4_v1", "L6-4_v2", "L6-5_v1", "L6-5_v2",
    }),
    "X": frozenset({
        "X1_rain_to_c2loss_to_forced_landing",
        "X2_gnss_spoof_to_geofence_violation",
        "X3_pedestrian_fall_to_emergency_response",
        "X4_fog_to_uav_conflict",
        "X5_comm_failure_to_pad_contention",
        "X6_crowd_evacuation_to_airspace_lockdown",
    }),
}
_PROFILE_PRODUCERS = {
    profile: producer
    for producer, profiles in _PROFILES_BY_PRODUCER.items()
    for profile in profiles
}
if len(_PROFILE_PRODUCERS) != sum(map(len, _PROFILES_BY_PRODUCER.values())):
    raise ValueError("ARM profile belongs to more than one producer")

_SHARED_GRAMMAR_BUILDERS: dict[str, Callable[[], ClosedRecordGrammar]] = {
    "compute_state": compute_grammar,
    "formal_predicate_truth": formal_predicate_truth_grammar,
    "formal_predicate_transitions": formal_predicate_transition_grammar,
    "formal_predicate_continuity_breaks": formal_predicate_continuity_break_grammar,
    "predicate_truth": supplement_predicate_truth_grammar,
    "predicate_truth_matrix": supplement_predicate_matrix_grammar,
    "formal_event_occurrences": formal_event_occurrence_grammar,
    "formal_event_outcomes": formal_event_outcome_grammar,
    "formal_world_truth_graph_base": world_truth_base_grammar,
    "formal_world_truth_graph_deltas": world_truth_delta_grammar,
    "communication_state": communication_state_grammar,
    "events": supplement_event_grammar,
}


def _producer_for_profile(profile: str, source: str) -> str:
    producer = _PROFILE_PRODUCERS.get(profile)
    if producer is None:
        raise ValueError(f"{source}: ARM profile has no reviewed producer binding: {profile!r}")
    return producer


@cache
def _profile_grammar(source_family: str, effective_family: str,
                     producer: str) -> ClosedRecordGrammar:
    builder = _PROFILE_GRAMMAR_BUILDERS.get(
        (source_family, effective_family, producer))
    if builder is None:
        raise ValueError(
            f"producer record grammar is not registered: {source_family!r}, "
            f"{effective_family!r}, {producer!r}")
    schema_builder, authority = builder
    schema_ref = ("Dataset/world_model/graph/" +
                  schema_builder.__module__.rsplit(".", 1)[-1] +
                  ".py:" + schema_builder.__name__)
    return compile_record_grammar(schema_builder(), schema_ref=schema_ref,
                                  authority=authority)


@cache
def _formal_profile_grammar(source_family: str,
                            effective_family: str,
                            level: str) -> ClosedRecordGrammar:
    schema_builder, authority, _ = _FORMAL_PROFILE_GRAMMAR_BUILDERS[
        (source_family, effective_family, level)]
    schema_ref = ("Dataset/world_model/graph/" +
                  schema_builder.__module__.rsplit(".", 1)[-1] +
                  ".py:" + schema_builder.__name__)
    return compile_record_grammar(schema_builder(), schema_ref=schema_ref,
                                  authority=authority)


def _producer_record_grammar(source_family: str, effective_family: str,
                    source: str, *, producer: str | None,
                    profile: str | None) -> ClosedRecordGrammar:
    if source_family == "arm_initial_events":
        if effective_family != source_family:
            raise ValueError(
                f"{source}: no reviewed effective record type for "
                f"{source_family!r} -> {effective_family!r}")
        if profile is not None:
            profile_producer = _producer_for_profile(profile, source)
            if producer is not None and producer != profile_producer:
                raise ValueError(
                    f"{source}: producer {producer!r} differs from profile {profile!r}")
            producer = profile_producer
        if producer not in {"L3", "L6"}:
            raise ValueError(f"{source}: unknown initial-events producer: {producer!r}")
        return initial_grammar(producer)
    if source_family in _CANDIDATE_MODULES:
        level = _candidate_level(source_family, source, profile)
        if effective_family not in _candidate_effective_families(source_family, level):
            raise ValueError(
                f"{source}: unregistered effective type {source_family!r} -> "
                f"{effective_family!r} at {level}")
        static_arm = (source_family, effective_family, level) in _PROFILE_GRAMMAR_BUILDERS
        static_formal = any(
            family == source_family and effective == effective_family and
            profile in binding[2]
            for (family, effective, _), binding in _FORMAL_PROFILE_GRAMMAR_BUILDERS.items()
        )
        if not static_arm and not static_formal:
            return _candidate_grammar(source_family, effective_family, source,
                                      level, profile)
    if source_family in _FORMAL_PROFILE_GRAMMAR_SOURCE_FAMILIES:
        if producer is not None:
            raise ValueError(f"{source}: formal record has no ARM producer: {producer!r}")
        levels = tuple(
            level for (family, effective, level), (_, _, profiles)
            in _FORMAL_PROFILE_GRAMMAR_BUILDERS.items()
            if family == source_family and effective == effective_family and
            profile in profiles)
        if len(levels) != 1:
            raise ValueError(
                f"{source}: no reviewed formal grammar for {source_family!r} "
                f"-> {effective_family!r} at profile {profile!r}")
        return _formal_profile_grammar(source_family, effective_family,
                                       levels[0])
    if source_family in _ARM_PROFILE_GRAMMAR_SOURCE_FAMILIES:
        if profile is None:
            raise ValueError(f"{source}: profile is required for {source_family!r}")
        profile_producer = _producer_for_profile(profile, source)
        if producer is not None and producer != profile_producer:
            raise ValueError(
                f"{source}: producer {producer!r} differs from profile {profile!r}")
        if (source_family, effective_family, profile_producer) not in _PROFILE_GRAMMAR_BUILDERS:
            raise ValueError(
                f"{source}: no reviewed grammar for {source_family!r} "
                f"-> {effective_family!r} "
                f"at profile {profile!r} (producer {profile_producer})")
        return _profile_grammar(source_family, effective_family, profile_producer)
    if effective_family != source_family:
        raise ValueError(
            f"{source}: no reviewed effective record type for "
            f"{source_family!r} -> {effective_family!r}")
    builder = _SHARED_GRAMMAR_BUILDERS.get(source_family)
    if builder is not None:
        return builder()
    if source_family == "arm_semantic_history":
        if profile is None:
            raise ValueError(f"{source}: semantic history has no declared profile")
        return _history_grammar(profile)
    _raise_unregistered_source_family(source_family, source)
    raise AssertionError("unreachable source family")


def _record_grammar(source_family: str, effective_family: str,
                    source: str, *, producer: str | None,
                    profile: str | None) -> ClosedRecordGrammar:
    grammar = _producer_record_grammar(source_family, effective_family, source,
                                       producer=producer, profile=profile)
    path = (REPO / source).resolve()
    if path.is_relative_to(REPO / "aw_data/objective_semantic_truth"):
        ref = grammar.schema_ref + ":published_without_integrity_metadata"
        projected = _COMPILED_BY_REF.get(ref)
        if projected is None:
            schema = published_record_schema(grammar.schema)
            if source_family in {"compute_state", "communication_state"}:
                from Dataset.semantic_simulation.compute_comm import SCHEMA_VERSION
                # The objective run persists the current simulation rows. The
                # separate supplement publisher's 2.0 envelope is not applied.
                for branch in schema["oneOf"]:
                    branch["properties"]["schema_version"] = {"const": SCHEMA_VERSION}
            projected = compile_record_grammar(schema,
                schema_ref=ref, authority=grammar.authority + ";Dataset/semantic_truth/provenance.py:without_integrity_metadata")
            _COMPILED_BY_REF[ref] = projected
        return projected
    return grammar


_ARM_DECLARATION_SUFFIXES = {
    "arm_initial_events": "objective/initial_events.json",
    "arm_actions": "raw/actions.jsonl",
    "arm_states": "raw/states.jsonl",
    "arm_trajectories": "raw/trajectories.jsonl",
    "arm_predicate_truth": "objective/predicate_truth.jsonl",
    "arm_predicate_truth_ticks": "objective/predicate_truth_ticks.jsonl",
    "arm_predicate_transitions": "objective/predicate_transitions.jsonl",
}
_FORMAL_DECLARATION_FILES = {
    "formal_weather_meta": "weather_meta.jsonl",
    "formal_roster": "global_entity_roster.json",
}


def _declaration_profile(declaration: SourceDeclaration) -> str:
    parts = declaration.sample.split("/")
    if len(parts) != 5 or parts[0] != "arms" or not all(parts[1:]):
        raise ValueError(f"ARM declaration has no branch sample: {declaration.sample!r}")
    expected = REPO / declaration.sample / _ARM_DECLARATION_SUFFIXES[declaration.source_family]
    if declaration.path != expected:
        raise ValueError(
            f"ARM declaration path differs from its source role: {declaration.path}")
    return parts[1]


def _formal_declaration_profile(declaration: SourceDeclaration) -> str:
    profile, separator, seed = declaration.sample.rpartition("__")
    if not separator or not profile or re.fullmatch(r"seed[0-9]{2}", seed) is None:
        raise ValueError(
            f"formal declaration has no episode profile/seed: {declaration.sample!r}")
    expected = (REPO / "aw_data/render_ready_episodes_capture_filtered" /
                declaration.sample / _FORMAL_DECLARATION_FILES[declaration.source_family])
    if declaration.path != expected:
        raise ValueError(
            f"formal declaration path differs from its source role: {declaration.path}")
    return profile


def _history_declaration_profile(declaration: SourceDeclaration) -> str:
    """Bind one history archive to its L6 profile through the branch manifest.

    The archive path is a manifest reference (``semantic_history``) with a
    producer digest in its name, so it is never guessed from a fixed suffix.
    """
    parts = declaration.sample.split("/")
    if len(parts) != 5 or parts[0] != "arms" or not all(parts[1:]):
        raise ValueError(f"ARM declaration has no branch sample: {declaration.sample!r}")
    branch = REPO / declaration.sample
    manifest_path = branch / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = manifest_reference(manifest, "semantic_history")
    if reference is None:
        raise ValueError(f"ARM manifest has no semantic history reference: {manifest_path}")
    if declaration.path != REPO / reference:
        raise ValueError(
            f"ARM history path differs from its manifest reference: {declaration.path}")
    return parts[1]


def source_grammar_for_declaration(
    declaration: SourceDeclaration, *, effective_family: str | None = None,
) -> ClosedRecordGrammar:
    """Resolve one complete record grammar from a declared role, without rows.

    A source with multiple effective row types needs an explicit reviewed
    mapping for each type. An absent or pending mapping raises here, so a
    caller cannot infer closure from source-family membership or file presence.
    """
    if not isinstance(declaration, SourceDeclaration):
        raise TypeError(f"expected SourceDeclaration, got {type(declaration).__name__}")
    source_family = declaration.source_family
    source = str(declaration.path)
    if declaration.structure_status != "field_scan_defined":
        raise ValueError(
            f"{source}: source declaration structure is not closed: "
            f"{declaration.structure_status!r}")
    effective = source_family if effective_family is None else effective_family
    if source_family in AUTHORING_SOURCE_FAMILIES:
        if effective != source_family:
            raise ValueError(
                f"{source}: no authoring grammar for effective type {effective!r}")
        return authoring_source_grammar(source_family, declaration.path)
    if source_family in _CANDIDATE_MODULES:
        if declaration_not_applicable(declaration):
            raise ValueError(f"source has no producer at this layer: {declaration.path}")
        profile = _candidate_profile_for_declaration(declaration)
        return _record_grammar(source_family, effective, source,
                               producer=None, profile=profile)
    if source_family not in (CLOSED_RECORD_FAMILIES |
                             PROFILE_GRAMMAR_SOURCE_FAMILIES |
                             {"arm_semantic_history"}):
        raise ValueError(
            f"{source}: source/effective grammar is not registered for "
            f"{source_family!r} -> {effective!r}")
    if source_family in _ARM_DECLARATION_SUFFIXES:
        profile = _declaration_profile(declaration)
    elif source_family in _FORMAL_DECLARATION_FILES:
        profile = _formal_declaration_profile(declaration)
    elif source_family == "arm_semantic_history":
        profile = _history_declaration_profile(declaration)
    else:
        profile = None
    return _record_grammar(source_family, effective, source, producer=None,
                           profile=profile)


def reviewed_source_grammar_declaration(declaration: SourceDeclaration) -> bool:
    """Check the explicit full-record registry without reading source rows."""
    if not isinstance(declaration, SourceDeclaration):
        raise TypeError(f"expected SourceDeclaration, got {type(declaration).__name__}")
    if declaration.structure_status != "field_scan_defined":
        return False
    family = declaration.source_family
    if family in _CANDIDATE_MODULES:
        if declaration_not_applicable(declaration):
            return False
        profile = _candidate_profile_for_declaration(declaration)
        level = _candidate_level(family, str(declaration.path), profile)
        return bool(_candidate_effective_families(family, level))
    if family in AUTHORING_SOURCE_FAMILIES or family in _SHARED_GRAMMAR_BUILDERS:
        return True
    if family == "arm_semantic_history":
        try:
            profile = _history_declaration_profile(declaration)
        except ValueError:
            return False
        return profile in HISTORY_L6_PROFILES
    if family in _FORMAL_DECLARATION_FILES:
        try:
            profile = _formal_declaration_profile(declaration)
        except ValueError:
            return False
        return any(source == family and profile in builder[2]
                   for (source, _, _), builder in _FORMAL_PROFILE_GRAMMAR_BUILDERS.items())
    if family not in _ARM_DECLARATION_SUFFIXES:
        return False
    try:
        profile = _declaration_profile(declaration)
    except ValueError:
        return False
    producer = _PROFILE_PRODUCERS.get(profile)
    if family == "arm_initial_events":
        return producer in {"L3", "L6"}
    return any(source == family and bound_producer == producer
               for source, _, bound_producer in _PROFILE_GRAMMAR_BUILDERS)


def declared_effective_families(declaration: SourceDeclaration) -> tuple[str, ...]:
    """Return the reviewed effective record types of one declaration.

    This query deliberately fails for a source role with an unresolved type
    split; discovering instances cannot turn a pending grammar into closure.
    It checks registry and source-role identity without compiling grammars.
    """
    if not isinstance(declaration, SourceDeclaration):
        raise TypeError(f"expected SourceDeclaration, got {type(declaration).__name__}")
    if not reviewed_source_grammar_declaration(declaration):
        raise ValueError(
            f"{declaration.path}: declaration has no reviewed complete record grammar: "
            f"{declaration.source_family!r}")
    family = declaration.source_family
    if family in _CANDIDATE_MODULES:
        profile = _candidate_profile_for_declaration(declaration)
        return _candidate_effective_families(
            family, _candidate_level(family, str(declaration.path), profile))
    if family in AUTHORING_SOURCE_FAMILIES:
        try:
            relative = declaration.path.relative_to(REPO)
        except ValueError as exc:
            raise ValueError(f"authoring source is outside repository: {declaration.path}") from exc
        if relative.parts[:2] == ("Dataset", "scenarios"):
            expected = ("scene_setup.json" if family == "formal_source_scene_setup"
                        else "event_script.json")
            if relative.name != expected:
                raise ValueError(f"{family} has the wrong authoring input: {relative}")
        elif family == "script_category_source":
            parts = relative.parts
            if (len(parts) != 5 or parts[0] != "design" or
                    parts[1] not in _PROFILES_BY_PRODUCER["L5"] or
                    re.fullmatch(r"seed[0-9]{2}", parts[2]) is None or
                    re.fullmatch(r"grp_[a-z0-9_]+", parts[3]) is None or
                    re.fullmatch(r"arm_[a-z0-9_]+\.script\.json", parts[4]) is None):
                raise ValueError(f"not a declared L5 branch script: {relative}")
        else:
            raise ValueError(f"{family} is not a scenario authoring input: {relative}")
        return (family,)
    if family == "arm_initial_events":
        producer = _producer_for_profile(_declaration_profile(declaration),
                                         str(declaration.path))
        if producer not in {"L3", "L6"}:
            raise ValueError(f"{declaration.path}: no initial-events producer: {producer}")
        return (family,)
    if family in _ARM_PROFILE_GRAMMAR_SOURCE_FAMILIES:
        profile = _declaration_profile(declaration)
        producer = _producer_for_profile(profile, str(declaration.path))
        effective_families = tuple(sorted({
            effective for source, effective, bound_producer in _PROFILE_GRAMMAR_BUILDERS
            if source == family and bound_producer == producer
        }))
        if not effective_families:
            raise ValueError(
                f"{declaration.path}: no reviewed grammar for {family!r} "
                f"at profile {profile!r}")
        return effective_families
    if family in _FORMAL_PROFILE_GRAMMAR_SOURCE_FAMILIES:
        profile = _formal_declaration_profile(declaration)
        effective_families = tuple(sorted({
            effective for (source, effective, _), (_, _, profiles)
            in _FORMAL_PROFILE_GRAMMAR_BUILDERS.items()
            if source == family and profile in profiles
        }))
        if not effective_families:
            raise ValueError(
                f"{declaration.path}: no reviewed formal grammar for {family!r} "
                f"at profile {profile!r}")
        return effective_families
    if family == "arm_semantic_history":
        if _history_declaration_profile(declaration) not in HISTORY_L6_PROFILES:
            raise ValueError(
                f"{declaration.path}: no reviewed L6 history profile")
        return (family,)
    if family not in _SHARED_GRAMMAR_BUILDERS:
        raise ValueError(f"{declaration.path}: no reviewed grammar for {family!r}")
    return (family,)


@cache
def _declared_structure_statuses() -> dict[str, str]:
    """Read the structure status of every family from the declared inventory."""
    statuses: dict[str, str] = {}
    for declaration in declared_source_inventory():
        known = statuses.setdefault(declaration.source_family,
                                    declaration.structure_status)
        if known != declaration.structure_status:
            raise ValueError(
                "declared source inventory assigns conflicting structure statuses "
                f"to {declaration.source_family}: {known!r} and "
                f"{declaration.structure_status!r}")
    return statuses


def _raise_unregistered_source_family(source_family: str, source: str) -> None:
    """Reject a source family outside the producer grammar registry."""
    status = _declared_structure_statuses().get(source_family)
    if status == "source_schema_pending":
        raise ValueError(
            f"{source}: source family has no registered producer grammar and "
            f"remains structure-pending: {source_family!r}")
    if status == "field_scan_defined":
        raise ValueError(
            f"{source}: source family has no producer record grammar and is "
            f"validated by source projection: {source_family!r}")
    if status is None:
        raise ValueError(
            f"{source}: source family is not declared in the current producer "
            f"source inventory: {source_family!r}")
    raise ValueError(
        f"{source}: source family has an unhandled declared structure status "
        f"{status!r}: {source_family!r}")


def validate_source_record(source_family: str, record: Any, source: str,
                           *, producer: str | None = None,
                           profile: str | None = None,
                           effective_family: str | None = None) -> None:
    effective = source_family if effective_family is None else effective_family
    if source_family in AUTHORING_SOURCE_FAMILIES:
        if effective != source_family:
            raise ValueError(
                f"{source}: no authoring grammar for effective type {effective!r}")
        validate_authoring_source_record(source_family, source, record)
    elif source_family == "arm_semantic_history":
        if effective != source_family:
            raise ValueError(
                f"{source}: no history record grammar for {effective!r}")
        validate_record(
            _record_grammar(source_family, effective, source,
                            producer=producer, profile=profile),
            record, source_path=source,
        )
    else:
        validate_record(
            _record_grammar(source_family, effective, source,
                            producer=producer, profile=profile),
            record, source_path=source,
        )
