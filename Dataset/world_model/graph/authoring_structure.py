"""Closed grammars for declared scenario scripts and scene authoring inputs.

The L5 branch script is a producer output. Its grammar comes from the exact
scenario authoring input and the checked entry-script transform, never from a
union of observed branch files. Source roles stay with their declarations;
this module does not assign sample identity or decide whether a conditional
reference was realized.
"""

from __future__ import annotations

import ast
from functools import cache
import json
from pathlib import Path
import re
from typing import Any

from Dataset.world_model.graph.contract import REPO
from Dataset.world_model.graph.sample_identity import manifest_reference
from Dataset.world_model.graph.structure_schema import (
    ClosedRecordGrammar, compile_authoring_input, compile_record_grammar,
    validate_record,
)


AUTHORING_SOURCE_FAMILIES = frozenset({
    "formal_source_script", "formal_source_scene_setup", "script_category_source",
})
_SCENARIOS = REPO / "Dataset/scenarios"
_L5_PROFILE = re.compile(r"L5-[0-9]+_v[0-9]+\Z")
_SEED = re.compile(r"seed[0-9]{2}\Z")
_GROUP = re.compile(r"grp_[a-z0-9_]+\Z")
_ARM_ID = re.compile(r"arm_[a-z0-9_]+\Z")
_ENTRY_BINDINGS = frozenset({
    "CONFIG", "AUTHORED_SCRIPT", "AUTHORED_SCENE", "OFFSET_TICKS",
})

# This is the complete current L5 entry program, with only its four bound
# declarations abstracted by _entry_signature. A new branch operation fails
# the gate until its producer structure is reviewed explicitly.
_L5_ENTRY_TEMPLATE = '''\
"""当前源脚本实跑；任务巡航独立调度，干预只延迟天气到达。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l5_arm_common import load_config
from Dataset.tools.l5_arm_branch import (
    write_branch, write_experiment, set_trigger, bind_branch_sources, measured_window)
from Dataset.tools.l5_arm_pipeline import execute_config

CONFIG = ROOT / 'config_placeholder'
AUTHORED_SCRIPT = ROOT / 'script_placeholder'
AUTHORED_SCENE = ROOT / 'scene_placeholder'
OFFSET_TICKS = 0

if __name__ == '__main__':
    config = load_config(CONFIG)
    script = json.loads(AUTHORED_SCRIPT.read_text())
    scene = json.loads(AUTHORED_SCENE.read_text())
    onset = scene['weather_profile']['transitions'][0]['tick']
    if OFFSET_TICKS:
        set_trigger(script, 'weather_onset', onset + OFFSET_TICKS)
        for transition in scene['weather_profile']['transitions']:
            transition['tick'] += OFFSET_TICKS
    design = ROOT / 'design' / config['group_key']
    paths = write_branch(design, script, scene, name=config['arm_id'])
    window, evidence = measured_window(paths['script'], paths['scene'],
        config['primary_entity_id'], config['window']['fork_tick'], config['episode_id'])
    write_experiment(design, arm_id=config['arm_id'], group_key=config['group_key'],
        episode_id=config['episode_id'], window=window,
        hazard={'onset_tick': onset + OFFSET_TICKS, 'offset_ticks': OFFSET_TICKS,
                'overrides': scene['weather_profile']['transitions'][0]['overrides']},
        kept=[e['event_id'] for e in script['events']], dropped=[], designed=[],
        target_event={'families': config['event_families'],
                      'basis': '预期：天气作用于独立任务进度；实际谓词和事件以本次 objective 为准。'},
        basis=evidence['basis'])
    bind_branch_sources(CONFIG, paths['script'], paths['scene'], window=window)
    execute_config(CONFIG)
'''


def _canonical_source(source_file: str | Path) -> tuple[Path, Path]:
    supplied = Path(source_file)
    source = (supplied if supplied.is_absolute() else REPO / supplied).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"authoring source is not a file: {source_file}")
    try:
        relative = source.relative_to(REPO)
    except ValueError as exc:
        raise ValueError(f"authoring source is outside repository: {source_file}") from exc
    if (supplied != source if supplied.is_absolute()
            else supplied.as_posix() != relative.as_posix()):
        raise ValueError(f"authoring source path is not canonical: {source_file}")
    return source, relative


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"{path}: non-finite JSON constant {constant}")))
    if type(value) is not dict:
        raise TypeError(f"{path}: expected a JSON object")
    return value


@cache
def _author_grammar(source: Path) -> ClosedRecordGrammar:
    return compile_authoring_input(source)


def _entry_signature(tree: ast.Module) -> tuple[str, dict[str, ast.AST]]:
    bindings: dict[str, ast.AST] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in _ENTRY_BINDINGS):
            name = node.targets[0].id
            if name in bindings:
                raise ValueError(f"duplicate L5 entry binding: {name}")
            bindings[name] = node.value
            node.value = ast.Name(id=f"BOUND_{name}", ctx=ast.Load())
    if bindings.keys() != _ENTRY_BINDINGS:
        raise ValueError(f"L5 entry bindings differ: {sorted(bindings)}")
    return ast.dump(tree, include_attributes=False), bindings


_EXPECTED_ENTRY_SIGNATURE, _ = _entry_signature(ast.parse(_L5_ENTRY_TEMPLATE))


def _root_path(binding: ast.AST, name: str, entry: Path) -> str:
    if not (isinstance(binding, ast.BinOp) and isinstance(binding.op, ast.Div) and
            isinstance(binding.left, ast.Name) and binding.left.id == "ROOT" and
            isinstance(binding.right, ast.Constant) and
            type(binding.right.value) is str):
        raise ValueError(f"{entry}: {name} is not a literal ROOT path")
    return binding.right.value


def _entry_bindings(entry: Path, *, config_ref: str, script_ref: str,
                    scene_ref: str) -> int:
    signature, bindings = _entry_signature(ast.parse(entry.read_text(encoding="utf-8")))
    if signature != _EXPECTED_ENTRY_SIGNATURE:
        raise ValueError(f"{entry}: L5 entry producer differs from the declared transform")
    for name, expected in (("CONFIG", config_ref),
                           ("AUTHORED_SCRIPT", script_ref),
                           ("AUTHORED_SCENE", scene_ref)):
        actual = _root_path(bindings[name], name, entry)
        if actual != expected:
            raise ValueError(f"{entry}: {name} source differs: {actual!r}")
    offset = bindings["OFFSET_TICKS"]
    if not isinstance(offset, ast.Constant) or type(offset.value) is not int:
        raise ValueError(f"{entry}: OFFSET_TICKS is not a literal integer")
    return offset.value


def _same_json(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _same_json(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right))
    return left == right


def _l5_derived_script_grammar(source: Path, relative: Path
                               ) -> tuple[ClosedRecordGrammar, dict[str, Any]]:
    parts = relative.parts
    if len(parts) != 5 or parts[0] != "design" or not parts[-1].endswith(".script.json"):
        raise ValueError(f"not a declared L5 branch script: {relative}")
    _, profile, seed, group, filename = parts
    arm_id = filename.removesuffix(".script.json")
    if not (_L5_PROFILE.fullmatch(profile) and _SEED.fullmatch(seed) and
            _GROUP.fullmatch(group) and _ARM_ID.fullmatch(arm_id)):
        raise ValueError(f"invalid L5 branch script identity: {relative}")
    group_key = f"{profile}/{seed}/{group}"
    branch = REPO / "arms" / group_key / arm_id
    config_ref = f"interventions/{group_key}/{arm_id}.yaml"
    entry_ref = f"scripts/arms/{group_key}/{arm_id}.py"
    script_ref = f"Dataset/scenarios/L5_environment/interaction/{profile}/event_script.json"
    scene_ref = f"Dataset/scenarios/L5_environment/interaction/{profile}/scene_setup.json"
    derived_scene_ref = f"design/{group_key}/{arm_id}.scene.json"
    manifest = _read_object(branch / "manifest.json")
    if (manifest["arm_id"] != arm_id or manifest["arm_key"] != f"{group_key}/{arm_id}" or
            manifest["configuration_ref"] != config_ref or
            manifest_reference(manifest, "script") != relative.as_posix() or
            manifest_reference(manifest, "scene") != derived_scene_ref):
        raise ValueError(f"{branch}: L5 source declarations differ from branch identity")
    from Dataset.tools.l5_arm_common import load_config
    config = load_config(REPO / config_ref)
    if (config["group_key"] != group_key or config["arm_id"] != arm_id or
            config["sources"]["script"] != relative.as_posix() or
            config["sources"]["scene"] != derived_scene_ref):
        raise ValueError(f"{config_ref}: L5 branch source declarations differ")
    offset = _entry_bindings(REPO / entry_ref, config_ref=config_ref,
                             script_ref=script_ref, scene_ref=scene_ref)
    authored_script = _read_object(REPO / script_ref)
    authored_scene = _read_object(REPO / scene_ref)
    events = authored_script["events"]
    triggers = authored_script["triggers"]
    transitions = authored_scene["weather_profile"]["transitions"]
    if (type(events) is not list or type(triggers) is not list or
            type(transitions) is not list or not transitions or
            any(type(item) is not dict or type(item.get("tick")) is not int
                for item in transitions)):
        raise ValueError(f"{entry_ref}: authored L5 event/weather shape differs")
    weather_events = [item for item in events if item["event_id"] == "weather_onset"]
    if len(weather_events) != 1:
        raise ValueError(f"{script_ref}: weather_onset event is not unique")
    weather_triggers = [item for item in triggers
                        if item["trigger_id"] == weather_events[0]["trigger_ref"]]
    if (len(weather_triggers) != 1 or
            set(weather_triggers[0]) != {"trigger_id", "type", "tick"} or
            weather_triggers[0]["type"] != "tick" or
            type(weather_triggers[0]["tick"]) is not int):
        raise ValueError(f"{script_ref}: weather_onset trigger cannot keep its shape")
    onset = transitions[0]["tick"]
    if offset:
        from Dataset.tools.l5_arm_branch import set_trigger
        set_trigger(authored_script, "weather_onset", onset + offset)
        for transition in transitions:
            transition["tick"] += offset
    authority = f"{entry_ref};Dataset/tools/l5_arm_branch.py:write_branch"
    grammar = compile_record_grammar(
        _author_grammar(REPO / script_ref).schema,
        schema_ref=f"{entry_ref}:write_branch(script)", authority=authority)
    scene_grammar = compile_record_grammar(
        _author_grammar(REPO / scene_ref).schema,
        schema_ref=f"{entry_ref}:write_branch(scene)", authority=authority)
    derived_script = _read_object(source)
    derived_scene = _read_object(REPO / derived_scene_ref)
    validate_record(grammar, derived_script, source_path=source)
    validate_record(scene_grammar, derived_scene, source_path=REPO / derived_scene_ref)
    if not _same_json(derived_script, authored_script):
        raise ValueError(f"{relative}: L5 script differs from its declared producer")
    if not _same_json(derived_scene, authored_scene):
        raise ValueError(f"{derived_scene_ref}: L5 scene differs from its declared producer")
    return grammar, authored_script


def authoring_source_grammar(source_family: str,
                             source_file: str | Path) -> ClosedRecordGrammar:
    """Return the closed grammar for one declared authoring source role/path."""
    if source_family not in AUTHORING_SOURCE_FAMILIES:
        raise ValueError(f"unsupported authoring source family: {source_family}")
    source, relative = _canonical_source(source_file)
    if source.is_relative_to(_SCENARIOS):
        expected = ("scene_setup.json" if source_family == "formal_source_scene_setup"
                    else "event_script.json")
        if relative.name != expected:
            raise ValueError(f"{source_family} has the wrong authoring input: {relative}")
        return _author_grammar(source)
    if source_family == "script_category_source":
        grammar, _ = _l5_derived_script_grammar(source, relative)
        return grammar
    raise ValueError(f"{source_family} is not a scenario authoring input: {relative}")


def validate_authoring_source_record(source_family: str, source_file: str | Path,
                                      record: Any) -> None:
    """Validate an unchanged source record, including L5 producer equality."""
    source, relative = _canonical_source(source_file)
    if source_family == "script_category_source" and relative.parts[0] == "design":
        grammar, expected = _l5_derived_script_grammar(source, relative)
        validate_record(grammar, record, source_path=source)
        if not _same_json(record, expected):
            raise ValueError(f"{relative}: record differs from its declared L5 producer")
        return
    validate_record(authoring_source_grammar(source_family, source), record,
                    source_path=source)
