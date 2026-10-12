"""L2 事件链窗口 ARM 执行器：共享引擎、语义与序列化底座。

设计约束（本会话用户决策）：
- 窗口按事件链片段设计，只保存 [start_tick, end_tick] 的逐 tick 状态，载入前史用于恢复。
- 每条臂的机制逐条写死在配置里；执行器不按 arm_id 猜动作。
- 预期（expected）只作假设；谓词与事件由当前语义契约实测计算。
"""
from __future__ import annotations

import copy
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
for _directory in (ROOT, ROOT / 'Dataset/tools', ROOT / 'Plugins/SumoImporter/Scripts'):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from Dataset.semantic_truth.provenance import open_artifact_text
from Dataset.tools.batch_generate import EpisodeStateEngine            # noqa: E402
from donghu_core.event_script_interpreter import EventScriptInterpreter  # noqa: E402
from Dataset.tools import l2_arm_semantics as semantics                # noqa: E402

SCHEMA = 'rules/l2_window.schema.json'
SCENARIO_ROOT = 'Dataset/scenarios/L2_infrastructure'
FACTUAL_ROOT = 'aw_data'


def load(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def stream_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False) + '\n')


def source_path(ref):
    path = (ROOT / ref).resolve()
    path.relative_to(ROOT)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_config(path):
    """Load and validate one L2 window configuration; enforce identity and factual root."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    schema = load(ROOT / SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(config),
                    key=lambda e: str(list(e.absolute_path)))
    if errors:
        raise ValueError(f'{path}: ' + '; '.join(e.message for e in errors))
    expected_path = ROOT / 'interventions' / config['group_key'] / f"{config['arm_id']}.yaml"
    if path != expected_path:
        raise ValueError(f'configuration identity/path mismatch: {path} != {expected_path}')
    profile, seed, _ = config['group_key'].split('/')
    if profile != config['profile'] or seed != config['seed']:
        raise ValueError('group_key disagrees with profile/seed')
    if config['episode_id'] != f'{profile}__{seed}':
        raise ValueError('episode_id disagrees with group_key')
    if config['mode'] != 'intervention' and (config['execution']['actions']
                                             or config['execution']['omit_action_ids']):
        raise ValueError('only intervention configurations may carry actions or omissions')
    expected_script = f"scripts/arms/{config['group_key']}/{config['arm_id']}.py"
    if config['script'] != expected_script or not (ROOT / expected_script).is_file():
        raise ValueError('dedicated arm script is missing or does not match configuration')
    if config['output'] != f"arms/{config['group_key']}/{config['arm_id']}":
        raise ValueError('output directory must match the ARM identity')
    if config['provenance']['parameter_sources']['factual_root'] != FACTUAL_ROOT:
        raise ValueError('L2 configurations read canonical factual only')
    window = config['window']
    if not 0 <= window['start_tick'] <= window['fork_tick'] < window['end_tick'] <= 900:
        raise ValueError('invalid window order')
    if any(window[k] % 5 for k in ('start_tick', 'fork_tick', 'end_tick')):
        raise ValueError('window endpoints must lie on the 5 tick formal grid')
    for step in config['execution']['actions']:
        if not window['fork_tick'] <= step['tick'] < window['end_tick']:
            raise ValueError('intervention action lies outside the execution window')
        if step['action']['entity_id'] not in config['entity_ids']:
            raise ValueError('intervention actor is outside declared entity scope')
    return config


def scene_script(config):
    scene = load(source_path(config['sources']['scene']))
    script = source_path(config['sources']['script'])
    return scene, script, load(script)


def execute_engine(config):
    """Execute the author scene from tick 0; apply omissions and injected actions.

    Phase order matches the window standard: pending runtime patches, state sample,
    snapshot, retained source actions, then intervention actions.
    """
    scene, script_path, script = scene_script(config)
    engine = EpisodeStateEngine(scene, script, script_path, config['window']['end_tick'])
    interpreter = EventScriptInterpreter(script_path, episode_id=config['episode_id'])
    script_actions = {a['action_id']: a for e in script['events'] for a in e['actions']}
    omit = set(config['execution']['omit_action_ids'])
    if not omit <= set(script_actions):
        raise ValueError(f'unknown omitted actions: {sorted(omit - set(script_actions))}')
    tick_now = [0]
    audit = []
    handlers = {
        'move_entity': engine._handle_move_entity,
        'set_visual_state': engine._handle_set_visual_state,
        'set_runtime_state': engine._handle_set_runtime_state,
        'spawn_entity': engine._handle_spawn_entity,
        'set_weather': engine._handle_set_weather,
        'set_pedestrian_activity': engine._handle_set_pedestrian_activity,
        'remove_entity': engine._handle_remove_entity,
        'capture_screenshot': engine._handle_capture_screenshot,
    }

    def dispatch(action, origin):
        tick = tick_now[0]
        if action['type'] not in handlers:
            raise ValueError(f"unimplemented action type: {action['type']}")
        if origin == 'source_script' and tick >= config['window']['fork_tick'] \
                and action['action_id'] in omit:
            result = {'status': 'omitted_by_explicit_intervention'}
        else:
            result = handlers[action['type']](action, tick)
            if result['status'] != 'ok':
                raise RuntimeError(f"{action['action_id']}: {result}")
        audit.append({'tick': tick, 'origin': origin, 'action': copy.deepcopy(action),
                      'result': copy.deepcopy(result)})
        return result

    for kind in handlers:
        interpreter.register_handler(kind, lambda action: dispatch(action, 'source_script'))

    snapshots = {}
    sampled = {}
    for tick in range(config['window']['end_tick'] + 1):
        tick_now[0] = tick
        engine._apply_pending_runtime_state_patches(tick)
        for weather in engine.weather_transitions.get(tick, []):
            engine.weather = weather
        rows = engine._record_tick_rows(tick)
        sampled[tick] = copy.deepcopy(rows)
        for row in rows:
            interpreter.update_entity_state(row['entity_id'], row['pos_enu'], {}, row['vel_mps'])
            if row['label_class'] == 'pedestrian':
                interpreter.update_entity_activity(row['entity_id'], row['activity_type'])
        interpreter.update_weather_state(engine.weather_rows[-1])
        if tick in (config['window']['start_tick'], config['window']['fork_tick']):
            snapshots[tick] = {
                'tick': tick, 'phase': 'after_state_sample_before_actions',
                'sampled_states': copy.deepcopy(sampled[tick]),
                'entity_definitions': copy.deepcopy(engine.entities),
                'keyframes': copy.deepcopy(engine.keyframes),
                'weather': copy.deepcopy(engine.weather),
                'pending_runtime_state_patches': copy.deepcopy(engine.pending_runtime_state_patches),
                'interpreter': {
                    'trigger_states': {k: getattr(v, 'is_active', False)
                                       for k, v in interpreter.trigger_states.items()},
                    'event_states': {k: getattr(v, 'fire_count', 0)
                                     for k, v in interpreter.event_states.items()},
                },
            }
        interpreter.tick(tick)
        for step in config['execution']['actions']:
            if step['tick'] == tick:
                dispatch(step['action'], 'intervention')
    omitted = [r for r in audit if r['result']['status'] == 'omitted_by_explicit_intervention']
    if {r['action']['action_id'] for r in omitted} != omit:
        raise ValueError('configured source omission was not dispatched inside the window')
    return engine, interpreter, audit, snapshots


def episode_record(config):
    profile, seed, _ = config['group_key'].split('/')
    eid = config['episode_id']
    capture = f'aw_data/render_ready_episodes_capture_filtered/{eid}'
    with open_artifact_text(ROOT / f'aw_data/objective_semantic_truth/{eid}'
                            / 'event_occurrences.jsonl') as stream:
        observed = sorted({json.loads(line)['event_family_id'] for line in stream})
    families = sorted(set(observed) | set(config['target_families']))
    return {'episode_id': eid, 'profile': profile, 'seed': seed,
            'scene': config['sources']['scene'], 'script': config['sources']['script'],
            'capture_core_path': capture,
            'required_chain': [{'event_family_id': f} for f in families],
            'observed_event_families': observed}


def entity_scope(config):
    episode = episode_record(config)
    ids = semantics.identities(episode)
    return sorted(set(ids['actors']) | set(ids['uavs'])), episode


def semantics_for(config, trajectories, weather, historical, snapshot, action_audit):
    entity_ids, episode = entity_scope(config)
    metadata = semantics.metadata_for(episode, snapshot or {}, historical=historical)
    scope = semantics.scope_for(episode, metadata, entity_ids)
    declared = set(config['target_families']) - set(scope['families'])
    if declared:
        raise ValueError(f'declared families missing from semantic scope: {sorted(declared)}')
    # 语义重算只吃有静态类型来源的实体；引擎自带的背景车流不进入本声明范围。
    typed = [r for r in trajectories if r['entity_id'] in metadata]
    semantic, mechanisms, info = semantics.recompute(
        episode, {**config, 'entity_ids': entity_ids}, typed, weather, metadata, scope, action_audit)
    return semantic, mechanisms, info, scope, metadata, entity_ids


def window_status(config, semantic, trajectories):
    formal = set(range(config['window']['start_tick'], config['window']['end_tick'] + 1))
    present = {r['tick'] for r in trajectories}
    outcomes = [{'event_id': o['event_id'], 'family': o['event_family_id'],
                 'status': o['status'], 'terminal_tick': o['terminal_tick']}
                for o in semantic['event_outcomes']]
    pending = [o['event_id'] for o in semantic['event_outcomes'] if o['status'] == 'pending']
    families = Counter(o['event_family_id'] for o in semantic['event_occurrences'])
    status = ('right_censored' if pending else
              'events_terminal' if semantic['event_occurrences'] else
              'no_event_in_window_fully_observed')
    return {'window': config['window'], 'status': status, 'event_count': len(semantic['event_occurrences']),
            'event_families': dict(sorted(families.items())), 'pending_event_ids': pending,
            'outcomes': outcomes, 'ticks_missing_from_trajectory': sorted(formal - present),
            'unknown_is_not_false': True, 'absence_is_within_window_only': True}


def build_chain(semantic):
    events, relations = [], []
    for occurrence in semantic['event_occurrences']:
        events.append({'node_id': occurrence['event_id'], 'family': occurrence['event_family_id'],
                       'bindings': occurrence['bindings'], 'trigger_tick': occurrence['trigger_tick'],
                       'trigger_predicate_id': occurrence['trigger_predicate_id']})
        for ref in occurrence['supporting_transition_ids']:
            relations.append({'relation': 'predicate_supports_event', 'from': ref,
                              'to': occurrence['event_id'],
                              'evidence': 'current expression recomputation'})
    for outcome in semantic['event_outcomes']:
        relations.append({'relation': 'observed_outcome', 'from': outcome['event_id'],
                          'to': outcome['outcome_id'], 'status': outcome['status'],
                          'terminal_tick': outcome['terminal_tick']})
    return events, relations


def state_rows(config, trajectories, historical):
    """One row per declared entity per window tick; missing observation is explicit."""
    observed = {(r['tick'], r['entity_id']): r for r in trajectories}
    if len(observed) != len(trajectories):
        raise ValueError('duplicate (tick, entity_id) rows in trajectory')
    rows = []
    for tick in range(config['window']['start_tick'], config['window']['end_tick'] + 1):
        for entity in config['entity_ids']:
            row = observed.get((tick, entity))
            if row is None:
                rows.append({'tick': tick, 'entity_id': entity, 'observed': False,
                             'missing_reason': 'no recorded sample for this entity at this tick'})
                continue
            rows.append({**row, 'observed': True,
                         'record_provenance': ('historical_capture_filtered' if historical
                                               else 'current_engine_execution')})
    return rows


def run_arm(spec: dict) -> dict:
    """Execute one arm whose mechanism is written in its own script.

    The script owns every mechanism field (window, entities, predicates, actions,
    omissions, expectation).  This function only carries it out and persists the
    window assets; it holds no per-arm knowledge and never derives actions from an
    arm id.
    """
    required = {'arm_id', 'mode', 'profile', 'seed', 'anchor_event', 'neighbor_event',
                'chain_nodes', 'entity_ids', 'predicate_ids', 'target_families',
                'window', 'omit_action_ids', 'actions', 'expected', 'window_basis',
                'rationale', 'falsified_if', 'factual_root'}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"{spec.get('arm_id')}: arm spec lacks {sorted(missing)}")
    arm_id = spec['arm_id']
    profile, seed = spec['profile'], spec['seed']
    # 组名以本脚本所在的目录为准：脚本路径就是这条臂的身份，规格不重复声明后缀。
    script_self = Path(__file__).resolve()
    # run_arm 由臂脚本调用，调用方脚本即本臂身份；用调用栈定位它。
    import inspect
    caller = inspect.stack()
    arm_script = None
    for frame in caller:
        f = Path(frame.filename).resolve()
        if 'scripts/arms/L2-' in str(f):
            arm_script = f
            break
    if arm_script is None:
        raise ValueError(f"{spec.get('arm_id')}: run_arm must be called from an arm script")
    group = arm_script.parent.relative_to(ROOT / 'scripts/arms').as_posix()
    episode_id = f'{profile}__{seed}'
    subdir = 'interaction' if profile.startswith('L2-5') else 'failure'
    capture = f'aw_data/render_ready_episodes_capture_filtered/{episode_id}'
    scene = f'{SCENARIO_ROOT}/{subdir}/{profile}/scene_setup.json'
    script = f'{SCENARIO_ROOT}/{subdir}/{profile}/event_script.json'
    window = {**spec['window'], 'phase': 'after_state_sample_before_actions',
              'intervention_order': 'source_actions_then_intervention'}
    config = {
        'schema': 'aeroworld_l2_window_config', 'version': 1,
        'arm_id': arm_id, 'group_key': group, 'episode_id': episode_id,
        'profile': profile, 'seed': seed, 'anchor_event': spec['anchor_event'],
        'target_families': spec['target_families'], 'mode': spec['mode'],
        'entity_ids': sorted(set(spec['entity_ids'])),
        'predicate_ids': spec['predicate_ids'], 'window': window,
        'sources': {'scene': scene, 'script': script,
                    'trajectories': f'{capture}/trajectories.jsonl',
                    'weather': f'{capture}/weather_meta.jsonl',
                    'roster': f'{capture}/global_entity_roster.json'},
        'execution': {'omit_action_ids': spec['omit_action_ids'] if spec['mode'] == 'intervention' else [],
                      'actions': spec['actions'] if spec['mode'] == 'intervention' else []},
        'expected': {'target_event': spec['expected'] if spec['mode'] == 'intervention' else 'observe',
                     'rationale': spec['rationale'] if spec['mode'] == 'intervention'
                                  else '当前 factual 原始过程片段',
                     'falsified_if': spec['falsified_if'] if spec['mode'] == 'intervention'
                                     else '不适用：对照臂'},
        'script': f'scripts/arms/{group}/{arm_id}.py',
        'output': f'arms/{group}/{arm_id}',
        'experiment': f'design/{group}/experiment.json',
        'provenance': {'experiment_ref': f'design/{group}/experiment.json',
                       'parameter_sources': {'factual_root': spec['factual_root'],
                                             'window_basis': spec['window_basis'],
                                             **spec.get('parameter_sources', {})}},
    }
    schema = load(ROOT / SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(config),
                    key=lambda e: str(list(e.absolute_path)))
    if errors:
        raise ValueError(f'{arm_id}: generated config invalid: ' + '; '.join(e.message for e in errors))
    write_json(ROOT / 'interventions' / group / f'{arm_id}.yaml', config)
    experiment_path = ROOT / 'design' / group / 'experiment.json'
    if not experiment_path.is_file():
        write_json(experiment_path, {
            'schema': 'aeroworld_l2_window_experiment', 'version': 1,
            'group_key': group, 'episode_id': episode_id,
            'profile': profile, 'seed': seed,
            'anchor_event': spec['anchor_event'], 'neighbor_event': spec['neighbor_event'],
            'chain_nodes': spec['chain_nodes'],
            'target_families': spec['target_families'],
            'predicate_ids': spec['predicate_ids'],
            'window': window, 'window_basis': spec['window_basis'],
            'study_entities': sorted(set(spec['entity_ids'])),
            'arm_ids': [spec['arm_id']],
            'factual_root': spec['factual_root'],
            'write_scope': [f'design/{group}', f'interventions/{group}',
                            f'scripts/arms/{group}', f'arms/{group}'],
        })
    return _execute(config, group, arm_id, window, spec)


def capture_directory(config):
    return str(Path(config['sources']['trajectories']).parent)


def _execute(config, group, arm_id, window, spec):
    out = ROOT / config['output']
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing ARM output: {out}')
    historical = config['mode'] == 'factual_observation'
    audit = []
    # factual 是本组保存的原始过程，干预由配置明确声明。
    engine, interpreter, audit, snapshots = execute_engine(config)
    full_traj = engine.trajectory_rows
    full_weather = engine.weather_rows
    # 窗口产物只列本臂研究实体；语义重算另取窗口内已类型化的全部实体，
    # 因为事件绑定可能引用研究实体之外的参与者（例如起降锚点、冲突对手机）。
    study = set(config['entity_ids'])
    in_window = [r for r in full_traj if window['start_tick'] <= r['tick'] <= window['end_tick']]
    trajectories = [r for r in in_window if r['entity_id'] in study]
    weather = [r for r in full_weather if window['start_tick'] <= r['tick'] <= window['end_tick']]
    if not trajectories:
        raise ValueError('window contains no trajectory rows for the declared study entities')
    # 语义重算吃完整前史（tick 0..end），窗口起点之前已触发的事件才能带着真实
    # 前置生命周期进入；随后把产物按窗口裁切，保持「窗口闭区间」的交付口径。
    semantic_full, mechanisms_full, info, scope, metadata, entity_ids = semantics_for(
        config, full_traj, full_weather, historical,
        snapshots.get(window['start_tick']) or snapshots.get(window['fork_tick']), audit)
    lo, hi = window['start_tick'], window['end_tick']
    def crop(rows, key='tick'):
        return [r for r in rows if lo <= r[key] <= hi]
    def in_window_or_spanning(onset, terminal):
        """窗内触发，或窗口起点前触发但窗口内仍在持续/恢复的事件。"""
        if lo <= onset <= hi:
            return True
        return onset < lo and (terminal is None or terminal >= lo)
    original = {
        o['event_id']: o for o in semantic_full['event_occurrences']
        if in_window_or_spanning(o['detection_tick'],
                                 next((x['terminal_tick'] for x in semantic_full['event_outcomes']
                                       if x['event_id'] == o['event_id']), None))
    }
    kept_outcomes = [r for r in semantic_full['event_outcomes'] if r['event_id'] in original]
    from Dataset.semantic_truth.window_outcome import arm_window_outcome
    kept_outcomes = [arm_window_outcome(row, start_tick=lo, end_tick=hi)
                     for row in kept_outcomes]
    term = {r['event_id']: r['terminal_tick'] for r in kept_outcomes}
    semantic = {
        'predicate_truth': crop(semantic_full['predicate_truth'], 'tick'),
        'predicate_truth_ticks': crop(semantic_full['predicate_truth_ticks'], 'tick'),
        'predicate_transitions': [r for r in semantic_full['predicate_transitions']
                                  if lo <= r['to_tick'] <= hi],
        'event_occurrences': list(original.values()),
        'event_outcomes': kept_outcomes,
        'discontinuities': crop(semantic_full['discontinuities'], 'tick'),
    }
    mechanisms = [r for r in mechanisms_full if lo <= r['tick'] <= hi]
    status = window_status(config, semantic, trajectories)
    events, relations = build_chain(semantic)
    audit = [a for a in audit if window['start_tick'] <= a['tick'] <= window['end_tick']]
    write_json(out / 'raw/initial_state.json', snapshots.get(window['start_tick'], {}))
    write_json(out / 'raw/fork_state.json', snapshots.get(window['fork_tick'], {}))
    write_rows(out / 'raw/actions.jsonl', audit or [])
    write_rows(out / 'raw/trajectories.jsonl', trajectories)
    write_rows(out / 'raw/weather.jsonl', weather)
    semantic_trajectories = [r for r in in_window if r['entity_id'] in entity_ids]
    write_rows(out / 'raw/states.jsonl', state_rows(
        {**config, 'entity_ids': entity_ids}, semantic_trajectories, False))
    write_json(out / 'raw/metadata.json', metadata)
    from Dataset.tools.arm_ue_export import write_engine_inputs
    write_engine_inputs(out, engine, interpreter, window, script=engine.script)
    write_rows(out / 'raw/mechanism_states.jsonl', mechanisms)
    write_rows(out / 'objective/predicate_truth.jsonl', semantic['predicate_truth'])
    write_rows(out / 'objective/predicate_truth_ticks.jsonl', semantic['predicate_truth_ticks'])
    write_rows(out / 'objective/predicate_transitions.jsonl', semantic['predicate_transitions'])
    write_rows(out / 'objective/event_occurrences.jsonl', semantic['event_occurrences'])
    write_rows(out / 'objective/event_outcomes.jsonl', semantic['event_outcomes'])
    write_json(out / 'objective/window_status.json', status)
    write_json(out / 'objective/semantic_scope.json', scope)
    write_rows(out / 'chain/events.jsonl', events)
    write_rows(out / 'chain/relations.jsonl', relations)
    files = [str(p.relative_to(out)) for p in sorted(out.rglob('*'))
             if p.is_file() and p != out / 'manifest.json'
             and 'ue' not in p.relative_to(out).parts]
    manifest = {
        'schema_name': 'l2_event_chain_window_arm', 'schema_version': '1.0.0',
        'episode_id': config['episode_id'], 'profile': config['profile'],
        'seed': config['seed'], 'group_key': config['group_key'], 'arm_id': arm_id,
        'mode': config['mode'], 'anchor_event': config['anchor_event'],
        'neighbor_event': spec['neighbor_event'], 'chain_nodes': spec['chain_nodes'],
        'target_families': config['target_families'],
        # 研究实体（窗口产物与 A09 口径）与语义重算实体集分开记录。
        'entity_ids': sorted(config['entity_ids']),
        'semantic_entity_ids': entity_ids,
        'predicate_ids': config['predicate_ids'], 'window': window,
        'tick_hz': 10, 'formal_step_ticks': 5,
        'sources': {name: {'path': path}
                    for name, path in config['sources'].items()},
        'config': {'path': f'interventions/{group}/{arm_id}.yaml'},
        'arm_script': {'path': f'scripts/arms/{group}/{arm_id}.py'},
        'code': ['Dataset/tools/l2_window.py',
                                            'Dataset/tools/l2_arm_semantics.py',
                                            'Dataset/tools/batch_generate.py',
                                            'rules/l2_window.schema.json'],
        'execution': {'omit_action_ids': config['execution']['omit_action_ids'],
                      'actions': [s['action']['action_id'] for s in config['execution']['actions']]},
        'semantic_model': info, 'files': files,
        'window_status': status['status'], 'event_count': status['event_count'],
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform()},
        'data_provenance': {
            'factual_observation': 'current_engine_no_intervention_factual_baseline',
            'intervention': 'current_engine_with_declared_mechanism',
        }[config['mode']],
        'historical_reference': 'aw_data/render_ready_episodes_capture_filtered/<episode>/ (对照引用，非基线)',
        'snapshot_contract': 'execute_from_tick0_not_direct_restore',
        'generation_method': {
            'factual_observation': 'current_engine_no_intervention_factual_baseline',
            'intervention': 'current_engine_with_declared_mechanism',
        }[config['mode']],
        'limitations': ['窗口只保存声明区间；区间外为不可变来源引用',
                        '分段线性运动模型，速度可突变，不构成真实飞行可行性认证',
                        '未生成 UE RGB/depth/segmentation',
                        'historical factual 不是当前引擎的逐字节复现'],
    }
    write_json(out / 'manifest.json', manifest)
    from Dataset.tools.arm_ue_export import export_arm
    export_arm(out, ROOT / capture_directory(config),
               source_path(config['sources']['scene']), source_path(config['sources']['script']))
    print(f"ARM {config['episode_id']} {arm_id} events={status['event_count']} "
          f"status={status['status']}", flush=True)
    return manifest


if __name__ == '__main__':
    raise SystemExit('import only; dedicated arm scripts call execute_config')
