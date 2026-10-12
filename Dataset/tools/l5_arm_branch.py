"""Structural helpers an L5 ARM entry script uses to record its own branch.

An L5 ARM is one dedicated entry script.  That script owns the branch's logic:
it decides which authored events survive, what the hazard becomes, and what
extra nodes the branch needs.  These helpers do no deciding — they only write
the branch definition to disk in the standard place and hand back the paths the
ARM's configuration must declare, so the executor can be a pure runner.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_branch(staging_dir, script: dict, scene: dict, *, name: str) -> dict:
    """Record this ARM's branch files under a staging directory.

    The ARM's output directory belongs to the executor (it refuses to overwrite),
    so the branch definition is staged outside it and referenced from the
    configuration's sources.
    """
    from pathlib import Path as _Path
    staging = _Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    script_path = staging / f'{name}.script.json'
    scene_path = staging / f'{name}.scene.json'
    script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2) + '\n')
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n')
    return {'script': script_path, 'scene': scene_path}


def authored_events(script: dict) -> dict:
    """Authored events by id, in authored order."""
    return {e['event_id']: e for e in script['events']}


def select_events(script: dict, keep_ids) -> list:
    """The authored events to retain, in authored order."""
    keep = list(keep_ids)
    by_id = authored_events(script)
    missing = [e for e in keep if e not in by_id]
    if missing:
        raise ValueError(f'kept ids are not authored events: {missing}')
    return [by_id[e] for e in keep]


def set_trigger(script: dict, event_id: str, tick: int) -> None:
    """Anchor an authored event on an explicit tick of the formal grid."""
    if int(tick) % 5:
        raise ValueError(f'{event_id}: tick {tick} is off the formal 5-tick grid')
    event = authored_events(script)[event_id]
    for trigger in script['triggers']:
        if trigger['trigger_id'] == event['trigger_ref']:
            trigger_id = trigger['trigger_id']
            trigger.clear()
            trigger.update({'trigger_id': trigger_id, 'type': 'tick', 'tick': int(tick)})
            return
    raise ValueError(f'{event_id}: trigger {event["trigger_ref"]!r} not found')


def drop_actions(event: dict, action_types) -> dict:
    """Return a copy of an authored event without the named action types."""
    drop = set(action_types)
    kept = [a for a in event.get('actions', []) if a.get('type') not in drop]
    if not kept:
        raise ValueError(f"{event['event_id']}: removing {sorted(drop)} leaves no action")
    out = dict(event)
    out['actions'] = kept
    return out


def add_event(script: dict, event_id: str, tick: int, actions: list, *,
              intent: str, rationale: str, target_roles=()) -> None:
    """Append an event this branch designs, on an explicit tick of the grid."""
    if int(tick) % 5:
        raise ValueError(f'{event_id}: tick {tick} is off the formal 5-tick grid')
    if not actions:
        raise ValueError(f'{event_id}: needs at least one action')
    if not rationale.strip():
        raise ValueError(f'{event_id}: needs a rationale')
    trigger_id = f'designed_{event_id}'
    if any(t['trigger_id'] == trigger_id for t in script['triggers']):
        raise ValueError(f'{trigger_id}: already declared')
    if any(e['event_id'] == event_id for e in script['events']):
        raise ValueError(f'{event_id}: collides with an authored event')
    script['triggers'] = [*script['triggers'], {'trigger_id': trigger_id, 'type': 'tick', 'tick': int(tick)}]
    script['events'] = [*script['events'], {
        'event_id': event_id, 'trigger_ref': trigger_id, 'priority': 10, 'max_fire_count': 1,
        'actions': actions, 'intent': intent, 'target_roles': list(target_roles),
        'designed_by': 'arm_entry_script', 'designed_rationale': rationale}]


def bind_branch_sources(config_path, script_path, scene_path, *, window=None) -> None:
    """Point a configuration's sources at this ARM's branch files.

    Called by the ARM entry script after it has written its branch files and
    before the executor runs.
    """
    from pathlib import Path as _Path
    import yaml
    config_path = _Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    repo = _Path(__file__).resolve().parents[2]
    config['provenance']['experiment_ref'] = (
        f"design/{config['group_key']}/{config['arm_id']}.experiment.json")
    for key, path in (('script', script_path), ('scene', scene_path)):
        config['sources'][key] = str(_Path(path).resolve().relative_to(repo))
    if window is not None:
        # Replace the window wholesale: a merge would let a stale run's extra keys
        # survive into the configuration, which the schema rejects.
        keys = ('start_tick', 'fork_tick', 'end_tick', 'phase', 'intervention_order')
        merged = {**config.get('window', {}), **window}
        config['window'] = {k: merged[k] for k in keys if k in merged}
        missing = [k for k in ('start_tick', 'fork_tick', 'end_tick', 'phase', 'intervention_order')
                   if k not in config['window']]
        if missing:
            raise ValueError(f'branch window lacks {missing}')
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


def write_experiment(design_dir, *, arm_id, group_key, episode_id, window, hazard, kept, dropped,
                     designed, target_event, basis) -> 'Path':
    """Record this ARM's own experiment definition next to its branch files."""
    from pathlib import Path as _Path
    design = _Path(design_dir)
    design.mkdir(parents=True, exist_ok=True)
    path = design / f'{arm_id}.experiment.json'
    path.write_text(json.dumps({
        'schema': 'aeroworld_l5_env_branch_experiment', 'version': 1,
        'group_key': group_key, 'episode_id': episode_id, 'arm_id': arm_id,
        'window': {**window, 'phase': 'after_state_sample_before_actions',
                   'intervention_order': 'source_actions_then_intervention'},
        'hazard': hazard,
        'events': {'kept': kept, 'dropped': [{'event_id': e, 'reason': r} for e, r in dropped],
                   'designed': designed},
        'target_event': target_event,
        'window_basis': basis}, ensure_ascii=False, indent=2) + '\n')
    return path


def measured_window(script_path, scene_path, primary, fork_tick, episode_id, *,
                    duration: int = 900, step: int = 5, hold_samples: int = 3):
    """The shortest window that covers THIS branch's own consequence chain.

    The window cannot be derived from the factual: a branch that removed the
    response chain starts its final descent from a different position, so its
    landing finishes at a different tick.  Instead run this branch's own script
    and measure when the primary aircraft actually stops moving, then leave room
    for the recovery-hold confirmation of the consequence predicate.

    Returns the window (schema-shaped) and the measured evidence separately, so the
    ARM can record why its window has this length without polluting the configuration.
    """
    from Dataset.tools.l5_arm_engine import execute
    from Dataset.tools.l5_arm_common import ROOT

    engine, interpreter, audit, snaps, owners = execute({
        'sources': {'script': Path(script_path).resolve().relative_to(ROOT).as_posix(),
                    'scene': Path(scene_path).resolve().relative_to(ROOT).as_posix()},
        'window': {'start_tick': 0, 'fork_tick': int(fork_tick), 'end_tick': int(duration),
                   'phase': 'after_state_sample_before_actions',
                   'intervention_order': 'source_actions_then_intervention'},
        'entity_ids': [primary], 'primary_entity_id': primary,
        'episode_id': episode_id,
        'execution': {'omit_action_ids': [], 'actions': []},
    })
    moving = [r['tick'] for r in engine.trajectory_rows
              if r['entity_id'] == primary and max(abs(v) for v in r['vel_mps']) > 1e-6]
    if not moving:
        raise ValueError(f'{episode_id}: the branch never moves the primary aircraft')
    last_motion = max(moving)
    need = last_motion + (int(hold_samples) - 1) * int(step)
    end = min(int(duration), ((need // int(step)) + 1) * int(step))
    window = {'start_tick': int(fork_tick) - int(step), 'fork_tick': int(fork_tick),
              'end_tick': end,
              'phase': 'after_state_sample_before_actions',
              'intervention_order': 'source_actions_then_intervention'}
    evidence = {'last_motion_tick': last_motion,
                'recovery_hold_samples': int(hold_samples),
                'formal_step_ticks': int(step),
                'basis': ('run this branch over the full duration and take the last tick the primary '
                          'aircraft moves, plus the recovery-hold confirmation; this is measured from '
                          'the branch itself, not inherited from the factual')}
    return window, evidence
