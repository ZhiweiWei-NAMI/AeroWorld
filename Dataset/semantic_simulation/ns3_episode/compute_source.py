"""Exact 1.6.0 frame-presence/rank adapter for local resource jobs.

Compute 1.6.0 sees all UAV/vehicle entries, including render-offstage entries.
This differs deliberately from the radio adapter's render-active selection.
No gateway, coordinates, target event labels, or predicate results are inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

STEP_NS = 100_000_000


def compile_compute_frames(frames, *, episode_id, duration_ticks, schedule, source_ref,
                           schedule_ref):
    """Normalize source-present epochs and inclusive-tick availability intervals.

    Category-local ranks are recomputed per frame, as _build_compute_rows does.
    Samples describe [tick,tick+1); H only closes observation, not actor life.
    """
    for row in schedule:
        if set(row) != {'entity_category', 'entity_rank', 'start_tick', 'end_tick'}:
            raise ValueError('source schedule must match the exact 1.6.0 fields')
        if row['entity_category'] not in ('uav', 'vehicle'):
            raise ValueError('unsupported compute category')
        if any(type(row[key]) is not int for key in ('entity_rank', 'start_tick', 'end_tick')):
            raise ValueError('source schedule rank/ticks must be integer')
        if row['entity_rank'] < 0 or not 0 <= row['start_tick'] <= row['end_tick']:
            raise ValueError('invalid source schedule interval/rank')
    h = duration_ticks * STEP_NS
    nodes, live, epochs, unavailable, open_outages = [], {}, {}, [], {}
    selector_spans, open_selectors = [], {}
    count = 0
    for tick, frame in enumerate(frames):
        count += 1
        if frame['episode_id'] != episode_id or frame['tick'] != tick or frame['tick_hz'] != 10:
            raise ValueError('compute source must use the exact episode 10Hz clock')
        if tick > duration_ticks:
            raise ValueError('source extends past the declared horizon')
        present = {}
        for entity in frame['entities']:
            owner, category = entity['entity_id'], entity['entity_category']
            if not isinstance(owner, str) or not owner or owner in present:
                raise ValueError('entity IDs must be unique nonempty strings per frame')
            if category in ('uav', 'vehicle'):
                present[owner] = category
        now = tick * STEP_NS
        for owner in tuple(live):
            if owner not in present:
                live.pop(owner)['death_ns'] = now
        for owner, category in sorted(present.items()):
            if owner in live and live[owner]['node_type'] != category:
                raise ValueError('category changes inside one source-present life epoch')
            if owner not in live:
                epoch = epochs.get(owner, 0)
                epochs[owner] = epoch + 1
                node = {'node_id': f'{episode_id}:{owner}:life{epoch}', 'owner': owner,
                        'life_epoch': epoch, 'node_type': category, 'birth_ns': now,
                        'death_ns': None, 'source_ref': source_ref,
                        'lifecycle_authority': 'compute source-present epochs; exit is source absence, not hardware death'}
                nodes.append(node)
                live[owner] = node
        down = set()
        bindings = {}
        if now < h:
            ranked = {category: sorted(owner for owner, kind in present.items() if kind == category)
                      for category in ('uav', 'vehicle')}
            for selector_index, selector in enumerate(schedule):
                if selector['start_tick'] <= tick <= selector['end_tick']:
                    owners = ranked[selector['entity_category']]
                    rank = selector['entity_rank']
                    owner = owners[rank] if rank < len(owners) else None
                    bindings[selector_index] = (owner, None if owner is None else live[owner]['life_epoch'])
            for category in ('uav', 'vehicle'):
                owners = ranked[category]
                for rank, owner in enumerate(owners):
                    if any(row['entity_category'] == category and row['entity_rank'] == rank
                           and row['start_tick'] <= tick <= row['end_tick'] for row in schedule):
                        down.add(owner)
        for selector_index in tuple(open_selectors):
            previous = open_selectors[selector_index]
            if bindings.get(selector_index) != (previous['owner_entity_id'], previous['life_epoch']):
                previous['end_ns'] = now
                selector_spans.append(open_selectors.pop(selector_index))
        for selector_index, (owner, epoch) in bindings.items():
            if selector_index not in open_selectors:
                open_selectors[selector_index] = {
                    'selector_index': selector_index, 'source_selector': dict(schedule[selector_index]),
                    'owner_entity_id': owner, 'life_epoch': epoch, 'start_ns': now,
                    'status': 'KNOWN' if owner is not None else 'UNKNOWN',
                    'reason': None if owner is not None else 'source_rank_has_no_present_owner',
                    'source_ref': schedule_ref+'/'+str(selector_index)}
        for owner in tuple(open_outages):
            if owner not in down:
                unavailable.append({'owner_entity_id': owner, 'start_ns': open_outages.pop(owner),
                                    'end_ns': now, 'source_ref': schedule_ref})
        for owner in down:
            if owner not in open_outages:
                open_outages[owner] = now
    if count != duration_ticks + 1:
        raise ValueError('compute source does not cover all horizon samples')
    if open_outages:
        raise ValueError('horizon closure failed')
    return {'episode_id': episode_id, 'duration_ns': h, 'nodes': nodes,
            'source_ref': source_ref, 'local_unavailability_intervals_ns': unavailable,
            'selector_binding_version': 'p09.compute.source-rank-owner-spans/v1',
            'selector_owner_spans': selector_spans,
            'presence_policy': 'same UAV/vehicle frame entries as compute 1.6.0; no render crop/filter here',
            'schedule_mapping': 'category-local lexicographic rank each tick; inclusive end_tick becomes end+1 tick; intervals clipped to H'}


def load_compute_episode(source_root, episode_id, legacy_profile, profile_ref):
    source = Path(source_root) / episode_id
    manifest = json.loads((source / 'episode_manifest.json').read_text())
    if manifest['episode_id'] != episode_id:
        raise ValueError('episode manifest binding differs')
    if legacy_profile['model']['model_version'] != '1.6.0':
        raise ValueError('exact compute source adapter requires version1.6.0')
    with (source / 'truth_frames.jsonl').open() as stream:
        episode = compile_compute_frames((json.loads(line) for line in stream),
            episode_id=episode_id, duration_ticks=manifest['duration_ticks'],
            schedule=legacy_profile['compute']['node_unavailability_schedule'],
            source_ref=str(source / 'truth_frames.jsonl'),
            schedule_ref=profile_ref+'#/compute/node_unavailability_schedule')
    episode['seed'] = manifest['seed']
    episode['manifest_ref'] = str(source / 'episode_manifest.json')
    return episode
