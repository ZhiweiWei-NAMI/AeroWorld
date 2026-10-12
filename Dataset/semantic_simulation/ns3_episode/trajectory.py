"""Frozen 10Hz truth replay, preserving source-present radio life epochs."""
from __future__ import annotations
import json
from pathlib import Path

STEP_NS = 100_000_000


def load_episode(root, episode_id, *, gateway_entity_id):
    root = Path(root)
    source = root / 'aw_data/render_ready_episodes' / episode_id
    manifest = json.loads((source / 'episode_manifest.json').read_text())
    duration_ns = int(manifest['duration_ticks']) * STEP_NS
    roster = json.loads((source / 'global_entity_roster.json').read_text())['entities']
    roster_order = {row['entity_id']: index for index, row in enumerate(roster)}
    selected = [row for row in roster if row['entity_id'] == gateway_entity_id]
    if len(selected) != 1:
        raise ValueError('explicit gateway must resolve to one exact roster entity')
    if selected[0]['semantic_scope']['ontology_class_id'] != 'world:CommunicationStation':
        raise ValueError('explicit gateway must be a typed CommunicationStation')
    station_id = gateway_entity_id
    nodes, mobility, live, epochs = [], [], {}, {}
    gateway = None
    with (source / 'truth_frames.jsonl').open() as stream:
        for index, line in enumerate(stream):
            frame = json.loads(line)
            if frame['episode_id'] != episode_id or frame['tick'] != index or frame['tick_hz'] != 10:
                raise ValueError('frozen episode truth clock differs from 10Hz source contract')
            time_ns = index * STEP_NS
            present = {}
            for entity in frame['entities']:
                owner = entity['entity_id']
                category = entity['entity_category']
                if category not in ('uav', 'vehicle') and owner != station_id:
                    continue
                if entity['render_presence']['offstage']:
                    continue
                position = list(entity['truth_pose']['position_enu_m'])
                if owner == station_id:
                    position[2] += 3.0
                    category = 'gateway'
                present[owner] = (category, position)
            for owner in tuple(live):
                if owner not in present:
                    live.pop(owner)['death_ns'] = time_ns
            for owner, (category, position) in present.items():
                if owner not in live:
                    epoch = epochs.get(owner, 0)
                    epochs[owner] = epoch + 1
                    node = {'node_id': f'{episode_id}:{owner}:life{epoch}', 'owner': owner,
                        'life_epoch': epoch, 'node_type': category, 'birth_ns': time_ns,
                        'death_ns': None, 'ns3_node_id': len(nodes),
                        'infra_id': owner if category == 'gateway' else None,
                        'ontology_class': {'uav':'world:UnmannedAircraft','vehicle':'world:GroundVehicle','gateway':'world:CommunicationStation'}[category],
                        'position_enu_m': position, 'source_ref': str(source / 'truth_frames.jsonl'),
                        'lifecycle_authority': 'source-present active intervals; horizon is censoring'}
                    nodes.append(node); live[owner] = node
                    if category == 'gateway':
                        if gateway is not None:
                            raise ValueError('gateway must have one continuous source-present life epoch')
                        gateway = node['node_id']
            mobility.append({'time_ns': time_ns, 'positions': [
                {'node_id': live[owner]['node_id'], 'position_enu_m': position, 'active': True}
                for owner, (_, position) in present.items()]})
    if len(mobility) != manifest['duration_ticks'] + 1 or mobility[-1]['time_ns'] != duration_ns:
        raise ValueError('truth replay does not cover the complete episode horizon')
    if gateway is None:
        raise ValueError('authored gateway is absent from frozen truth')
    uav_ids = sorted({node['owner'] for node in nodes if node['node_type'] == 'uav'}, key=roster_order.__getitem__)
    return {'episode_id': episode_id, 'duration_ns': duration_ns, 'nodes': nodes,
        'mobility': mobility, 'source_ref': str(source / 'truth_frames.jsonl'),
        'gateway_node_id': gateway, 'roster_uav_ids': uav_ids,
        'gateway_entity_id': station_id,
        'mobility_clock_ns': STEP_NS, 'gateway_antenna_offset_m': [0,0,3]}
