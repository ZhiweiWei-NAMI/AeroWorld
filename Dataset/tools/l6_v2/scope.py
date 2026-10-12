"""Declared scope from the engine's own entity metadata and the authored script.

Actor identities come from actual source commands and declared roles. Station,
pad and GCS identities come from the engine entity definitions. No historical
capture file is consulted.
"""
from __future__ import annotations

from typing import Any, Mapping

from Dataset.tools.l6_v2.common import ROOT, load


def select_scope(engine: Any, script: Mapping[str, Any]) -> dict:
    definitions = engine.entities
    moving = {a['entity_id'] for e in script['events'] for a in (e.get('actions') or [])
              if a.get('type') == 'move_entity' and a.get('entity_id')}
    aircraft = sorted(e for e in moving
                      if e in definitions and definitions[e].get('label_class') == 'uav')
    acted = {a['entity_id'] for e in script['events']
             for a in (e.get('actions') or []) if a.get('entity_id')}
    communication = load(ROOT / 'Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json')
    backup = communication['communication']['station_profiles']['base_station']['backup_station_id']
    # A declared handover destination is an observed station even when no action
    # changes it. Include the real scene entity, never synthesize a backup site.
    if backup in definitions:
        acted.add(backup)
    runtime_targets = {a['entity_id'] for e in script['events']
                       for a in (e.get('actions') or [])
                       if a.get('type') == 'set_runtime_state' and a.get('entity_id')}
    stations = sorted(
        e for e, v in definitions.items()
        if e in acted
        and str(v.get('entity_kind') or '').startswith('facility.base_station'))
    gcs = sorted(e for e, v in definitions.items()
                 if str(v.get('entity_kind') or '') == 'ground_station.ground_control_station')
    pads = sorted({definitions[e]['lifecycle']['home_pad_entity_id'] for e in aircraft})
    entities = sorted(set(aircraft) | set(stations) | set(gcs) | set(pads) | runtime_targets)
    if not aircraft:
        raise ValueError('no authored aircraft found in engine definitions')
    if not stations and not gcs:
        raise ValueError('no authored communication station or GCS found')
    if any(e not in definitions for e in entities):
        raise ValueError('declared scope entity absent from engine definitions')
    return {'entity_ids': entities, 'aircraft_ids': aircraft, 'station_ids': stations,
            'gcs_ids': gcs, 'home_pad_ids': pads}


def aircraft_geometry(engine: Any, scope: Mapping[str, Any]) -> dict:
    geometry = {}
    for entity_id in scope['aircraft_ids']:
        entity = engine.entities[entity_id]
        lifecycle = entity['lifecycle']
        geometry[entity_id] = {
            'home_pad_id': lifecycle['home_pad_entity_id'],
            'ground_reference_z_m': entity['ground_reference_z_m'],
            'assigned_altitude_m': entity['assigned_altitude_m'],
        }
    return geometry
