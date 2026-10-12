"""Decision-time role binding from typed prospective and observed causal context.

No post-decision state, future occurrence label, or arm-specific actor list is
an input. Entity spelling is used only as identity and for canonical ordering
of role sets explicitly declared unordered by the predicate contract.
"""
from collections import defaultdict
from collections.abc import Mapping


FIXED_GEOMETRY_PREDICATES = frozenset({
    'airspace.aircraft_airborne', 'airspace.aircraft_touchdown',
    'airspace.aircraft_outside_assigned_altitude_band',
    'safety.aircraft_pair_below_separation', 'safety.aircraft_near_pedestrian',
    'safety.aircraft_near_ground_vehicle', 'safety.aircraft_near_building_structure',
    'safety.vehicle_pedestrian_conflict',
})


def scene_geometry_classes(scene):
    category_classes = {'uav': 'world:UnmannedAircraft', 'vehicle': 'world:GroundVehicle',
                        'pedestrian': 'world:Pedestrian', 'building': 'world:BuildingStructure',
                        'structure': 'world:BuildingStructure', 'facade_anchor': 'world:BuildingStructure'}
    result = {}
    for entity in scene['entities']:
        identity = entity['entity_id']
        if identity in result:
            raise ValueError('duplicate scene entity identity')
        classes = []
        category = entity.get('entity_category', entity.get('category'))
        if category in category_classes:
            classes.append(category_classes[category])
        result[identity] = classes
    return result


def _targets(event):
    result = []
    for source in (event, event.get('log_event', {})):
        if isinstance(source, Mapping):
            for value in source.get('target_ids', []):
                if isinstance(value, str) and value and value not in result:
                    result.append(value)
    for action in event.get('actions', []):
        for field in ('entity_id', 'actor_id', 'target_id', 'ped_id'):
            value = action.get(field)
            if isinstance(value, str) and value and value not in result:
                result.append(value)
    return result


def bind_causal_query(*, template, script, anchor_event_id, fork_tick, entity_classes, realizations):
    events = {event['event_id']: event for event in script['events']}
    if len(events) != len(script['events']):
        raise ValueError('duplicate causal event identity')
    triggers = {row['trigger_id']: row for row in script['triggers']}
    if len(triggers) != len(script['triggers']):
        raise ValueError('duplicate trigger identity')
    if anchor_event_id not in events:
        raise ValueError('query anchor is not an authored event')
    available = set()
    for row in realizations:
        tick = row.get('dispatch_tick')
        if isinstance(tick, int) and not isinstance(tick, bool) and tick <= fork_tick:
            available.add(row['event_id'])
    roles_by_class = defaultdict(list)
    for role in template['argument_roles']:
        roles_by_class['world:'+role['class']].append(role['key'])
    unordered = {frozenset(group) for group in template['grounding_spec']['unordered_role_sets']}
    candidates = {kind: [] for kind in roles_by_class}
    chain = []
    current = anchor_event_id
    reason = 'no_available_causal_predecessor'
    while current:
        if current in chain:
            raise ValueError('causal query dependency cycle')
        chain.append(current)
        event = events[current]
        target_ids = _targets(event)
        trigger = triggers.get(event.get('trigger_ref'))
        if isinstance(trigger, Mapping) and trigger.get('type') == 'entity_proximity':
            target_ids.extend([trigger['entity_a'], trigger['entity_b']])
        for kind, roles in roles_by_class.items():
            if len(candidates[kind]) == len(roles):
                continue
            for entity in target_ids:
                if kind in entity_classes.get(entity, ()) and entity not in candidates[kind]:
                    candidates[kind].append(entity)
            if len(candidates[kind]) > len(roles):
                return {'status': 'ambiguous', 'bindings': {}, 'causal_event_ids': chain,
                        'reason': 'multiple_typed_targets_for_role', 'candidates': candidates}
        if all(len(candidates[kind]) == len(roles) for kind, roles in roles_by_class.items()):
            bindings = {}
            for kind, roles in roles_by_class.items():
                choices = candidates[kind]
                if len(roles) > 1:
                    if frozenset(roles) not in unordered:
                        return {'status': 'ambiguous', 'bindings': {}, 'causal_event_ids': chain,
                                'reason': 'same_class_roles_lack_an_explicit_ordering_contract', 'candidates': candidates}
                    choices = sorted(choices)
                bindings.update(zip(roles, choices))
            return {'status': 'resolved', 'bindings': bindings, 'causal_event_ids': chain,
                    'reason': 'typed_anchor_and_available_causal_context', 'candidates': candidates}
        trigger = triggers.get(event.get('trigger_ref'))
        if isinstance(trigger, Mapping) and trigger.get('type') == 'event_fired_after':
            predecessor = trigger['event_id']
            current = predecessor if predecessor in available else None
        else:
            intent = event.get('causal_predecessor_intent')
            matching = [identity for identity, source in events.items()
                        if intent and identity in available and identity != current and source.get('intent') == intent]
            if len(matching) > 1:
                reason = 'ambiguous_available_causal_predecessor'
            current = matching[0] if len(matching) == 1 else None
    return {'status': 'unknown', 'bindings': {}, 'causal_event_ids': chain,
            'reason': reason, 'candidates': candidates}
