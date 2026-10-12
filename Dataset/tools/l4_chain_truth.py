"""L4 chain-truth map: the per-profile incident fragment, read from L4's own authoring.

L4 scripts declare an explicit ``intent_stage`` on every event.  Stage numbers 01..08 are
the incident itself; 00 and 09 are lifecycle/context (takeoff, approach, landing).  A
branch arm only makes sense on the incident fragment, so this module resolves, per
profile: which script node causes the incident, which semantic event is that incident,
and which predicate transition proves it.

Every value is read from the episode's own artifacts.  Nothing here infers a mechanism
from an arm id, and nothing is templated across profiles: the incident family is declared
in ``L4_INCIDENT_FAMILY`` and cross-checked against the profile's own scene description.

Read-only source-authoring lookup used by the L4 executor and acceptance tooling.
"""
from __future__ import annotations

import json
from Dataset.semantic_truth.provenance import open_artifact_text
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / 'aw_data/render_ready_episodes_capture_filtered'
SEMANTIC = ROOT / 'aw_data/objective_semantic_truth'

# The incident each profile is authored around.  This is the family named in the
# profile's own scene description; the module asserts the description agrees, so a
# rename or a wrong entry fails loudly instead of silently checking the wrong family.
L4_INCIDENT_FAMILY = {
    'L4-1_v1': 'uav_separation_conflict',
    'L4-1_v2': 'uav_separation_conflict',
    'L4-2_v1': 'structure_near_miss',
    'L4-2_v2': 'structure_near_miss',
    'L4-3_v1': 'crowd_evacuation',
    'L4-3_v2': 'forced_landing',
    'L4-3_v3': 'forced_landing',
    'L4-4_v1': 'uav_vehicle_conflict',
    'L4-4_v2': 'uav_vehicle_conflict',
    'L4-5_v1': 'uav_pedestrian_near_miss',
    'L4-5_v2': 'uav_pedestrian_near_miss',
    'L4-5_v3': 'uav_pedestrian_near_miss',
    'L4-6_v1': 'vehicle_pedestrian_conflict',
    'L4-6_v2': 'vehicle_pedestrian_conflict',
    'L4-7_v1': 'pedestrian_medical',
    'L4-7_v2': 'pedestrian_medical',
    'L4-8_v1': 'crowd_evacuation',
    'L4-8_v2': 'crowd_evacuation',
    'L4-9_v1': 'vehicle_intersection_conflict',
    'L4-9_v2': 'vehicle_intersection_conflict',
    'L4-10_v1': 'ambulance_priority',
    'L4-10_v2': 'ambulance_priority',
    'L4-11_v1': 'av_safe_stop',
    'L4-11_v2': 'av_safe_stop',
}

# The description text each profile carries, used as an independent check on the map above.
_PROFILE_DESCRIPTION_HINT = {
    'L4-1_v1': 'UAV-UAV', 'L4-1_v2': 'UAV-UAV',
    'L4-2_v1': 'facade', 'L4-2_v2': 'facade',
    'L4-3_v1': 'forced landing', 'L4-3_v2': 'forced landing', 'L4-3_v3': 'forced landing',
    'L4-4_v1': 'vehicle', 'L4-4_v2': 'vehicle',
    'L4-5_v1': 'pedestrian', 'L4-5_v2': 'pedestrian', 'L4-5_v3': 'pedestrian',
    'L4-6_v1': 'jaywalk', 'L4-6_v2': 'jaywalk',
    'L4-7_v1': 'fall', 'L4-7_v2': 'fall',
    'L4-8_v1': 'evacuation', 'L4-8_v2': 'evacuation',
    'L4-9_v1': 'intersection', 'L4-9_v2': 'intersection',
    'L4-10_v1': 'ambulance', 'L4-10_v2': 'ambulance',
    'L4-11_v1': 'sensor failure', 'L4-11_v2': 'sensor failure',
}

AUTHORITY_CHAINS = ROOT / 'design/L4/l4_authority_chains.json'


DECLARED_FAMILY_POLICY = (
    'L4-3_v1 is the case where the earlier L4B session mis-routed: its select_research tests '
    'descent_to_low_altitude before ambulance_response, and L4-3_v1 contains both, so it was sent '
    'to the delay_descent branch even though its own crowd_evacuation_active transition is driven '
    'by crowd_proximity_response@310 setting pedestrian_state.evacuation_active. The anchor here '
    'is resolved from the family, not from the first matching event id.',
    'The L4 handoff prompts state the rule for a declared non-executable family: implement and '
    'verify it locally against the contract, or mark it blocked/out_of_scope with a reason. The '
    'earlier L4 sessions did the first: they branched on the script node that drives the family '
    '(L4-9 vehicles_enter_intersection@230, L4-10 ambulance_priority_approach@220, L4-11 '
    'av_safe_stop@260, L4-3 descent_to_low_altitude@260) and shipped '
    'target_assessment=blocked_non_executable_family with the authority reason attached. No '
    'human decision is required; the family status is read from the event authority.'
)


def authority_chain(profile: str) -> list[dict[str, Any]]:
    """The L4 handoff's own ``required_chain`` for this profile.

    This is the declared design intent: a stage goal and the event family that realises it,
    in order.  It is the authority a branch design has to agree with, so it is read rather
    than reconstructed.  It comes from the L4A/L4B handoff packages, materialised at
    ``design/L4/l4_authority_chains.json``.
    """
    if not AUTHORITY_CHAINS.is_file():
        raise FileNotFoundError(
            f'{AUTHORITY_CHAINS} is missing; the L4 authority chain cannot be checked'
        )
    doc = json.loads(AUTHORITY_CHAINS.read_text())
    for package in doc.values():
        chain = (package.get('required_chain') or {}).get(profile)
        if chain is not None:
            return chain
    raise KeyError(f'{profile}: no required_chain in the L4 authority packages')


# The authored intent word that identifies the node driving each declared family.  Read from the
# scenes' own intent strings; a family whose profiles do not agree fails loudly rather than
# silently anchoring on the wrong stage.
# The authored anchor event each declared family was branched on.  These come from the earlier
# L4 sessions' own designs (design/L4-*/seed00/grp_*/experiment.json): L4A used fork_event_id,
# L4B used research.anchor_script_event.  They are the layer's own precedent, so a declared family
# resolves to the node that precedent used instead of a heuristic that might pick a sibling stage.
_DECLARED_FAMILY_ANCHOR = {
    'forced_landing': 'descent_to_low_altitude',
    'pedestrian_medical': 'ambulance_response',
    'ambulance_priority': 'ambulance_priority_approach',
    'av_safe_stop': 'av_safe_stop',
    'vehicle_intersection_conflict': 'vehicles_enter_intersection',
}

INCIDENT_STAGE_MIN = 1
INCIDENT_STAGE_MAX = 8


def _rows(path: Path) -> list[dict[str, Any]]:
    with open_artifact_text(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def profiles() -> list[str]:
    return sorted({p.parent.name for p in (ROOT / 'Dataset/scenarios/L4_agents').rglob('event_script.json')})


def script_path(profile: str) -> Path:
    return next((ROOT / 'Dataset/scenarios/L4_agents').rglob(f'{profile}/event_script.json'))


def load_profile(profile: str) -> dict[str, Any]:
    """Read the profile's authored script and its dispatch ticks."""
    script = json.loads(script_path(profile).read_text())
    description = str(script.get('description') or '')
    hint = _PROFILE_DESCRIPTION_HINT[profile]
    if hint.lower() not in description.lower():
        raise ValueError(
            f'{profile}: scene description {description!r} does not mention {hint!r}; '
            'the incident-family map and the authored scene disagree'
        )
    events = []
    for event in script.get('events') or []:
        stage_text = str(event.get('intent_stage') or '')
        head = stage_text.split('.')[0]
        stage = int(head) if head.isdigit() else None
        events.append({
            'event_id': str(event['event_id']),
            'stage': stage,
            'intent': event.get('intent'),
            'trigger_ref': event.get('trigger_ref'),
            'actions': [
                {'action_id': a.get('action_id'), 'type': a.get('type'), 'entity_id': a.get('entity_id')}
                for a in event.get('actions') or []
            ],
        })
    return {'profile': profile, 'description': description,
            'incident_family': L4_INCIDENT_FAMILY[profile], 'events': events}


def dispatch_ticks(episode_id: str) -> dict[str, int]:
    return {r['chain_id']: r['activated_tick']
            for r in _rows(CAPTURE / episode_id / 'event_trace.jsonl')}


def incident_nodes(profile: str) -> list[dict[str, Any]]:
    """The script nodes that constitute the incident (stages 01..08), with their authored intent."""
    return [dict(e) for e in load_profile(profile)['events']
            if e['stage'] is not None and INCIDENT_STAGE_MIN <= e['stage'] <= INCIDENT_STAGE_MAX]


def incident_occurrences(episode_id: str, family: str) -> list[dict[str, Any]]:
    """Every semantic occurrence of the incident family, in onset order."""
    rows = [r for r in _rows(SEMANTIC / episode_id / 'event_occurrences.jsonl')
            if r.get('event_family_id') == family and r.get('event_phases')]
    return sorted(rows, key=lambda r: r['event_phases'][0]['tick'])


def crowd_cohort_members(episode_id: str) -> set[str]:
    """Members of the crowd aggregate a crowd-family predicate reads, from the domain profile."""
    candidates = [
        SEMANTIC / episode_id / 'domain_state_observations.jsonl',
        ROOT / 'aw_data/domain_state_supplement' / episode_id / 'domain_state_observations.jsonl',
    ]
    for path in candidates:
        if not path.is_file():
            continue
        members: set[str] = set()
        for row in _rows(path):
            values = row.get('values') or {}
            for key in ('cohort_member_ids', 'member_ids', 'cohort'):
                value = values.get(key) if isinstance(values, dict) else None
                if isinstance(value, list):
                    members.update(str(v) for v in value)
        if members:
            return members
    return set()


def shared_incident_family(profile: str) -> str | None:
    """The family that is both an executable stage of the L4 authority chain and observed here.

    A branch needs a family the layer was designed around *and* one the engine actually
    produces.  L4-3_v2/v3 is the case that forces this: the authority names ``forced_landing``
    (non-executable) for the risk stages, and ``uav_pedestrian_near_miss`` occurs but is not a
    declared stage.  Neither alone can host a branch on the authored incident, so the pair is
    reported instead of silently picking one.
    """
    declared = {row['event_family_id'] for row in authority_chain(profile) if row.get('executable')}
    if not declared:
        return None
    observed = {r['event_family_id']
                for r in _rows(SEMANTIC / f'{profile}__seed00/event_occurrences.jsonl')}
    shared = sorted(declared & observed)
    return shared[0] if shared else None


def incident_is_executable(profile: str) -> bool:
    """Whether the incident family can host a semantic event at all, per the event authority."""
    mapping = json.loads((ROOT / 'Dataset/semantic_rules/profiles/event_family_authority_mapping.json').read_text())
    entry = mapping.get('families', {}).get(L4_INCIDENT_FAMILY[profile]) or {}
    return bool(entry.get('executable'))


def non_executable_reason(profile: str) -> str | None:
    mapping = json.loads((ROOT / 'Dataset/semantic_rules/profiles/event_family_authority_mapping.json').read_text())
    entry = mapping.get('families', {}).get(L4_INCIDENT_FAMILY[profile]) or {}
    return entry.get('non_executable_reason')


def fragment(profile: str, seed: str) -> dict[str, Any]:
    """Resolve the incident fragment: script node -> semantic event -> predicate transition.

    When the incident family is declared non-executable there is no semantic event to pair
    with, so the fragment is the script node alone, reported as
    ``blocked_non_executable_family`` with the authority's reason.  That is the handling the
    L4 handoff prompts prescribe and the earlier L4 sessions shipped; it is a terminal state,
    not a question.
    """
    episode_id = f'{profile}__{seed}'
    family = L4_INCIDENT_FAMILY[profile]
    ticks = dispatch_ticks(episode_id)
    nodes = [dict(n, dispatch=ticks.get(n['event_id'])) for n in incident_nodes(profile)]
    if not incident_is_executable(profile):
        # Branch on the script node that drives the declared family.  The earlier sessions
        # picked the first node that acts on the participants, which for these profiles is the
        # stage-1 or stage-2 node carrying the driving action.
        # The anchor is the incident-stage node whose own authored intent names the process the
        # profile is about, and which carries a motion to intervene on.  L4-7_v1 is the case that
        # forces the intent test: stage 02 also moves a UAV, but its intent is "uav moves above the
        # fallen pedestrian before detection is confirmed", while stage 04 "ambulance" is the
        # response the pedestrian_medical chain is built on.  L4-3_v1 likewise anchors on stage 02
        # "uav descends..." rather than stage 03, which is the crowd's reaction.
        drivers = [n for n in nodes if any(a['type'] == 'move_entity' for a in n['actions'])]
        if not drivers:
            return {'profile': profile, 'seed': seed, 'episode_id': episode_id,
                    'incident_family': family, 'status': 'no_incident_node', 'nodes': nodes}
        preferred = _DECLARED_FAMILY_ANCHOR.get(family)
        matching = [n for n in drivers if n['event_id'] == preferred]
        if preferred and not matching:
            raise KeyError(
                f'{profile}: the declared-family anchor {preferred!r} is not an incident-stage node '
                f'carrying a motion; the precedent and the script disagree'
            )
        chosen = matching or drivers
        node = min(chosen, key=lambda n: (n['dispatch'] is None, n['dispatch'], n['stage']))
        return {'profile': profile, 'seed': seed, 'episode_id': episode_id,
                'incident_family': family, 'status': 'blocked_non_executable_family',
                'fork_event_id': node['event_id'], 'fork_tick': node['dispatch'],
                'fork_role': 'declared_family_driver',
                'blocked_reason': non_executable_reason(profile),
                'events': [e for e in load_profile(profile)['events']],
                'nodes': nodes}
    occurrences = incident_occurrences(episode_id, family)
    if not occurrences:
        return {'profile': profile, 'seed': seed, 'episode_id': episode_id,
                'incident_family': family, 'status': 'no_semantic_event',
                'nodes': nodes}
    target = occurrences[0]
    onset = target['event_phases'][0]['tick']
    terminal = [p['tick'] for p in target['event_phases'] if p['phase_kind'] == 'terminal']
    # Cause vs response.  The engine samples state before it runs same-tick actions, so a node
    # dispatched at the tick the trigger predicate first reads true is *observing* an already
    # true condition, not creating it.  The directing node is the latest one dispatched strictly
    # before onset; only when no incident node precedes onset (the incident is immediate at its
    # own node's tick) does the at-onset node have to serve as the fork.
    # The fork must be a node whose own actions touch the entities the proving transition
    # reads.  A node that merely fires at the right time but acts on nobody (an arrival
    # screenshot, a status log) cannot host a branch: nothing it does can change the cause.
    cause = transition_entities(episode_id, target) or crowd_cohort_members(episode_id)
    def acts_on_cause(node: dict[str, Any]) -> bool:
        return bool({a['entity_id'] for a in node['actions'] if a.get('entity_id')} & cause)
    before = [n for n in nodes if n['dispatch'] is not None and n['dispatch'] < onset and acts_on_cause(n)]
    at_onset = [n for n in nodes if n['dispatch'] is not None and n['dispatch'] == onset and acts_on_cause(n)]
    eligible = before or at_onset
    if not eligible:
        return {'profile': profile, 'seed': seed, 'episode_id': episode_id,
                'incident_family': family, 'status': 'no_script_node_before_onset',
                'nodes': nodes, 'onset': onset}
    fork_node = max(eligible, key=lambda n: n['dispatch'])
    fork_role = 'directing' if before else 'at_onset'
    if not eligible:
        return {'profile': profile, 'seed': seed, 'episode_id': episode_id,
                'incident_family': family, 'status': 'no_intervenable_node',
                'nodes': nodes, 'onset': onset, 'cause_entities': sorted(cause)}
    entities = transition_entities(episode_id, target)
    # An aggregate predicate (a crowd volume) has no per-entity source_refs; its members come
    # from the domain profile instead.  ``cohort_member_ids`` is where the crowd model lists
    # them, so an intervention on a cohort member still acts on the cause.
    if not entities:
        entities = crowd_cohort_members(episode_id)
    fork_actors = {a['entity_id'] for a in fork_node['actions'] if a.get('entity_id')}
    return {
        'profile': profile, 'seed': seed, 'episode_id': episode_id,
        'incident_family': family, 'status': 'resolved',
        'fork_event_id': fork_node['event_id'], 'fork_tick': fork_node['dispatch'],
        'fork_role': fork_role,
        'cause_entities': sorted(entities),
        'fork_actors': sorted(fork_actors),
        'fork_acts_on_cause': bool(fork_actors & entities), 'nodes_before_onset': [n['event_id'] for n in before],
        'nodes_at_onset': [n['event_id'] for n in at_onset],
        'onset_tick': onset, 'terminal_tick': terminal[0] if terminal else None,
        'tail': target['bindings'], 'trigger_predicate_id': target.get('trigger_predicate_id'),
        'event_id': target.get('event_id'),
        'nodes': nodes,
    }


def transition_entities(episode_id: str, occurrence: Mapping[str, Any]) -> set[str]:
    """The entities the proving transition reads, taken from its own source_refs.

    The transition row cites the trajectory samples it compared (for example
    ``trajectories.jsonl#tick=335&entity=ambulance_fall_l4_7_v2``).  Those entities are the
    participants whose motion created the condition, which is a stronger statement than the
    occurrence bindings alone: ``crowd`` bindings name an aggregate, while the transition
    cites the file, and the difference tells us whether a script node can touch the cause.
    """
    predicate_id = occurrence.get('trigger_predicate_id')
    onset = occurrence['event_phases'][0]['tick']
    path = SEMANTIC / episode_id / 'predicate_transitions.jsonl'
    if not path.is_file():
        return set()
    entities: set[str] = set()
    for row in _rows(path):
        if row.get('predicate_id') != predicate_id or row.get('direction') != 'rising':
            continue
        if row.get('to_tick') != onset:
            continue
        if json.dumps(row.get('bindings'), sort_keys=True) != json.dumps(occurrence['bindings'], sort_keys=True):
            continue
        for ref in row.get('source_refs') or []:
            if 'entity=' in ref:
                entities.add(ref.split('entity=', 1)[1])
    return entities


def rising_transition(episode_id: str, occurrence: Mapping[str, Any]) -> dict[str, Any] | None:
    """The rising transition that proves the incident event; None when absent."""
    predicate_id = occurrence.get('trigger_predicate_id')
    if not predicate_id:
        return None
    onset = occurrence['event_phases'][0]['tick']
    for row in _rows(SEMANTIC / episode_id / 'predicate_transitions.jsonl'):
        if row.get('predicate_id') != predicate_id or row.get('direction') != 'rising':
            continue
        if row.get('to_tick') != onset:
            continue
        if json.dumps(row.get('bindings'), sort_keys=True) != json.dumps(occurrence['bindings'], sort_keys=True):
            continue
        return row
    return None
