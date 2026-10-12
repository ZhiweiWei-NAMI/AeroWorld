"""Closed record structure for L2 ARM predicate transition rows.

Source family ``arm_predicate_transitions``; the L2 profile set is L2-1_v1,
L2-1_v2, L2-2_v1, L2-2_v2, L2-3_v1, L2-3_v2, L2-4_v1, L2-4_v2 (failure) and
L2-5_v1 (interaction), 108 declared JSONL sources
(``arms/L2-*/seed*/**/objective/predicate_transitions.jsonl``).  Producer
chain: ``Dataset/tools/l2_window.py`` line 453 writes
``objective/predicate_transitions.jsonl`` from the semantic payload
(``Dataset/tools/l2_arm_semantics.py`` 479), whose rows are
``_adapt_transitions`` (``Dataset/semantic_truth/minimal_semantics_adapter.py``
868-929) applied to ``build_transitions`` output
(``Dataset/semantic_truth/minimal_semantics.py`` 238-295) and stamped with the
two ARM identity keys ``parent_episode_id``/``arm_key``
(``l2_arm_semantics.py`` 476-478).  The window crop keeps a row when
``start_tick <= to_tick <= end_tick`` (``l2_window.py`` 429-430), so
``from_tick`` may precede the window start while both ticks stay inside the
formal 0..900 corpus range.

Row identity.  ``episode_id = f'{episode_id}::{arm_key}'`` with
``arm_key = f'{group_key}/{arm_id}'`` (``l2_arm_semantics.py`` 332-333) and
``group_key`` the arm-script path under ``scripts/arms`` (``l2_window.py``
319); ``parent_episode_id = f'{profile}__{seed}'`` (``l2_window.py`` 320).
The declared L2 profiles themselves carry underscores (``L2-1_v1`` ...
``L2-5_v1``), so both episode patterns pin the profile segment with the exact
nine-name alternation declared for this layer instead of an underscore-free
wildcard that would reject every real row.  The ``__seed`` join and the
``::`` arm suffix stay structural: ``arm_id`` ``^arm_[a-z0-9_]+$`` and
``group_key`` ``^[^/]+/seed[0-9]+/[^/]+$`` follow
``rules/l2_window.schema.json`` (whose config-level seed enum narrows real
seeds to ``seed00``/``seed01``/``seed02`` while the group-key segment contract
is the ``seed[0-9]+`` pattern used verbatim here).  ``tuple_id``/
``transition_id``/truth ids and ``rule_digest`` are the producer identifiers
of ``Dataset/semantic_truth/provenance.py`` (``stable_identifier`` = 24
lowercase hex, ``digest_object`` = ``sha256:`` + 64 lowercase hex):
``tuple_id`` is ``'arm_tuple:' + digest([arm_key, pid, bindings])[:24]`` and
``tuple_type`` is ``'grounded_roles__' + '__'.join(sorted(bindings))``
(``l2_arm_semantics.py`` 457-459).

Each arm-path profile segment is restricted to the declared L2 profiles.
Equality of profile, seed and group across the three identity fields remains
a producer invariant enforced by the window and semantic writer.

Predicate and binding vocabulary.  ``scope_for`` instantiates exactly fifteen
executable registry predicates - every ``add(...)`` call in
``l2_arm_semantics.py`` 200-245 names one of the module constants at lines
82-96 - and raises on any binding-key set that differs from the template
``argument_roles`` (194-199), so each branch below pins the predicate id and
its exact role-key set without re-judging predicate business definitions.
The other executable predicates have no L2 instantiation path and
non-executable registry predicates cannot be asserted at all
(``_assert_core_vocabulary``), so neither can reach a row.  Binding values
are authoritative entity/service identifiers: scene ids, the governed GNSS
service id from the domain profile (237), or the landing-zone authority -
``assigned_landing_zone:{uav_id}`` for an off-pad terminal waypoint and the
home-pad entity id otherwise (``l2_arm_semantics.py`` 212-220, mirrored by
``Dataset/semantic_simulation/predicate_state_computers.py`` 3020-3023).

Direction and values.  ``build_transitions`` emits only ``rising``/``falling``
for adjacent formal-tick boolean pairs with ``to_tick - from_tick ==
formal_step_ticks = 5`` (``minimal_semantics.py`` 238-295;
``PARAMS['formal_step_ticks'] = 5`` from the governed defaults); assertions
exist only on the formal grid because ``l2_arm_semantics.py`` 456 skips every
non-formal tick, so both ticks are multiples of 5 in 0..900.  The arithmetic
``to_tick == from_tick + 5`` between two properties is not expressible in
this closed-record grammar and remains a producer invariant.  Continuity
breaks never enter this family: they go to the separate ``discontinuities``
collection (``l2_arm_semantics.py`` 480) that L2 never writes as transition
rows.  ``evidence_truth_ids`` is exactly the before/after truth pair
(``minimal_semantics_adapter.py`` 910); the two ids derive from
``_truth_id(episode_id, tick, tuple_id, predicate_id)`` (adapter 1775-1776)
with the before/after ticks necessarily different, so the pair cannot repeat
and is declared ``uniqueItems``.  ``source_refs`` is the sorted deduplicated
union of the two truth rows' references (adapter 911-914); per truth row the
writer emits either ``raw/states.jsonl#tick={tick}&entity={entity_id}`` per
bound entity or the ``raw/metadata.json`` fallback when no bound value is a
declared window entity, and corridor rows use the all-scope-entities form
plus ``raw/metadata.json`` (``l2_arm_semantics.py`` 389-390, 415), so the
items rule pins exactly those two producer formats and the union is declared
``uniqueItems``.  ``template_parameters`` is the producer's hardcoded empty
mapping (adapter 921).

Branch split.  ``predicate_id`` + ``bindings`` form the first ``oneOf``
(15 branches); a second ``oneOf`` carries the two direction triplets
(``direction`` + ``from_value`` + ``to_value`` consts: rising =
``false``->``true``, falling = ``true``->``false``).  The base ``bindings``
declaration carries only the JSON type; its closure comes entirely from the
per-branch role sets, so the compiler walk sees each role key typed by the
active branch.  ``rule_digest`` is required here exactly as the producer
writes it; it is also a publication-excluded path
(``Dataset/world_model/graph/structure_schema.py``
``_PUBLICATION_EXCLUDED_PATHS``, entry ``("rule_digest",)``), so the
publication projection drops it globally while raw validation keeps it.
"""

_L2_PROFILES = (
    "L2-1_v1", "L2-1_v2", "L2-2_v1", "L2-2_v2", "L2-3_v1",
    "L2-3_v2", "L2-4_v1", "L2-4_v2", "L2-5_v1",
)
_PROFILE_ALTERNATION = "(" + "|".join(_L2_PROFILES) + ")"

_FORMAL_STEP_TICKS = 5
_TICK_MAX = 900
_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# ``scope_for`` add() calls (``l2_arm_semantics.py`` 200-245): the only
# predicate instantiations an L2 row can carry, with their exact role keys.
_REACHABLE_PREDICATES = (
    ("airspace.aircraft_airborne", ("aircraft",)),
    ("airspace.aircraft_at_home_pad", ("aircraft", "home_pad")),
    ("airspace.aircraft_in_landing_zone", ("aircraft", "landing_zone")),
    ("airspace.aircraft_outside_assigned_altitude_band", ("actor",)),
    ("airspace.aircraft_touchdown", ("aircraft",)),
    ("airspace.airspace_corridor_over_capacity", ("corridor",)),
    ("communication.link_degraded", ("actor", "station")),
    ("communication.link_unavailable", ("actor", "station")),
    ("facility.charging_queue_over_capacity", ("station",)),
    ("facility.charging_station_unavailable", ("station",)),
    ("facility.landing_pad_contention",
     ("first_aircraft", "pad", "second_aircraft")),
    ("ground.traffic_signal_malfunction_active", ("signal",)),
    ("positioning.gnss_quality_degraded", ("actor", "service")),
    ("safety.aircraft_near_building_structure", ("aircraft", "structure")),
    ("safety.aircraft_pair_below_separation",
     ("first_aircraft", "second_aircraft")),
)


def _nonempty_string() -> dict:
    return {"type": "string", "minLength": 1}


def _identifier(prefix: str) -> dict:
    return {"type": "string",
            "pattern": "^" + prefix + ":[0-9a-f]{24}$"}


def _tick() -> dict:
    return {"type": "integer", "minimum": 0, "maximum": _TICK_MAX,
            "multipleOf": _FORMAL_STEP_TICKS}


def _parent_episode_id() -> dict:
    return {"type": "string",
            "pattern": "^" + _PROFILE_ALTERNATION + "__seed[0-9]+$"}


def _arm_key() -> dict:
    return {"type": "string",
            "pattern": "^" + _PROFILE_ALTERNATION + "/seed[0-9]+/[^/]+/arm_[a-z0-9_]+$"}


def _episode_id() -> dict:
    return {"type": "string",
            "pattern": ("^" + _PROFILE_ALTERNATION + "__seed[0-9]+::"
                        + _PROFILE_ALTERNATION + "/seed[0-9]+/[^/]+/arm_[a-z0-9_]+$")}


def _source_refs() -> dict:
    """The two producer reference formats of one truth row's ``source_refs``.

    ``l2_arm_semantics.py`` 389-390 writes
    ``raw/states.jsonl#tick={tick}&entity={entity_id}`` per bound declared
    entity, falling back to the exact ``raw/metadata.json`` reference when no
    bound value is a declared window entity; corridor rows write the
    all-scope-entities states refs plus ``raw/metadata.json`` (415).  The
    transition row carries the sorted deduplicated union of the before/after
    truth rows' references (``minimal_semantics_adapter.py`` 911-914).  The
    ref tick is the truth row's tick (``range(0, end_tick + 1)`` with
    ``end_tick <= 900``, ``l2_window.py`` 100-103), written by f-string as a
    canonical non-negative decimal, so the item pattern pins the exact
    0..900 domain of ``_TICK_MAX`` on the five-tick assertion grid.
    """
    return {
        "type": "array",
        "items": {
            "oneOf": [
                {"type": "string",
                 "pattern": ("^raw/states\\.jsonl#tick="
                             "([05]|[1-9][05]|[1-8][0-9][05]|900)"
                             "&entity=.+$")},
                {"const": "raw/metadata.json"},
            ],
        },
        "minItems": 1,
        "uniqueItems": True,
    }


def _predicate_branch(predicate_id: str, roles: tuple[str, ...]) -> dict:
    return {
        "properties": {
            "predicate_id": {"const": predicate_id},
            "bindings": {
                "type": "object", "additionalProperties": False,
                "required": list(roles),
                "properties": {role: _nonempty_string() for role in roles},
            },
        },
    }


def _direction_triplet(direction: str, from_value: str, to_value: str) -> dict:
    """One ``oneOf`` branch pinning ``direction`` with its value pair."""
    return {
        "properties": {
            "direction": {"const": direction},
            "from_value": {"const": from_value},
            "to_value": {"const": to_value},
        },
    }


def arm_predicate_transitions_schema() -> dict:
    """Return the closed record schema for one ``arm_predicate_transitions``
    L2 JSONL row.

    One closed object with all 21 producer keys required; ``oneOf`` over the
    fifteen producer-reachable predicate branches (id + exact binding role
    keys) and a second ``oneOf`` pinning the rising and falling value pairs.
    Pure schema constructor: no I/O, no registry or compiler interaction.
    """
    return {
        "$schema": _SCHEMA,
        "$id": ("https://aeroworld.local/schema/"
                "arm_predicate_transition_l2.record.schema.json"),
        "title": ("arm_predicate_transitions L2 whole-record grammar "
                  "(objective/predicate_transitions.jsonl row)"),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_name", "schema_version", "annotation_layer", "source_layer",
            "materialization_policy", "transition_id", "episode_id",
            "predicate_id", "tuple_id", "bindings", "from_tick", "to_tick",
            "from_value", "to_value", "direction", "evidence_truth_ids",
            "source_refs", "rule_digest", "template_parameters",
            "parent_episode_id", "arm_key",
        ],
        "properties": {
            "schema_name": {"const": "predicate_transition"},
            "schema_version": {"const": "3.0.0"},
            "annotation_layer": {"const": "L1"},
            "source_layer": {"const": "L0"},
            "materialization_policy": {"const": "predicate_delta"},
            "transition_id": _identifier("transition"),
            "episode_id": _episode_id(),
            "parent_episode_id": _parent_episode_id(),
            "arm_key": _arm_key(),
            "tuple_id": _identifier("arm_tuple"),
            "bindings": {"type": "object"},
            "predicate_id": {
                "type": "string",
                "enum": [pid for pid, _roles in _REACHABLE_PREDICATES],
            },
            "from_tick": _tick(),
            "to_tick": _tick(),
            "from_value": {"type": "string", "enum": ["true", "false"]},
            "to_value": {"type": "string", "enum": ["true", "false"]},
            "direction": {"type": "string", "enum": ["rising", "falling"]},
            "evidence_truth_ids": {
                "type": "array",
                "items": _identifier("truth"),
                "minItems": 2, "maxItems": 2, "uniqueItems": True,
            },
            "source_refs": _source_refs(),
            "rule_digest": {
                "type": "string", "pattern": "^sha256:[0-9a-f]{64}$",
            },
            "template_parameters": {
                "type": "object", "additionalProperties": False,
                "required": [],
            },
        },
        "allOf": [
            {"oneOf": [_predicate_branch(pid, roles)
                       for pid, roles in _REACHABLE_PREDICATES]},
            {"oneOf": [
                _direction_triplet("rising", "false", "true"),
                _direction_triplet("falling", "true", "false"),
            ]},
        ],
    }
