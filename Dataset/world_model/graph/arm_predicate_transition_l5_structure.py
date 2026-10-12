"""Closed record structure for L5 ARM predicate-transition rows.

Source family ``arm_predicate_transitions``; the L5 profile set is L5-1_v1,
L5-1_v2, L5-1_v3, L5-2_v1, L5-2_v2, L5-3_v1, L5-3_v2 under
``grp_l5_env_branch_v1``.  Each window writes ``arm_factual`` and, where the
branch declares a delayed arm, ``arm_time_delay``.

Producer chain (every link read this turn):

- ``Dataset/tools/l5_arm_pipeline.py`` ``execute_config`` is the only writer:
  line 116 calls ``calculate``, line 127
  ``dump_rows(out/'objective'/(key+'.jsonl'), values)`` dumps
  ``semantic['predicate_transitions']`` verbatim as one compact-JSON object
  per line.  ``Dataset/tools/l5_env_branch_delivery.py`` (line 65-67) and
  ``Dataset/tools/l5_arm_pipeline.py`` (``record_counts``, line 142) only
  read the file.
- ``Dataset/tools/l5_arm_semantics.py`` ``calculate`` (lines 191-206) builds
  ``semantic['predicate_transitions'] = converted`` from
  ``build_transitions(assertions, params)`` through
  ``_adapt_transitions(transitions, truth_map)`` and raises on any continuity
  break (lines 197-198).
- ``Dataset/semantic_truth/minimal_semantics.py`` ``build_transitions``
  (lines 238-295) pairs adjacent assertions of one exact tuple:
  ``after_tick - before_tick != step`` becomes ``continuity_break`` and
  only the ``false->true`` (``rising``) and ``true->false`` (``falling``)
  pairs are kept — ``unknown`` and same-value pairs take ``continue`` —
  so exactly the two boolean flips with a 5-tick
  formal step reach this file (``formal_step_ticks`` comes from the governed
  registry defaults; L5 windows sit on the 5-tick grid per
  ``Dataset/tools/l5_arm_common.py::load_config`` lines 102-105).
- ``Dataset/semantic_truth/minimal_semantics_adapter.py``
  ``_adapt_transitions`` (lines 868-929) fixes the 19-key record shape:
  constants ``predicate_transition`` / ``3.0.0`` / ``L1`` / ``L0`` /
  ``predicate_delta``, the two boundary truth ids, the unioned
  ``source_refs`` and the ``rule_digest``.
- Assertions come only from ``calculate``'s local ``add`` over the four
  constant families ``ENV_PREDICATES`` (6 ids), ``MAIN_PREDICATES`` (5 ids),
  ``emergency.mission_aborted`` (per-window, gated by
  ``config['predicate_ids']`` membership, l5_arm_semantics.py line 151) and
  ``safety.aircraft_near_pedestrian`` (gated by
  ``config['semantic_pedestrian_ids']`` being nonempty, line 170), sampled
  on the formal grid only (``if tick % step: return``, line 140).  No other
  assertion source feeds ``build_transitions`` in this pipeline, so
  ``predicate_id`` closes over those 13 ids; there is no per-profile
  predicate specialization — the profile config only decides which of the
  13 fire.

Bindings.  ``add`` validates the binding key set against the registry
template's argument roles (line 129) and each call site fixes one exact
key set, so the schema carries one ``allOf`` branch of ``oneOf`` variants, one per
predicate id, with const ``predicate_id`` plus a closed ``bindings``
object of exactly that predicate's role keys: ``region`` (six environment
ids, one call site, lines 156-158), ``actor``
(``airspace.aircraft_outside_assigned_altitude_band`` line 161 and
``emergency.mission_aborted`` line 153), ``aircraft`` alone
(``airborne``/``touchdown``, lines 159-168), ``aircraft`` + ``home_pad``
(lines 164-165), ``aircraft`` + ``landing_zone`` (lines 166-167) and
``aircraft`` + ``pedestrian`` (lines 174-177).  Binding values are entity
or region identifiers written from config-scoped ids (nonempty strings,
the same identifier domain the sibling L5 modules keep for config ids).
``emergency.mission_aborted`` rows exist only where the window config
declares that id in ``config['predicate_ids']`` and
``safety.aircraft_near_pedestrian`` rows only where
``config['semantic_pedestrian_ids']`` is nonempty.  The bounded sample
contains mission-abort transitions but no near-pedestrian transition.

Ticks.  ``add`` runs over ``range(start, end+1)`` and transitions pair
window ticks, while ``load_config`` accepts only
``0 <= start < fork < end <= 900`` on the 5-tick grid, so ``from_tick`` is
a multiple of 5 in [0, 895] and ``to_tick`` is a multiple of 5 in [5, 900].
``to_tick == from_tick + 5`` and
``to_tick > from_tick`` span two properties and are not expressible in JSON
Schema; they stay producer-side invariants of ``build_transitions``.

Values and direction.  ``build_transitions`` emits only the exact boolean
flips, and the adapter copies ``before_truth.value``/``after_truth.value``
as the strings ``"true"``/``"false"``, so ``from_value``/``to_value`` are the
two-value string enums and the second ``allOf`` branch pins the pair to the
direction exactly as ``Dataset/world_model/graph/predicate_structure.py``
does for the formal producer (lines 454-476).

Record identity.  ``transition_id`` is ``stable_identifier("transition",
...)`` — ``"transition:"`` plus 24 lowercase hex digits
(``Dataset/semantic_truth/provenance.py`` lines 42-44).  ``tuple_id`` is
``'l5_tuple:'`` plus the first 24 hex digits of ``object_hash([arm_key,
pid, bindings])`` (l5_arm_semantics.py line 134,
``Dataset/tools/l5_arm_common.py::object_hash``).  ``episode_id`` is the
ARM key composed at l5_arm_semantics.py line 124:
``config['episode_id'] + '::' + group_key + '/' + arm_id`` where
``load_config`` (l5_arm_common.py lines 93-96) enforces
``episode_id == profile + '__' + seed``, ``group_key == profile + '/' +
seed + '/' + group`` and ``re.fullmatch('grp_[a-z0-9_]+', group)``; the
pattern below encodes that composition with a backreference, keeps the
``profile``/``seed``/``arm_id`` segments to the path-segment span the
config layout gives them (no character contract exists for them), and pins
the group segment to the producer's own fullmatch rule.  The equality of
``evidence_truth_ids`` with the truth ids of the two boundary tuples, the
5-tick adjacency and the equality of the ``source_ref`` tick/entity spans
with ``from_tick``/``to_tick`` and the configured entity ids similarly span
properties or files and stay producer-side.

source_refs.  The adapter writes ``sorted(set(before)|set(after))`` of the
two boundary rows' evidence refs (adapter lines 911-914): unique by
construction, nonempty because every ``add`` call site passes a literal
nonempty ref list.  Across the four ``add`` call sites the emitted refs use
exactly four formats — the weather ref ``f'raw/weather.jsonl#tick={tick}'``
(environment, line 158), the trajectory ref
``f'raw/trajectories.jsonl#tick={tick}&entity={entity}'`` with the
identifier interpolated verbatim (airspace mainref line 150 and pedestrian
line 177; the pattern pins the literal fragment prefix and leaves the
identifier span to match any characters, matching the config contract's
character-free nonempty-string rule as in ``arm_states_l5_structure.py``),
the geometry ref ``f'raw/geometry.jsonl#tick={tick}'`` (airspace, line
168) and the fixed ``raw/binding_catalog.json`` (airspace, line 169) — so
the root items rule lists those four formats.  Each predicate branch also
fixes the sorted union's structure: two weather refs for environment, the
binding-catalog ref followed by two geometry and two trajectory refs for
airspace, and two trajectory refs for mission abort.  Near-pedestrian has
two trajectory refs per boundary; set deduplication yields two refs if its
roles have the same identifier, otherwise four.  Ref tick fragments use
canonical decimal integers in [0, 900].  The equality of those ticks with
``from_tick``/``to_tick`` remains a producer-side invariant.

template_parameters is the adapter's literal ``{}``
(``_delta_base`` -> ``_base_properties`` -> ``_empty_object``, copied
unchanged by ``_adapt_transitions`` line 921): a closed object with
``maxProperties: 0``, not an open object.

Integrity metadata.  ``rule_digest`` is the only digest-keyed field; it is
required here with its ``sha256:<64 hex>`` shape (``digest_object``,
provenance.py lines 27-28) and its exact key path ``("rule_digest",)`` is
already in ``_PUBLICATION_EXCLUDED_PATHS``
(``structure_schema.py`` line 27), so no global exclusion rule change and
no additional publication path is needed.  No other integrity-keyed field
exists at any depth: the keyset is fixed and the only nested objects are
``bindings``, ``template_parameters`` and the string arrays.

Evidence scope.  The schema compiles under the current compiler and was
checked against the real records of six of the seven profiles (L5-1_v1,
L5-1_v2, L5-1_v3, L5-2_v1, L5-2_v2, L5-3_v1, arm_factual seed00 files —
61 rows), which exercise the region, actor, aircraft, aircraft+home_pad and
aircraft+landing_zone branches and ten of the 13 predicate ids;
``environment.low_illumination_active`` and
``environment.temperature_outside_operating_range`` share the unconditional
``ENV_PREDICATES`` call site of the observed environment ids, while
``safety.aircraft_near_pedestrian`` is the unobserved gated call site.
Those three predicate branches remain 尚无实例验证 in this bounded sample.
L5-3_v2 was not read this turn (the task's six-JSONL read budget was
exhausted); it is produced by the same ``execute_config`` path.  A window
can also legitimately produce zero transitions, in which case ``dump_rows``
writes an empty file — that is a file-level outcome outside a row grammar.
Branch coverage, compilation and these checks are not a full-root closure
claim: the bounded record validation of the assigned scope is the
scheduler's step.
"""

from __future__ import annotations

__all__ = ["arm_predicate_transitions_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"
_REF_TICK = r"(?:0|[1-9][0-9]?|[1-8][0-9]{2}|900)"
_WEATHER_REF = rf"^raw/weather\.jsonl#tick={_REF_TICK}$"
_GEOMETRY_REF = rf"^raw/geometry\.jsonl#tick={_REF_TICK}$"
_TRAJECTORY_REF = (
    rf"^raw/trajectories\.jsonl#tick={_REF_TICK}&entity=[\s\S]+$"
)

# The exact assertion vocabulary of Dataset/tools/l5_arm_semantics.py
# calculate(): ENV_PREDICATES + MAIN_PREDICATES + the mission-abort call site
# + the gated pedestrian call site.  Role keys per call site.
_ROLE_KEYS = {
    "environment.rain_active": ("region",),
    "environment.fog_active": ("region",),
    "environment.visibility_below_threshold": ("region",),
    "environment.wind_speed_above_threshold": ("region",),
    "environment.low_illumination_active": ("region",),
    "environment.temperature_outside_operating_range": ("region",),
    "airspace.aircraft_airborne": ("aircraft",),
    "airspace.aircraft_at_home_pad": ("aircraft", "home_pad"),
    "airspace.aircraft_touchdown": ("aircraft",),
    "airspace.aircraft_in_landing_zone": ("aircraft", "landing_zone"),
    "airspace.aircraft_outside_assigned_altitude_band": ("actor",),
    "emergency.mission_aborted": ("actor",),
    "safety.aircraft_near_pedestrian": ("aircraft", "pedestrian"),
}


def _identifier() -> dict:
    return {"type": "string", "minLength": 1}


def _truth_id() -> dict:
    """``stable_identifier("truth", ...)``: prefix plus 24 hex digits."""
    return {"type": "string", "pattern": "^truth:[0-9a-f]{24}$"}


def _fixed_refs(*members: dict) -> dict:
    return {
        "type": "array",
        "prefixItems": list(members),
        "items": False,
        "minItems": len(members),
        "maxItems": len(members),
        "uniqueItems": True,
    }


def _source_refs_variant(predicate_id: str) -> dict:
    weather = {"type": "string", "pattern": _WEATHER_REF}
    geometry = {"type": "string", "pattern": _GEOMETRY_REF}
    trajectory = {"type": "string", "pattern": _TRAJECTORY_REF}
    if predicate_id.startswith("environment."):
        return _fixed_refs(weather, weather)
    if predicate_id.startswith("airspace."):
        return _fixed_refs(
            {"const": "raw/binding_catalog.json"},
            geometry, geometry, trajectory, trajectory,
        )
    if predicate_id == "emergency.mission_aborted":
        return _fixed_refs(trajectory, trajectory)
    if predicate_id == "safety.aircraft_near_pedestrian":
        # If both roles have the same identifier, the set union has two
        # entries; otherwise it has four.  The writer does not check this
        # identity relation here.
        return {"oneOf": [
            _fixed_refs(trajectory, trajectory),
            _fixed_refs(trajectory, trajectory, trajectory, trajectory),
        ]}
    raise ValueError(f"unregistered L5 transition predicate: {predicate_id}")


def _binding_variant(predicate_id: str) -> dict:
    """One ``add`` call-site shape: exact predicate, roles and evidence refs."""
    return {
        "properties": {
            "predicate_id": {"const": predicate_id},
            "bindings": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_ROLE_KEYS[predicate_id]),
                "properties": {role: _identifier()
                               for role in _ROLE_KEYS[predicate_id]},
            },
            "source_refs": _source_refs_variant(predicate_id),
        },
        "required": ["predicate_id", "bindings", "source_refs"],
    }


def _direction_variant(direction: str, from_value: str, to_value: str) -> dict:
    """The exact boolean flip ``build_transitions`` emits for one
    direction; the 5-tick adjacency stays a producer-side invariant."""
    return {
        "properties": {
            "direction": {"const": direction},
            "from_value": {"const": from_value},
            "to_value": {"const": to_value},
        },
        "required": ["direction", "from_value", "to_value"],
    }


def arm_predicate_transitions_schema() -> dict:
    """Return the closed record schema for one ``arm_predicate_transitions``
    L5 JSONL row.

    The root is the exact 19-key ``_adapt_transitions`` shape with every
    key required and ``additionalProperties: false``.  The first ``allOf``
    branch carries one ``oneOf`` variant per producer call site (const
    predicate id, exact binding key set and branch-specific source refs;
    the root-level ``bindings`` property only declares the key for the
    root's closed boundary, the variant carries its closed shape), the second pins the
    ``direction``/``from_value``/``to_value`` pair to the two boolean flips
    ``build_transitions`` emits; continuity-break records cannot reach this
    file because the producer raises on them.
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/"
               "arm_predicate_transitions_l5.record.schema.json",
        "title": "L5 ARM predicate transition record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_name", "schema_version", "annotation_layer",
            "source_layer", "materialization_policy", "transition_id",
            "episode_id", "predicate_id", "tuple_id", "bindings",
            "from_tick", "to_tick", "from_value", "to_value", "direction",
            "evidence_truth_ids", "source_refs", "rule_digest",
            "template_parameters",
        ],
        "properties": {
            "schema_name": {"const": "predicate_transition"},
            "schema_version": {"const": "3.0.0"},
            "annotation_layer": {"const": "L1"},
            "source_layer": {"const": "L0"},
            "materialization_policy": {"const": "predicate_delta"},
            "transition_id": {
                "type": "string", "pattern": "^transition:[0-9a-f]{24}$"},
            # config['episode_id'] + '::' + group_key + '/' + arm_id with
            # episode_id == profile + '__' + seed, group_key == profile +
            # '/' + seed + '/' + group (l5_arm_semantics.py:124,
            # l5_arm_common.py:93-96); profile/seed/arm_id keep the
            # path-segment span the config layout gives them, the group
            # segment keeps the producer's own grp_[a-z0-9_]+ fullmatch.
            "episode_id": {
                "type": "string",
                "pattern": ("^([^/]+)__([^/]+)::\\1/\\2/"
                            "grp_[a-z0-9_]+/[^/]+$"),
            },
            "predicate_id": {"enum": sorted(_ROLE_KEYS)},
            "tuple_id": {
                "type": "string", "pattern": "^l5_tuple:[0-9a-f]{24}$"},
            # Root-level base rule (as in predicate_structure.py
            # _base_properties): it declares the key for the root's
            # additionalProperties boundary; the exact closed shape per
            # predicate id comes from the first allOf oneOf branch below.
            "bindings": {"type": "object"},
            "from_tick": {"type": "integer", "minimum": 0, "maximum": 895,
                          "multipleOf": 5},
            "to_tick": {"type": "integer", "minimum": 5, "maximum": 900,
                        "multipleOf": 5},
            "from_value": {"enum": ["true", "false"]},
            "to_value": {"enum": ["true", "false"]},
            "direction": {"enum": ["rising", "falling"]},
            # [before["truth_id"], after["truth_id"]] (adapter line 910).
            "evidence_truth_ids": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "prefixItems": [_truth_id(), _truth_id()],
                "items": False, "uniqueItems": True,
            },
            # sorted(set(before)|set(after)) of the two boundary evidence
            # ref lists: unique by construction, nonempty at every call
            # site; the items rule is the union of the producer's ref
            # shapes (weather pair, trajectory+catalog set, mission-abort
            # pair) with the literal fragment prefixes pinned.
            "source_refs": {
                "type": "array", "minItems": 2, "maxItems": 5,
                "uniqueItems": True,
                "items": {"oneOf": [
                    {"type": "string",
                     "pattern": _WEATHER_REF},
                    {"type": "string",
                     "pattern": _TRAJECTORY_REF},
                    {"type": "string",
                     "pattern": _GEOMETRY_REF},
                    {"const": "raw/binding_catalog.json"},
                ]},
            },
            "rule_digest": {
                "type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            # The adapter's literal {} (_empty_object copied unchanged).
            "template_parameters": {
                "type": "object", "additionalProperties": False,
                "maxProperties": 0,
            },
        },
        "allOf": [
            # One ``add`` call-site branch per predicate id: const
            # predicate_id plus exactly that id's binding key set.  The
            # predicate-id discrimination has to sit at this level — all six
            # environment ids share the {region} binding shape, so a
            # bindings-only oneOf could not stay exclusive, and the root
            # level only carries the open base rule for its own
            # additionalProperties boundary.
            {"oneOf": [_binding_variant(predicate_id)
                       for predicate_id in sorted(_ROLE_KEYS)]},
            {"oneOf": [
                _direction_variant("rising", "false", "true"),
                _direction_variant("falling", "true", "false"),
            ]},
        ],
    }
