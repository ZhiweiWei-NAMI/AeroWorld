"""Closed record structure for one formal weather metadata row, L5 layer.

Source family ``formal_weather_meta``; profiles L5-1_v1, L5-1_v2, L5-1_v3,
L5-2_v1, L5-2_v2, L5-3_v1, L5-3_v2, L5-4_v1, L5-5_v1 (9 profiles x 3 seeds
= 27 declarations; the transfer boundary holds exactly 27
``weather_meta.jsonl`` files under those episode directories this session).

Producer chain: ``Dataset/tools/batch_generate.py`` builds the L0 episode
weather — ``weather_payload`` (:1701-1764) resolves the profile against the
``WEATHER_PROFILES`` registry (:86-200: clear, rain, fog, wind, dusk, heat,
"light smoke"), rejects unknown override keys outside
``WEATHER_OVERRIDE_FIELDS`` and non-finite values, and sets ``condition``
from the profile key (``_handle_set_weather`` :2205-2208,
``initial_weather_state`` :1767-1769, ``scheduled_weather_transitions``
:1772-1788); ``_record_tick_rows`` (:2352-2361) writes one row per tick
``{"tick": tick, **weather}`` and ``generate_episode`` persists
``weather_meta.jsonl`` (:2534).  ``Dataset/tools/convert_to_render_ready.py``
loads those rows (:4614), ``expand_weather_rows`` (:1934-1955) requires one
source row for every tick 0..duration_ticks (a missing tick raises), and
``normalize_weather_row`` (:1866-1931) writes the L5 row: unknown source
keys are rejected against ``WEATHER_SOURCE_FIELDS``
(``Dataset/tools/weather_fields.py`` :14-16), the ``fog``/``visibility``
aliases are folded into the canonical fields (:1870-1890), the six base
fields are required finite numbers (:1874-1908), ``rain``/``fog_density``
are clamped to [0, 1] (``WEATHER_FRACTION_FIELDS`` :1906-1908), the optional
numeric/boolean channels are copied only when the source row carries them
(:1904-1925), ``temperature_source`` requires ``temperature_c`` on the same
source row (:1926-1930), and ``condition`` is stripped and lowercased
(:1892-1897).  Rows are written with ``write_jsonl(output_episode_dir /
"weather_meta.jsonl", weather_rows)`` (:5733).
``Dataset/tools/filter_render_ready_truth_for_capture.py`` copies
``weather_meta.jsonl`` byte-for-byte through to
``aw_data/render_ready_episodes_capture_filtered``
(``copy_static_episode_files`` :490-501; the file is not in
``REGENERATED_FILES`` :64-71), so every boundary byte is the converter's
output.

Record identity: ``structure_sources.py`` streams each nonblank JSONL row as
one record (``formal_weather_meta`` is not in ``_ENTITY_FILES``), so the
record root of this grammar is ONE per-tick weather row; ``tick`` is present
on every producer row (``record_tick_present``).

Branches — the converter emits a single record type with no per-profile key
branch; keyset differences are exactly its copy-if-present optional fields.
Required keys are the six the writer raises on when missing
(``required_number`` :1874-1886 plus the base dict :1895-1903).  The
optional keys exist because the writer keeps a channel absent when the L0
generator did not provide it (:1904-1925, "Keep them absent when the source
generator did not provide them so the semantic layer fails closed").  The
default "light smoke" payload includes ``dust``; ``WEATHER_OVERRIDE_FIELDS``
also permits a ``dust`` override on any weather profile.  No default profile
writes ``temperature_source``.  The optional status follows the writer
contract, not the observed keyset.  ``condition`` is the closed 7-member
registry domain: ``weather_payload`` raises "Unknown deterministic weather
profile" for anything outside ``WEATHER_PROFILES`` and the converter
lowercases the value.  The nine current L5 scene setups and event scripts
select only clear, rain, fog, wind, dusk and heat.  "light smoke" remains a
producer-accepted profile through ``weather_payload`` and
``_handle_set_weather`` with no L5-specific exclusion.  It is not selected
by the current L5 authored inputs or the bounded sample (尚无实例验证).
``rain``/``fog_density`` carry the enforced [0, 1] bounds; the
other numerics carry no producer range constraint beyond finiteness, so none
is invented here.  No field is ever null at the boundary: the writer either
stores a validated value or omits the key, so absence is the only missing
status and no ``oneOf``/nullable branch is needed.

Without-instance-verification: ``temperature_source`` is accepted and copied
by the writer (:1926-1930, and only together with ``temperature_c``) but no
executable producer path writes it — ``weather_payload`` overrides accept
only ``WEATHER_OVERRIDE_FIELDS`` (numeric/boolean/alias keys; a string
override is rejected :1710-1718) and no ``WEATHER_PROFILES`` member contains
it.  The branch is declared from the writer contract and has no verified L5
instance (尚无实例验证).

Checked scope this session: the nine L5 scene setups and event scripts have
no duration override and select six condition values; their seed00 frozen
manifests declare duration 900 and point back to those scripts.  This schema
was compiled and 18 weather rows were validated at ticks 0/450/900 from six
seed00 files: L5-1_v1, L5-1_v3, L5-2_v1, L5-3_v1, L5-4_v1 and L5-5_v1.
Those sampled ticks carry clear, rain, fog and heat; all 18 carry ``dust``
and none carries ``temperature_source``.  The sample does not exercise wind,
dusk, "light smoke" or ``temperature_source``.  It does not establish a
whole-file or whole-family validation result.

Publication exclusion: weather rows carry no digest/integrity/hash key and
no key matches ``_PUBLICATION_EXCLUDED_PATHS`` (the global
``("sources", "weather", "digest")`` rule belongs to truth-frame source
blocks, not this family), so this module declares no excluded path and needs
none.

Boundary: this module exports the pure schema constructor only.  It is not
registered in ``structure_closure.py`` (``formal_weather_meta`` is absent
from ``CLOSED_RECORD_FAMILIES`` and from the ``validate_source_record``
dispatch; the declared inventory status is ``field_scan_defined``, so the
family is currently "validated by source projection") — registration and
dispatch wiring belong to G1.  ``effective_family == "formal_weather_meta"``
(no ``_effective_family`` branch in ``structure_sources.py``).  The module
scans no data at import and compiles nothing at import.
"""

from __future__ import annotations

__all__ = ["formal_weather_meta_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

# WEATHER_PROFILES registry keys (batch_generate.py :86-200), lowercased by
# the L5 writer (convert_to_render_ready.py :1897).
_CONDITION_VALUES = (
    "clear",
    "dusk",
    "fog",
    "heat",
    "light smoke",
    "rain",
    "wind",
)


def _float_leaf() -> dict:
    # normalize_weather_row.required_number stores float(value) (:1883-1886),
    # so the writer never emits an integer or boolean on these channels.
    return {"type": "number", "x-source-value-types": ["float"]}


def _fraction_leaf() -> dict:
    # WEATHER_FRACTION_FIELDS clamp (convert_to_render_ready.py :1906-1908).
    return {
        "type": "number",
        "x-source-value-types": ["float"],
        "minimum": 0.0,
        "maximum": 1.0,
    }


def formal_weather_meta_schema() -> dict:
    """Full closed schema of one L5 per-tick formal weather JSONL row."""
    return {
        "$schema": _SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "tick",
            "condition",
            "rain",
            "wetness",
            "fog_density",
            "wind_speed",
            "visibility_m",
        ],
        "properties": {
            # batch_generate writes int ticks 0..duration_ticks
            # (:2352-2361); the nine L5 scripts use the default 900 and
            # their frozen manifests confirm duration_ticks=900.
            "tick": {"type": "integer", "minimum": 0, "maximum": 900},
            "condition": {"enum": list(_CONDITION_VALUES)},
            "rain": _fraction_leaf(),
            "wetness": _float_leaf(),
            "fog_density": _fraction_leaf(),
            # Optional channels: copied only when the L0 row carries them
            # (convert_to_render_ready.py :1904-1925); absence is the
            # declared missing status, never null.
            "dust": _float_leaf(),
            "wind_speed": _float_leaf(),
            "visibility_m": _float_leaf(),
            "wind_direction_deg": _float_leaf(),
            "temperature_c": _float_leaf(),
            "illumination_lux": _float_leaf(),
            "hazard_concentration_ppm": _float_leaf(),
            "hazard_radius_m": _float_leaf(),
            "hazard_source_active": {"type": "boolean"},
            # Writer accepts and copies (:1926-1930) but no current producer
            # path writes it — declared from the writer contract only
            # (尚无实例验证).
            "temperature_source": {"type": "string", "pattern": "\\S"},
        },
        "if": {"required": ["temperature_source"]},
        "then": {"required": ["temperature_c"]},
    }
