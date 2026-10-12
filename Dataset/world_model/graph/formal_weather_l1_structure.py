"""Closed record structure for one formal_weather_meta JSONL row.

Source family ``formal_weather_meta``; L1 profiles L1-1_v1, L1-1_v2,
L1-2_v1, L1-3_v1, L1-3_v2, L1-4_v1, L1-4_v2.  The current inventory
declares exactly 21 files for this family and level (7 profiles x 3
seeds, ``aw_data/render_ready_episodes_capture_filtered/<episode>/
weather_meta.jsonl``), all present with structure status
``field_scan_defined``.

Producer chain (single writer, profile-independent — profiles differ only
by episode directory):
- ``Dataset/tools/convert_to_render_ready.py`` reads the source episode
  rows (``load_jsonl(source_episode_dir / "weather_meta.jsonl")`` :4614)
  and expands exactly one normalized row per formal tick
  (``expand_weather_rows`` :1934-1955, called at :5479): a row is
  required at every tick, duplicate and missing source ticks raise.
  Each row is rebuilt by ``normalize_weather_row`` (:1866-1931), which is
  the authoritative keyset and requiredness contract, and the result is
  written verbatim as JSONL
  (``write_jsonl(output_episode_dir / "weather_meta.jsonl", weather_rows)``
  :5733).  ``required_number`` (:1874-1886) accepts only finite
  ``int``/``float`` source values (bools rejected) and stores
  ``float(value)``, so every numeric leaf of the written row is a float.
- ``Dataset/tools/filter_render_ready_truth_for_capture.py`` copies the
  file unchanged to the frozen transfer boundary:
  ``copy_static_episode_files`` (:490-501) copies every file not listed
  in ``REGENERATED_FILES`` (:64-71), and ``weather_meta.jsonl`` is not
  regenerated; the filter rewrites only artifact path strings in the
  manifests (:512, :569).  Every weather row at the frozen boundary is
  therefore the converter's output.

Record identity: ``structure_sources.py`` streams one JSONL row per
record (:218-245) and ``formal_weather_meta`` is not in ``_ENTITY_FILES``
(:34), so ``_project_full_or_entity_root`` (:250-262) projects the row
root directly.  The record root of this grammar is ONE weather row; the
sampled files hold 901 rows each (formal ticks 0..900).  The L1 source
files sampled this session (seed00 of L1-1_v1, L1-1_v2, L1-2_v1, L1-3_v1,
L1-3_v2, L1-4_v1) all carry the same 14-key keyset: the seven core keys
plus ``dust``, ``wind_direction_deg``, ``temperature_c``,
``illumination_lux``, ``hazard_concentration_ppm``, ``hazard_radius_m``
and ``hazard_source_active``; ``temperature_source`` is not written.
Sampled values (``condition: "clear"``, ``visibility_m: 20000.0`` ...)
are examples only — the grammar keeps every value producer-typed, never
pinned, and does not judge weather-model truth.

Keyset decisions, all from the producer dict literal and conditionals:
- Required keys ``tick, condition, rain, wetness, fog_density,
  wind_speed, visibility_m`` are unconditional members of the literal
  (:1895-1903).
- ``dust`` (:1904-1905), ``wind_direction_deg``/``temperature_c``/
  ``illumination_lux``/``hazard_concentration_ppm``/``hazard_radius_m``
  (:1912-1921), ``hazard_source_active`` (:1922-1925) and
  ``temperature_source`` (:1926-1930) are written only when the source
  row carried them, so presence and absence are both declared producer
  statuses rather than an observed-union guess.
- ``temperature_source`` additionally requires ``temperature_c`` on the
  same source row (:1926-1930 raises otherwise); the root ``if`` states
  that co-occurrence.  No L1 instance of ``temperature_source`` exists in
  the sampled rows: that branch is declared from producer evidence and is
  still without instance verification.
- ``condition`` is the stripped+lowered source value, which must be a
  nonempty string (:1892-1894, :1897); the producer enforces no
  condition vocabulary, so no enum is pinned here.
- ``rain`` and ``fog_density`` carry the producer range check
  ``0 <= value <= 1`` over ``WEATHER_FRACTION_FIELDS`` (:1906-1908;
  ``Dataset/tools/weather_fields.py`` :10); ``wetness`` and all other
  numerics carry no range because the producer checks none.  ``fog`` and
  ``visibility`` are source-row aliases consumed by the converter
  (``WEATHER_ALIASES``, weather_fields.py :9) and are never written to
  the output, so the output keyset is closed to the fifteen declared
  names.

Integrity metadata: the producer writes no digest/hash/sha256-style key
and no weather-row path matches ``_PUBLICATION_EXCLUDED_PATHS``
(structure_schema.py :31-62), so this module declares no publication
exclusion path and removes nothing from the raw record.

Boundary: this module exports the pure schema constructor only.  It is
not registered in ``structure_closure.py`` (``formal_weather_meta`` is
absent from ``CLOSED_RECORD_FAMILIES`` :159-167 and from the
``validate_source_record`` dispatch :205-245; until wired, the family
reports itself through ``_raise_unregistered_source_family`` :185-202 as
validated by source projection).  Registration and dispatch wiring belong
to the integration stage.  The module imports nothing beyond
``__future__``, scans no data at import, and compiles nothing at import.
"""

from __future__ import annotations

__all__ = ["formal_weather_meta_schema"]

_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _float_leaf() -> dict:
    # required_number stores float(value) (:1874-1886), so the source value
    # type of every numeric weather leaf is float, never int or bool.
    return {"type": "number", "x-source-value-types": ["float"]}


def _fraction_leaf() -> dict:
    # Producer range check over WEATHER_FRACTION_FIELDS (:1906-1908).
    return {"type": "number", "minimum": 0, "maximum": 1,
            "x-source-value-types": ["float"]}


def formal_weather_meta_schema() -> dict:
    """Return the closed record schema for one ``formal_weather_meta`` row.

    The producer writes one fixed-key object per tick: the seven core keys
    of the dict literal (:1895-1903) are always present and the eight
    remaining keys are written only when the source row carried them
    (:1904-1930).  ``temperature_source`` implies ``temperature_c`` on the
    same source row; the root ``if`` captures that producer-impossible
    combination (:1926-1930).
    """
    return {
        "$schema": _SCHEMA,
        "$id": "https://aeroworld.local/schema/formal_weather_l1.record.schema.json",
        "title": "AERO_WORLD formal_weather_meta L1 weather row record schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "condition", "fog_density", "rain", "tick", "visibility_m",
            "wetness", "wind_speed",
        ],
        "properties": {
            "condition": {
                # :1892-1894 raises unless the source value is a nonempty
                # string; the stripped+lowered value is written (:1897).  The
                # "\\S" pattern is that emptiness check; no condition
                # vocabulary is producer-enforced, so no enum.
                "type": "string", "pattern": "\\S",
            },
            "tick": {
                # int(tick) over the formal tick grid (:1896); a non-int
                # source tick raises in expand_weather_rows (:1944-1945).
                "type": "integer", "minimum": 0, "maximum": 900,
            },
            "rain": _fraction_leaf(),
            "wetness": _float_leaf(),
            "fog_density": _fraction_leaf(),
            "wind_speed": _float_leaf(),
            "visibility_m": _float_leaf(),
            # Written only when the source row carries the key (:1904-1905).
            "dust": _float_leaf(),
            # Optional numeric channels (:1912-1921): kept absent when the
            # source generator did not provide them — the producer comment
            # at :1909-1911 requires fail-closed missingness, not a default.
            "wind_direction_deg": _float_leaf(),
            "temperature_c": _float_leaf(),
            "illumination_lux": _float_leaf(),
            "hazard_concentration_ppm": _float_leaf(),
            "hazard_radius_m": _float_leaf(),
            # :1922-1925 — written only when present and boolean.
            "hazard_source_active": {"type": "boolean"},
            # :1926-1930 — written only when the source row carries a
            # nonempty string AND temperature_c.  "\\S" mirrors the
            # source.strip() emptiness check; the value is copied unstripped.
            "temperature_source": {"type": "string", "pattern": "\\S"},
        },
        "if": {"required": ["temperature_source"]},
        "then": {"required": ["temperature_c"]},
    }
