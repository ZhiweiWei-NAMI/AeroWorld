"""S01.04 supplement: source identity, split-group coupling, producer policy.

Parent cross-review supplement rules implemented here:

- ``transported_q`` (historical diagnostic: TRUE future values mixed with
  TRAIN-frozen Gaussian noise) is an oracle diagnostic only and must never
  enter new prediction inputs;
- ``source_family`` and ``split_group`` are derived from the ORIGINAL
  scenario and seed: every old/new copy, 36overlay, normal/congested branch
  and every supervision window derived from the same source share exactly
  one split group - versions are not independent examples;
- producer mechanism, available_time/cutoff and same-run capture binding are
  recorded where established; anything not established stays explicitly
  unknown (never guessed).

The decorated-episode grammar (copy/overlay/branch suffixes) is parent-
declared lineage; no decorated identifier exists in this package's sources,
so the parser preserves unknown decoration marks verbatim and flags the
grammar as unverified instead of inventing splits from them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ORIGINAL_ID_RE = re.compile(
    r"^(?P<scenario>.+?)_v(?P<version>\d+)__seed(?P<seed>\d+)$")


class SplitGroupError(ValueError):
    """Raised when split-group coupling or transported-input rules break."""


@dataclass(frozen=True)
class OriginalIdentity:
    """Original scenario/seed behind an episode id, plus variant marks."""

    scenario: str
    version: int
    seed: int
    variant_marks: tuple[str, ...]
    grammar_verified: bool
    _raw_key: str = ""

    @property
    def original_key(self) -> str:
        # the raw pre-decoration head, preserving zero-padding exactly
        return self._raw_key

    def document(self) -> dict[str, Any]:
        return {
            "original_key": self.original_key,
            "scenario": self.scenario,
            "version": self.version,
            "seed": self.seed,
            "variant_marks": list(self.variant_marks),
            "grammar_verified": self.grammar_verified,
        }


def original_identity(episode_id: str, *, variant_delimiters: tuple[str, ...] = ("::", "++")) -> OriginalIdentity:
    """Derive the ORIGINAL scenario/seed identity from any episode id.

    Known-verified grammar: ``<scenario>_v<N>__seed<M>`` (the form used by
    every declared capture cohort entry).  Any additional decoration (copies,
    36overlay marks, normal/congested branch suffixes) is expected to appear
    after a variant delimiter or trailing suffix; such marks are preserved
    verbatim and ``grammar_verified`` is False when they are present, so a
    caller can never mistake an unverified decoration for a known id form.
    """
    text = episode_id
    marks: list[str] = []
    for delim in variant_delimiters:
        if delim in text:
            head, tail = text.split(delim, 1)
            marks.append(tail)
            text = head
    m = ORIGINAL_ID_RE.match(text)
    if m is None:
        raise SplitGroupError(
            f"episode id '{episode_id}' does not match the original-identity grammar "
            f"<scenario>_v<N>__seed<M> (decorations seen: {marks or 'none'})")
    return OriginalIdentity(scenario=m.group("scenario"), version=int(m.group("version")),
                            seed=int(m.group("seed")), variant_marks=tuple(marks),
                            grammar_verified=not marks, _raw_key=text)


def split_group(episode_id: str, split_table: Mapping[str, str]) -> str:
    """Resolve the one split group shared by all versions of one source.

    ``split_table`` maps ORIGINAL keys (``<scenario>_v<N>__seed<M>``) to a
    group label and must be loaded from a versioned config.  Every copy,
    36overlay, normal/congested branch and window of the same original
    resolves to that original's single entry (decorated ids share the
    original's group by construction, so versions are never independent
    examples); an UNDECORATED original missing from the table is an error,
    never a silent default.
    """
    ident = original_identity(episode_id)
    if ident.original_key not in split_table:
        raise SplitGroupError(
            f"original '{ident.original_key}' (from '{episode_id}') is not in the "
            "split table; refusing to invent a split group")
    return split_table[ident.original_key]


# Whole scenarios (all versions and seeds of e.g. L4-3) belong to exactly one split,
# so held-out windows never share a scenario layout with TRAIN.
SOURCE_SPLITS = {
    'L1-1_v1__seed00': 'TRAIN',
    'L1-1_v1__seed01': 'TRAIN',
    'L1-4_v1__seed00': 'TRAIN',
    'L2-2_v1__seed00': 'TRAIN',
    'L2-2_v2__seed00': 'TRAIN',
    'L2-2_v1__seed01': 'TRAIN',
    'L3-3_v1__seed00': 'TRAIN',
    'L3-3_v1__seed01': 'TRAIN',
    'L4-1_v1__seed00': 'TRAIN',
    'L4-5_v1__seed00': 'TRAIN',
    'L4-8_v1__seed00': 'TRAIN',
    'L4-2_v1__seed00': 'TRAIN',
    'L4-4_v1__seed00': 'TRAIN',
    'L4-6_v1__seed00': 'TRAIN',
    'L5-2_v1__seed00': 'TRAIN',
    'L5-3_v1__seed00': 'TRAIN',
    'L6-3_v1__seed00': 'TRAIN',
    'L6-3_v1__seed01': 'TRAIN',
    'L1-2_v1__seed01': 'VALID',
    'L2-3_v2__seed00': 'VALID',
    'L4-3_v1__seed00': 'VALID',
    'L4-7_v1__seed00': 'VALID',
    'L5-4_v1__seed00': 'VALID',
    'L6-5_v1__seed00': 'VALID',
    'L1-3_v1__seed00': 'TEST',
    'L2-1_v2__seed00': 'TEST',
    'L4-10_v1__seed00': 'TEST',
    'L4-11_v1__seed00': 'TEST',
    'L5-5_v1__seed00': 'TEST',
}


def load_split_table() -> dict[str, str]:
    return dict(SOURCE_SPLITS)


def resolve_source_split(episode_id: str,
                         *, split_table: Mapping[str, str] | None = None) -> str:
    """Split group of an episode's ORIGINAL source via the declared table."""
    return split_group(episode_id,
                       split_table if split_table is not None else load_split_table())


# ----------------------- transported_q producer policy --------------------

TRANSPORTED_Q_FAMILY = "transported_q"
# Parent-declared historical mechanism (B01 supplement): transported_q was
# produced from TRUE future values with TRAIN-frozen Gaussian noise added.
# No code in this repository re-derives that mechanism; it is recorded as
# lineage and used only to keep it out of prediction inputs.
def content_available_at(rec: Mapping[str, Any], *, cutoff: int) -> bool:
    """Per-field-policy content gate (v2, native-review D2).

    A record may enter the input branch only when its content was actually
    observable at the cutoff: the record tick must be <= cutoff AND any
    declared ``available_time.tick`` (late-arriving observations, hindsight
    truth, capture bindings) must also be <= cutoff.  Records whose
    availability rule is known to be archive-only are never input.  This is
    the field-level observer-available cutoff; raw archive content is left
    unchanged on the record itself.
    """
    tick = rec.get("tick")
    if isinstance(tick, int) and tick > cutoff:
        return False
    avail = (rec.get("provenance") or {}).get("available_time") or {}
    rule = avail.get("rule")
    if rule == "archive_only":
        return False
    if rule in ("capture_bound", "hindsight_truth"):
        at = avail.get("tick", tick)
        return isinstance(at, int) and at <= cutoff
    if rule is not None:  # undeclared/unknown rule: never guessed in
        return False
    return True


# ------------------- provenance and capture binding -----------------------

def _record_source_family(rec: Mapping[str, Any]) -> str | None:
    """Declared source family of a record: ``source_family`` first, then
    provenance ``producer_mechanism.source_family``.  Never a guessed id
    match; undeclared stays None."""
    fam = rec.get("source_family")
    if fam is None:
        prov = rec.get("provenance") or {}
        fam = prov.get("producer_mechanism", {}).get("source_family")
    return fam


def capture_binding_record(episode_id: str, *, frame_tick: int,
                           observations: Iterable[Mapping[str, Any]],
                           run_id: str | None = None,
                           capture_root: str | None = None) -> dict[str, Any]:
    """Same-run capture binding between a truth frame and its modalities.

    Established for the pilot rgb/lidar supplement: an observation at tick t
    is bound to the truth frame at the same tick of the same run.  Returns
    the binding document the window builder stores in provenance.

    Same-run identity (v2, native-review D2): the binding records the
    episode, the capture run (``run_id``, the importer-declared capture run identity) and, when provided, the capture root it was bound
    within.  Cross-episode observations never bind: every observation must
    declare the binding's ``episode_id``.

    Run verification (native 095638 followup, repair 2): a requested
    ``run_id`` is verified against each observation's ACTUAL run identity,
    not merely stored.  An observation declares its run through
    ``source.capture.run_id`` (set by the importer) or an explicit
    ``run_id`` on the record; an
    observation of another run cannot establish this binding.  When no
    requested run is given, the actual run is read from the observations
    and must agree on one run; observations that declare no run identity
    leave ``run_id`` None (unknown stays unknown, never guessed).
    """
    obs = list(observations)
    ticks = {o.get("tick") for o in obs}
    if ticks and ticks != {frame_tick}:
        raise SplitGroupError(
            f"capture binding for tick {frame_tick} got observations at {sorted(ticks)}; "
            "same-run binding requires matching ticks")
    for o in obs:
        o_ep = o.get("episode_id", episode_id)
        if o_ep != episode_id:
            raise SplitGroupError(
                f"capture binding for episode '{episode_id}' got an observation of "
                f"episode '{o_ep}'; cross-episode capture cannot bind to the same run")

    def _obs_run(o: Mapping[str, Any]) -> str | None:
        # declared run identity of one observation; never guessed from ids
        r = o.get("run_id")
        if isinstance(r, str) and r:
            return r
        src = o.get("source") or {}
        declared = (src.get("capture") or {}).get("run_id")
        if isinstance(declared, str) and declared:
            return declared
        return None

    actual_runs = {r for r in (_obs_run(o) for o in obs) if r is not None}
    if run_id is not None:
        if actual_runs and run_id not in actual_runs:
            raise SplitGroupError(
                f"capture binding for episode '{episode_id}' tick {frame_tick} was "
                f"requested for run '{run_id}' but the observations belong to "
                f"{sorted(actual_runs)}; observations of another run cannot "
                "establish this same-run binding")
        doc_run = run_id
    elif actual_runs:
        if len(actual_runs) > 1:
            raise SplitGroupError(
                f"capture binding for episode '{episode_id}' tick {frame_tick} got "
                f"observations of several runs {sorted(actual_runs)}; one binding "
                "binds one run")
        doc_run = next(iter(actual_runs))
    else:
        doc_run = None  # no observation declares a run: unknown stays unknown
    doc: dict[str, Any] = {
        "rule": "same_run_capture_binding",
        "episode_id": episode_id,
        "frame_tick": frame_tick,
        "run_id": doc_run,
        "modalities": sorted({o.get("fields", {}).get("observation.modality")
                              for o in obs if o.get("fields")}),
        "established": True,
    }
    if capture_root is not None:
        doc["capture_root"] = capture_root
    return doc


# ---------------- point 1: package vs per-dimension revisions -------------

class ProvenanceError(ValueError):
    """Raised when a revision/provenance document is structurally invalid."""


def package_revision_document(*, package_source_revision: str, dimensions: Mapping[str, Any]) -> dict[str, Any]:
    """Separate the v14 PACKAGE source_revision from per-dimension revisions.

    A package label (e.g. an inherited 36x8 mark) never proves that all
    eight producers changed.  The package carries exactly one
    ``source_revision``; each of the eight dimensions must carry its own
    ``producer_revision`` plus the actual source files mapped for it.  An
    absent dimension or an absent file list is recorded as null with an
    explicit note - unknown stays unknown.
    """
    doc = {
        "rule": "package_source_revision_vs_per_dimension_producer_revision",
        "package_source_revision": package_source_revision,
        "dimensions": {},
    }
    required = ("truth_frames", "world_truth_graph_deltas", "event_occurrences",
                "entity_roster", "rgb", "lidar")
    for name in required:
        d = dimensions.get(name)
        if d is None:
            doc["dimensions"][name] = {
                "producer_revision": None,
                "source_files": [],
                "note": "dimension not provided; producer_revision unknown",
            }
        else:
            files = d.get("source_files", [])
            if not files:
                raise ProvenanceError(
                    f"dimension '{name}' must map its actual source files; "
                    "a package revision never proves a producer changed")
            doc["dimensions"][name] = {
                "producer_revision": d.get("producer_revision"),
                "source_files": list(files),
                "note": d.get("note"),
            }
    return doc


def changed_dimensions(revision_doc: Mapping[str, Any]) -> list[str]:
    """Dimension names whose producer_revision differs from the package's.

    Comparisons use explicit per-dimension values only; a dimension with an
    unknown (null) producer_revision is never counted as changed or
    unchanged - it is returned separately in the ``unknown`` bucket by
    :func:`revision_summary`.
    """
    pkg = revision_doc["package_source_revision"]
    return sorted(
        name for name, d in revision_doc["dimensions"].items()
        if d.get("producer_revision") is not None
        and d["producer_revision"] != pkg)


def revision_summary(revision_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Package-vs-dimension summary; unknowns stay explicit."""
    pkg = revision_doc["package_source_revision"]
    changed = changed_dimensions(revision_doc)
    unknown = sorted(
        name for name, d in revision_doc["dimensions"].items()
        if d.get("producer_revision") is None)
    same = sorted(
        name for name, d in revision_doc["dimensions"].items()
        if d.get("producer_revision") is not None and d["producer_revision"] == pkg)
    return {
        "package_source_revision": pkg,
        "changed_vs_package": changed,
        "same_as_package": same,
        "unknown_producer_revision": unknown,
        "note": "a shared package label never proves all producers changed",
    }


# ---------------------- point 5: L/P definition ids -----------------------

L_DEFINITION_LEGACY_Q = "L/p01-legacy-q-bounded-observation"
L_DEFINITION_TTL = "L/p01-ttl-observation-actual-available-time"
#: B00 lineage: old q-based L/P and native mature accepted-cohort TTL L/P are
#: different definitions, never one metric under two cohorts.


def timely_L(cases: Iterable[Mapping[str, Any]], *, cutoff: int) -> dict[str, Any]:
    """TTL L over cases with actual available_time <= cutoff.

    A case without a timely RX (reception) has L UNDEFINED for it - it is
    excluded from the mean and counted in ``l_undefined``, never folded into
    L as zero.  Cases with pending/censored compute tasks contribute no
    hindsight outcome; they are counted, not imputed.
    """
    values: list[float] = []
    undefined = 0
    pending = 0
    for c in cases:
        at = c.get("available_time_tick")
        rx = c.get("rx_tick")
        if not isinstance(at, int) or at > cutoff or not isinstance(rx, int):
            undefined += 1
            continue
        values.append(float(c["latency_ticks"]))
    # pending/censored compute tasks never contribute hindsight outcomes
    pending = sum(1 for c in cases if c.get("compute_status") in ("pending", "censored"))
    return {
        "definition_id": L_DEFINITION_TTL,
        "cutoff": cutoff,
        "n_timely": len(values),
        "l_undefined_cases": undefined,
        "pending_or_censored": pending,
        "L": (sum(values) / len(values)) if values else None,
    }
