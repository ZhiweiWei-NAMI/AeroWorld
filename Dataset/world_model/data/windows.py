"""Canonical supervision windows from the observed prefix and explicit future targets."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable, Mapping

from Dataset.world_model.data.provenance import (  # noqa: F401  (re-exported)
    SplitGroupError,
    content_available_at,
    load_split_table,
    original_identity,
    resolve_source_split,
    split_group,
)
from Dataset.world_model.data.provenance import _record_source_family  # noqa: F401

class FutureLeakError(ValueError):
    """Raised when future information would enter the input branch."""


@dataclass
class SupervisionSample:
    """One causally-split supervision window."""

    episode_id: str
    split: str
    cutoff: int
    target_times: list[int]
    source_family: str | None = None
    split_group_id: str | None = None
    input_records: list[dict[str, Any]] = dc_field(default_factory=list)
    target_records: list[dict[str, Any]] = dc_field(default_factory=list)




def build_supervision_window(records: Iterable[Mapping[str, Any]], *, episode_id: str,
                             split: str, cutoff: int,
                             target_times: Iterable[int],
                             split_table: Mapping[str, str] | None = None) -> SupervisionSample:
    """Split canonical records into input/target branches at ``cutoff``.

    Split authority (v2): ``split`` must equal the group of the episode's
    ORIGINAL scenario/seed resolved through the declared split table (the
    versioned ``data.provenance.SOURCE_SPLITS`` unless an explicit table is
    passed).  A caller may not build a TEST sample from a TRAIN source or
    vice versa; unknown sources are refused, never invented; every
    copy/overlay/branch/window of one source resolves to the same group.

    The input predicate is exactly ``tick <= cutoff`` evaluated on the record
    tick; field values (including visibility) never influence membership.
    Records without a tick (entity/episode) are context and stay on the input
    branch.  Target times must all be > cutoff.
    """
    authority = resolve_source_split(episode_id, split_table=split_table)
    if split != authority:
        raise SplitGroupError(
            f"caller split '{split}' does not match the declared source group "
            f"'{authority}' for original '{original_identity(episode_id).original_key}' "
            f"(episode '{episode_id}')")
    targets = sorted(int(t) for t in target_times)
    if not targets:
        raise FutureLeakError("empty target_times")
    if any(t <= cutoff for t in targets):
        raise FutureLeakError(
            f"target_times must be strictly after cutoff {cutoff}; got {targets}")
    sample = SupervisionSample(episode_id=episode_id, split=split,
                               cutoff=cutoff, target_times=targets,
                               source_family=original_identity(episode_id).original_key,
                               split_group_id=authority)
    allowed_targets = set(targets)
    for rec in records:
        if rec.get("episode_id") not in (None, episode_id):
            raise FutureLeakError(
                f"cross-episode record '{rec.get('episode_id')}' in window build")
        tick = rec.get("tick")
        if tick is None or (isinstance(tick, int) and tick <= cutoff):
            # v2 (native-review D2): input-branch admission applies the shared
            # per-field available_time/content gate - a late-arriving
            # observation, hindsight payload or archive-only record never
            # enters the context even with tick <= cutoff.  The raw record
            # (archive fields included) is left unchanged.
            if not content_available_at(rec, cutoff=cutoff):
                continue
            # v2 (native third replay): the MODEL-INPUT projection of an
            # admitted record strips verbatim archive families and applies
            # the field-level as-of to dated event fields.  The original
            # record in the archive stays complete; only this projection is
            # reduced.
            sample.input_records.append(model_input_projection(rec, cutoff=cutoff))
        elif tick in allowed_targets and rec["record_kind"] in ("frame", "observation"):
            sample.target_records.append(dict(rec))
        # records beyond cutoff that are not declared targets are dropped from
        # both branches (outside this supervision window's span)
    return sample


#: Event fields whose content is dated by their own tick value.  At a cutoff,
#: a field whose date lies after the cutoff was not observable yet and is
#: marked absent in the model-input projection (the archive keeps the value).
DATED_EVENT_FIELDS = ("event.detection_tick", "event.end_tick", "event.start_tick")


def model_input_projection(rec: Mapping[str, Any], *, cutoff: int) -> dict[str, Any]:
    """Model-input view of one admitted record at ``cutoff``.

    v2 (native third replay): the historical model-input projection removes
    every verbatim ``archive.*`` family (the lossless archive stays on the
    original record, never in model input) and applies the field-level as-of
    to dated event fields: ``event.*_tick`` values beyond the cutoff are not
    yet observable and are replaced by the explicit non-present marker
    ``{"kind": "missing"}`` (the schema's legal non-present shape for event
    fields) - never dropped silently, never guessed.

    The original record is never mutated; future targets and the archive
    remain complete.
    """
    fields = rec.get("fields") or {}
    out_fields = {k: v for k, v in fields.items()
                  if not k.startswith("archive.")}
    changed = len(out_fields) != len(fields)
    for name in DATED_EVENT_FIELDS:
        if name in out_fields and isinstance(out_fields[name], int) \
                and out_fields[name] > cutoff:
            out_fields[name] = {"kind": "missing"}
            changed = True
    if not changed:
        return dict(rec)
    out = dict(rec)
    out["fields"] = out_fields
    return out
