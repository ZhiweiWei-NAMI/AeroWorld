"""S01.04 normalization with explicit fit-scope provenance.

Scalers are fitted only on the allowed input-branch history of TRAIN-split
episodes (tick <= cutoff).  VALID/TEST fit attempts and transform-time split
violations raise instead of silently proceeding: test-fit normalization is
rejected by construction, not by convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from p01v4.data.provenance import content_available_at, resolve_source_split

NORM_VERSION = "p01v4-norm-1"
ALLOWED_FIT_SPLITS = ("TRAIN",)


class NormalizationScopeError(ValueError):
    """Raised when a scaler is fitted or applied outside its declared scope."""


@dataclass
class Scaler:
    """Per-field standardization scaler with explicit fit-scope provenance."""

    field: str
    episode_id: str
    split: str
    max_fit_tick: int
    mean: float
    std: float
    count: int
    version: str = NORM_VERSION

    def scope_document(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "field": self.field,
            "fit_episode_id": self.episode_id,
            "fit_split": self.split,
            "max_fit_tick": self.max_fit_tick,
            "count": self.count,
        }

    def transform(self, value: float) -> float:
        if self.split not in ALLOWED_FIT_SPLITS:
            raise NormalizationScopeError(
                f"scaler fitted on split '{self.split}' may not transform values")
        if self.count == 0 or self.std == 0.0:
            raise NormalizationScopeError("degenerate scaler: zero count or zero std")
        return (float(value) - self.mean) / self.std


def fit_scaler(records: Iterable[Mapping[str, Any]], *, field: str, episode_id: str,
               split: str, cutoff: int,
               split_table: Mapping[str, str] | None = None) -> Scaler:
    """Fit mean/std on input-branch values only (tick <= cutoff, TRAIN split).

    VALID/TEST splits and future records are excluded by construction: the
    fit input is the caller-declared allowed history, and the split gate
    rejects any non-TRAIN attempt (test-fit normalization is impossible).

    Source/split binding at the actual fit entry (v2, native-review D3): the
    caller-declared ``split`` is not trusted on its own.  The episode's
    ORIGINAL source is resolved through the declared split table (versioned
    ``configs/split-table.json`` unless an explicit table is passed) and must
    be a TRAIN original; a caller may not fit a scaler for a VALID/TEST/other
    source by declaring ``split="TRAIN"``.  Every fitted record must belong
    to ``episode_id`` (a foreign-episode fit is an error, never silently
    averaged in).

    Actual availability at the fit entry (native 095638 followup, repair 3):
    ``cutoff`` is enforced on the record's actual observability, not on the
    tick alone.  A record whose content only became observable after the
    cutoff (declared ``available_time.tick`` > cutoff, archive-only or
    unknown rule) never enters the fit - the same per-field gate the window
    builder applies on the input branch.
    """
    if split not in ALLOWED_FIT_SPLITS:
        raise NormalizationScopeError(
            f"scaler fit attempted on split '{split}'; allowed: {ALLOWED_FIT_SPLITS}")
    authority = resolve_source_split(episode_id, split_table=split_table)
    if authority != "TRAIN":
        raise NormalizationScopeError(
            f"scaler fit refused: source of episode '{episode_id}' resolves to split "
            f"group '{authority}' through the declared split table; only TRAIN "
            "sources may be fitted, regardless of the caller-declared split")
    values: list[float] = []
    max_tick = -1
    for rec in records:
        rec_episode = rec.get("episode_id")
        if rec_episode is not None and rec_episode != episode_id:
            raise NormalizationScopeError(
                f"scaler fit input contains a record of episode '{rec_episode}' while "
                f"fitting episode '{episode_id}'; foreign-episode fit is rejected")
        tick = rec.get("tick")
        if not isinstance(tick, int) or tick > cutoff:
            continue  # future records never enter the fit
        if not content_available_at(rec, cutoff=cutoff):
            continue  # content not actually observable at the cutoff
        v = rec.get("fields", {}).get(field)
        if isinstance(v, Mapping) or v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            values.append(float(v))
            max_tick = max(max_tick, tick)
    if not values:
        raise NormalizationScopeError(f"no fit values for field '{field}' within cutoff")
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return Scaler(field=field, episode_id=episode_id, split=split,
                  max_fit_tick=max_tick, mean=mean, std=var ** 0.5, count=n)
