# SPDX-License-Identifier: Apache-2.0
"""Core types for Tarot.

These types are the enforcement point for the conventions in `CLAUDE.md`:

- probabilities are floats in [0, 1] — never percentages, never strings
- all datetimes are timezone-aware and normalized to UTC; naive datetimes are
  rejected at validation, not silently interpreted
- `as_of` is required on every forecast call, with no default and no `None`

Schemas follow `docs/00-product-spec.md` (ForecastRequest / ForecastResponse)
and `docs/01-data-pipeline.md` (QuestionRecord / EvidenceDoc). Where this module
extends those schemas, the field carries an `EXTENSION:` comment saying why.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from tarot.nonpredictive import Confidence

__all__ = [
    "EvidenceDoc",
    "ForecastRequest",
    "ForecastResponse",
    "Outcome",
    "PricePoint",
    "Probability",
    "QuestionRecord",
    "SoftTargetProvenance",
    "TimestampProvenance",
    "UtcDatetime",
]


# --------------------------------------------------------------------------
# Scalar types
# --------------------------------------------------------------------------


def _reject_naive_and_normalize(value: datetime) -> datetime:
    """Reject naive datetimes; normalize aware ones to UTC.

    A naive datetime is a bug (`CLAUDE.md` conventions), not something to guess
    a zone for: guessing is exactly how an evidence doc ends up on the wrong
    side of an `as_of` boundary.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            "naive datetime rejected: timestamps must be timezone-aware (ISO 8601 with offset)"
        )
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_reject_naive_and_normalize)]


def _reject_non_numeric_probability(value: Any) -> Any:
    """Reject the two coercions pydantic would otherwise perform silently."""
    if isinstance(value, str):
        raise ValueError(
            f"probability must be a float, not a string (got {value!r}); see CLAUDE.md conventions"
        )
    if isinstance(value, bool):
        raise ValueError("probability must be a float, not a bool")
    return value


Probability = Annotated[
    float,
    BeforeValidator(_reject_non_numeric_probability),
    Field(ge=0.0, le=1.0),
]

# Resolved binary outcome. Unresolved questions carry `None` — see QuestionRecord.
Outcome = Literal[0, 1]


# `metaculus:12345`, `synthetic:cpi:2026-03` — a lowercase source namespace, a
# colon, then a source-local identifier that may itself contain colons.
QUESTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*:[A-Za-z0-9_.:+-]+$")


def _validate_question_id(value: str) -> str:
    if not QUESTION_ID_RE.match(value):
        raise ValueError(
            f"question_id {value!r} is not namespaced by source "
            "(expected e.g. 'metaculus:12345' or 'synthetic:cpi:2026-03')"
        )
    return value


QuestionId = Annotated[str, AfterValidator(_validate_question_id)]

NonEmptyStr = Annotated[str, Field(min_length=1)]


class SoftTargetProvenance(StrEnum):
    """Where a soft target came from. Required whenever one is present.

    `docs/05` weights low-variance targets over terminal binary outcomes, so the
    provenance is not decoration: it is the thing that says how much variance the
    target carries and therefore how heavily it should be weighted.
    """

    #: Aggregated human forecasts at a point in time (Metaculus community
    #: prediction, a Manifold MKT resolution).
    CROWD_DERIVED = "crowd_derived"
    #: A traded price, i.e. money-backed consensus.
    MARKET_CONSENSUS = "market_consensus"
    #: Computed from a reference class rather than elicited.
    COMPUTED_BASE_RATE = "computed_base_rate"
    #: Mean of a model ensemble.
    MODEL_ENSEMBLE = "model_ensemble"


class TimestampProvenance(StrEnum):
    """How a doc's `timestamp` was obtained.

    EXTENSION beyond `docs/01`. The doc schema annotates `timestamp` as
    "publication time, verified", but the same doc lists the four ways that
    claim goes wrong (wrong, missing, crawl-date-not-publish-date, silently
    revised). The detector cannot apply the failure posture ("when in doubt,
    drop the doc") unless the record says how much doubt there is.
    """

    #: Publisher-issued immutable timestamp (SEC filing header, wire archive).
    PUBLISHER_VERIFIED = "publisher_verified"
    #: Publisher-asserted but not independently verified (article byline date).
    PUBLISHER_CLAIMED = "publisher_claimed"
    #: The time we fetched it. Says nothing about publication time.
    CRAWL = "crawl"
    #: Derived heuristically (URL slug, in-text date, sibling docs).
    INFERRED = "inferred"
    #: Not recorded by the ingestion path.
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


class _Frozen(BaseModel):
    """Immutable, no-extra-fields base.

    Records flow from the corpus through the leakage detector into eval. Making
    them frozen means a doc cannot be mutated after it was screened, so a
    verdict cannot go stale behind our back.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceDoc(_Frozen):
    """A single retrieved document offered as evidence for a question."""

    doc_id: NonEmptyStr
    snippet: str
    source: NonEmptyStr
    """Source identifier, matched against the trust-tier registry in `leakage`."""

    timestamp: UtcDatetime | None
    """Publication time.

    ASSUMPTION (deviates from `docs/01`, which types this as required): optional,
    because `docs/01` explicitly lists missing publication timestamps as a thing
    that happens. If the type layer refused to represent a doc with no timestamp,
    the detector's "missing timestamp -> drop" rule would be unreachable and the
    drop would instead happen as a parse crash somewhere in ingestion, uncounted.
    A dropped doc must be *counted*, not thrown.
    """

    provenance: TimestampProvenance = TimestampProvenance.UNKNOWN
    """EXTENSION. How `timestamp` was obtained. Default is the honest default."""

    last_modified: UtcDatetime | None = None
    """EXTENSION. Latest known revision time, if the source exposes one.

    Catches "articles get silently updated after publication" (`docs/01`): a doc
    published before `as_of` but revised after it may carry post-`as_of` content.
    """

    retrieved_at: UtcDatetime | None = None
    """EXTENSION. When we fetched it. Used as a sanity bound on `timestamp`."""

    url: str | None = None
    """EXTENSION. Provenance for manual audit of a rejection."""


class PricePoint(_Frozen):
    """One observation of a market price.

    `docs/01` types `price_series` as `list[(datetime, float)]`. A two-field model
    instead of a bare tuple so the tz-aware and [0, 1] rules apply to it too — a
    market price is a probability and is subject to the same conventions.
    """

    timestamp: UtcDatetime
    price: Probability


class QuestionRecord(_Frozen):
    """A corpus record: the question, its ground truth, and its evidence."""

    question_id: QuestionId
    text: NonEmptyStr
    resolution_criteria: str
    open_date: UtcDatetime

    scheduled_resolve_date: UtcDatetime
    """When the question was *expected* to resolve.

    This is the date the forecaster is told about, and it is the one that slips:
    an expected-March question routinely resolves in June. It therefore says
    nothing about when outcome information became available, and must never be
    used as a temporal boundary. See `resolved_at`.
    """

    resolved_at: UtcDatetime | None = None
    """When the question *actually* resolved, or `None` if it is still open.

    This is the load-bearing date. Two things depend on it and neither can use
    `scheduled_resolve_date`:

    - the temporal split (`docs/01`): "resolves before cutoff -> train" is a
      statement about actual resolution
    - admissible forecast times: `as_of` after `resolved_at` is not a forecast

    Set together with `outcome`; see `_check_resolution_consistency`.
    """

    outcome: Outcome | None = None
    """Resolved binary outcome, or `None` if the question has not resolved.

    Deviates from `docs/01`, which types this as a required int: the live rolling
    eval (`docs/07`) pipes *currently-open* markets in and scores them when they
    close, so unresolved records must be representable. Metrics refuse `None`;
    use `is_resolved` to filter.
    """

    soft_target: Probability | None = None
    """A probability-valued target, as opposed to a realized binary outcome.

    A question can resolve to a probability rather than to an event: a Manifold
    MKT resolution, a Metaculus community prediction, a closing market price.
    Those are not unresolved questions and they are not 0/1 outcomes — they are a
    third thing, and `docs/05` argues they are the *better* training signal,
    because they carry less variance than a terminal outcome.

    Independent of `outcome`. A record may carry a binary outcome, a soft target,
    or both.
    """

    soft_target_provenance: SoftTargetProvenance | None = None
    soft_target_at: UtcDatetime | None = None
    """When the soft target was observed. A soft target is a measurement at an
    instant, not a resolution, so it carries its own time — and that time is what
    bounds admissible `as_of` when the target is used for supervision."""

    category: NonEmptyStr
    base_rate_class: NonEmptyStr

    nonpredictive: Confidence | None = None
    """Whether the market's own creator labelled it as not a forecasting question.

    **`None` means not assessed, which is not the same as assessed-and-clean.**
    `Confidence.NONE` is the clean verdict. The distinction is load-bearing:
    12.9% of the Manifold corpus carries no group slug, so the classifier cannot
    reach it, and a sentinel that conflated "we did not look" with "we looked and
    it is fine" would silently understate contamination — which is exactly the
    error the 2026-08-20 corpus figure made before it was corrected.

    Carried on the record rather than applied to the corpus so that filtering is
    a choice each artifact declares (`tarot.nonpredictive.filter_nonpredictive`)
    rather than a decision baked into the data. Training wants these questions;
    a published evaluation does not; a comparison between subsets must drop them
    or repeat the confound that inflated the 2026-08-22 mechanism result."""

    price_series: tuple[PricePoint, ...] = ()
    evidence: tuple[EvidenceDoc, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.outcome is not None

    @property
    def has_soft_target(self) -> bool:
        return self.soft_target is not None

    @property
    def has_supervision(self) -> bool:
        """Whether this record carries any training signal at all."""
        return self.outcome is not None or self.soft_target is not None

    @property
    def supervision_time(self) -> datetime | None:
        """When this record's supervision became available. `None` if it has none.

        The **latest** of the times attached to the signals it carries. Deliberately
        the latest rather than the earliest: this is what the temporal split keys
        on, and a record is only safely on the training side if *every* signal it
        carries predates the boundary. Taking the earliest would let a record whose
        binary outcome landed after the freeze sit in training on the strength of
        an older soft target.

        For a record with only a binary outcome this is exactly `resolved_at`,
        which is why the split rule did not have to change shape to accommodate
        soft targets — only to stop assuming `outcome` is the only signal.
        """
        times = [t for t in (self.resolved_at, self.soft_target_at) if t is not None]
        return max(times) if times else None

    @model_validator(mode="after")
    def _check_dates(self) -> QuestionRecord:
        if self.scheduled_resolve_date <= self.open_date:
            raise ValueError(
                f"scheduled_resolve_date ({self.scheduled_resolve_date.isoformat()}) "
                f"must be after open_date ({self.open_date.isoformat()})"
            )
        if self.resolved_at is not None and self.resolved_at <= self.open_date:
            raise ValueError(
                f"resolved_at ({self.resolved_at.isoformat()}) must be after "
                f"open_date ({self.open_date.isoformat()})"
            )
        return self

    @model_validator(mode="after")
    def _check_resolution_consistency(self) -> QuestionRecord:
        # `outcome` and `resolved_at` must arrive together. An outcome without a
        # resolution time cannot be placed on either side of the temporal split
        # (docs/01) and cannot bound an admissible `as_of`, so it is a record that
        # looks usable and silently is not.
        if (self.outcome is None) != (self.resolved_at is None):
            raise ValueError(
                "outcome and resolved_at must be set together: "
                f"outcome={self.outcome!r}, resolved_at={self.resolved_at!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_soft_target_consistency(self) -> QuestionRecord:
        # Same shape of rule as outcome/resolved_at, for the same reason: a
        # target with no provenance cannot be weighted (docs/05), and a target
        # with no observation time cannot bound an admissible as_of.
        present = {
            "soft_target": self.soft_target is not None,
            "soft_target_provenance": self.soft_target_provenance is not None,
            "soft_target_at": self.soft_target_at is not None,
        }
        if len(set(present.values())) != 1:
            missing = [name for name, ok in present.items() if not ok]
            raise ValueError(
                "soft_target, soft_target_provenance and soft_target_at must be set "
                f"together; missing: {missing}"
            )
        if self.soft_target_at is not None and self.soft_target_at < self.open_date:
            raise ValueError(
                f"soft_target_at ({self.soft_target_at.isoformat()}) is before "
                f"open_date ({self.open_date.isoformat()})"
            )
        return self

    @model_validator(mode="after")
    def _check_price_series_ordered(self) -> QuestionRecord:
        # "Market price at as_of" (docs/07 baselines) is a lookup of the last
        # point at or before as_of. That lookup is only well defined on an
        # ordered series, so require it here rather than re-sorting downstream.
        timestamps = [p.timestamp for p in self.price_series]
        if any(later < earlier for earlier, later in pairwise(timestamps)):
            raise ValueError("price_series must be ordered by non-decreasing timestamp")
        return self


class ForecastRequest(_Frozen):
    """The input half of the interface contract (`docs/00`).

    Every tier takes exactly this. Prior tier passes `evidence=()`.
    """

    question: NonEmptyStr
    resolution_criteria: str

    scheduled_resolve_date: UtcDatetime
    """The expected resolution date, which is all the forecaster gets to know.

    Deliberately NOT validated against `as_of`. A question can be open long past
    its scheduled date, so `as_of > scheduled_resolve_date` is an ordinary
    forecast on a slipped question, not an error. The check that matters is
    `as_of` against the record's `resolved_at`, and that lives in
    `tarot.leakage.check_forecast_time` — a request cannot carry `resolved_at`,
    because the actual resolution time is exactly the thing the model must not
    see.
    """

    as_of: UtcDatetime
    """Required. No default, no `None` — everything the model may condition on is
    defined relative to this instant."""

    evidence: tuple[EvidenceDoc, ...] = ()

    question_id: QuestionId | None = None
    """EXTENSION. Optional join key back to the corpus record, so an eval run can
    line up responses with ground truth without positional matching."""


class ForecastResponse(_Frozen):
    """The output half of the interface contract (`docs/00`)."""

    probability: Probability
    reasoning: str
    model_version: NonEmptyStr
    as_of: UtcDatetime
    """Echoed back from the request, so a stored response is self-describing."""
