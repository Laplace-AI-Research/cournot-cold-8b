# SPDX-License-Identifier: Apache-2.0
"""The temporal split, and the guard that keeps dev numbers out of model cards.

`docs/01` "Temporal split", `docs/07`, and the freeze-date entry in
the internal decisions log. Three things this module exists to make structural:

1. **The split keys on `resolved_at`** — actual resolution, never `open_date`
   and never `scheduled_resolve_date`. A question that was scheduled for March
   and resolved in June resolved in June.
2. **`dev` and `published` are different artifacts with different status.** dev
   is carved from the training side and may be contaminated via base
   pretraining; it gates iteration and is never published. published is
   contamination-free by construction and is the only source of a headline
   number.
3. **A card-bound artifact cannot be built from dev.** Not by convention — the
   guard re-derives the split from `resolved_at` at emission time rather than
   trusting a field, so there is nothing to set wrongly and nothing to remember.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from tarot.metrics import (
    Binning,
    BrierDecomposition,
    ECEResult,
    HistogramStats,
    LogScoreResult,
    brier,
    ece,
    log_score,
    output_histogram,
)
from tarot.types import QuestionRecord

__all__ = [
    "DEV_START",
    "FREEZE",
    "CardMetrics",
    "EvalSlice",
    "HorizonBucket",
    "IterationMetrics",
    "MetricSet",
    "MissingForecastTimesError",
    "NotPublishableError",
    "Split",
    "SplitCensus",
    "TemporalSplit",
    "census",
    "forecast_horizon",
    "format_census",
    "iteration_metrics",
    "model_card_metrics",
    "question_lifetime",
]


#: The global freeze. Recorded in the internal decisions log (entry dated 2026-08-13).
#:
#: A clean UTC midnight that was strictly in the future when the decision was
#: committed, so "nothing in `published` resolved before we committed to the
#: freeze" is checkable by anyone holding the entry and a clock. A backdated
#: freeze is the first thing an adversarial reader looks for.
#:
#: DO NOT MOVE THIS. Moving it forward invalidates comparability of every
#: published figure and of every point on the track record that straddles the
#: change; all of them must be regenerated and re-dated. It is a constant here
#: so that a change to it shows up in a diff.
FREEZE = datetime(2026, 8, 15, tzinfo=UTC)

#: Start of the dev window, carved from the training side. See
#: the internal decisions log for why twelve months.
DEV_START = FREEZE - timedelta(days=365)


class Split(StrEnum):
    """Which side of the temporal split a question falls on."""

    #: Resolved before the dev window. Trained on.
    TRAIN = "train"
    #: Resolved inside the dev window. Carved from the training side, so it may
    #: be contaminated via base pretraining and we cannot rule that out. All
    #: iteration and all gating run here. NEVER published, never in a model card.
    DEV = "dev"
    #: Resolved after the freeze. Contamination-free by construction, since the
    #: freeze is after every base model's pretraining cutoff. Accumulates
    #: forward. The only source of a published number.
    PUBLISHED = "published"
    #: Carries no supervision signal of any kind — no binary outcome and no soft
    #: target. Belongs to neither side; it joins `published` when it acquires
    #: one, since every future resolution is after the freeze.
    OPEN = "open"

    @property
    def publishable(self) -> bool:
        """Whether a number computed on this split may leave the building."""
        return self is Split.PUBLISHED

    @property
    def trainable(self) -> bool:
        """Whether records on this split may enter a training set.

        `dev` is trainable-in-principle — it is on the training side — but doing
        so destroys its only purpose, so it is excluded here too. `CLAUDE.md` #2
        covers `published`: never, under any circumstances.
        """
        return self is Split.TRAIN


class MissingForecastTimesError(ValueError):
    """A card-bound artifact was requested from a slice with no `as_ofs`.

    Not a convention violation to be documented — an omission that cannot be
    repaired later. Forecast horizons are `resolved_at - as_of`, and once a run
    has discarded its `as_of` values nothing can reconstruct them, so the
    per-horizon reliability diagrams `docs/07` requires become permanently
    unavailable for that run. Nothing fails at the time it happens, which is the
    shape of guard this repo has been caught by twice.
    """


class NotPublishableError(RuntimeError):
    """Raised when a card-bound artifact is requested from a non-published split.

    This is the failure mode `docs/09` #1 ends in: an internal number, computed
    on data the base model may have seen, quoted externally. It is an exception
    rather than a warning because there is no correct way to proceed.
    """


@dataclass(frozen=True)
class TemporalSplit:
    """The freeze, the dev boundary, and the assignment rule.

    Intervals, stated once so the boundaries are not re-derived by eye, over
    `record.supervision_time` — the latest time at which any of the record's
    training signals became available:

        TRAIN      supervision_time <  dev_start
        DEV        dev_start        <= supervision_time <= freeze
        PUBLISHED  freeze           <  supervision_time
        OPEN       supervision_time is None

    For a record carrying only a binary outcome, `supervision_time` *is*
    `resolved_at`, so this is the same rule it always was. It changed only to
    stop assuming a binary outcome is the only thing a record can be supervised
    on — a soft target is supervision too, and a question that resolved to a
    probability is not an open question.
    """

    freeze: datetime = FREEZE
    dev_start: datetime = DEV_START

    def __post_init__(self) -> None:
        for name, value in (("freeze", self.freeze), ("dev_start", self.dev_start)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.dev_start >= self.freeze:
            raise ValueError(
                f"dev_start ({self.dev_start.isoformat()}) must be before "
                f"freeze ({self.freeze.isoformat()})"
            )

    def assign(self, record: QuestionRecord) -> Split:
        """Assign one record, keying on `record.supervision_time`.

        A question whose `resolved_at` is exactly the freeze instant goes to the
        training side (`dev`), not to `published`. Same asymmetry as the
        `timestamp == as_of` boundary in `tarot.leakage`: resolution times are
        routinely recorded at day granularity, so an exactly-equal timestamp
        usually means "some time on freeze day", part of which is before the
        freeze. `published` is the split whose whole value is the claim that
        contamination is impossible, and a question that *might* have resolved
        before the freeze cannot carry that claim. Losing one question costs
        nothing; weakening the claim costs the product.
        """
        supervision_time = record.supervision_time
        if supervision_time is None:
            return Split.OPEN
        if supervision_time > self.freeze:
            return Split.PUBLISHED
        if supervision_time >= self.dev_start:
            return Split.DEV
        return Split.TRAIN

    def partition(
        self, records: Sequence[QuestionRecord]
    ) -> Mapping[Split, tuple[QuestionRecord, ...]]:
        """Split a corpus. Every `Split` key is present, possibly empty."""
        buckets: dict[Split, list[QuestionRecord]] = {s: [] for s in Split}
        for record in records:
            buckets[self.assign(record)].append(record)
        return {split: tuple(items) for split, items in buckets.items()}


DEFAULT_SPLIT = TemporalSplit()


# --------------------------------------------------------------------------
# Census — the counts that go in the decision entry
# --------------------------------------------------------------------------


def question_lifetime(record: QuestionRecord) -> timedelta:
    """`supervision_time - open_date`. How long the question was open.

    A property of the RECORD. Use for corpus description: how much of the corpus
    is long-running questions. Not what reliability diagrams break out by.
    """
    end = record.supervision_time
    if end is None:
        raise ValueError(f"{record.question_id} carries no supervision; it has no lifetime yet")
    return end - record.open_date


def forecast_horizon(record: QuestionRecord, as_of: datetime) -> timedelta:
    """`resolved_at - as_of`. How far ahead the forecast was looking.

    A property of the FORECAST. This is what `docs/07`'s per-horizon reliability
    diagrams break out by, because calibration degrades with how far ahead you
    are predicting — not with how long the question happened to be open.

    The two diverge worst exactly where horizon analysis matters: a question open
    for two years and forecast a week before it resolves has a lifetime of 730
    days and a horizon of 7. Bucketing that forecast as long-horizon would put a
    near-certain call in with the genuinely hard ones and flatter the diagram.
    """
    if record.resolved_at is None:
        raise ValueError(f"{record.question_id} is unresolved; it has no horizon")
    if as_of >= record.resolved_at:
        raise ValueError(
            f"as_of {as_of.isoformat()} is at or after resolved_at "
            f"{record.resolved_at.isoformat()} for {record.question_id}; that is not "
            "a forecast (see tarot.leakage.check_forecast_time)"
        )
    return record.resolved_at - as_of


class HorizonBucket(StrEnum):
    """Half-open duration buckets, applicable to either horizon quantity.

    Deliberately holds no opinion about which one it is bucketing — the caller
    picks, via `of_lifetime` or `of_forecast`, and the two are never
    interchangeable. See `question_lifetime` and `forecast_horizon`.
    """

    UNDER_30D = "<30d"
    D30_TO_180D = "30-180d"
    OVER_180D = "180d+"

    @staticmethod
    def containing(duration: timedelta) -> HorizonBucket:
        days = duration.total_seconds() / 86400.0
        if days < 30.0:
            return HorizonBucket.UNDER_30D
        if days < 180.0:
            return HorizonBucket.D30_TO_180D
        return HorizonBucket.OVER_180D

    @staticmethod
    def of_lifetime(record: QuestionRecord) -> HorizonBucket | None:
        """Bucket by how long the question was open. `None` if still open."""
        if record.supervision_time is None:
            return None
        return HorizonBucket.containing(question_lifetime(record))

    @staticmethod
    def of_forecast(record: QuestionRecord, as_of: datetime) -> HorizonBucket:
        """Bucket by how far ahead the forecast looked. What diagrams use."""
        return HorizonBucket.containing(forecast_horizon(record, as_of))


@dataclass(frozen=True)
class SplitCensus:
    split: Split
    n: int
    by_lifetime: Mapping[HorizonBucket, int]
    """Bucketed by `question_lifetime`, NOT by `forecast_horizon`.

    The census describes a corpus, and a corpus has no `as_of` — the same
    question is forecast at many horizons. Per-horizon reliability diagrams use
    `forecast_horizon` and are computed per eval slice, not here.
    """

    n_without_lifetime: int
    """Open questions, which have no lifetime yet."""


def census(
    records: Sequence[QuestionRecord], *, temporal_split: TemporalSplit = DEFAULT_SPLIT
) -> Mapping[Split, SplitCensus]:
    """Count questions per split, bucketed by QUESTION LIFETIME.

    For the decision entry and the dataset manifest. See `SplitCensus.by_lifetime`
    for why this is not the forecast horizon.
    """
    partitioned = temporal_split.partition(records)
    out: dict[Split, SplitCensus] = {}
    for split, items in partitioned.items():
        lifetimes: Counter[HorizonBucket] = Counter()
        missing = 0
        for record in items:
            bucket = HorizonBucket.of_lifetime(record)
            if bucket is None:
                missing += 1
            else:
                lifetimes[bucket] += 1
        out[split] = SplitCensus(
            split=split, n=len(items), by_lifetime=dict(lifetimes), n_without_lifetime=missing
        )
    return out


def format_census(counts: Mapping[Split, SplitCensus]) -> str:
    """Render the census as a markdown table, for pasting into the internal decisions log."""
    header = "| Split | n | lifetime <30d | 30-180d | 180d+ |\n|---|---:|---:|---:|---:|"
    rows = [
        f"| {split.value} | {c.n} | "
        + " | ".join(str(c.by_lifetime.get(b, 0)) for b in HorizonBucket)
        + " |"
        for split, c in ((s, counts[s]) for s in Split if s in counts)
    ]
    return "\n".join([header, *rows])


# --------------------------------------------------------------------------
# Scored slices and the publishability guard
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSlice:
    """Forecasts paired with the records they were made on.

    Holds the records themselves rather than a split label, so `split` is
    *derived* on demand from `resolved_at`. There is no field anyone can set to
    make dev data look publishable.
    """

    records: tuple[QuestionRecord, ...]
    probabilities: tuple[float, ...]
    as_ofs: tuple[datetime, ...] | None = None
    """When each forecast was made. Optional, because scoring does not need it —
    but per-horizon reliability diagrams do, and they cannot be reconstructed
    later, so an eval runner should always supply it."""

    label: str = ""

    def __post_init__(self) -> None:
        if len(self.records) != len(self.probabilities):
            raise ValueError(
                f"length mismatch: {len(self.records)} records, "
                f"{len(self.probabilities)} probabilities"
            )
        if not self.records:
            raise ValueError("empty eval slice")
        unresolved = [r.question_id for r in self.records if r.outcome is None]
        if unresolved:
            raise ValueError(f"cannot score unresolved questions: {unresolved[:3]}")
        if self.as_ofs is not None:
            if len(self.as_ofs) != len(self.records):
                raise ValueError(
                    f"length mismatch: {len(self.records)} records, {len(self.as_ofs)} as_ofs"
                )
            # Validates each as_of against its record's resolved_at.
            self.forecast_horizons()

    @property
    def outcomes(self) -> tuple[int, ...]:
        # `__post_init__` rejects unresolved records, so no outcome is None here.
        return tuple(r.outcome for r in self.records if r.outcome is not None)

    @property
    def n(self) -> int:
        return len(self.records)

    def forecast_horizons(self) -> tuple[timedelta, ...]:
        """`resolved_at - as_of` per forecast. Requires `as_ofs`."""
        if self.as_ofs is None:
            raise ValueError(
                "this slice has no as_ofs, so forecast horizons are unknown; "
                "they cannot be reconstructed from the records — a question's "
                "lifetime is not its forecast horizon"
            )
        return tuple(
            forecast_horizon(record, as_of)
            for record, as_of in zip(self.records, self.as_ofs, strict=True)
        )

    def by_forecast_horizon(self) -> Mapping[HorizonBucket, EvalSlice]:
        """Sub-slices grouped by forecast horizon, for `docs/07`'s per-horizon
        reliability diagrams. Only non-empty buckets appear.

        Sub-slices keep their records, so the publishability guard still applies
        to each one — a per-horizon diagram is as card-bound as the aggregate.
        """
        grouped: dict[HorizonBucket, list[int]] = {}
        for i, horizon in enumerate(self.forecast_horizons()):
            grouped.setdefault(HorizonBucket.containing(horizon), []).append(i)
        assert self.as_ofs is not None  # forecast_horizons() would have raised
        return {
            bucket: EvalSlice(
                records=tuple(self.records[i] for i in idx),
                probabilities=tuple(self.probabilities[i] for i in idx),
                as_ofs=tuple(self.as_ofs[i] for i in idx),
                label=f"{self.label}:{bucket.value}" if self.label else bucket.value,
            )
            for bucket, idx in grouped.items()
        }

    def split(self, temporal_split: TemporalSplit = DEFAULT_SPLIT) -> Split:
        """The split every record in this slice belongs to.

        Refuses a mixed slice. A number computed across the freeze is neither
        publishable nor interpretable — it averages contaminated and
        contamination-free questions and reports one figure.
        """
        splits = {temporal_split.assign(r) for r in self.records}
        if len(splits) > 1:
            raise ValueError(
                "eval slice spans multiple splits "
                f"({sorted(s.value for s in splits)}); partition before scoring"
            )
        return splits.pop()


@dataclass(frozen=True)
class MetricSet:
    """The numbers themselves, with no claim about where they may be used."""

    n: int
    brier: BrierDecomposition
    ece: ECEResult
    log_score: LogScoreResult
    histogram: HistogramStats

    def summary(self) -> str:
        return "\n".join(
            [
                self.brier.summary(),
                self.ece.summary(),
                self.log_score.summary(),
                self.histogram.summary(),
            ]
        )


def _compute(slice_: EvalSlice, binning: Binning) -> MetricSet:
    probabilities, outcomes = slice_.probabilities, slice_.outcomes
    return MetricSet(
        n=slice_.n,
        brier=brier(probabilities, outcomes, binning=binning),
        ece=ece(probabilities, outcomes, n_bins=binning.n_bins),
        log_score=log_score(probabilities, outcomes),
        histogram=output_histogram(probabilities, binning=binning),
    )


@dataclass(frozen=True)
class IterationMetrics:
    """Metrics for gating and iteration. Runs on any resolved split.

    Carries `split` so a number pasted into a message says where it came from.
    Deliberately a different type from `CardMetrics`: anything that renders a
    model card asks for `CardMetrics`, and this will not satisfy it.
    """

    metrics: MetricSet
    split: Split
    label: str

    def summary(self) -> str:
        return f"[{self.split.value}:{self.label or 'unlabelled'}] {self.metrics.summary()}"


@dataclass(frozen=True)
class CardMetrics:
    """Metrics cleared for a model card, a reliability diagram, or any external
    figure. Only `model_card_metrics` produces one."""

    metrics: MetricSet
    by_horizon: Mapping[HorizonBucket, MetricSet]
    """Per-forecast-horizon breakdown, which `docs/07` requires to be published:
    calibration degrades with horizon and a single aggregate diagram hides it.

    Computing this is why a card-bound slice must carry `as_ofs` — the
    requirement is not a check bolted onto the guard, it is what the card is
    made of."""

    split: Split
    freeze: datetime
    label: str

    def summary(self) -> str:
        horizons = ", ".join(
            f"{bucket.value}: n={m.n}" for bucket, m in sorted(self.by_horizon.items())
        )
        return (
            f"[published since {self.freeze.date().isoformat()}:"
            f"{self.label or 'unlabelled'}] {self.metrics.summary()}\n"
            f"by forecast horizon — {horizons}"
        )


DEFAULT_CARD_BINNING = Binning()


def iteration_metrics(
    slice_: EvalSlice,
    *,
    temporal_split: TemporalSplit = DEFAULT_SPLIT,
    binning: Binning = DEFAULT_CARD_BINNING,
) -> IterationMetrics:
    """Score a slice for iteration or gating. Any resolved split is fine here —
    that is what `dev` is for."""
    return IterationMetrics(
        metrics=_compute(slice_, binning),
        split=slice_.split(temporal_split),
        label=slice_.label,
    )


def model_card_metrics(
    slice_: EvalSlice,
    *,
    temporal_split: TemporalSplit = DEFAULT_SPLIT,
    binning: Binning = DEFAULT_CARD_BINNING,
) -> CardMetrics:
    """Score a slice for external publication. Refuses anything but `published`.

    Requires `as_ofs`, because the card carries a per-horizon breakdown and that
    cannot be computed without them — nor reconstructed afterwards. Scoring stays
    agnostic (`iteration_metrics` does not care); producing something publishable
    does not.

    There is deliberately no override parameter. The whole point of the split is
    that this path cannot be taken on dev data, and a `force=True` would make the
    guard a formality — someone under deadline would find it.
    """
    # Split first, deliberately: "this is dev data" is the stronger refusal and
    # must never be masked by a missing-as_ofs error that a caller could fix and
    # then walk straight into publishing dev numbers.
    split = slice_.split(temporal_split)
    if not split.publishable:
        raise NotPublishableError(
            f"refusing to build a card-bound artifact from split {split.value!r}: "
            "only 'published' (resolved after the freeze "
            f"{temporal_split.freeze.isoformat()}) is contamination-free by "
            "construction. Use iteration_metrics() for gating numbers."
        )
    if slice_.as_ofs is None:
        raise MissingForecastTimesError(
            f"slice {slice_.label or '<unlabelled>'!r} has no as_ofs, so its "
            "forecast horizons are unknown and the per-horizon breakdown docs/07 "
            "requires cannot be built. Horizons cannot be recovered after the "
            "fact — re-run the eval recording as_of per forecast."
        )
    return CardMetrics(
        metrics=_compute(slice_, binning),
        by_horizon={
            bucket: _compute(sub, binning) for bucket, sub in slice_.by_forecast_horizon().items()
        },
        split=split,
        freeze=temporal_split.freeze,
        label=slice_.label,
    )
