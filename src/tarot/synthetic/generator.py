# SPDX-License-Identifier: Apache-2.0
"""Generate threshold-over-series questions with contemporaneous evidence.

`docs/13`. The Likelihood training source, because the real-market corpus cannot
supply question-evidence pairs at scale.

The two properties that make a synthetic question worth generating, and that this
module exists to guarantee rather than intend:

1. **The evidence is provably contemporaneous.** It is the series truncated at
   `as_of`, and the truncation is asserted, not assumed.
2. **The threshold was set without look-ahead.** `X` is chosen from the
   distribution of changes observable *before* `as_of`. Choosing it from the
   realized value would encode the answer in the question — the same
   look-ahead-selection failure rejected for price targets, where selecting on
   how much the price moved would have conditioned the corpus on the outcome.

Both are enforced here and both are mutation-checked. A generator that got either
wrong would produce a corpus that trains a model to read its own construction.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from itertools import pairwise

from tarot.types import Probability

__all__ = [
    "LookAheadError",
    "ObservationPoint",
    "SeriesQuestion",
    "ThresholdDirection",
    "generate_question",
    "generate_questions",
]


class ThresholdDirection(StrEnum):
    ABOVE = "above"
    BELOW = "below"


class LookAheadError(AssertionError):
    """A generated question was contaminated by information after `as_of`.

    An assertion rather than a value: there is no partially-usable question here,
    and a corpus is worth nothing if any member of it can be like this.
    """


@dataclass(frozen=True)
class ObservationPoint:
    """One dated value of a series."""

    timestamp: datetime
    value: float


@dataclass(frozen=True)
class SeriesQuestion:
    """A threshold question, its evidence, and its answer."""

    series_id: str
    as_of: datetime
    decision_date: datetime
    threshold: float
    direction: ThresholdDirection
    outcome: int
    evidence: tuple[ObservationPoint, ...]
    """The series truncated to strictly before `as_of`. This *is* the evidence
    bundle; there is no retrieval step and nothing to screen."""

    implied_probability: Probability
    """The generator's own estimate at `as_of`, from the pre-`as_of` distribution
    of changes. A low-variance, unbiased target in the `docs/05` sense — the
    closest thing available to that document's oracle arm, and unlike a market
    price it carries no bias to correct."""

    def text(self) -> str:
        return (
            f"Will {self.series_id} be {self.direction.value} {self.threshold:g} "
            f"on {self.decision_date.date().isoformat()}?"
        )

    def __post_init__(self) -> None:
        # The guarantees, asserted where the question is built.
        if self.decision_date <= self.as_of:
            raise LookAheadError(
                f"decision_date {self.decision_date.isoformat()} is not after "
                f"as_of {self.as_of.isoformat()}: nothing is being forecast"
            )
        late = [p for p in self.evidence if p.timestamp >= self.as_of]
        if late:
            raise LookAheadError(
                f"{len(late)} evidence points at or after as_of "
                f"{self.as_of.isoformat()}; first is {late[0].timestamp.isoformat()}"
            )


def _log_returns(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for a, b in pairwise(values):
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


@dataclass(frozen=True)
class _Context:
    """Everything about an `(as_of, horizon)` pair that does not depend on the
    target probability.

    Split out for speed, and the speed is not incidental: recomputing the return
    distribution per target made a full sweep take longer than it took to notice
    it was wrong. It also puts the leakage-relevant work in ONE place — `history`
    is derived here and nowhere else, so there is a single line to audit.
    """

    history: tuple[ObservationPoint, ...]
    horizon_returns: tuple[float, ...]
    last: float
    realized: float
    decision_date: datetime


def _context(
    series: Sequence[ObservationPoint],
    stamps: Sequence[datetime],
    as_of: datetime,
    horizon: timedelta,
    min_history: int,
    max_staleness: timedelta | None = None,
) -> _Context | None:
    # Strictly before as_of: the evidence, and the only thing the threshold may
    # be computed from.
    cut = bisect_right(stamps, as_of - timedelta(microseconds=1))
    history = series[:cut]
    if len(history) < min_history:
        return None

    # Steps per horizon, measured on the history's own spacing. The spacing also
    # sets the tolerance below: an observation is an acceptable stand-in for a
    # requested date only if it is within one step of it.
    span = (history[-1].timestamp - history[0].timestamp) / max(len(history) - 1, 1)
    if span.total_seconds() <= 0:
        return None
    steps = max(int(horizon / span), 1)

    # The anchor must be fresh. A stale last observation would put the threshold
    # on a price the question is no longer about.
    #
    # `max_staleness` separates *publication lag* from *observation spacing*.
    # They coincide for a traded price, which is what this rule was written for,
    # and diverge for anything released on a schedule: CPI describes a month and
    # appears about two months later, so `as_of - last_observation` is ~61 days
    # against a 30-day spacing and NO valid as_of can satisfy the default. That
    # is the normal state of knowledge for a macro forecaster, not staleness, and
    # measuring it against spacing rejected 100% of macro questions
    # (the internal decisions log, 2026-08-22).
    #
    # Defaults to `span`, so behaviour is unchanged unless a caller states the
    # lag its series actually has.
    if as_of - history[-1].timestamp > (max_staleness if max_staleness is not None else span):
        return None

    decision_date = as_of + horizon
    idx = bisect_right(stamps, decision_date) - 1
    if idx < 0 or stamps[idx] < as_of:
        return None
    # The series must actually REACH the decision date. Without this the last
    # available point is silently accepted instead, so a question asked at a
    # 100-day horizon against a series ending in 4 days would be labelled from
    # day 4 and recorded as a 100-day question. That is a horizon corruption
    # rather than a missing question, and it is invisible downstream: the record
    # looks complete and its stated horizon is a lie. Fail closed.
    if decision_date - stamps[idx] > span:
        return None

    returns = _log_returns([p.value for p in history])
    if len(returns) < min_history // 2:
        return None
    horizon_returns = sorted(
        sum(returns[i : i + steps]) for i in range(max(len(returns) - steps + 1, 1))
    )
    if not horizon_returns or history[-1].value <= 0:
        return None

    return _Context(
        history=tuple(history),
        horizon_returns=tuple(horizon_returns),
        last=history[-1].value,
        realized=series[idx].value,
        decision_date=stamps[idx],
    )


def _from_context(
    series_id: str,
    as_of: datetime,
    ctx: _Context,
    target_probability: float,
    direction: ThresholdDirection,
) -> SeriesQuestion:
    # Quantile of the pre-as_of change distribution giving the wanted probability.
    # P(above X) = target  =>  X sits at the (1 - target) quantile.
    q = 1.0 - target_probability if direction is ThresholdDirection.ABOVE else target_probability
    returns = ctx.horizon_returns
    pos = min(max(q * (len(returns) - 1), 0.0), len(returns) - 1.0)
    lo = int(pos)
    hi = min(lo + 1, len(returns) - 1)
    chosen = returns[lo] + (pos - lo) * (returns[hi] - returns[lo])
    threshold = ctx.last * math.exp(chosen)

    exceed = sum(1 for r in returns if ctx.last * math.exp(r) > threshold)
    implied = exceed / len(returns)
    if direction is ThresholdDirection.BELOW:
        implied = 1.0 - implied

    hit = (
        ctx.realized > threshold
        if direction is ThresholdDirection.ABOVE
        else ctx.realized < threshold
    )
    return SeriesQuestion(
        series_id=series_id,
        as_of=as_of,
        decision_date=ctx.decision_date,
        threshold=threshold,
        direction=direction,
        outcome=1 if hit else 0,
        evidence=ctx.history,
        implied_probability=min(max(implied, 0.0), 1.0),
    )


def generate_question(
    series_id: str,
    series: Sequence[ObservationPoint],
    *,
    as_of: datetime,
    horizon: timedelta,
    target_probability: float,
    direction: ThresholdDirection = ThresholdDirection.ABOVE,
    min_history: int = 30,
    max_staleness: timedelta | None = None,
) -> SeriesQuestion | None:
    """One question at `as_of`, or `None` if the series cannot support it.

    The threshold is placed so that, under the distribution of `h`-step changes
    **observed before `as_of`**, the implied probability is near
    `target_probability`. Difficulty is therefore approximate by construction —
    the exact version would need the realized value, which is precisely the
    contamination this avoids. `docs/13` accepts the approximation for that
    reason.
    """
    if not (0.0 < target_probability < 1.0):
        raise ValueError(f"target_probability must be in (0, 1), got {target_probability}")
    ordered = tuple(sorted(series, key=lambda p: p.timestamp))
    stamps = [p.timestamp for p in ordered]
    ctx = _context(ordered, stamps, as_of, horizon, min_history, max_staleness)
    if ctx is None:
        return None
    return _from_context(series_id, as_of, ctx, target_probability, direction)


def generate_questions(
    series_id: str,
    series: Sequence[ObservationPoint],
    *,
    as_ofs: Sequence[datetime],
    horizon: timedelta,
    target_probabilities: Sequence[float],
    direction: ThresholdDirection = ThresholdDirection.ABOVE,
    min_history: int = 30,
    max_staleness: timedelta | None = None,
) -> list[SeriesQuestion]:
    """Sweep `as_of` and target difficulty.

    `docs/01`: vary the threshold systematically so the label distribution is not
    degenerate and the model sees the full probability range. Sweeping the
    *target probability* rather than the raw threshold is what makes that
    controllable — a fixed grid of thresholds produces mostly-trivial questions,
    because most thresholds sit far from anything plausible.

    The context is built once per `as_of` and shared across targets. That is a
    pure speed change: targets differ only in which quantile of an identical
    distribution they take.
    """
    for target in target_probabilities:
        if not (0.0 < target < 1.0):
            raise ValueError(f"target_probability must be in (0, 1), got {target}")

    ordered = tuple(sorted(series, key=lambda p: p.timestamp))
    stamps = [p.timestamp for p in ordered]
    out: list[SeriesQuestion] = []
    for as_of in as_ofs:
        ctx = _context(ordered, stamps, as_of, horizon, min_history, max_staleness)
        if ctx is None:
            continue
        out.extend(
            _from_context(series_id, as_of, ctx, target, direction)
            for target in target_probabilities
        )
    return out
