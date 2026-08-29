# SPDX-License-Identifier: Apache-2.0
"""Supervision points from a price series, in the shape the internal decisions log decided.

For a forecast at `as_of`, the target is the price at
`τ = as_of + lam*(resolved_at - as_of)` — never `price(as_of)`, which is the
evaluation baseline `docs/07` requires beating. `τ` is a dial between a circular
low-variance target and a non-circular high-variance one; see the 2026-08-14
entry for why it does not touch bias, and why debiasing is a separate step.

Pure functions only: no IO, no polars. The scale work lives in
`scripts/price_targets.py`, which implements the same rule independently over the
whole corpus and cross-checks against this module on a sample — `docs/11` rule 1.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from cournot.types import PricePoint, Probability, SoftTargetProvenance

__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_MIN_SEPARATION",
    "DEFAULT_TERMINAL_EXCLUSION",
    "PriceRejection",
    "SupervisionPoint",
    "select_supervision_points",
    "validate_price_series",
]

#: Fraction of the remaining time from `as_of` to resolution at which the target
#: price is read. Swept, not assumed: λ→0 is the circular contemporaneous target
#: and λ→1 is the outcome-only case, so the sweep contains both endpoints of the
#: decision this default sits inside.
DEFAULT_LAMBDA = 0.5

#: Prices within this of resolution are dropped from both roles. On Manifold the
#: price snaps to 0/1 when the resolver acts, so a near-terminal price is the
#: outcome leaking backwards rather than a forecast.
DEFAULT_TERMINAL_EXCLUSION = timedelta(hours=1)

#: A target must sit at least this far after its `as_of`, whatever lambda says.
#:
#: Lambda is a *fraction* of the remaining horizon, so on a densely-observed
#: question a small lambda resolves to a target minutes away — which is the
#: contemporaneous price, i.e. the evaluation baseline, reached by a route the
#: lambda=0 exclusion does not cover. Measured and priced in the internal decisions log
#: (2026-08-14, minimum absolute separation).
DEFAULT_MIN_SEPARATION = timedelta(hours=6)


class PriceRejection(StrEnum):
    """Why a price point cannot be used. Counted, never silently dropped.

    The first four are `docs/11` rule-3 checks: each is an invariant against
    something outside the price row itself — the market's own dates, the
    snapshot's generation time, the unit interval.
    """

    OUT_OF_RANGE = "out_of_range"
    """`prob` outside [0, 1]. 2 rows in the Manifold corpus."""

    BEFORE_MARKET_OPENED = "before_market_opened"
    """Observed before the question existed."""

    AFTER_RESOLUTION = "after_resolution"
    """Observed after the question resolved. 141 rows in the Manifold corpus."""

    IN_TERMINAL_WINDOW = "in_terminal_window"
    """Inside the resolution-jump window; the outcome leaking backwards."""

    NO_LATER_OBSERVATION = "no_later_observation"
    """Usable as an `as_of`, but no price exists strictly after it to be the
    target. The last observation of every series lands here by construction."""

    AMBIGUOUS_AT_TIMESTAMP = "ambiguous_at_timestamp"
    """Two observations share a timestamp but disagree on the price, so there is
    no fact about what the price was at that instant.

    5,825 such timestamps exist in the Manifold binary subset. Exact repeats —
    same timestamp, same price — are 99.3% of duplicates and are deduplicated
    silently, because they carry no ambiguity. These are dropped instead: when in
    doubt, drop, and picking one arbitrarily would make the target depend on row
    order rather than on the data."""

    SEPARATION_TOO_SMALL = "separation_too_small"
    """A later observation exists within the lambda window, but not far enough
    after `as_of` to be anything other than the contemporaneous price."""


@dataclass(frozen=True)
class SupervisionPoint:
    """One `(question, as_of) -> target` training example."""

    question_id: str
    as_of: datetime
    target: Probability
    target_at: datetime
    provenance: SoftTargetProvenance = SoftTargetProvenance.MARKET_CONSENSUS

    def __post_init__(self) -> None:
        # The guarantee the whole decision rests on, asserted where the point is
        # constructed rather than trusted downstream.
        if self.target_at <= self.as_of:
            raise ValueError(
                f"target_at {self.target_at.isoformat()} is not strictly after "
                f"as_of {self.as_of.isoformat()}: that is the contemporaneous "
                "price, which is the evaluation baseline"
            )


def validate_price_series(
    prices: Sequence[PricePoint],
    *,
    open_date: datetime,
    resolved_at: datetime,
    terminal_exclusion: timedelta = DEFAULT_TERMINAL_EXCLUSION,
) -> tuple[list[PricePoint], dict[PriceRejection, int]]:
    """Drop unusable observations, counting each by reason (`docs/01` posture)."""
    kept: list[PricePoint] = []
    counts: dict[PriceRejection, int] = {}

    def reject(reason: PriceRejection) -> None:
        counts[reason] = counts.get(reason, 0) + 1

    # Resolve timestamp collisions before anything else. A timestamp carrying
    # two different prices has no answer to "what was the price then", and
    # leaving it in would make target selection depend on input order.
    by_stamp: dict[datetime, set[float]] = {}
    for point in prices:
        by_stamp.setdefault(point.timestamp, set()).add(point.price)
    ambiguous = {t for t, values in by_stamp.items() if len(values) > 1}

    seen: set[datetime] = set()
    cutoff = resolved_at - terminal_exclusion
    for point in prices:
        if point.timestamp in ambiguous:
            reject(PriceRejection.AMBIGUOUS_AT_TIMESTAMP)
            continue
        if point.timestamp in seen:
            continue  # exact repeat: same instant, same price, no ambiguity
        seen.add(point.timestamp)
        if not (0.0 <= point.price <= 1.0):
            reject(PriceRejection.OUT_OF_RANGE)
        elif point.timestamp < open_date:
            reject(PriceRejection.BEFORE_MARKET_OPENED)
        elif point.timestamp > resolved_at:
            reject(PriceRejection.AFTER_RESOLUTION)
        elif point.timestamp >= cutoff:
            reject(PriceRejection.IN_TERMINAL_WINDOW)
        else:
            kept.append(point)
    return kept, counts


def select_supervision_points(
    question_id: str,
    prices: Sequence[PricePoint],
    *,
    resolved_at: datetime,
    open_date: datetime,
    lam: float = DEFAULT_LAMBDA,
    terminal_exclusion: timedelta = DEFAULT_TERMINAL_EXCLUSION,
    min_separation: timedelta = DEFAULT_MIN_SEPARATION,
) -> tuple[list[SupervisionPoint], dict[PriceRejection, int]]:
    """One supervision point per usable observation, target read λ of the way out.

    The target is the latest observation at or before
    `as_of + lam*(resolved_at - as_of)` that is strictly after `as_of`. Reading the
    latest at-or-before, rather than interpolating, keeps every target a price
    that actually traded.

    The target must also sit at least `min_separation` after `as_of`. Lambda
    bounds it *relatively*; the minimum bounds it *absolutely*. Points that
    cannot satisfy both are dropped rather than having their target pushed past
    the lambda window — pushing it would silently redefine what lambda means and
    make the sweep arms incomparable.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam}: 0 is the baseline, 1 is the outcome")

    usable, counts = validate_price_series(
        prices,
        open_date=open_date,
        resolved_at=resolved_at,
        terminal_exclusion=terminal_exclusion,
    )
    usable.sort(key=lambda p: p.timestamp)
    stamps = [p.timestamp for p in usable]

    points: list[SupervisionPoint] = []
    for point in usable:
        horizon = resolved_at - point.timestamp
        wanted = point.timestamp + horizon * lam
        # Latest observation at or before `wanted`...
        j = bisect_right(stamps, wanted) - 1
        # ...but it has to be strictly later than as_of, or it is the
        # contemporaneous price under another name.
        #
        # Compared on TIMESTAMPS, not on indices. Real price series contain
        # observations sharing a timestamp to the millisecond, so `j > i` can
        # still select a point at the same instant as `as_of` — an index is not
        # a time. The __post_init__ guard on SupervisionPoint caught this.
        if j < 0 or stamps[j] <= point.timestamp:
            counts[PriceRejection.NO_LATER_OBSERVATION] = (
                counts.get(PriceRejection.NO_LATER_OBSERVATION, 0) + 1
            )
            continue
        if stamps[j] - point.timestamp < min_separation:
            counts[PriceRejection.SEPARATION_TOO_SMALL] = (
                counts.get(PriceRejection.SEPARATION_TOO_SMALL, 0) + 1
            )
            continue
        target = usable[j]
        points.append(
            SupervisionPoint(
                question_id=question_id,
                as_of=point.timestamp,
                target=target.price,
                target_at=target.timestamp,
            )
        )
    return points, counts


#: How far outside [0, 1] a stored price may sit before it is treated as corrupt
#: rather than as float noise. Manifold's `price_history` is reconstructed from
#: bet `probBefore`/`probAfter`, and 2 of 23,152,840 points exceed 1.0 — by
#: 1.4e-14. Clamping silently would also swallow a genuinely bad 1.5, so the
#: tolerance is narrow and anything beyond it raises.
PRICE_EPSILON = 1e-9


class ImplausiblePriceError(ValueError):
    """A stored price is outside [0, 1] by more than float noise."""


def clamp_price(value: float) -> float:
    """Snap a stored price into [0, 1], or refuse it.

    `PricePoint` validates its range, so a 1.000000000000014 read straight from
    the corpus raises during ingest — 2 rows in 23M, enough to kill a full
    corpus build after several minutes of work.
    """
    if value < -PRICE_EPSILON or value > 1.0 + PRICE_EPSILON:
        raise ImplausiblePriceError(
            f"price {value!r} is outside [0, 1] by more than {PRICE_EPSILON:g}; "
            "that is corrupt data rather than floating-point noise"
        )
    return min(max(value, 0.0), 1.0)
