"""Confidence intervals for eval metrics, by paired question-clustered bootstrap.

`docs/07` mandates a decomposed Brier, ECE under two binning schemes, a clipped
log score and an output histogram — and until 2026-08-22 attached an interval to
none of them. The published card therefore carried a point estimate at n=127,
which is the one artifact that leaves the building.

An external review found that roughly half of published LLM-forecasting work
reports intervals, and that three independent systems converge on the same
method: a **paired, question-clustered bootstrap**. This is that.

## Why clustered, and why it is not optional here

The published run scores 127 questions at five `as_of` placements — 629 forecasts,
but only 127 independent things. The five arms of one question share its
difficulty, its resolution and its wording, so resampling forecasts independently
would treat five correlated observations as five independent ones and shrink the
interval by roughly `sqrt(5)`.

That is the same pseudo-replication the phi arms were already flagged for
(the internal decisions log, market parity: "the arms are not independent — same question per
arm"). Fixing it in the design and then losing it in the interval would be worse
than not computing one.

## Why paired

Model and baseline are scored on the *same* questions, so the quantity of
interest is the mean of per-question differences, not the difference of two
independently resampled means. Pairing removes question difficulty — much larger
than the effect — from the interval.

## Method

Percentile bootstrap over cluster indices. Chosen over the normal approximation
because Brier differences are skewed at this sample size, and over BCa because
BCa's acceleration term needs a jackknife over clusters that costs more than the
interval is worth here. The choice is recorded on the result rather than left
implicit, so a later reader knows which was used.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

#: Enough that the interval's endpoints are stable to about the third decimal.
DEFAULT_RESAMPLES = 10_000

#: Fixed so an interval is reproducible from the same inputs. A bootstrap that
#: moved between runs would make two reports of the same result disagree.
DEFAULT_SEED = 20260822


class IntervalMethod(StrEnum):
    PERCENTILE = "percentile"


@dataclass(frozen=True)
class Interval:
    """A point estimate and its interval, carrying how it was produced."""

    point: float
    lower: float
    upper: float
    level: float
    method: IntervalMethod
    n_resamples: int
    n_clusters: int
    n_observations: int

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0

    def summary(self) -> str:
        return (
            f"{self.point:+.4f} [{self.lower:+.4f}, {self.upper:+.4f}] "
            f"({self.level:.0%} {self.method.value}, {self.n_clusters:,} clusters "
            f"over {self.n_observations:,} observations)"
        )


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not ordered:
        raise ValueError("no resamples")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def clustered_bootstrap(
    values: Sequence[float],
    clusters: Sequence[object],
    *,
    level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Interval for the mean of `values`, resampling whole clusters.

    `clusters[i]` names the independent unit observation `i` belongs to — for a
    multi-`as_of` eval that is the question id. Observations sharing a cluster
    are always drawn together, which is what stops five arms of one question
    counting as five independent data points.
    """
    if len(values) != len(clusters):
        raise ValueError(
            f"{len(values)} values but {len(clusters)} cluster labels; every "
            "observation needs to say which independent unit it belongs to"
        )
    if not values:
        raise ValueError("no observations to bootstrap")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    grouped: dict[object, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        grouped.setdefault(cluster, []).append(value)
    keys = list(grouped)

    point = sum(values) / len(values)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        drawn: list[float] = []
        for _ in range(len(keys)):
            drawn.extend(grouped[keys[rng.randrange(len(keys))]])
        means.append(sum(drawn) / len(drawn))
    means.sort()

    tail = (1.0 - level) / 2.0
    return Interval(
        point=point,
        lower=_percentile(means, tail),
        upper=_percentile(means, 1.0 - tail),
        level=level,
        method=IntervalMethod.PERCENTILE,
        n_resamples=n_resamples,
        n_clusters=len(keys),
        n_observations=len(values),
    )


def paired_bootstrap(
    arm: Sequence[float],
    baseline: Sequence[float],
    clusters: Sequence[object],
    *,
    level: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Interval for the mean per-observation difference `arm - baseline`.

    Both sequences must be scored on the same observations in the same order.
    The difference is taken first and the *differences* are bootstrapped, which
    is what makes the interval paired: resampling the two arms separately would
    put question difficulty back in.
    """
    if not (len(arm) == len(baseline) == len(clusters)):
        raise ValueError(
            f"lengths must match: arm={len(arm)}, baseline={len(baseline)}, "
            f"clusters={len(clusters)}. Unequal lengths mean the arms were not "
            "scored on the same observations, so the difference is not paired."
        )
    diffs = [a - b for a, b in zip(arm, baseline, strict=True)]
    return clustered_bootstrap(diffs, clusters, level=level, n_resamples=n_resamples, seed=seed)
