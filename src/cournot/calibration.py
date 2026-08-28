"""Isotonic calibration maps, and the cross-fitted debiaser for price targets.

`docs/06` describes this machinery pointed at the model. This module points it at
the *teacher*: a market price is a low-variance target and a biased one
(the internal decisions log, 2026-08-14), so before a price becomes a training target it is passed
through a map fitted on realized outcomes. Realized outcomes are unbiased and
useless per-question; at corpus scale they are exactly what is needed to estimate
a systematic distortion.

Two properties the naive version does not have:

- **Cross-fitting.** A map fitted on the same questions whose targets it corrects
  has seen their outcomes, so its apparent correction is optimistic. Folds are
  assigned **by question**, never by supervision point: points from one question
  share an outcome, so splitting by point would put the same outcome on both
  sides of the fold and reintroduce exactly what cross-fitting removes.
- **Stratification as a parameter.** Market bias is category-dependent, and
  categories do not exist yet (the internal decisions log, 2026-08-14). `fit_cross_fitted` takes a
  stratum key per row; passing `None` gives the global map, and passing a category
  later is a call-site change rather than a rewrite.

Isotonic rather than temperature scaling: `docs/06` prefers temperature by
default because it is hard to overfit on a small split, and reaches for isotonic
"only with sufficient volume". Here the volume is 8.48M points and the distortion
being corrected is not sigmoid-shaped — favorite-longshot bias is a monotone
distortion of unknown form, which is what isotonic is for.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

__all__ = [
    "DEFAULT_FOLDS",
    "BetaMap",
    "CalibrationMap",
    "CrossFittedDebiaser",
    "IsotonicMap",
    "PriceBin",
    "fit_beta",
    "fit_cross_fitted",
    "fit_isotonic",
    "fold_for_question",
]

DEFAULT_FOLDS = 5

#: Below this many observations a stratum falls back to the global map rather
#: than fitting its own. A thin stratum produces a map that is mostly noise, and
#: an over-fitted correction is worse than an under-fitted one — it moves targets
#: confidently in the wrong direction.
MIN_STRATUM_OBSERVATIONS = 10_000


@dataclass(frozen=True)
class PriceBin:
    """Aggregated `(price, outcome)` evidence: one bin of the fitting data.

    Fitting on aggregates rather than raw pairs is exact, not an approximation:
    weighted isotonic regression over bin means with bin counts as weights gives
    the same solution as running it over the underlying rows, provided the
    binning does not split a tie. It also makes 8.48M rows a 1,000-row fit.
    """

    price: float
    """Mean raw price in the bin."""

    outcomes: float
    """Sum of realized outcomes in the bin."""

    n: int

    @property
    def frequency(self) -> float:
        return self.outcomes / self.n if self.n else 0.0


@dataclass(frozen=True)
class IsotonicMap:
    """A monotone non-decreasing step function from raw price to corrected value.

    Stored as block upper edges and block values, which is what PAVA produces.
    Monotone by construction: a debiasing map that could reorder two prices would
    be asserting the market ranks questions wrongly, which is a much stronger
    claim than "it is miscalibrated" and not one this data supports.
    """

    edges: tuple[float, ...]
    """Upper price edge of each block except the last."""

    values: tuple[float, ...]
    """Corrected value per block. `len(values) == len(edges) + 1`."""

    n_fit: int
    """Observations behind the fit."""

    def __post_init__(self) -> None:
        if len(self.values) != len(self.edges) + 1:
            raise ValueError(
                f"{len(self.edges)} edges need {len(self.edges) + 1} values, got {len(self.values)}"
            )
        if any(b < a for a, b in pairwise(self.values)):
            raise ValueError("isotonic map values must be non-decreasing")

    def __call__(self, price: float) -> float:
        if not (0.0 <= price <= 1.0):
            raise ValueError(f"price must be in [0, 1], got {price!r}")
        return self.values[bisect_left(self.edges, price)]

    @property
    def n_blocks(self) -> int:
        return len(self.values)


def fit_isotonic(bins: Sequence[PriceBin]) -> IsotonicMap:
    """Weighted pool-adjacent-violators. Exact, O(n) after the sort.

    Each bin starts as its own block; whenever a block's value falls below its
    predecessor's, the two merge into their count-weighted mean. The result is
    the unique monotone least-squares fit to the data.
    """
    # Bins sharing a price are one point, not two: isotonic regression cannot
    # assign two values to the same x, and leaving them separate would let the
    # map return whichever happened to sort first. This is how folds arrive —
    # the same price bin, once per fold.
    merged: dict[float, list[float]] = {}
    for b in bins:
        if b.n <= 0:
            continue
        entry = merged.setdefault(b.price, [0.0, 0.0])
        entry[0] += b.outcomes
        entry[1] += b.n
    if not merged:
        raise ValueError("no observations to fit")
    populated = [
        PriceBin(price=price, outcomes=outcomes, n=int(count))
        for price, (outcomes, count) in sorted(merged.items())
    ]

    # (right edge, summed outcomes, summed count)
    blocks: list[list[float]] = []
    for b in populated:
        blocks.append([b.price, b.outcomes, float(b.n)])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks[-1]
            left[0] = right[0]
            left[1] += right[1]
            left[2] += right[2]

    return IsotonicMap(
        edges=tuple(block[0] for block in blocks[:-1]),
        values=tuple(block[1] / block[2] for block in blocks),
        n_fit=int(sum(block[2] for block in blocks)),
    )


def fold_for_question(question_id: str, k: int = DEFAULT_FOLDS) -> int:
    """Deterministic fold for a question.

    blake2b rather than `hash()`: Python's string hash is salted per process, so
    a fold assignment built on it would differ between runs and between the
    module and anything reproducing it. A corpus build has to be reproducible.
    """
    if k < 2:
        raise ValueError(f"cross-fitting needs at least 2 folds, got {k}")
    digest = hashlib.blake2b(question_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % k


class CalibrationMap(Protocol):
    """Anything that maps a raw probability to a corrected one."""

    def __call__(self, price: float) -> float: ...


#: Probabilities are clipped this far from the ends before taking logs. A raw
#: 0.0 or 1.0 is a finite claim from a model that emits two decimal places, not
#: an infinite log-odds, and treating it as one would let one forecast dominate
#: the fit.
BETA_CLIP = 1e-4

#: Ridge added to the Hessian diagonal. Without it a separable fold — every low
#: forecast resolving NO — sends the coefficients to infinity and the map
#: becomes a step function, which is the behaviour beta calibration is chosen to
#: avoid.
BETA_RIDGE = 1e-6


@dataclass(frozen=True)
class BetaMap:
    """`sigmoid(a*ln(p) - b*ln(1-p) + c)` — smooth, monotone, three parameters.

    The alternative to `IsotonicMap`, and the reason to have one. Isotonic wins
    marginally on Brier and pools adjacent inputs to a common output: measured on
    the 2026-08-20 Qwen3-8B run it collapsed 22 distinct forecasts to 13 and
    pushed 29.9% of the mass within 0.05 of a half. A three-parameter smooth map
    cannot do that — distinct inputs stay distinct — and it costs 0.002 Brier.

    `CLAUDE.md` #5 makes histogram collapse the characteristic failure to watch
    for, and it is not visible in Brier, so the 0.002 is worth paying.

    `a` and `b` must be non-negative or the map is not monotone. `fit_beta`
    enforces that, but the check lives here too: a `BetaMap` built by any other
    route is equally capable of reversing the model's ordering, and a
    calibration step is not entitled to make that claim.

    **It cannot lift the floor.** If a model's low forecasts genuinely resolve YES
    30% of the time, every honest map sends them to ~0.30. That is the model's
    low end carrying no information, and no post-hoc correction reaches it.
    """

    a: float
    b: float
    c: float
    n_fit: int

    def __post_init__(self) -> None:
        if self.a < 0.0 or self.b < 0.0:
            raise ValueError(
                f"BetaMap requires a >= 0 and b >= 0 for monotonicity, got "
                f"a={self.a}, b={self.b}. A negative slope reverses the model's "
                "ordering — see Kull et al. (AISTATS 2017)."
            )

    def __call__(self, price: float) -> float:
        if not (0.0 <= price <= 1.0):
            raise ValueError(f"price must be in [0, 1], got {price!r}")
        p = min(max(price, BETA_CLIP), 1.0 - BETA_CLIP)
        z = self.a * math.log(p) - self.b * math.log(1.0 - p) + self.c
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def _solve_sym(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting, for any square system.

    `_solve3` is fixed at 3x3; the monotonicity refit drops features, so the
    system can be 1x1 or 2x2. A singular pivot raises `ZeroDivisionError`, which
    the IRLS loop already treats as "stop iterating" rather than as an error.
    """
    n = len(rhs)
    aug = [[*row, r] for row, r in zip(matrix, rhs, strict=True)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-300:
            raise ZeroDivisionError("singular system")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for row in reversed(range(n)):
        total = aug[row][n] - sum(aug[row][c] * out[c] for c in range(row + 1, n))
        out[row] = total / aug[row][row]
    return out


def _solve3(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting on a 3x3 system."""
    m = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ZeroDivisionError("singular system in beta calibration")
        m[col], m[pivot] = m[pivot], m[col]
        for row in range(3):
            if row == col:
                continue
            factor = m[row][col] / m[col][col]
            for k in range(col, 4):
                m[row][k] -= factor * m[col][k]
    return [m[i][3] / m[i][i] for i in range(3)]


def fit_beta(bins: Sequence[PriceBin], *, iterations: int = 50) -> BetaMap:
    """Weighted logistic regression on `(ln p, -ln(1-p))`, by IRLS.

    Fitting on aggregates is exact here for the same reason it is for isotonic:
    every observation in a bin shares one price, so the bin's likelihood is
    exactly binomial and weighting by count reproduces the per-row fit.

    IRLS rather than gradient descent — it converges in around ten iterations
    with no step size to choose, so there is no tuning constant whose value
    silently determines the result.

    **Monotonicity is enforced, not assumed.** Kull et al. (AISTATS 2017) require
    `a >= 0` and `b >= 0`; the unconstrained fit does not respect that, and on
    2026-08-22 a three-bin case returned `a = -27.5`, mapping 0.1 -> 1.00 and
    0.5 -> 0.00 — a calibration map that reverses the model's ordering, which is
    a claim no calibration step is entitled to make. Following the reference
    `betacal` implementation, a negative coefficient means its feature is dropped
    and the model refitted without it.
    """
    rows = [b for b in bins if b.n > 0]
    if not rows:
        raise ValueError("no observations to fit a beta calibration map on")
    total = sum(b.n for b in rows)

    design: list[tuple[float, float, float]] = []
    for b in rows:
        p = min(max(b.price, BETA_CLIP), 1.0 - BETA_CLIP)
        design.append((math.log(p), -math.log(1.0 - p), 1.0))

    theta = [0.0, 0.0, 0.0]
    for _ in range(iterations):
        hessian = [[BETA_RIDGE if i == j else 0.0 for j in range(3)] for i in range(3)]
        gradient = [0.0, 0.0, 0.0]
        for x, b in zip(design, rows, strict=True):
            z = sum(t * xi for t, xi in zip(theta, x, strict=True))
            q = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            weight = b.n * q * (1.0 - q)
            residual = b.outcomes - b.n * q
            for i in range(3):
                gradient[i] += residual * x[i]
                for j in range(3):
                    hessian[i][j] += weight * x[i] * x[j]
        try:
            step = _solve3(hessian, gradient)
        except ZeroDivisionError:
            break
        theta = [t + s for t, s in zip(theta, step, strict=True)]
        if max(abs(s) for s in step) < 1e-10:
            break
    return _enforce_monotone(theta, design, rows, total, iterations)


def _fit_theta(
    design: Sequence[tuple[float, float, float]],
    rows: Sequence[PriceBin],
    active: Sequence[int],
    iterations: int,
) -> list[float]:
    """IRLS over a subset of the three features; inactive ones stay at zero."""
    theta = [0.0, 0.0, 0.0]
    k = len(active)
    for _ in range(iterations):
        hessian = [[BETA_RIDGE if i == j else 0.0 for j in range(k)] for i in range(k)]
        gradient = [0.0] * k
        for x, b in zip(design, rows, strict=True):
            z = sum(theta[c] * x[c] for c in active)
            q = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            weight = b.n * q * (1.0 - q)
            residual = b.outcomes - b.n * q
            for i, ci in enumerate(active):
                gradient[i] += residual * x[ci]
                for j, cj in enumerate(active):
                    hessian[i][j] += weight * x[ci] * x[cj]
        try:
            step = _solve_sym(hessian, gradient)
        except ZeroDivisionError:
            break
        for i, ci in enumerate(active):
            theta[ci] += step[i]
        if max(abs(s) for s in step) < 1e-10:
            break
    return theta


def _enforce_monotone(
    theta: Sequence[float],
    design: Sequence[tuple[float, float, float]],
    rows: Sequence[PriceBin],
    total: int,
    iterations: int,
) -> BetaMap:
    """Drop a negatively-fitted slope and refit, as `betacal` does.

    `a < 0` or `b < 0` makes the map non-monotone. Clamping to zero in place
    would leave the intercept fitted against a coefficient that is no longer
    there, so the model is refitted without the offending feature instead.
    """
    a, b, c = theta[0], theta[1], theta[2]
    if a >= 0.0 and b >= 0.0:
        return BetaMap(a=a, b=b, c=c, n_fit=total)

    active = [i for i in (0, 1) if theta[i] >= 0.0] + [2]
    refit = _fit_theta(design, rows, active, iterations)
    # A second negative can appear once the first is gone; drop it too.
    if refit[0] < 0.0 or refit[1] < 0.0:
        refit = _fit_theta(design, rows, [2], iterations)
    return BetaMap(a=max(refit[0], 0.0), b=max(refit[1], 0.0), c=refit[2], n_fit=total)


@dataclass(frozen=True)
class CrossFittedDebiaser:
    """Per-fold maps, each fitted on the *other* folds.

    Applying `maps[f]` to a question in fold `f` means the correction was
    estimated without that question's outcome. Out-of-fold by construction, so
    the measured correction is what it would achieve on unseen questions rather
    than on the ones it was fitted to.
    """

    maps: Mapping[int, CalibrationMap]
    k: int
    global_map: CalibrationMap
    """Fitted on everything. For reporting and as the fallback for strata too
    thin to support their own."""

    strata: Mapping[tuple[Hashable, int], CalibrationMap] | None = None
    """`(stratum, fold) -> map`, when fitted with stratification."""

    def debias(self, price: float, question_id: str, stratum: Hashable = None) -> float:
        """Correct one price, using a map that never saw this question's outcome."""
        fold = fold_for_question(question_id, self.k)
        if self.strata is not None:
            specific = self.strata.get((stratum, fold))
            if specific is not None:
                return specific(price)
        return self.maps[fold](price)

    @property
    def stratified(self) -> bool:
        return self.strata is not None


@dataclass(frozen=True)
class FoldedBins:
    """Binned fitting evidence, already partitioned by fold and stratum."""

    fold: int
    bins: Sequence[PriceBin]
    stratum: Hashable = None


def fit_cross_fitted(
    folded: Sequence[FoldedBins],
    *,
    k: int = DEFAULT_FOLDS,
    min_stratum_observations: int = MIN_STRATUM_OBSERVATIONS,
    fitter: Callable[[Sequence[PriceBin]], CalibrationMap] = fit_isotonic,
) -> CrossFittedDebiaser:
    """Fit `k` maps, each excluding one fold, plus optional per-stratum maps.

    `folded` carries counts already aggregated per (fold, stratum, price bin) —
    the caller does that aggregation, because at corpus scale it belongs in the
    query engine, and because keeping this function pure keeps it testable
    without a corpus.
    """
    if k < 2:
        raise ValueError(f"cross-fitting needs at least 2 folds, got {k}")

    by_fold: dict[int, list[PriceBin]] = {f: [] for f in range(k)}
    by_stratum_fold: dict[tuple[Hashable, int], list[PriceBin]] = {}
    strata_seen: set[Hashable] = set()

    for entry in folded:
        if entry.fold not in by_fold:
            raise ValueError(f"fold {entry.fold} is outside range(0, {k})")
        by_fold[entry.fold].extend(entry.bins)
        if entry.stratum is not None:
            strata_seen.add(entry.stratum)
            by_stratum_fold.setdefault((entry.stratum, entry.fold), []).extend(entry.bins)

    everything = [b for bins in by_fold.values() for b in bins]
    global_map = fitter(everything)

    maps = {
        held_out: fitter([b for f, bins in by_fold.items() if f != held_out for b in bins])
        for held_out in range(k)
    }

    strata: dict[tuple[Hashable, int], IsotonicMap] | None = None
    if strata_seen:
        strata = {}
        for stratum in strata_seen:
            for held_out in range(k):
                training = [
                    b
                    for (s, f), bins in by_stratum_fold.items()
                    if s == stratum and f != held_out
                    for b in bins
                ]
                if sum(b.n for b in training) < min_stratum_observations:
                    continue  # falls back to the global per-fold map
                strata[(stratum, held_out)] = fit_isotonic(training)

    return CrossFittedDebiaser(maps=maps, k=k, global_map=global_map, strata=strata)
