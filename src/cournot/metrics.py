# SPDX-License-Identifier: Apache-2.0
"""Scoring metrics for Cournot.

`docs/07`. The shapes here encode two non-negotiables from `CLAUDE.md`:

- **#4 — the Brier decomposition is mandatory, not optional.** `brier()` returns
  a `BrierDecomposition`; there is no function that returns a bare float. A model
  can lower aggregate Brier purely by hedging toward the base rate, and the only
  way to see that is calibration vs. resolution.
- **#5 — watch the output histogram.** `output_histogram()` is a first-class
  metric, not a debugging aid. Collapse onto 0.5 or onto a handful of round
  values looks fine on every average in this module.

Pure Python, no numeric dependency: these run on eval-set-sized inputs, and the
arithmetic being readable line by line is worth more here than speed.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Bin",
    "Binning",
    "BinningScheme",
    "BrierDecomposition",
    "CoarseBinningWarning",
    "ECEBinned",
    "ECEResult",
    "HistogramStats",
    "LogScoreResult",
    "SoftTargetScore",
    "brier",
    "ece",
    "ece_binned",
    "log_score",
    "output_histogram",
    "soft_target_score",
]


class CoarseBinningWarning(UserWarning):
    """The Brier decomposition's residual is a material share of the score.

    The identity `score = cal - res + unc + residual` closes regardless, so this
    condition is invisible unless something says it out loud.
    """


#: Above this share of the score, the residual means the binning is too coarse
#: to read calibration and resolution off. 5% is a starting line, not a result.
MATERIAL_RESIDUAL_FRACTION = 0.05

#: Bin count used by both ECE schemes and by the default Brier binning.
DEFAULT_N_BINS = 10


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


class BinningScheme(StrEnum):
    #: n bins of equal width on [0, 1]. The conventional reliability-diagram
    #: scheme; bins in the tails are often near-empty.
    EQUAL_WIDTH = "equal_width"
    #: n bins of (approximately) equal count. Edges are data-derived, so they
    #: must be reported alongside any number computed with them.
    EQUAL_MASS = "equal_mass"


@dataclass(frozen=True)
class Binning:
    """The binning scheme, carried on every result that depends on one.

    `docs/07` requires ECE to be reported "with the binning scheme stated". The
    cheapest way to guarantee that is to make it impossible to hold the number
    without the scheme.
    """

    scheme: BinningScheme = BinningScheme.EQUAL_WIDTH
    n_bins: int = DEFAULT_N_BINS

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

    def describe(self) -> str:
        return f"{self.n_bins} bins, {self.scheme.value}"


@dataclass(frozen=True)
class Bin:
    """One bin of a reliability diagram.

    Empty bins are retained with `n == 0` and `None` statistics: a reliability
    diagram needs to show where there was no data rather than silently skip it.
    """

    lower: float
    upper: float
    n: int
    mean_forecast: float | None
    observed_frequency: float | None

    @property
    def gap(self) -> float | None:
        """Signed calibration gap, forecast minus observed. Positive means the
        model was overconfident in this bin."""
        if self.mean_forecast is None or self.observed_frequency is None:
            return None
        return self.mean_forecast - self.observed_frequency


DEFAULT_BINNING = Binning()


def _validate(probabilities: Sequence[float], outcomes: Sequence[int]) -> None:
    if len(probabilities) != len(outcomes):
        raise ValueError(
            f"length mismatch: {len(probabilities)} probabilities, {len(outcomes)} outcomes"
        )
    if not probabilities:
        raise ValueError("no forecasts to score")
    for i, p in enumerate(probabilities):
        if not (0.0 <= p <= 1.0) or math.isnan(p):
            raise ValueError(f"probability at index {i} is not in [0, 1]: {p!r}")
    for i, o in enumerate(outcomes):
        if o not in (0, 1):
            raise ValueError(f"outcome at index {i} is not 0 or 1: {o!r}")


def _equal_width_index(value: float, n_bins: int) -> int:
    """Bin index for `value` under [k/n, (k+1)/n) bins, last bin closed.

    The naive `int(value * n_bins)` misbins values whose product falls just below
    an integer: `0.29 * 100 == 28.999999999999996`, which puts 0.29 in
    [0.28, 0.29). At 10 bins on 2dp outputs nothing is affected — the cases that
    bite are fine binnings, which is what the reliability diagrams and the
    residual gate push you toward. Compare against the edges themselves rather
    than reasoning about when the arithmetic happens to be exact.
    """
    k = min(int(value * n_bins), n_bins - 1)
    while k + 1 < n_bins and value >= (k + 1) / n_bins:
        k += 1
    while k > 0 and value < k / n_bins:
        k -= 1
    return k


def _equal_width_groups(values: Sequence[float], n_bins: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(n_bins)]
    for i, v in enumerate(values):
        groups[_equal_width_index(v, n_bins)].append(i)
    return groups


def _equal_mass_groups(values: Sequence[float], n_bins: int) -> list[list[int]]:
    """Split into ~equal-count bins, never splitting a run of identical values.

    Identical forecasts must land in the same bin; otherwise the same predicted
    value gets two different observed frequencies and calibration is undefined
    for it. Ties therefore win over the size quota, and bins come out uneven when
    the forecast distribution is lumpy — which, per non-negotiable #5, is exactly
    the distribution we expect to be looking at.

    Bin sizes are re-planned against what is left rather than against a fixed
    quota per bin. A large block of identical forecasts — which is the expected
    input, not a corner case — overshoots its quota, and a fixed schedule then
    reads every later bin as already over budget and emits singletons for the
    rest of the data.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    groups: list[list[int]] = []
    current: list[int] = []
    placed = 0

    for pos, idx in enumerate(order):
        if current and len(groups) < n_bins - 1:
            target = math.ceil((n - placed) / (n_bins - len(groups)))
            if len(current) >= target and values[idx] != values[order[pos - 1]]:
                groups.append(current)
                placed += len(current)
                current = []
        current.append(idx)

    if current:
        groups.append(current)
    return groups


def _build_bins(
    probabilities: Sequence[float], outcomes: Sequence[int], binning: Binning
) -> list[Bin]:
    if binning.scheme is BinningScheme.EQUAL_WIDTH:
        groups = _equal_width_groups(probabilities, binning.n_bins)
        edges = [(k / binning.n_bins, (k + 1) / binning.n_bins) for k in range(binning.n_bins)]
    else:
        groups = _equal_mass_groups(probabilities, binning.n_bins)
        edges = [
            (min(probabilities[i] for i in g), max(probabilities[i] for i in g))
            if g
            else (0.0, 0.0)
            for g in groups
        ]

    bins: list[Bin] = []
    for (lower, upper), group in zip(edges, groups, strict=True):
        if not group:
            bins.append(
                Bin(lower=lower, upper=upper, n=0, mean_forecast=None, observed_frequency=None)
            )
            continue
        bins.append(
            Bin(
                lower=lower,
                upper=upper,
                n=len(group),
                mean_forecast=sum(probabilities[i] for i in group) / len(group),
                observed_frequency=sum(outcomes[i] for i in group) / len(group),
            )
        )
    return bins


# --------------------------------------------------------------------------
# Brier
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BrierDecomposition:
    """Brier score with the Murphy calibration / resolution / uncertainty split.

    Identity, exactly as computed here:

        score = calibration - resolution + uncertainty + residual

    - `calibration` (a.k.a. reliability): how far each bin's observed frequency
      sits from its mean forecast. Lower is better; 0 is perfectly calibrated.
    - `resolution`: how far bins' observed frequencies sit from the base rate.
      **Higher is better** — it is the model saying something. A model that hedges
      to the base rate scores calibration ~0 and resolution ~0, which is the
      failure mode non-negotiable #4 exists to expose.
    - `uncertainty`: base_rate * (1 - base_rate). A property of the eval set, not
      the model. It is why Brier scores are not comparable across sets.
    - `residual`: within-bin variation, the price of binning continuous forecasts.
      It is 0 when every forecast in a bin is identical. A large residual means
      the decomposition is being read through too coarse a binning — the identity
      still closes, so nothing looks wrong, but `calibration` and `resolution`
      are attributing to bins what actually happened inside them. See
      `residual_fraction` and `binning_is_too_coarse`.
    """

    score: float
    calibration: float
    resolution: float
    uncertainty: float
    residual: float
    base_rate: float
    n: int
    binning: Binning
    bins: tuple[Bin, ...]
    residual_threshold: float = MATERIAL_RESIDUAL_FRACTION

    @property
    def residual_fraction(self) -> float:
        """|residual| as a share of the score. `inf` if the score is 0 and the
        residual is not, which cannot happen but should not read as clean."""
        if self.score == 0.0:
            return 0.0 if self.residual == 0.0 else math.inf
        return abs(self.residual) / self.score

    @property
    def binning_is_too_coarse(self) -> bool:
        """True when the residual is a material share of the score, i.e. the
        calibration/resolution split should not be quoted as-is."""
        return self.residual_fraction > self.residual_threshold

    def summary(self) -> str:
        warn = " COARSE-BINNING" if self.binning_is_too_coarse else ""
        return (
            f"brier={self.score:.4f} "
            f"(cal={self.calibration:.4f}, res={self.resolution:.4f}, "
            f"unc={self.uncertainty:.4f}, resid={self.residual:.4f} "
            f"[{self.residual_fraction:.1%}]{warn}) "
            f"n={self.n}, {self.binning.describe()}"
        )


def brier(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    binning: Binning = DEFAULT_BINNING,
    residual_threshold: float = MATERIAL_RESIDUAL_FRACTION,
    warn_on_coarse_binning: bool = True,
) -> BrierDecomposition:
    """Brier score, always decomposed.

    There is deliberately no `brier_score() -> float`. Per `CLAUDE.md` #4, a
    Brier improvement does not count until it has been attributed to calibration
    or resolution, so the decomposition is the only shape this returns.

    Warns (`CoarseBinningWarning`) when the residual is a material share of the
    score. The identity closes either way, so a silent field would let a
    misleading decomposition read as a clean one; the fix is finer bins.
    """
    _validate(probabilities, outcomes)
    n = len(probabilities)

    score = sum((p - o) ** 2 for p, o in zip(probabilities, outcomes, strict=True)) / n
    base_rate = sum(outcomes) / n
    uncertainty = base_rate * (1.0 - base_rate)

    bins = _build_bins(probabilities, outcomes, binning)

    calibration = 0.0
    resolution = 0.0
    for b in bins:
        if b.n == 0 or b.mean_forecast is None or b.observed_frequency is None:
            continue
        calibration += b.n * (b.mean_forecast - b.observed_frequency) ** 2
        resolution += b.n * (b.observed_frequency - base_rate) ** 2
    calibration /= n
    resolution /= n

    residual = score - (calibration - resolution + uncertainty)

    result = BrierDecomposition(
        score=score,
        calibration=calibration,
        resolution=resolution,
        uncertainty=uncertainty,
        residual=residual,
        base_rate=base_rate,
        n=n,
        binning=binning,
        bins=tuple(bins),
        residual_threshold=residual_threshold,
    )

    if warn_on_coarse_binning and result.binning_is_too_coarse:
        warnings.warn(
            f"Brier residual is {result.residual_fraction:.1%} of the score "
            f"(threshold {residual_threshold:.0%}) under {binning.describe()}. "
            "The calibration/resolution split is attributing to bins what happened "
            "inside them; use finer bins before quoting it.",
            CoarseBinningWarning,
            stacklevel=2,
        )

    return result


# --------------------------------------------------------------------------
# ECE
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ECEBinned:
    """ECE under one specific binning scheme."""

    value: float
    max_calibration_error: float
    """Largest absolute gap in any non-empty bin. An aggregate ECE can look fine
    while one bin is badly wrong."""

    n: int
    binning: Binning
    bins: tuple[Bin, ...]

    @property
    def n_populated_bins(self) -> int:
        return sum(1 for b in self.bins if b.n > 0)

    @property
    def smallest_populated_bin(self) -> int:
        """Count in the thinnest non-empty bin. A bin holding a handful of
        questions estimates an observed frequency out of noise, and under
        equal-width binning most of the bins usually look like that."""
        populated = [b.n for b in self.bins if b.n > 0]
        return min(populated) if populated else 0

    def summary(self) -> str:
        return (
            f"ece={self.value:.4f} mce={self.max_calibration_error:.4f} "
            f"({self.binning.describe()}, {self.n_populated_bins} populated, "
            f"min bin n={self.smallest_populated_bin})"
        )


@dataclass(frozen=True)
class ECEResult:
    """ECE under BOTH binning schemes, because the scheme moves the number.

    There is no `.value`. `docs/06` and `docs/07` both publish ECE, and a single
    number whose scheme was picked by default is the kind of figure that gets
    challenged — so getting a scalar out of this requires naming a scheme.

    The two schemes fail differently and neither is right:

    - **equal-width** puts most of a well-calibrated forecaster's predictions on
      a skewed question set into two or three bins, and the rest hold counts too
      small to estimate an observed frequency. ECE then reads low largely because
      the sparse bins are noise averaging out.
    - **equal-mass** keeps every bin populated but its edges are data-derived, so
      they move between checkpoints and the bins are not comparable across runs
      or across category slices.

    Report both. `spread` is the honest measure of how much the choice mattered.
    """

    equal_width: ECEBinned
    equal_mass: ECEBinned
    n: int
    n_bins: int

    @property
    def spread(self) -> float:
        """|equal-width ECE - equal-mass ECE|. Large means the number is being
        set by the binning as much as by the model."""
        return abs(self.equal_width.value - self.equal_mass.value)

    def for_scheme(self, scheme: BinningScheme) -> ECEBinned:
        return self.equal_width if scheme is BinningScheme.EQUAL_WIDTH else self.equal_mass

    def summary(self) -> str:
        return (
            f"ece[equal_width]={self.equal_width.value:.4f} "
            f"ece[equal_mass]={self.equal_mass.value:.4f} "
            f"spread={self.spread:.4f} n={self.n}, {self.n_bins} bins"
        )


def ece_binned(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    binning: Binning = DEFAULT_BINNING,
) -> ECEBinned:
    """Count-weighted mean absolute gap between mean forecast and observed rate,
    under one named scheme. Prefer `ece`, which computes both."""
    _validate(probabilities, outcomes)
    n = len(probabilities)
    bins = _build_bins(probabilities, outcomes, binning)

    total = 0.0
    worst = 0.0
    for b in bins:
        gap = b.gap
        if b.n == 0 or gap is None:
            continue
        total += b.n * abs(gap)
        worst = max(worst, abs(gap))

    return ECEBinned(
        value=total / n,
        max_calibration_error=worst,
        n=n,
        binning=binning,
        bins=tuple(bins),
    )


def ece(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> ECEResult:
    """ECE under both binning schemes."""
    return ECEResult(
        equal_width=ece_binned(
            probabilities, outcomes, binning=Binning(BinningScheme.EQUAL_WIDTH, n_bins)
        ),
        equal_mass=ece_binned(
            probabilities, outcomes, binning=Binning(BinningScheme.EQUAL_MASS, n_bins)
        ),
        n=len(probabilities),
        n_bins=n_bins,
    )


# --------------------------------------------------------------------------
# Log score
# --------------------------------------------------------------------------

#: ASSUMPTION: `docs/07` says "log score, clipped" without naming a bound. 1e-6
#: caps a single confident miss at ~13.8 nats, which is punishing but not
#: unbounded. It is a reporting choice, so it is carried on the result and must
#: be stated wherever the number is: two log scores with different eps are not
#: comparable. Record the final choice in the internal decisions log.
DEFAULT_LOG_SCORE_EPS = 1e-6


@dataclass(frozen=True)
class LogScoreResult:
    """Mean clipped negative log likelihood, in nats. Lower is better."""

    value: float
    eps: float
    n: int
    n_clipped: int
    """Forecasts that hit the clip. A rising count means the tail of the reported
    score is being set by the clip rather than by the model."""

    def summary(self) -> str:
        return (
            f"logscore={self.value:.4f} nats (eps={self.eps:g}, clipped={self.n_clipped}/{self.n})"
        )


def log_score(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    eps: float = DEFAULT_LOG_SCORE_EPS,
) -> LogScoreResult:
    """Clipped log score. Probabilities are clamped to [eps, 1 - eps] first."""
    _validate(probabilities, outcomes)
    if not (0.0 < eps < 0.5):
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")

    total = 0.0
    n_clipped = 0
    for p, o in zip(probabilities, outcomes, strict=True):
        if p < eps or p > 1.0 - eps:
            n_clipped += 1
        clipped = min(max(p, eps), 1.0 - eps)
        total += -math.log(clipped if o == 1 else 1.0 - clipped)

    n = len(probabilities)
    return LogScoreResult(value=total / n, eps=eps, n=n, n_clipped=n_clipped)


# --------------------------------------------------------------------------
# Output histogram
# --------------------------------------------------------------------------

#: ASSUMPTION: distinct-value counting needs a rounding tolerance, since the
#: model emits decimal strings and float noise would otherwise inflate the count.
#: 2dp matches the `<probability>0.37</probability>` output format in `CLAUDE.md`.
DEFAULT_HISTOGRAM_DECIMALS = 2

#: Half-width of the band around 0.5 used for `mass_near_half`.
DEFAULT_CENTER_HALF_WIDTH = 0.05


@dataclass(frozen=True)
class HistogramStats:
    """Shape of the model's output distribution (`CLAUDE.md` #5).

    Every other metric in this module is an average and all of them can look
    healthy while the model emits three values. These are the numbers that catch
    that, so they are logged at every checkpoint alongside loss.
    """

    n: int
    distinct_values: int
    """Count of distinct outputs after rounding to `decimals`."""

    value_entropy_bits: float
    """Entropy of the distribution over distinct rounded values. Capped at
    log2(distinct_values); 0 means the model emits a single number."""

    bin_entropy_bits: float
    """Entropy over the binning, so it is comparable across checkpoints even when
    the set of distinct values changes."""

    mass_near_half: float
    """Fraction of outputs within `center_half_width` of 0.5 — collapse toward
    the base rate, stated directly."""

    top_values: tuple[tuple[float, int], ...]
    """Most frequent rounded values with counts. Catches the pile-up on round
    numbers (0.5 / 0.7 / 0.9) that a bin entropy can miss."""

    min: float
    max: float
    mean: float
    binning: Binning
    decimals: int
    center_half_width: float

    def summary(self) -> str:
        top = ", ".join(f"{v:g}x{c}" for v, c in self.top_values)
        return (
            f"n={self.n} distinct={self.distinct_values} "
            f"H_value={self.value_entropy_bits:.3f}b H_bin={self.bin_entropy_bits:.3f}b "
            f"near_half={self.mass_near_half:.3f} top=[{top}]"
        )


def _entropy_bits(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue  # 0 * log(0) := 0
        p = c / total
        h -= p * math.log2(p)
    return h


def output_histogram(
    probabilities: Sequence[float],
    *,
    binning: Binning = DEFAULT_BINNING,
    decimals: int = DEFAULT_HISTOGRAM_DECIMALS,
    top_k: int = 5,
    center_half_width: float = DEFAULT_CENTER_HALF_WIDTH,
) -> HistogramStats:
    """Distributional stats over forecasts alone — no outcomes needed.

    Runs on any set of outputs, including an unresolved live-eval batch, which is
    the point: hedging collapse is visible immediately, without waiting for
    questions to resolve.
    """
    if not probabilities:
        raise ValueError("no forecasts to summarize")
    for i, p in enumerate(probabilities):
        if not (0.0 <= p <= 1.0) or math.isnan(p):
            raise ValueError(f"probability at index {i} is not in [0, 1]: {p!r}")

    rounded = [round(p, decimals) for p in probabilities]
    value_counts = Counter(rounded)

    groups = (
        _equal_width_groups(probabilities, binning.n_bins)
        if binning.scheme is BinningScheme.EQUAL_WIDTH
        else _equal_mass_groups(probabilities, binning.n_bins)
    )

    n = len(probabilities)
    # Symmetric bounds rather than `abs(p - 0.5) <= w`. That formulation is
    # asymmetric in floating point: |0.45 - 0.5| rounds to 0.04999999999999999
    # and counts, while |0.55 - 0.5| rounds to 0.050000000000000044 and does
    # not, so the band silently included one edge and excluded the other.
    # Comparing against precomputed edges is symmetric by construction. Same
    # family of defect as the misbinning `_equal_width_index` documents.
    lower_edge = 0.5 - center_half_width
    upper_edge = 0.5 + center_half_width
    near_half = sum(1 for p in probabilities if lower_edge <= p <= upper_edge)

    return HistogramStats(
        n=n,
        distinct_values=len(value_counts),
        value_entropy_bits=_entropy_bits(list(value_counts.values())),
        bin_entropy_bits=_entropy_bits([len(g) for g in groups]),
        mass_near_half=near_half / n,
        # Ties broken by value so the output is deterministic across runs.
        top_values=tuple(sorted(value_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]),
        min=min(probabilities),
        max=max(probabilities),
        mean=sum(probabilities) / n,
        binning=binning,
        decimals=decimals,
        center_half_width=center_half_width,
    )


# --------------------------------------------------------------------------
# Soft targets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftTargetScore:
    """Agreement with a probability-valued target. **Not** a calibration number.

    Deliberately a separate type from `BrierDecomposition`, with no
    calibration / resolution / uncertainty, because the Murphy decomposition does
    not survive the substitution:

    - `uncertainty` is `o(1 - o)`, the variance of a Bernoulli outcome. For a
      continuous target `Var(t) != mean(t)*(1 - mean(t))` unless every `t` is 0 or 1, so the
      substituted term is not the variance of anything.
    - `resolution` measures how far bins' observed *frequencies* sit from the
      base rate. Against soft targets a bin's mean is a mean of probabilities,
      not a frequency, and "the model separates questions that resolved
      differently" is not a statement the quantity supports.
    - `calibration` against a consensus measures agreement with that consensus.
      Calling it calibration would name the product's central claim after a
      number that is not about reality.

    The identity would still close arithmetically. That is exactly the danger —
    see `CoarseBinningWarning` for the other case where a closing identity hides
    a meaningless decomposition.
    """

    mse: float
    """Mean squared error against the target. Lower is better."""

    mae: float
    mean_signed_error: float
    """Mean of `p - t`. Positive means the model sits above the consensus
    systematically, which is a fact about disagreement, not about accuracy."""

    n: int

    def summary(self) -> str:
        return (
            f"soft-target agreement: mse={self.mse:.4f} mae={self.mae:.4f} "
            f"signed={self.mean_signed_error:+.4f} n={self.n} "
            "(agreement with a consensus, NOT calibration)"
        )


def soft_target_score(probabilities: Sequence[float], targets: Sequence[float]) -> SoftTargetScore:
    """Score forecasts against probability-valued targets.

    Squared error against a probability, not against a realization. Use for
    training diagnostics and for `docs/05` target-weighting work. It cannot enter
    a model card: `CardMetrics` does not accept it, and there is no conversion.
    """
    if len(probabilities) != len(targets):
        raise ValueError(
            f"length mismatch: {len(probabilities)} probabilities, {len(targets)} targets"
        )
    if not probabilities:
        raise ValueError("no forecasts to score")
    for name, values in (("probability", probabilities), ("target", targets)):
        for i, v in enumerate(values):
            if not (0.0 <= v <= 1.0) or math.isnan(v):
                raise ValueError(f"{name} at index {i} is not in [0, 1]: {v!r}")

    n = len(probabilities)
    errors = [p - t for p, t in zip(probabilities, targets, strict=True)]
    return SoftTargetScore(
        mse=sum(e * e for e in errors) / n,
        mae=sum(abs(e) for e in errors) / n,
        mean_signed_error=sum(errors) / n,
        n=n,
    )
