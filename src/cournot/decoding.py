"""Reading a probability out of a token distribution, by expectation not argmax.

## Why this module exists

the internal decisions log (2026-08-24) records the correction that motivates it. The first SFT
arm returned **392 of 490 forecasts as exactly 0.0**, and that was diagnosed as a
corpus defect: half the training targets were the literal string `0.00`. The
corpus *was* defective. But it was not the only cause.

**Greedy decoding returns the mode, and the mode of a Bernoulli-ish distribution
is 0 or 1 whenever p != 0.5.** A model that has correctly learned "this question
resolves YES about 28% of the time" places its single most likely tenths digit at
`0` — because P(digit=0) can exceed P(digit=2) while the *expectation* over
digits is 2.8. Argmax discards exactly the information a forecaster is for.

So an argmax read of a probability model is a category error, and this module is
the correct read: the expectation over the digit distribution.

## The estimator

For a forecast written `0.d1d2`,

    E[p] = E[d1]/10 + E[d2 | d1 = argmax]/100

The first term is exact. The second conditions on the greedy tenths digit rather
than marginalising over all ten, because marginalising needs ten forward passes
and the term it corrects is bounded by 0.09. **That approximation is stated
rather than hidden**, and `hundredths_mass` on the result lets a caller see how
much of the distribution it rested on.

## Truncated distributions

Serving stacks return top-k logprobs, so the digit mass rarely sums to 1. The
estimator renormalises over the digits present and **reports the mass it
captured**. A low `tenths_mass` means the model was mostly predicting something
that is not a digit, and the read should not be trusted — which is why it is a
field on the result rather than something this module silently absorbs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

#: Below this share of top-k mass on digit tokens, the position was not really a
#: digit position and the read is refused. Not tuned; a read resting on less than
#: half the distribution is not a read.
MINIMUM_DIGIT_MASS = 0.5

#: ASCII digits only. `str.isdigit()` is True for Unicode digit characters --
#: superscripts, subscripts, and other scripts' numerals -- which `int()` then
#: refuses. A model that emits one of those is not making a forecast, and the
#: naive check crashes the whole run on it. Observed in the wild: a top-k entry
#: for a subscript zero killed a 500-question scoring pass at record 2.
ASCII_DIGITS = frozenset("0123456789")


def is_digit(token: str) -> bool:
    """True only for a single ASCII digit. See `ASCII_DIGITS`."""
    return len(token) == 1 and token in ASCII_DIGITS


#: The tag the format contract puts a probability behind (`CLAUDE.md`).
PROBABILITY_MARKER = "<probability>"

#: How far past the marker a decimal point may sit and still be *this* forecast's
#: decimal point. Long enough for `0.`, short enough that a full stop later in a
#: malformed response is not mistaken for one.
MAXIMUM_DOT_OFFSET = 3


def digit_positions(text: str) -> tuple[int, int] | None:
    """Character offsets of the two digits in the first `<probability>0.dd`.

    Returns `None` rather than a guess when the response does not reach a digit,
    so a caller cannot silently read the distribution of a neighbouring token.
    """
    start = text.find(PROBABILITY_MARKER)
    if start < 0:
        return None
    after = start + len(PROBABILITY_MARKER)
    dot = text.find(".", after)
    if dot < 0 or dot - after > MAXIMUM_DOT_OFFSET:
        return None
    if dot + 1 >= len(text) or not is_digit(text[dot + 1]):
        return None
    return dot + 1, dot + 2


class DigitReadFailure(Exception):
    """The token distribution does not describe a digit.

    A distinct type, deliberately. A failed read that returned 0.0 would be
    indistinguishable from a confident forecast of zero -- which is precisely the
    collapse this module exists to measure.
    """


@dataclass(frozen=True)
class DigitRead:
    """An expectation-decoded probability, with the evidence it rested on."""

    probability: float
    tenths_mass: float
    """Share of the top-k mass at the tenths position that sat on digit tokens."""
    hundredths_mass: float
    """Same at the hundredths position. 0.0 when no hundredths position was read."""
    greedy_probability: float
    """What argmax would have returned. The comparison is the point."""

    @property
    def mode_mean_gap(self) -> float:
        """How far argmax sits from the expectation. Large means argmax was lying."""
        return self.probability - self.greedy_probability


def _digit_distribution(distribution: Mapping[str, float]) -> tuple[dict[int, float], float]:
    """Restrict a token distribution to single-digit tokens and renormalise.

    Returns the normalised digit distribution and the share of the input mass it
    accounted for. Tokens are matched after stripping surrounding whitespace, so
    a leading-space variant of a digit counts as that digit.
    """
    total = sum(distribution.values())
    if total <= 0.0:
        raise DigitReadFailure("token distribution carries no mass")

    digits: dict[int, float] = {}
    for token, mass in distribution.items():
        stripped = token.strip()
        if is_digit(stripped):
            digits[int(stripped)] = digits.get(int(stripped), 0.0) + mass

    captured = sum(digits.values())
    share = captured / total
    if captured <= 0.0:
        raise DigitReadFailure("no single-digit token in the distribution")
    return {d: m / captured for d, m in digits.items()}, share


def _expectation(digits: Mapping[int, float]) -> float:
    return sum(d * m for d, m in digits.items())


def read_probability(
    tenths: Mapping[str, float],
    hundredths: Mapping[str, float] | None = None,
    *,
    minimum_digit_mass: float = MINIMUM_DIGIT_MASS,
) -> DigitRead:
    """Expectation-decode `0.d1d2` from the token distributions at each position.

    Raises `DigitReadFailure` when a position is not a digit position, rather
    than returning a number that would be mistaken for a forecast.
    """
    tenths_digits, tenths_mass = _digit_distribution(tenths)
    if tenths_mass < minimum_digit_mass:
        raise DigitReadFailure(
            f"tenths position holds {tenths_mass:.1%} digit mass, "
            f"below the {minimum_digit_mass:.0%} floor"
        )

    expected = _expectation(tenths_digits) / 10.0
    greedy_tenths = max(tenths_digits, key=lambda d: tenths_digits[d])
    greedy = greedy_tenths / 10.0
    hundredths_mass = 0.0

    if hundredths is not None:
        try:
            hundredths_digits, hundredths_mass = _digit_distribution(hundredths)
        except DigitReadFailure:
            hundredths_mass = 0.0
        else:
            if hundredths_mass >= minimum_digit_mass:
                expected += _expectation(hundredths_digits) / 100.0
                greedy += max(hundredths_digits, key=lambda d: hundredths_digits[d]) / 100.0
            else:
                hundredths_mass = 0.0

    # A probability of exactly 1.0 is reachable only as 0.99 + rounding; the
    # estimator cannot exceed 0.99 by construction, so no clamp is needed above.
    return DigitRead(
        probability=expected,
        tenths_mass=tenths_mass,
        hundredths_mass=hundredths_mass,
        greedy_probability=greedy,
    )
