# SPDX-License-Identifier: Apache-2.0
"""The event-partition gate for the synthetic/real mixture sweep.

`docs/13` states the constraint the mixture ratio is bounded by:

> The event slice is a **gate, not a diagnostic**. An arm that improves aggregate
> Brier while flat or worse on the event partition has learned the training
> distribution, not forecasting, and does not pass.

That is a sentence, and a sentence is not a gate. This module is the gate.

**Written before any arm has run.** A threshold built after seeing the numbers it
judges is fitted to them, and the choice of "flat" would be made by whichever
arm was in front of us. Two guards in this file exist only because the
alternative is a gate that always passes.

## What it decides

Synthetic training questions are threshold-over-series; the corpus that pays for
this is largely event-driven. So an arm can raise aggregate Brier purely by
getting better at the mechanism it was trained on, while the questions the
product is sold on stay flat. Aggregate Brier cannot see that, and neither can a
per-category breakdown -- the split that matters here is by *mechanism*.

## Why paired, and why the event slice alone decides

Arms are scored on the **same questions**, so per-question differences are paired
and the pairing removes question difficulty from the comparison. Comparing two
unpaired means would put the variance of question difficulty -- much larger than
the effect -- back into the interval.

The gate reads the **event slice on its own terms**. A common alternative,
requiring the event slice to improve *as much as* the aggregate, sounds stricter
but is not: it passes an arm that degrades both equally.

## The asymmetry

"Not degraded" is the bar for passing, not "improved". Requiring a significant
event-slice improvement at every rung would reject a ratio that buys threshold
skill for free, which is the whole point of adding synthetic data. What is
rejected is *paying for aggregate gains with event-slice losses*.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from cournot.types import Outcome

#: Two-sided 95%.
DEFAULT_Z = 1.96


class EventGateOutcome(StrEnum):
    PASS = "pass"
    #: Aggregate improved while the event slice did not. The failure `docs/13`
    #: names: skill bought on the trained mechanism, not on the sold one.
    TRAINED_DISTRIBUTION_ONLY = "trained_distribution_only"
    #: The event slice got significantly worse, whatever the aggregate did.
    EVENT_SLICE_DEGRADED = "event_slice_degraded"
    #: Too few event questions to say anything. Never reported as a pass.
    UNDERPOWERED = "underpowered"


@dataclass(frozen=True)
class PairedDelta:
    """Mean paired Brier change and its interval. Negative means improved."""

    mean: float
    lower: float
    upper: float
    n: int

    @property
    def improved(self) -> bool:
        """Significantly better: the whole interval is below zero."""
        return self.upper < 0.0

    @property
    def degraded(self) -> bool:
        """Significantly worse: the whole interval is above zero."""
        return self.lower > 0.0


@dataclass(frozen=True)
class EventGateVerdict:
    outcome: EventGateOutcome
    aggregate: PairedDelta
    event: PairedDelta
    threshold: PairedDelta | None

    @property
    def passed(self) -> bool:
        return self.outcome is EventGateOutcome.PASS

    def explain(self) -> str:
        if self.outcome is EventGateOutcome.UNDERPOWERED:
            return (
                f"only {self.event.n} event questions — too few to gate on. "
                "Not a pass: widen the eval set."
            )
        if self.outcome is EventGateOutcome.EVENT_SLICE_DEGRADED:
            return (
                f"event slice significantly worse ({self.event.mean:+.4f}, "
                f"95% CI [{self.event.lower:+.4f}, {self.event.upper:+.4f}])"
            )
        if self.outcome is EventGateOutcome.TRAINED_DISTRIBUTION_ONLY:
            return (
                f"aggregate improved ({self.aggregate.mean:+.4f}) but the event slice "
                f"did not ({self.event.mean:+.4f}, 95% CI [{self.event.lower:+.4f}, "
                f"{self.event.upper:+.4f}]) — skill on the trained mechanism, not the sold one"
            )
        return (
            f"event slice not degraded ({self.event.mean:+.4f}, 95% CI "
            f"[{self.event.lower:+.4f}, {self.event.upper:+.4f}]), n={self.event.n}"
        )


def _brier(probability: float, outcome: Outcome) -> float:
    return (probability - outcome) ** 2


def paired_delta(
    arm: Sequence[float],
    baseline: Sequence[float],
    outcomes: Sequence[Outcome],
    *,
    z: float = DEFAULT_Z,
) -> PairedDelta:
    """Mean per-question Brier change from `baseline` to `arm`, with a CI.

    Paired because both arms score the *same* questions. The standard error is
    of the paired differences, not of either arm's mean -- pooling them would
    carry question difficulty into the interval and swamp the effect.
    """
    if not (len(arm) == len(baseline) == len(outcomes)):
        raise ValueError(
            f"lengths must match: arm={len(arm)}, baseline={len(baseline)}, "
            f"outcomes={len(outcomes)}. Unequal lengths mean the arms were not "
            "scored on the same questions, and the comparison is not paired."
        )
    n = len(arm)
    if n == 0:
        return PairedDelta(mean=0.0, lower=0.0, upper=0.0, n=0)

    diffs = [_brier(a, o) - _brier(b, o) for a, b, o in zip(arm, baseline, outcomes, strict=True)]
    mean = sum(diffs) / n
    if n == 1:
        return PairedDelta(mean=mean, lower=-math.inf, upper=math.inf, n=1)
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half = z * math.sqrt(variance / n)
    return PairedDelta(mean=mean, lower=mean - half, upper=mean + half, n=n)


def event_gate(
    arm: Sequence[float],
    baseline: Sequence[float],
    outcomes: Sequence[Outcome],
    mechanisms: Sequence[str],
    *,
    minimum_event_questions: int,
    z: float = DEFAULT_Z,
) -> EventGateVerdict:
    """Gate one mixture arm against the baseline arm on the event partition.

    `mechanisms` holds `"event"` / `"threshold"` / `"ambiguous"` per question,
    from `scripts/mechanism_split.classify`. Ambiguous questions count toward the
    aggregate and toward neither slice: forcing them into one would let the
    classifier's uncertainty decide the gate.

    `minimum_event_questions` has **no default**. A caller who gates on a thin
    slice states the number they accepted, in their code rather than in a
    footnote -- the same rule as `EvalRunResult.scored_slice`.
    """
    if minimum_event_questions < 1:
        raise ValueError(f"minimum_event_questions must be >= 1, got {minimum_event_questions}")
    if not (len(arm) == len(baseline) == len(outcomes) == len(mechanisms)):
        raise ValueError(
            f"lengths must match: arm={len(arm)}, baseline={len(baseline)}, "
            f"outcomes={len(outcomes)}, mechanisms={len(mechanisms)}"
        )

    def slice_for(want: str) -> PairedDelta:
        idx = [i for i, m in enumerate(mechanisms) if m == want]
        return paired_delta(
            [arm[i] for i in idx],
            [baseline[i] for i in idx],
            [outcomes[i] for i in idx],
            z=z,
        )

    aggregate = paired_delta(arm, baseline, outcomes, z=z)
    event = slice_for("event")
    threshold = slice_for("threshold")

    # Order matters. Underpowered is checked first so a thin slice can never be
    # reported as a pass, and degradation before the aggregate comparison so an
    # arm that hurts the event slice fails even if the aggregate is flat.
    if event.n < minimum_event_questions:
        outcome = EventGateOutcome.UNDERPOWERED
    elif event.degraded:
        outcome = EventGateOutcome.EVENT_SLICE_DEGRADED
    elif aggregate.improved and not event.improved:
        outcome = EventGateOutcome.TRAINED_DISTRIBUTION_ONLY
    else:
        outcome = EventGateOutcome.PASS

    return EventGateVerdict(
        outcome=outcome,
        aggregate=aggregate,
        event=event,
        threshold=threshold if threshold.n else None,
    )
