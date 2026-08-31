# SPDX-License-Identifier: Apache-2.0
"""Marking Manifold markets that are not forecasting questions.

Manifold users tag their own markets, and some of those tags say plainly that the
market is not a forecast: a lottery, a personal goal, a joke, a bet on the market
itself. Those questions cannot measure forecasting skill — no model beats the base
rate on "will I attend Manifest", which turns on one person's undisclosed
intention.

**This is deliberately a bracket, not a classifier.** The first version was a
single hand-written slug list, and checking it against an independently derived
reference — the question TITLE, which the slug author did not write with our test
in mind — refuted its precision in both directions:

- `destinygg` flagged *"Will Destiny reach 630k subs on YouTube by 3/01?"*, a real
  forecast about a public metric. The slug marks a community, not a content type.
- `new-years-resolutions-2024` flagged *"Will OPTIC hold a competition in
  London?"*, likewise real.
- Meanwhile `personal` and untagged first-person markets were missed entirely:
  *"Will I drop out of college?"*, *"Will my flight be cancelled?"*, *"When this
  market closes, I will flip a coin; will the result be heads?"*

So a point estimate would carry false precision. `CERTAIN` slugs state the content
type outright and give a lower bound; `COMMUNITY` slugs are topic groups that
merely correlate with it and widen the estimate to an upper bound. Callers get
both and are expected to report both.

`docs/12` reserves the corpus-filtering decision. Nothing here filters anything —
it labels, and the label is the market creator's rather than ours.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, cast

__all__ = [
    "Confidence",
    "DropPolicy",
    "FilterResult",
    "HasNonPredictiveFlag",
    "NonPredictiveVerdict",
    "classify_slugs",
    "filter_nonpredictive",
]

#: Slugs that state the market is not a forecast. Lower bound.
CERTAIN = re.compile(
    r"^(nonpredictive|nonpredictive-profits|personal-goals|personal|"
    r"self-resolving|free-lottery|fairlyrandom|manifold-[0-9a-f]{12})$"
)

#: Community and event groups that correlate with non-forecasting content but
#: also contain genuine questions. Upper bound only.
COMMUNITY = re.compile(
    r"^(fun|destinygg|new-years-resolutions-\d+|manifoldlove-\w+|"
    r"change-my-mind|gambling|proofniks)$"
)


class Confidence(StrEnum):
    CERTAIN = "certain"
    """A slug naming the content type. Counts toward the lower bound."""

    POSSIBLE = "possible"
    """A community or event slug. Counts only toward the upper bound."""

    NONE = "none"


@dataclass(frozen=True)
class NonPredictiveVerdict:
    confidence: Confidence
    matched: tuple[str, ...]

    @property
    def lower(self) -> bool:
        """Counts in the conservative estimate."""
        return self.confidence is Confidence.CERTAIN

    @property
    def upper(self) -> bool:
        """Counts in the generous estimate."""
        return self.confidence is not Confidence.NONE


def classify_slugs(slugs: object) -> NonPredictiveVerdict:
    """Classify one market from its group slugs.

    Accepts anything list-like; a market with no slugs is `NONE`, which is a
    genuine limitation rather than a clean result — 13.6% of the corpus carries
    no group at all and cannot be reached this way.
    """
    if not isinstance(slugs, list | tuple):
        return NonPredictiveVerdict(Confidence.NONE, ())
    values = [str(item) for item in cast(Sequence[object], slugs)]
    certain = tuple(s for s in values if CERTAIN.match(s))
    if certain:
        return NonPredictiveVerdict(Confidence.CERTAIN, certain)
    community = tuple(s for s in values if COMMUNITY.match(s))
    if community:
        return NonPredictiveVerdict(Confidence.POSSIBLE, community)
    return NonPredictiveVerdict(Confidence.NONE, ())


# --------------------------------------------------------------------------
# Filtering — a choice each artifact declares, not a property of the corpus
# --------------------------------------------------------------------------


class DropPolicy(StrEnum):
    """What an artifact chose to exclude. There is no default.

    The three answers are right for different jobs, which is why the flag lives
    on the record and the choice lives here:

    `NOTHING` — training. A calibrated forecaster *should* say 50% to a coin
    flip, so learning that some questions carry no signal is a thing to teach it
    rather than contamination to remove.

    `CERTAIN` — when recall matters more than precision. Drops only what a
    creator's own slug states outright.

    `CERTAIN_AND_POSSIBLE` — published evaluation, and **mandatory for any
    comparison between subsets**. The 2026-08-22 mechanism audit found 21.6%
    contamination in the threshold arm against 8.2% in the event arm; that
    differential inflated a headline result until it was caught, and no other
    policy would have prevented it.
    """

    NOTHING = "nothing"
    CERTAIN = "certain"
    CERTAIN_AND_POSSIBLE = "certain_and_possible"


class HasNonPredictiveFlag(Protocol):
    """Anything carrying the flag. `QuestionRecord` does; so can a lighter row."""

    nonpredictive: Confidence | None


R = TypeVar("R", bound=HasNonPredictiveFlag)

#: Which verdicts each policy removes. Written out so the mapping is one named
#: thing rather than a literal inside the function that applies it.
_DROPS: dict[DropPolicy, frozenset[Confidence]] = {
    DropPolicy.NOTHING: frozenset(),
    DropPolicy.CERTAIN: frozenset({Confidence.CERTAIN}),
    DropPolicy.CERTAIN_AND_POSSIBLE: frozenset({Confidence.CERTAIN, Confidence.POSSIBLE}),
}


@dataclass(frozen=True)
class FilterResult(Generic[R]):
    """What survived, what was dropped, and what could not be reached.

    Carries `unassessed` because a record with `nonpredictive=None` was never
    classified — 12.9% of the Manifold corpus has no group slug — and counting
    those as clean is the error that understated the corpus figure on
    2026-08-20. They are kept, and reported, never silently absorbed.
    """

    kept: tuple[R, ...]
    policy: DropPolicy
    dropped_certain: int
    dropped_possible: int
    unassessed: int
    """Kept, but never classified. The residual contamination lives here."""

    @property
    def n_dropped(self) -> int:
        return self.dropped_certain + self.dropped_possible

    def summary(self) -> str:
        total = len(self.kept) + self.n_dropped
        share = 100 * self.unassessed / len(self.kept) if self.kept else float("nan")
        return (
            f"[{self.policy.value}] kept {len(self.kept):,}/{total:,}, "
            f"dropped {self.dropped_certain:,} certain + {self.dropped_possible:,} possible, "
            f"{self.unassessed:,} unassessed ({share:.1f}% of kept, residual)"
        )


def filter_nonpredictive(records: Sequence[R], *, policy: DropPolicy) -> FilterResult[R]:
    """Apply a stated drop policy to records carrying a `nonpredictive` flag.

    `policy` is keyword-only and has no default, on purpose. Whoever builds an
    artifact from this corpus states what they excluded, and it appears in their
    code rather than in a footnote — the same discipline
    `tarot.evalrun.scored_slice` applies to coverage.
    """
    drop = _DROPS[policy]

    kept: list[R] = []
    dropped_certain = dropped_possible = unassessed = 0
    for record in records:
        flag = record.nonpredictive
        if flag is None:
            unassessed += 1
            kept.append(record)
            continue
        if flag in drop:
            if flag is Confidence.CERTAIN:
                dropped_certain += 1
            else:
                dropped_possible += 1
            continue
        kept.append(record)
    return FilterResult(
        kept=tuple(kept),
        policy=policy,
        dropped_certain=dropped_certain,
        dropped_possible=dropped_possible,
        unassessed=unassessed,
    )
