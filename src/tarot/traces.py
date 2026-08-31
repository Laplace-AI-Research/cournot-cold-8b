# SPDX-License-Identifier: Apache-2.0
"""Select SFT traces, including the wrong ones that have to be kept.

`docs/04` states the rule and the reason:

> - Keep traces whose stated probability lands on the correct side of the base
>   rate, for questions the strong model got right
> - **Deliberately keep a fraction of confidently-wrong-but-well-reasoned
>   traces.** The target behavior is good reasoning followed by an honest
>   commitment, not "always be right". If every training trace is vindicated by
>   the outcome, the model learns that its reasoning is reliable, which is
>   exactly the wrong lesson for a forecaster.

So this is not a quality filter with an exception bolted on. Keeping some losses
is the *point*: a corpus in which every confident trace was right teaches that
confidence is warranted, and a forecaster that has learned that is broken in the
way this project cares about most.

## Three classes

A trace is scored against the corpus base rate, not against 0.5. "Correct side"
means it moved away from the base rate in the direction the outcome went — which
is the only sense in which a probabilistic forecast can be individually right.

| class | kept |
|---|---|
| `CORRECT_SIDE` | all |
| `CONFIDENTLY_WRONG` | a **stated fraction** |
| `HEDGED_WRONG` | none |

`HEDGED_WRONG` is dropped because it teaches nothing either way: a trace that
barely moved off the base rate and still missed carries no reasoning worth
imitating and no honest-commitment lesson either.

## What this cannot check

`docs/04` says *well-reasoned* confidently-wrong traces. Nothing here reads the
reasoning. Keeping a fraction of confidently-wrong traces therefore keeps
badly-reasoned ones at the same rate as well-reasoned ones, which is an
approximation of the rule rather than the rule. Stated here rather than hidden:
narrowing it needs either a judge model or hand labels, and the 2026-08-22
surface-form round is the precedent for how cheap the latter is.

## Determinism

The kept fraction is selected by hashing the question id, not by sampling. Two
runs over the same traces produce the same corpus, and a corpus that changes
between runs would make every downstream comparison unreproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from tarot.types import Outcome


class TraceClass(StrEnum):
    CORRECT_SIDE = "correct_side"
    CONFIDENTLY_WRONG = "confidently_wrong"
    HEDGED_WRONG = "hedged_wrong"


class HasTrace(Protocol):
    """The fields selection reads. Anything richer is carried through untouched."""

    question_id: str
    probability: float
    outcome: Outcome


T = TypeVar("T", bound=HasTrace)


@dataclass(frozen=True)
class TraceSelection(Generic[T]):
    """What survived, and what it took — reported, never inferred downstream."""

    kept: tuple[T, ...]
    n_correct_side: int
    n_confidently_wrong_kept: int
    n_confidently_wrong_dropped: int
    n_hedged_wrong_dropped: int
    base_rate: float
    confident_margin: float
    wrong_fraction: float

    @property
    def n_total(self) -> int:
        return (
            self.n_correct_side
            + self.n_confidently_wrong_kept
            + self.n_confidently_wrong_dropped
            + self.n_hedged_wrong_dropped
        )

    @property
    def wrong_share_of_kept(self) -> float:
        """Share of the training corpus that is a confident miss.

        The number `docs/04`'s rule is really about. Zero means the corpus
        teaches that confidence is always vindicated.
        """
        kept = self.n_correct_side + self.n_confidently_wrong_kept
        return self.n_confidently_wrong_kept / kept if kept else 0.0

    def summary(self) -> str:
        return (
            f"kept {len(self.kept):,}/{self.n_total:,} — "
            f"{self.n_correct_side:,} correct-side, "
            f"{self.n_confidently_wrong_kept:,} confident misses "
            f"({self.wrong_share_of_kept:.1%} of corpus); dropped "
            f"{self.n_confidently_wrong_dropped:,} confident misses and "
            f"{self.n_hedged_wrong_dropped:,} hedged misses"
        )


def classify_trace(
    probability: float,
    outcome: Outcome,
    *,
    base_rate: float,
    confident_margin: float,
) -> TraceClass:
    """Which of the three classes a trace falls in.

    Scored against `base_rate` rather than 0.5: on a corpus resolving YES 42% of
    the time, a forecast of 0.45 on a YES is barely informative, and calling it
    "correct" because it exceeds one half would credit the model for the corpus's
    own asymmetry.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0,1], got {probability}")
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0,1), got {base_rate}")
    if confident_margin < 0.0:
        raise ValueError(f"confident_margin must be >= 0, got {confident_margin}")

    moved_up = probability > base_rate
    correct = moved_up if outcome == 1 else not moved_up
    if correct:
        return TraceClass.CORRECT_SIDE
    return (
        TraceClass.CONFIDENTLY_WRONG
        if abs(probability - base_rate) >= confident_margin
        else TraceClass.HEDGED_WRONG
    )


def select_traces(
    traces: Sequence[T],
    *,
    base_rate: float,
    confident_margin: float,
    wrong_fraction: float,
    salt: str = "trace",
) -> TraceSelection[T]:
    """Apply `docs/04`'s rule, keeping `wrong_fraction` of confident misses.

    Every parameter is keyword-only with no default. `wrong_fraction` especially:
    it decides what share of the training corpus contradicts its own confidence,
    which is a claim about what the model should learn, not a tuning knob. A
    default would settle it silently.
    """
    if not 0.0 <= wrong_fraction <= 1.0:
        raise ValueError(f"wrong_fraction must be in [0,1], got {wrong_fraction}")

    correct: list[T] = []
    confident_wrong: list[T] = []
    hedged_wrong = 0
    for trace in traces:
        verdict = classify_trace(
            trace.probability,
            trace.outcome,
            base_rate=base_rate,
            confident_margin=confident_margin,
        )
        if verdict is TraceClass.CORRECT_SIDE:
            correct.append(trace)
        elif verdict is TraceClass.CONFIDENTLY_WRONG:
            confident_wrong.append(trace)
        else:
            hedged_wrong += 1

    # Hash-ordered, not sampled: the same traces must give the same corpus.
    ordered = sorted(
        confident_wrong,
        key=lambda t: hashlib.blake2b(
            f"{salt}:{t.question_id}".encode(), digest_size=8
        ).hexdigest(),
    )
    n_keep = round(len(ordered) * wrong_fraction)
    keep_wrong = ordered[:n_keep]

    return TraceSelection(
        kept=tuple(correct) + tuple(keep_wrong),
        n_correct_side=len(correct),
        n_confidently_wrong_kept=len(keep_wrong),
        n_confidently_wrong_dropped=len(ordered) - len(keep_wrong),
        n_hedged_wrong_dropped=hedged_wrong,
        base_rate=base_rate,
        confident_margin=confident_margin,
        wrong_fraction=wrong_fraction,
    )
