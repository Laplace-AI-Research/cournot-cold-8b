# SPDX-License-Identifier: Apache-2.0
"""Assemble SFT training examples, choosing what each one is trained against.

`docs/04` names the overconfidence trap as this stage's dominant failure:

> Every outcome label is 0 or 1, so naive SFT on outcome-labeled data teaches the
> model that the world is deterministic and drives it toward the extremes.
>
> Mitigation: train a substantial fraction of the mixture against **soft
> targets** rather than terminal outcomes [...] A soft target carries far more
> information per example than a single Bernoulli draw.

The 2026-08-22 trace pilot turned that from precaution into requirement. The
teacher is overconfident on both tails — it issued forecasts of <= 0.05 fifty
times and the event happened **18%** of the time — so distilling its numbers
wholesale would import ECE 0.140 into a model whose entire thesis is that
calibration is the fixable half.

This module decides, per example, whether the target is a soft one or the
terminal outcome, and refuses to let that share be settled by accident.

## The fraction is stated and achieved, or it raises

`soft_fraction` has no default. It is capped by **coverage**: a soft target
exists only where a question has a usable supervision point, and asking for 60%
soft when 40% of questions have one is not a rounding problem — it is a
different corpus from the one requested. Same discipline as `cournot.mixture`,
where an early version silently returned 0.833 for a requested 0.900.

## Provenance travels with the target

`SoftTargetProvenance` is carried on every soft example rather than inferred.
`docs/05` weights low-variance targets over terminal outcomes, so provenance is
the field that says how much variance a target carries — a market consensus and
a computed base rate are not interchangeable, and a corpus that has forgotten
which is which cannot be weighted at all.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from cournot.types import Outcome, SoftTargetProvenance


class TargetKind(StrEnum):
    SOFT = "soft"
    #: The realized binary outcome. One Bernoulli draw, maximum variance.
    TERMINAL = "terminal"


class HasTrace(Protocol):
    question_id: str
    probability: float
    outcome: Outcome
    reasoning: str | None


class SoftTargetUnavailableError(ValueError):
    """The requested soft fraction exceeds what supervision points can supply."""


@dataclass(frozen=True)
class TrainingExample:
    """One SFT example: a trace, and the number it is trained against."""

    question_id: str
    reasoning: str | None
    target: float
    kind: TargetKind
    provenance: SoftTargetProvenance | None
    """Set exactly when `kind` is SOFT. A soft target without provenance cannot
    be weighted by `docs/05`, and a terminal one with provenance is a category
    error, so both are rejected at construction."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.target <= 1.0:
            raise ValueError(f"target must be in [0,1], got {self.target}")
        if (self.kind is TargetKind.SOFT) != (self.provenance is not None):
            raise ValueError(
                f"kind={self.kind.value} with provenance={self.provenance!r}: "
                "a soft target must carry provenance and a terminal one must not"
            )


@dataclass(frozen=True)
class SftCorpus:
    examples: tuple[TrainingExample, ...]
    requested_soft_fraction: float
    n_soft: int
    n_terminal: int
    n_without_supervision: int
    """Questions with no usable supervision point, so terminal by necessity
    rather than by choice. Reported because it bounds what any later run can
    request."""

    @property
    def achieved_soft_fraction(self) -> float:
        total = self.n_soft + self.n_terminal
        return self.n_soft / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{len(self.examples):,} examples — {self.n_soft:,} soft "
            f"({self.achieved_soft_fraction:.1%}), {self.n_terminal:,} terminal; "
            f"{self.n_without_supervision:,} had no supervision point available"
        )


def assemble_sft(
    traces: Sequence[HasTrace],
    soft_targets: Mapping[str, tuple[float, SoftTargetProvenance]],
    *,
    soft_fraction: float,
    salt: str = "sft",
    tolerance: float = 0.005,
) -> SftCorpus:
    """Build the corpus, training `soft_fraction` of it against soft targets.

    `soft_targets` maps question id to `(target, provenance)`. A question absent
    from it has no usable supervision point and is trained on its terminal
    outcome — counted separately, because "terminal because we chose to" and
    "terminal because nothing else existed" are different facts and only the
    second one bounds future runs.

    Which questions get the soft target is decided by hashing the id, not by
    sampling: the same inputs must give the same corpus, or no two training runs
    are comparable.
    """
    if not 0.0 <= soft_fraction <= 1.0:
        raise ValueError(f"soft_fraction must be in [0,1], got {soft_fraction}")

    eligible = [t for t in traces if t.question_id in soft_targets]
    ineligible = [t for t in traces if t.question_id not in soft_targets]

    wanted = round(len(traces) * soft_fraction)
    if wanted > len(eligible):
        raise SoftTargetUnavailableError(
            f"soft_fraction {soft_fraction:.3f} needs {wanted:,} soft targets "
            f"against {len(eligible):,} questions that have one "
            f"({len(eligible) / max(len(traces), 1):.1%} coverage). Fetch more "
            "supervision points or lower the fraction — do not let it round down."
        )

    ordered = sorted(
        eligible,
        key=lambda t: hashlib.blake2b(
            f"{salt}:{t.question_id}".encode(), digest_size=8
        ).hexdigest(),
    )
    soft_ids = {t.question_id for t in ordered[:wanted]}

    examples: list[TrainingExample] = []
    for trace in list(eligible) + list(ineligible):
        if trace.question_id in soft_ids:
            target, provenance = soft_targets[trace.question_id]
            examples.append(
                TrainingExample(
                    question_id=trace.question_id,
                    reasoning=trace.reasoning,
                    target=target,
                    kind=TargetKind.SOFT,
                    provenance=provenance,
                )
            )
        else:
            examples.append(
                TrainingExample(
                    question_id=trace.question_id,
                    reasoning=trace.reasoning,
                    target=float(trace.outcome),
                    kind=TargetKind.TERMINAL,
                    provenance=None,
                )
            )

    corpus = SftCorpus(
        examples=tuple(examples),
        requested_soft_fraction=soft_fraction,
        n_soft=sum(1 for e in examples if e.kind is TargetKind.SOFT),
        n_terminal=sum(1 for e in examples if e.kind is TargetKind.TERMINAL),
        n_without_supervision=len(ineligible),
    )
    if abs(corpus.achieved_soft_fraction - soft_fraction) > tolerance:
        raise SoftTargetUnavailableError(
            f"soft_fraction {soft_fraction:.3f} requested but "
            f"{corpus.achieved_soft_fraction:.3f} assembled, outside tolerance "
            f"{tolerance}."
        )
    return corpus


#: Share of targets at exactly 0 or 1 above which a corpus is refused for text
#: SFT. `docs/04`: "NO TARGET IS EVER EXACTLY 0 OR 1" -- under next-token
#: prediction `0.00` is a string to emit, not a Bernoulli draw to average over.
#:
#: The rule was written after v1 (27,697 of 55,394 terminal targets -> 392 of
#: 490 forecasts at exactly 0.0) and broken again by v4 on 2026-08-25
#: (`--soft-fraction 0.50` -> 50.2% extreme -> 2,727 of 3,000). Between the two
#: the rule existed, in capital letters, citing v1. A document does not refuse a
#: run.
#:
#: A share rather than zero: rejecting a corpus for one stray value would make
#: the guard impractical, and an impractical guard gets switched off.
MAXIMUM_EXTREME_SHARE = 0.02


class ExtremeTargetError(ValueError):
    """A text-SFT corpus trains against 0/1 targets. Twice observed."""


def check_extreme_share(
    targets: Sequence[float], *, maximum: float = MAXIMUM_EXTREME_SHARE
) -> float:
    """Return the share of targets at exactly 0 or 1, raising above `maximum`.

    Not applicable to `docs/14`'s scalar head: under Brier loss a 0/1 label is a
    proper Bernoulli observation. This is a text-SFT rule and only a text-SFT
    rule.
    """
    if not targets:
        return 0.0
    extreme = sum(1 for t in targets if t <= 0.0 or t >= 1.0)
    share = extreme / len(targets)
    if share > maximum:
        raise ExtremeTargetError(
            f"{extreme:,} of {len(targets):,} targets ({share:.1%}) are exactly "
            f"0 or 1, above the {maximum:.0%} ceiling.\n\n"
            "docs/04: under next-token prediction a 0/1 target is an instruction "
            "to emit that string, not a Bernoulli draw. This collapsed v1 "
            "(392/490 at 0.00) and v4 (2,727/3,000).\n\n"
            "Most likely cause: --soft-fraction below 1.0 in "
            "scripts/sft_corpus.py, which trains the remainder against the "
            "realized outcome. Raise it to 1.0."
        )
    return share
