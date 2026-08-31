# SPDX-License-Identifier: Apache-2.0
"""Score an eval slice per group — by category, by mechanism, by anything.

Three documents require this and none of it existed. `docs/07` asks for
reliability diagrams "broken out by category and by time horizon"; `docs/09`
risk 5 detection #1 requires per-mechanism reporting; `docs/13` makes the
per-mechanism version **mandatory at every checkpoint**, on the grounds that a
pooled number cannot show the failure the mixture sweep exists to catch — an arm
that improves aggregate Brier while going flat or worse on event-driven
questions.

`tarot.eventgate` decides whether that happened. This shows *how*.

## Small groups are reported, not dropped

A group of four questions has an ECE that means nothing, and a group of one has a
Brier of zero or one. Both would look like findings.

So `minimum_group` is required with no default, and groups below it are returned
in `too_small` **with their counts** rather than merged into a neighbour or
silently discarded. A reader can then see that "politics" was excluded at n=6,
which is a different fact from politics not appearing.

## The grouping key is supplied, not guessed

Callers pass a function from record to group label. Two reasons: `category` on
`QuestionRecord` is `"unclassified"` corpus-wide (the classifier measured 54.7%
against hand labels and is not fit for use), so a hardcoded category breakdown
would today produce exactly one group and imply a breakdown had happened. And
mechanism lives in `scripts/mechanism_split`, which is a script rather than a
library — importing it here would make the package depend on the scripts
directory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tarot.metrics import DEFAULT_BINNING, Binning, BrierDecomposition, brier
from tarot.types import Outcome, QuestionRecord


@dataclass(frozen=True)
class GroupScore:
    """One group's score, and how much of the slice it was."""

    label: str
    n: int
    share: float
    brier: BrierDecomposition
    base_rate: float


@dataclass(frozen=True)
class Breakdown:
    """Per-group scores, plus everything that did not get one."""

    key: str
    """What the slice was split on, e.g. "mechanism". Carried so a breakdown
    cannot be read without knowing what it broke down by."""

    groups: tuple[GroupScore, ...]
    too_small: Mapping[str, int]
    """Group label -> count, for groups below `minimum_group`. Reported because
    "excluded at n=6" and "absent" are different facts."""

    minimum_group: int
    n_scored: int
    n_total: int

    @property
    def coverage(self) -> float:
        return self.n_scored / self.n_total if self.n_total else 0.0

    def summary(self) -> str:
        lines = [
            f"by {self.key}: {len(self.groups)} groups, "
            f"{self.n_scored:,}/{self.n_total:,} scored ({self.coverage:.1%})"
        ]
        for group in self.groups:
            lines.append(
                f"  {group.label:<16} n={group.n:>5}  brier={group.brier.score:.4f}  "
                f"(cal {group.brier.calibration:.4f}, res {group.brier.resolution:.4f})  "
                f"base rate {group.base_rate:.3f}"
            )
        for label, count in sorted(self.too_small.items()):
            lines.append(f"  {label:<16} n={count:>5}  below the {self.minimum_group} minimum")
        return "\n".join(lines)


def breakdown(
    records: Sequence[QuestionRecord],
    probabilities: Sequence[float],
    key: Callable[[QuestionRecord], str],
    *,
    key_name: str,
    minimum_group: int,
    binning: Binning = DEFAULT_BINNING,
) -> Breakdown:
    """Group a slice by `key` and score each group that is big enough.

    `minimum_group` has no default. The number below which a per-group score
    becomes noise depends on what is being measured — a Brier stabilises far
    sooner than an ECE — so the caller states it, and it appears in their code
    rather than in a footnote.
    """
    if len(records) != len(probabilities):
        raise ValueError(
            f"{len(records)} records but {len(probabilities)} probabilities; "
            "every forecast needs the record it was made on"
        )
    if minimum_group < 1:
        raise ValueError(f"minimum_group must be >= 1, got {minimum_group}")

    buckets: dict[str, list[tuple[float, Outcome]]] = {}
    for record, probability in zip(records, probabilities, strict=True):
        if record.outcome is None:
            continue
        buckets.setdefault(key(record), []).append((probability, record.outcome))

    total = sum(len(rows) for rows in buckets.values())
    scored: list[GroupScore] = []
    too_small: dict[str, int] = {}
    for label, rows in buckets.items():
        if len(rows) < minimum_group:
            too_small[label] = len(rows)
            continue
        probs = [p for p, _ in rows]
        outcomes = [o for _, o in rows]
        scored.append(
            GroupScore(
                label=label,
                n=len(rows),
                share=len(rows) / total if total else 0.0,
                brier=brier(probs, outcomes, binning=binning),
                base_rate=sum(outcomes) / len(outcomes),
            )
        )

    scored.sort(key=lambda g: (-g.n, g.label))
    return Breakdown(
        key=key_name,
        groups=tuple(scored),
        too_small=dict(too_small),
        minimum_group=minimum_group,
        n_scored=sum(g.n for g in scored),
        n_total=total,
    )
