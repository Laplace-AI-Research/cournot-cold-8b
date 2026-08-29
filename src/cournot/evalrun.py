# SPDX-License-Identifier: Apache-2.0
"""The eval runner: forecaster in, scoreable slice out.

    result = run_eval(forecaster, records, as_ofs=as_ofs, label="prior-v1")
    print(result.summary())
    metrics = iteration_metrics(result.scored_slice(minimum_coverage=0.9))

**The whole job is not losing track of what was not scored.** A forecaster that
fails on some questions and has the failures dropped is scored on a subset it
selected itself, and the direction is not neutral: models fail on the questions
they handle worst. Dropping 20% of an eval can improve every metric on it while
the model gets worse.

So there is no attribute holding a slice. `scored_slice` is a method that takes
a `minimum_coverage`, and a caller who wants to score 61% of an eval has to
write the number down.

**A crashed call and a malformed answer are different things** and are counted
separately, by `cournot.outcomes`, which exists because this repo counted Mistral's
rate limits as format failures once already. A 429 says nothing about a model's
compliance; a `<probability>37%</probability>` says a great deal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from cournot.outcomes import OutcomeTally
from cournot.parsing import FormatFailure, parse_forecast
from cournot.splits import EvalSlice
from cournot.types import QuestionRecord

__all__ = ["EvalRunResult", "InsufficientCoverageError", "run_eval"]

#: A forecaster takes a record and an `as_of` and returns raw model text.
Forecaster = Callable[[QuestionRecord, datetime], str]


class InsufficientCoverageError(RuntimeError):
    """Fewer questions were scoreable than the caller agreed to accept."""


@dataclass(frozen=True)
class EvalRunResult:
    """What a run produced, including everything it did not produce."""

    records: tuple[QuestionRecord, ...]
    probabilities: tuple[float, ...]
    as_ofs: tuple[datetime, ...]
    attempted: int
    tally: OutcomeTally
    format_failures: Counter[FormatFailure] = field(default_factory=Counter[FormatFailure])
    transport_failures: Counter[str] = field(default_factory=Counter[str])
    label: str = ""

    @property
    def scored(self) -> int:
        return len(self.records)

    @property
    def coverage(self) -> float:
        """Share of attempted questions that yielded a probability."""
        return self.scored / self.attempted if self.attempted else float("nan")

    @property
    def compliance(self) -> float:
        """Coverage among calls that actually completed.

        Distinct from `coverage`, and the distinction is the point: a run that
        was rate-limited on half its questions has poor coverage and says nothing
        about compliance.
        """
        completed = self.scored + sum(self.format_failures.values())
        return self.scored / completed if completed else float("nan")

    def scored_slice(self, *, minimum_coverage: float) -> EvalSlice:
        """The slice, if coverage clears the floor the caller names.

        There is deliberately no default. Whoever scores a partial eval states
        what partial they accepted, and it appears in their code rather than in
        a footnote nobody reads.
        """
        if not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError(f"minimum_coverage must be in [0,1], got {minimum_coverage}")
        if self.scored == 0:
            raise InsufficientCoverageError(
                f"{self.label or 'run'}: nothing was scoreable out of {self.attempted} "
                f"attempted. Failures: {self.summary_failures()}"
            )
        if self.coverage < minimum_coverage:
            raise InsufficientCoverageError(
                f"{self.label or 'run'}: scored {self.scored}/{self.attempted} = "
                f"{self.coverage:.1%}, below the {minimum_coverage:.1%} floor. The "
                "unscored questions are not a random sample — models fail on what "
                f"they handle worst. Failures: {self.summary_failures()}"
            )
        return EvalSlice(
            records=self.records,
            probabilities=self.probabilities,
            as_ofs=self.as_ofs,
            label=self.label,
        )

    def summary_failures(self) -> str:
        parts = [f"{k.value}={v}" for k, v in sorted(self.format_failures.items())]
        parts += [f"{k}={v}" for k, v in sorted(self.transport_failures.items())]
        return ", ".join(parts) or "none"

    def summary(self) -> str:
        return (
            f"[{self.label or 'unlabelled'}] scored {self.scored}/{self.attempted} "
            f"(coverage {self.coverage:.1%}, compliance {self.compliance:.1%}) — "
            f"{self.summary_failures()}"
        )


def run_eval(
    forecaster: Forecaster,
    records: Sequence[QuestionRecord],
    *,
    as_ofs: Sequence[datetime],
    label: str = "",
) -> EvalRunResult:
    """Run `forecaster` over `records`, keeping every outcome.

    `as_ofs` is required rather than defaulted. `CLAUDE.md` says `as_of` has no
    default and no `None`, and the per-horizon breakdown a model card needs
    cannot be reconstructed after the run.
    """
    if len(as_ofs) != len(records):
        raise ValueError(f"{len(records)} records but {len(as_ofs)} as_ofs")

    kept_records: list[QuestionRecord] = []
    kept_probabilities: list[float] = []
    kept_as_ofs: list[datetime] = []
    tally = OutcomeTally()
    format_failures: Counter[FormatFailure] = Counter()
    transport_failures: Counter[str] = Counter()

    for record, as_of in zip(records, as_ofs, strict=True):
        try:
            text = forecaster(record, as_of)
        except Exception as exc:  # a forecaster is arbitrary caller code
            label_ = f"transport_{type(exc).__name__}"
            transport_failures[label_] += 1
            tally.transport(label_)
            continue
        parsed = parse_forecast(text)
        if parsed.probability is None:
            assert parsed.failure is not None  # ParseResult enforces the exclusivity
            format_failures[parsed.failure] += 1
            tally.content(parsed.failure.value)
            continue
        tally.content("scored")
        kept_records.append(record)
        kept_probabilities.append(parsed.probability)
        kept_as_ofs.append(as_of)

    return EvalRunResult(
        records=tuple(kept_records),
        probabilities=tuple(kept_probabilities),
        as_ofs=tuple(kept_as_ofs),
        attempted=len(records),
        tally=tally,
        format_failures=format_failures,
        transport_failures=transport_failures,
        label=label,
    )
