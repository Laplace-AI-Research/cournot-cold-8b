# SPDX-License-Identifier: Apache-2.0
"""The `docs/07` baselines. "Report against all four, always."

| baseline | here |
|---|---|
| market price at `as_of` | `market_price` |
| category base rate | `BaseRateForecaster` |
| unmodified base model | a model run — `scripts/bakeoff_run.py` |
| frontier model, zero-shot | a model run — `scripts/bakeoff_api.py` |

Only the first two are pure functions of the corpus; the other two are inference
and already have runners. What is here is the half that can be wrong silently.

**Both are leakage surfaces, and that is the whole design problem.** A market
price is a forecast made by someone else at a moment in time, and reading it at
the wrong moment reads the answer. A base rate is fitted on other questions, and
fitting it on the questions being scored is the same mistake wearing a different
hat. `CLAUDE.md` #1 governs both.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from cournot.splits import DEFAULT_SPLIT, TemporalSplit
from cournot.types import QuestionRecord

__all__ = [
    "ALWAYS_HALF_BRIER",
    "RANDOM_UNIFORM_BRIER",
    "BaseRateForecaster",
    "NotTrainableError",
    "UnfittableBaseRateError",
    "always_half",
    "market_price",
    "random_uniform",
]

#: Brier of forecasting 0.5 on everything. `(0.5 - y)^2 = 0.25` for y in {0, 1},
#: so it is exactly 0.25 whatever the base rate — which is what makes it an
#: anchor rather than a measurement.
ALWAYS_HALF_BRIER = 0.25

#: Brier of an independent Uniform(0,1) forecast. For an outcome y,
#: `E[(U - y)^2] = E[U^2] - 2y E[U] + y^2 = 1/3 - y + y^2`, which is **1/3 for
#: both y = 0 and y = 1**. Also independent of the base rate.
RANDOM_UNIFORM_BRIER = 1.0 / 3.0


class NotTrainableError(ValueError):
    """A base rate was about to be fitted on data it will later be scored on."""


class UnfittableBaseRateError(ValueError):
    """Nothing resolved to fit on."""


def market_price(record: QuestionRecord, as_of: datetime) -> float | None:
    """The last quoted price at or before `as_of`, or None if there is none.

    **Strictly at or before.** The series is time-ordered, so the temptation is
    the nearest point in either direction; the nearest point is frequently in the
    future, and a market price from after `as_of` is the crowd's answer *after*
    it learned something the forecaster had not. That is not a hard baseline, it
    is the outcome leaking through a price.

    Returns None rather than a default when the series opens after `as_of`. A
    question with no price yet has no market baseline, and substituting 0.5 would
    quietly add a hedged forecast to the baseline's score and flatter it.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware (CLAUDE.md)")
    latest: float | None = None
    for point in record.price_series:
        if point.timestamp > as_of:
            break
        latest = point.price
    return latest


@dataclass(frozen=True)
class BaseRateForecaster:
    """P(YES) by reference class, fitted on questions outside the eval slice.

    Catches "the model learned nothing but the prior" (`docs/07`). For that to
    mean anything the prior has to be a real one — fitted on other questions,
    never on the questions being scored.
    """

    by_class: Mapping[str, float]
    overall: float
    minimum_count: int
    n_fitted: int

    @classmethod
    def fit(
        cls,
        records: Iterable[QuestionRecord],
        *,
        minimum_count: int = 30,
        temporal_split: TemporalSplit = DEFAULT_SPLIT,
    ) -> BaseRateForecaster:
        """Fit on resolved, *trainable* records.

        Refuses any record whose split is not trainable. Fitting on `dev` or
        `published` and then scoring there is peeking, and it would show up as a
        baseline that is mysteriously hard to beat — which reads as a strong
        market rather than as a bug.

        `minimum_count` exists because a class holding three questions produces a
        base rate of 0.0 or 1.0, and a baseline that predicts certainty from three
        observations is not a prior, it is noise with an extreme value. Classes
        below the floor fall back to the overall rate. Shrinkage would be the
        better estimator; a hard floor is the one whose behaviour is obvious at a
        glance, and this is a baseline rather than the product.
        """
        yes: Counter[str] = Counter()
        seen: Counter[str] = Counter()
        total_yes = total = 0
        for record in records:
            if record.outcome is None:
                continue
            split = temporal_split.assign(record)
            if not split.trainable:
                raise NotTrainableError(
                    f"{record.question_id} is in split {split.value!r}, which is not "
                    "trainable. Fitting a base-rate baseline on the slice it will be "
                    "scored on is peeking, and it looks like a strong baseline."
                )
            key = record.base_rate_class
            seen[key] += 1
            yes[key] += record.outcome
            total += 1
            total_yes += record.outcome
        if total == 0:
            raise UnfittableBaseRateError("no resolved trainable records to fit a base rate on")
        overall = total_yes / total
        by_class = {key: yes[key] / count for key, count in seen.items() if count >= minimum_count}
        return cls(by_class=by_class, overall=overall, minimum_count=minimum_count, n_fitted=total)

    def __call__(self, record: QuestionRecord, as_of: datetime | None = None) -> float:
        """The class rate, or the overall rate for an unseen or thin class.

        `as_of` is accepted and ignored: a base rate does not move with time, and
        taking the argument keeps the baseline interchangeable with the others.
        """
        return self.by_class.get(record.base_rate_class, self.overall)

    def covered(self, records: Sequence[QuestionRecord]) -> float:
        """Share of `records` whose class had enough support to get its own rate.

        Worth reporting: a base-rate baseline that fell back to the overall rate
        for 95% of the eval is a global base rate wearing a category label, and
        the two answer different questions.
        """
        if not records:
            return float("nan")
        hits = sum(1 for r in records if r.base_rate_class in self.by_class)
        return hits / len(records)


def always_half(_record: QuestionRecord, _as_of: datetime) -> float:
    """Forecast 0.5 on everything. Scores exactly `ALWAYS_HALF_BRIER`.

    The field's usual anchor — ForecastBench rescales its whole leaderboard so
    this equals 0.25, and most papers state it in prose. Cheap to report and it
    makes every other number legible without the reader computing anything.
    """
    return 0.5


def random_uniform(seed: int) -> Callable[[QuestionRecord, datetime], float]:
    """An independent Uniform(0,1) forecast per question. Scores exactly 1/3.

    The baseline `CLAUDE.md` #5 was missing, and the only one that separates the
    two failures it names. A model collapsed onto 0.5 and a model with a wide but
    useless spread are both "no skill", and `always_half` cannot tell them apart:
    the collapsed model scores 0.25 and *beats* it, which reads as competence.

    Against both anchors the diagnosis is immediate:

    | score | reading |
    |---|---|
    | ~0.25 with a narrow histogram | collapsed onto the middle |
    | ~1/3 with a wide histogram | spread, no information |
    | below `p(1-p)` | genuinely better than the base rate |

    Seeded and required, so a reported number is reproducible; an unseeded
    baseline would move between runs and make two reports of one result disagree.
    """
    rng = random.Random(seed)

    def forecaster(_record: QuestionRecord, _as_of: datetime) -> float:
        return rng.random()

    return forecaster
