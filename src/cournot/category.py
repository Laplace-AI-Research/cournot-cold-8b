"""A title-only category classifier, and the honesty it requires.

    model = NaiveBayesCategoriser.fit(examples)
    model.predict("Will the Fed cut rates in June?")

Multinomial naive Bayes over lowercased word tokens, pure Python and no
dependencies, matching `cournot.metrics` — these are corpus-sized fits, not a
training run, and a readable model is worth more than speed. It is also the
right *shape* of model for this job: the point is a measured accuracy number for
`docs/12`, and a baseline whose failures are legible beats a stronger one whose
failures are not.

**Two things this cannot do, recorded because both are load-bearing.**

It cannot predict a class it never saw. Manifold's `*-default` slugs are the only
free labels, and they carry none of the "other" and "unclear" markets that make
up **26.0%** of the strata those slugs do not cover. A model fitted on them will
assign one of seven categories to a lottery, confidently.

And it inherits its training distribution. The free labels are 5.8% culture; the
hand-labelled unlabelled strata are 27.7% (2026-08-20). A prior fitted on the
former is wrong about the latter by a factor of nearly five, and naive Bayes
leans on its prior.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["NaiveBayesCategoriser", "Prediction", "tokenize"]

_TOKEN = re.compile(r"[a-z0-9']+")
#: Tokens carrying no category signal. Deliberately short: an aggressive list is
#: a modelling decision smuggled in as preprocessing, and this is a baseline.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "will",
        "be",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "and",
        "or",
        "if",
        "then",
        "than",
        "that",
        "this",
        "it",
        "its",
        "as",
        "from",
        "any",
        "before",
        "after",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, stopwords removed, order discarded."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Prediction:
    """A label and the model's confidence, which is reported rather than hidden.

    A naive-Bayes posterior is badly calibrated by construction — the
    independence assumption compounds evidence that is not independent — so this
    is a ranking signal, not a probability. Named `score` for that reason.
    """

    label: str
    score: float
    runner_up: str | None


@dataclass(frozen=True)
class NaiveBayesCategoriser:
    """Multinomial naive Bayes with add-one smoothing."""

    log_prior: dict[str, float]
    log_likelihood: dict[str, dict[str, float]]
    vocabulary: frozenset[str]
    n_fitted: int

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.log_prior))

    @classmethod
    def fit(cls, examples: Iterable[tuple[str, str]]) -> NaiveBayesCategoriser:
        """Fit on `(text, label)` pairs.

        A market carrying two `*-default` slugs appears once per label rather
        than being collapsed to the first: the source assigns both, and picking
        one would invent a single-label ground truth it does not supply.
        """
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        totals: Counter[str] = Counter()
        documents: Counter[str] = Counter()
        vocabulary: set[str] = set()
        n = 0
        for text, label in examples:
            tokens = tokenize(text)
            documents[label] += 1
            n += 1
            for token in tokens:
                counts[label][token] += 1
                totals[label] += 1
                vocabulary.add(token)
        if not documents:
            raise ValueError("no examples to fit on")

        vocab_size = len(vocabulary) or 1
        log_prior = {label: math.log(count / n) for label, count in documents.items()}
        log_likelihood: dict[str, dict[str, float]] = {}
        for label in documents:
            denominator = totals[label] + vocab_size
            log_likelihood[label] = {
                token: math.log((counts[label][token] + 1) / denominator) for token in vocabulary
            }
        return cls(
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            vocabulary=frozenset(vocabulary),
            n_fitted=n,
        )

    def scores(self, text: str) -> dict[str, float]:
        """Log posterior per label, up to a constant.

        **Out-of-vocabulary tokens are dropped, not smoothed.** A token unseen in
        training contributes `log(1 / (tokens_c + |V|))` to every class, and those
        denominators differ, so smoothing it in would tilt the result toward
        whichever class had the fewest training tokens. That is a document-length
        artifact rather than evidence about this title, and it is strongest
        exactly where the model knows least. Dropping is the honest default; a
        title of entirely unseen words falls back to the prior, which is the
        correct answer when there is no evidence.
        """
        tokens = [t for t in tokenize(text) if t in self.vocabulary]
        return {
            label: self.log_prior[label] + sum(self.log_likelihood[label][t] for t in tokens)
            for label in self.log_prior
        }

    def predict(self, text: str) -> Prediction:
        """The best label. Ties break alphabetically so the output is stable."""
        ranked = sorted(self.scores(text).items(), key=lambda kv: (-kv[1], kv[0]))
        best, best_score = ranked[0]
        return Prediction(
            label=best,
            score=best_score,
            runner_up=ranked[1][0] if len(ranked) > 1 else None,
        )

    def accuracy(self, examples: Sequence[tuple[str, str]]) -> float:
        """Share of `examples` whose true label the model predicts."""
        if not examples:
            return float("nan")
        return sum(1 for text, label in examples if self.predict(text).label == label) / len(
            examples
        )
