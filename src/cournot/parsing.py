# SPDX-License-Identifier: Apache-2.0
"""Parsing the rigid output format, and measuring compliance with it.

`CLAUDE.md` fixes the format and calls it non-negotiable:

    <reasoning>...</reasoning>
    <probability>0.37</probability>

This module is the format gate. Two jobs, deliberately kept together: extract the
probability, and say precisely *how* a non-compliant output failed. The second is
criterion 4 of `docs/02` and a `docs/05` anti-hacking measure — "format
degradation under RL, mitigated by strict parsing and format gates at every
checkpoint" — and neither is served by a boolean.

Strictness posture, matching the rest of the repo: a response that is *probably*
fine is not fine. `37%` is rejected rather than divided by 100, and a response
carrying two probabilities is rejected rather than resolved to the first. A
parser that repairs output hides exactly the degradation the gate exists to
catch, and the repair silently becomes load-bearing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "PROBABILITY_PLACES",
    "FormatFailure",
    "ParseResult",
    "compliance_rate",
    "parse_forecast",
    "render_forecast",
]

#: `CLAUDE.md` fixes the output as `0.XX`. Two places, not more: a renderer that
#: emitted three would produce output this module's own gate treats differently
#: from every previously scored run.
PROBABILITY_PLACES = 2

_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_PROBABILITY_RE = re.compile(r"<probability>(.*?)</probability>", re.DOTALL | re.IGNORECASE)
#: A bare decimal. Deliberately does not accept `%`, `1/3`, or `37 percent`.
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")
#: Conservative: only consulted when there is no probability tag at all, so a
#: compliant answer that happens to contain "I cannot" in its reasoning is
#: unaffected.
_REFUSAL_RE = re.compile(
    r"\bi\s*(?:'m|\u2019m|\s+am)?\s*"
    r"(?:cannot|can(?:'|\u2019)?t|can not|won(?:'|\u2019)?t|will not|"
    r"unable to|not able to|must decline|do not speculate|don(?:'|\u2019)?t speculate)"
    r"|\bas an ai\b",
    re.IGNORECASE,
)


class FormatFailure(StrEnum):
    """Why an output did not comply. Counted, so degradation has a shape."""

    MISSING_PROBABILITY = "missing_probability"
    REFUSAL = "refusal"
    """The model declined to forecast at all.

    A distinct failure from malformed output, and a real property of a candidate
    rather than a parsing accident: Llama-3.1-8B answers some forecasting prompts
    with "I cannot create content that could be used to predict the outcome of a
    future event." A base that refuses the task is differently unsuitable from
    one that answers in the wrong shape, and `docs/02` criterion 4 should not
    conflate them."""

    MISSING_REASONING = "missing_reasoning"
    MULTIPLE_PROBABILITIES = "multiple_probabilities"
    """Two or more probability tags. Resolving to the first would let a model
    hedge by emitting several and relying on the parser to choose."""

    EMPTY_PROBABILITY = "empty_probability"
    PERCENTAGE = "percentage"
    """`37%` or `37 percent`. `CLAUDE.md`: never percentages."""

    NOT_A_NUMBER = "not_a_number"
    OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class ParseResult:
    """Either a probability, or the reason there is not one."""

    probability: float | None
    reasoning: str | None
    failure: FormatFailure | None
    raw: str

    @property
    def compliant(self) -> bool:
        return self.failure is None

    def __post_init__(self) -> None:
        if (self.probability is None) == (self.failure is None):
            raise ValueError(
                "a parse result carries exactly one of a probability or a failure; "
                f"got probability={self.probability!r}, failure={self.failure!r}"
            )


def parse_forecast(text: str) -> ParseResult:
    """Parse one model response. Never raises; failures are values."""
    probabilities = _PROBABILITY_RE.findall(text)
    reasonings = _REASONING_RE.findall(text)

    def fail(reason: FormatFailure) -> ParseResult:
        return ParseResult(probability=None, reasoning=None, failure=reason, raw=text)

    if not probabilities:
        # Refusal is checked only here, where no probability exists at all.
        return fail(
            FormatFailure.REFUSAL if _REFUSAL_RE.search(text) else FormatFailure.MISSING_PROBABILITY
        )
    if len(probabilities) > 1:
        return fail(FormatFailure.MULTIPLE_PROBABILITIES)
    if not reasonings:
        return fail(FormatFailure.MISSING_REASONING)

    body = probabilities[0].strip()
    if not body:
        return fail(FormatFailure.EMPTY_PROBABILITY)
    if "%" in body or "percent" in body.lower():
        return fail(FormatFailure.PERCENTAGE)
    if not _NUMBER_RE.match(body):
        return fail(FormatFailure.NOT_A_NUMBER)

    value = float(body)
    if not (0.0 <= value <= 1.0):
        return fail(FormatFailure.OUT_OF_RANGE)

    return ParseResult(probability=value, reasoning=reasonings[0].strip(), failure=None, raw=text)


def compliance_rate(results: list[ParseResult]) -> float:
    """Criterion 4 of `docs/02`. Undefined on an empty set, so it refuses."""
    if not results:
        raise ValueError("no responses to score")
    return sum(1 for r in results if r.compliant) / len(results)


def render_forecast(reasoning: str, probability: float) -> str:
    """Render a forecast in the format `parse_forecast` accepts.

    Rendering lives beside parsing deliberately. `cournot.heads`-style callers --
    the scalar head in `scripts/scalar_train.py` is the first -- produce a float
    and no text, but every scorer in this repo reads a `<probability>` tag. If
    each such caller wrote its own string, the scalar arm would be scored by a
    second implementation of the contract, which is exactly the coupling
    `docs/11` rule 1 warns about.

    Clamped into `[0, 1]` before formatting: a sigmoid can return 1.0 - 1e-9,
    and a renderer that emitted `1` or `1.000` would be scored as a **format
    failure** rather than as a confident forecast, silently converting a
    calibration question into a compliance one.
    """
    bounded = min(1.0, max(0.0, probability))
    return (
        f"<reasoning>{reasoning}</reasoning>\n"
        f"<probability>{bounded:.{PROBABILITY_PLACES}f}</probability>"
    )
