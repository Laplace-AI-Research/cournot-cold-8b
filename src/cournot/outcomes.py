"""Counters that cannot conflate a transport failure with a content result.

Twice a rate limit has been recorded as a property of the thing being measured:
Mistral's 429s counted as format failures, then a throttled Wikipedia search
counted as "no article match" — the second written *after* the first was fixed,
because the fix was applied to a code path rather than made structural.

The construction rule (`docs/11`): **a transport outcome and a content outcome
must never share a counter.** This module enforces it in the type system rather
than asking anyone to remember it.

    tally = OutcomeTally()
    tally.transport("http_429")
    tally.content("genuinely_no_article")

    tally.total()            # refuses — "total of what?"
    tally.transport_total()  # 1
    tally.content_total()    # 1

Merging is defined only between like kinds. Crossing them requires
`reclassify()`, which is deliberately verbose and exists for the one legitimate
case: discovering that something you counted as transport was really content.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = ["OutcomeKind", "OutcomeTally", "TallyConflict"]


class OutcomeKind(StrEnum):
    """Why a call produced no usable content."""

    TRANSPORT = "transport"
    """The call did not complete: timeout, 429, 5xx, DNS, malformed body. Says
    nothing about the subject — only that we failed to ask."""

    CONTENT = "content"
    """The call completed and the answer was negative: no such article, no
    resolution recorded, outcome type unsupported. A fact about the subject."""


class TallyConflict(ValueError):
    """A label was used for both kinds. Almost always the conflation bug."""


@dataclass
class OutcomeTally:
    """Two counters that do not add up, on purpose."""

    _transport: Counter[str] = field(default_factory=Counter[str])
    _content: Counter[str] = field(default_factory=Counter[str])

    def transport(self, label: str, n: int = 1) -> None:
        if label in self._content:
            raise TallyConflict(
                f"{label!r} is already counted as a CONTENT outcome. A label that "
                "means both is the conflation this type exists to prevent; give "
                "the transport case its own name."
            )
        self._transport[label] += n

    def content(self, label: str, n: int = 1) -> None:
        if label in self._transport:
            raise TallyConflict(
                f"{label!r} is already counted as a TRANSPORT outcome. A label that "
                "means both is the conflation this type exists to prevent; give "
                "the content case its own name."
            )
        self._content[label] += n

    def total(self) -> int:
        """Deliberately unavailable.

        "How many failures were there" is the question that produced both bugs:
        it invites summing 287 throttled calls with 288 absent articles and
        reporting 575 of something. Ask for one kind or the other.
        """
        raise TypeError(
            "OutcomeTally has no total: transport and content outcomes do not sum. "
            "Use transport_total(), content_total(), or as_dict()."
        )

    def transport_total(self) -> int:
        return sum(self._transport.values())

    def content_total(self) -> int:
        return sum(self._content.values())

    def reclassify(self, label: str, to: OutcomeKind) -> None:
        """Move a label between kinds. Verbose on purpose.

        The legitimate case: discovering that something counted as transport was
        really content, or the reverse. Doing it by hand means the reclassifier
        has to say which way and why.
        """
        source = self._transport if to is OutcomeKind.CONTENT else self._content
        target = self._content if to is OutcomeKind.CONTENT else self._transport
        if label not in source:
            raise KeyError(f"{label!r} is not counted under the other kind")
        target[label] += source.pop(label)

    def merge(self, other: OutcomeTally) -> OutcomeTally:
        """Merge two tallies. Like kinds only — the separation survives."""
        merged = OutcomeTally()
        merged._transport = self._transport + other._transport
        merged._content = self._content + other._content
        overlap = set(merged._transport) & set(merged._content)
        if overlap:
            raise TallyConflict(f"labels used as both kinds across the merge: {sorted(overlap)}")
        return merged

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {"transport": dict(self._transport), "content": dict(self._content)}

    def summary(self) -> str:
        t = ", ".join(f"{k}={v}" for k, v in sorted(self._transport.items())) or "none"
        c = ", ".join(f"{k}={v}" for k, v in sorted(self._content.items())) or "none"
        return f"transport[{self.transport_total()}]: {t} | content[{self.content_total()}]: {c}"
