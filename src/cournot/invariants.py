"""Force every load-bearing field to carry a declared external invariant.

The forcing half of `docs/11` rule 3. Rule 3 has fired three times, and its fires
do not share one preventable shape:

- duplicate-instant prices — structurally preventable
- the Metaculus credential — partly (fetch the field the work consumes)
- Kalshi `resolve_ts` — **not preventable internally.** Nothing inside the record
  can know the field means scheduled expiry; only an outside fact does.

So this module does what construction *can* do: it makes **omission** a hard
failure. A source declares an invariant for each of its load-bearing fields, and
ingestion refuses to emit records if the declaration does not cover them.
"Nobody thought to check `resolve_ts`" becomes an error at import.

**It cannot make "checked the wrong thing" an error.** That residual stays
detection, and `docs/11` records it rather than papering over it. A declared
invariant that tests nothing will pass here and fail in production, which is why
the declaration carries a human-readable `external_anchor`: the reviewable claim
is *what outside fact this is checked against*, not that a check exists.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

__all__ = ["FieldInvariant", "InvariantRegistry", "UndeclaredFieldError"]


class UndeclaredFieldError(LookupError):
    """A load-bearing field has no declared invariant."""


@dataclass(frozen=True)
class FieldInvariant:
    """One field, one check, and the outside fact it is checked against."""

    field: str
    external_anchor: str
    """Plain English: what *outside the record* makes this checkable. Reviewable
    on its own — "precedes the snapshot's generation time" is a claim someone can
    disagree with; "is validated" is not."""

    check: Callable[[object], bool]

    def __post_init__(self) -> None:
        if not self.external_anchor.strip():
            raise ValueError(
                f"{self.field!r} needs an external anchor. A field checked against "
                "nothing outside itself is not validated, it is well-formed."
            )


class InvariantRegistry:
    """Declared invariants for one source."""

    def __init__(self, source: str, invariants: Iterable[FieldInvariant]) -> None:
        self.source = source
        self._by_field: dict[str, FieldInvariant] = {}
        for inv in invariants:
            if inv.field in self._by_field:
                raise ValueError(f"{inv.field!r} declared twice for {source!r}")
            self._by_field[inv.field] = inv

    def require(self, load_bearing: Iterable[str]) -> None:
        """Raise unless every load-bearing field has a declaration.

        Called at import or at the top of ingestion, so a source cannot ship with
        an unchecked field that a downstream decision keys on.
        """
        missing = sorted(set(load_bearing) - set(self._by_field))
        if missing:
            raise UndeclaredFieldError(
                f"{self.source!r} has load-bearing fields with no declared external "
                f"invariant: {missing}. Declare one, or argue in the decisions log why the "
                "field is not load-bearing."
            )

    def violations(self, record: Mapping[str, object]) -> dict[str, str]:
        """Fields whose declared check fails, mapped to their anchor."""
        out: dict[str, str] = {}
        for field, inv in self._by_field.items():
            if field in record and not inv.check(record[field]):
                out[field] = inv.external_anchor
        return out

    @property
    def declared(self) -> frozenset[str]:
        return frozenset(self._by_field)
