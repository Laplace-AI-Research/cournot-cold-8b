# SPDX-License-Identifier: Apache-2.0
"""Where the series comes from, and how it becomes evidence.

Two jobs kept apart on purpose.

**Loading** is pluggable because the production macro source and the source that
can be used today are not the same thing. `docs/13` names ALFRED — FRED's vintage
archive — as the correct macro source, because a FRED series read today gives the
*current* value of a past date, and revised macro data is look-ahead wearing a
timestamp. ALFRED requires an API key, and the standing guardrails forbid signing
up for anything, so the first end-to-end series is one that is never revised and
needs no credential. The interface is the same either way; the vintage question is
a property of the source, declared by the source.

**Bridging to evidence** is where the leakage guarantee is cashed. The series
becomes `EvidenceDoc`s and goes through `cournot.leakage.screen_bundle` — the same
detector every other evidence path uses. Writing a second truncation here would
have been shorter and would have created exactly the thing `CLAUDE.md` #1 exists
to prevent: two implementations of "before `as_of`", one of them untested.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from cournot.leakage import (
    DEFAULT_TRUST_REGISTRY,
    LeakageReport,
    SourceTrustRegistry,
    screen_bundle,
)
from cournot.synthetic.generator import ObservationPoint, SeriesQuestion
from cournot.types import EvidenceDoc, TimestampProvenance

__all__ = [
    "CsvSeriesSource",
    "RevisionPolicy",
    "SeriesSource",
    "SeriesSpec",
    "UnvintagedSeriesError",
    "load_csv_series",
    "series_evidence",
]


class RevisionPolicy(StrEnum):
    """Whether a source's past values can change after the fact.

    The single most important property of a series for this purpose, and the one
    a caller is most likely to assume rather than check.
    """

    NEVER_REVISED = "never_revised"
    """A closing price is what it was. Nothing to vintage."""

    VINTAGED = "vintaged"
    """Revised, but the source serves the as-of-date vintage (ALFRED). Safe."""

    REVISED_IN_PLACE = "revised_in_place"
    """Revised, and the source serves only the current value (FRED's plain API,
    most convenience wrappers). Using this is look-ahead: the number attached to
    2024-03 is what we know about 2024-03 *now*, not what was published then."""


class UnvintagedSeriesError(ValueError):
    """A `REVISED_IN_PLACE` series was offered as evidence.

    Refused rather than warned. This is the failure that looks like nothing:
    every timestamp is before `as_of`, the detector passes it, and the values are
    still contaminated — because the contamination is in the *number*, not the
    date. No timestamp check can catch it, so it is caught at the source.
    """


@dataclass(frozen=True)
class SeriesSpec:
    """A series and the facts about it that bear on leakage."""

    series_id: str
    source: str
    """Matched against the trust-tier registry, like any other evidence source."""

    revision_policy: RevisionPolicy
    units: str = ""
    description: str = ""


class SeriesSource(Protocol):
    """Anything that can produce a dated series."""

    def spec(self) -> SeriesSpec: ...

    def load(self) -> tuple[ObservationPoint, ...]: ...


def load_csv_series(
    path: Path | str, *, date_column: str = "Date", value_column: str = "Close"
) -> tuple[ObservationPoint, ...]:
    """Read a dated CSV into a sorted series.

    Rows with an unparseable date or a non-numeric value are dropped — but the
    count is returned nowhere, which would violate the "a dropped row is counted,
    not thrown" posture if this were an ingestion path. It is not: it is a loader
    for series whose format is known, and a malformed row is a loud failure.
    """
    out: list[ObservationPoint] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_date, raw_value = row.get(date_column), row.get(value_column)
            if not raw_date or not raw_value:
                raise ValueError(f"row missing {date_column!r}/{value_column!r}: {row!r}")
            stamp = datetime.fromisoformat(raw_date.strip())
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)  # CLAUDE.md: naive datetimes are a bug
            out.append(ObservationPoint(timestamp=stamp, value=float(raw_value)))
    if not out:
        raise ValueError(f"{path} produced no observations")
    return tuple(sorted(out, key=lambda p: p.timestamp))


@dataclass(frozen=True)
class CsvSeriesSource:
    """A series held on disk. The proving path: no credential, no network."""

    path: Path
    series_spec: SeriesSpec
    date_column: str = "Date"
    value_column: str = "Close"

    def spec(self) -> SeriesSpec:
        return self.series_spec

    def load(self) -> tuple[ObservationPoint, ...]:
        return load_csv_series(
            self.path, date_column=self.date_column, value_column=self.value_column
        )


def _docs(
    points: Iterable[ObservationPoint], spec: SeriesSpec, question: SeriesQuestion
) -> list[EvidenceDoc]:
    return [
        EvidenceDoc(
            doc_id=f"{spec.series_id}@{p.timestamp.date().isoformat()}",
            snippet=f"{spec.series_id} on {p.timestamp.date().isoformat()}: {p.value:g}"
            + (f" {spec.units}" if spec.units else ""),
            source=spec.source,
            timestamp=p.timestamp,
            # A closing print is publisher-issued and immutable. Claiming anything
            # weaker would be dishonest; claiming this for a REVISED_IN_PLACE
            # series would be the lie the guard above refuses.
            provenance=TimestampProvenance.PUBLISHER_VERIFIED,
        )
        for p in points
    ]


def series_evidence(
    question: SeriesQuestion,
    spec: SeriesSpec,
    *,
    registry: SourceTrustRegistry = DEFAULT_TRUST_REGISTRY,
    window: int | None = None,
) -> tuple[tuple[EvidenceDoc, ...], LeakageReport]:
    """The question's evidence bundle, screened by the real detector.

    Returns the report as well as the docs, and the caller is expected to look at
    it. A bundle that comes back short is not a smaller bundle — it is a
    generator that produced contaminated evidence, and the report is the only
    place that shows up.

    `window` trims to the last N observations. Trimming happens *before*
    screening, so the report's `n_input` is the number of docs actually offered
    and the kept/rejected counts mean what they say.
    """
    if spec.revision_policy is RevisionPolicy.REVISED_IN_PLACE:
        raise UnvintagedSeriesError(
            f"{spec.series_id} is served revised-in-place by {spec.source}: the value "
            f"attached to a past date is today's estimate of it, not the value known "
            f"at that date. Every timestamp would pass the detector and every number "
            f"would still be look-ahead. Use the vintage archive instead."
        )

    points: Sequence[ObservationPoint] = question.evidence
    if window is not None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        points = points[-window:]

    report = screen_bundle(_docs(points, spec, question), question.as_of, registry=registry)
    return report.kept, report
