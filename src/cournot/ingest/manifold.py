# SPDX-License-Identifier: Apache-2.0
"""Manifold ingestion: normalized corpus rows to `QuestionRecord`, every drop counted.

Source choice is argued in the internal decisions log (2026-08-14). Short version:
Metaculus carries no outcomes in the available snapshot, Polymarket has no
`created_ts`, and Kalshi's `resolve_ts` is not an actual settlement time — 39% of
its "resolved" rows carry a `resolve_ts` after the snapshot was generated, which
a settled market cannot have. Manifold is the one source whose `resolve_ts` is
internally consistent, and it is the field the entire temporal split keys on.

Input is a row of the shared normalized schema in `~/Laplace/data` (see its
`DATASET.md`): `platform`, `platform_market_id`, `title`, `description`,
`category`, `created_ts`, `close_ts`, `resolve_ts`, `status`, `outcome_type`,
`resolution`, `resolution_prob`. Timestamps are UTC epoch milliseconds.

**Posture, from `docs/01`: a record that does not make it through is counted and
categorized, never silently dropped.** `normalize` returns either a
`QuestionRecord` or a `Rejected` carrying a reason — there is no path that
returns nothing, and `IngestReport.check_accounting` asserts the two sides sum to
the input. The rejection breakdown is as much the output as the records are.

This module does no IO. Reading parquet belongs to the caller
(`scripts/ingest_manifold.py`), so the normalizer is tested without a corpus and
a report can be regenerated from a saved snapshot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from cournot.types import QuestionRecord, SoftTargetProvenance

__all__ = [
    "SOURCE",
    "UNASSIGNED_BASE_RATE_CLASS",
    "UNCLASSIFIED_CATEGORY",
    "IngestRejection",
    "IngestReport",
    "Rejected",
    "normalize",
    "normalize_all",
]

SOURCE = "manifold"

#: Fields the source does not supply. Explicit sentinels rather than invented
#: values — and an open question in the internal decisions log, because a sentinel makes "not yet
#: assigned" indistinguishable from a class genuinely named "unassigned".
UNCLASSIFIED_CATEGORY = "unclassified"
UNASSIGNED_BASE_RATE_CLASS = "unassigned"


class IngestRejection(StrEnum):
    """Why a raw row did not become a `QuestionRecord`. All of these are counted."""

    WRONG_PLATFORM = "wrong_platform"
    NOT_BINARY = "not_binary"
    """`outcome_type` is not binary — multiple choice, numeric, poll."""

    NOT_RESOLVED = "not_resolved"
    """Still open or closed-but-unresolved. Not an error; not a record yet."""

    MISSING_SOFT_TARGET = "missing_soft_target"
    """Resolved MKT but with no `resolution_prob`, so the probability it resolved
    to was not recorded. Resolved to something unknown is not a target."""

    ANNULLED = "annulled"
    """Resolved CANCEL: resolution withdrawn, stakes returned. Resolved, again
    with no binary outcome."""

    UNKNOWN_RESOLUTION = "unknown_resolution"
    """A resolution value this normalizer does not recognize. Loud on purpose: a
    resolution state nobody has looked at should stop a record, not be guessed."""

    MISSING_FIELD = "missing_field"
    BAD_TIMESTAMP = "bad_timestamp"
    SCHEMA_REJECTED = "schema_rejected"
    """`QuestionRecord` validation refused it; the detail says which rule."""

    DUPLICATE_ID = "duplicate_id"


@dataclass(frozen=True)
class Rejected:
    raw_id: str
    reason: IngestRejection
    detail: str


@dataclass(frozen=True)
class IngestReport:
    """Records that made it, and an accounting of everything that did not."""

    records: tuple[QuestionRecord, ...]
    rejected: tuple[Rejected, ...]
    n_raw: int
    source: str = SOURCE

    @property
    def rejected_by_reason(self) -> Mapping[IngestRejection, int]:
        return dict(Counter(r.reason for r in self.rejected))

    @property
    def n_kept(self) -> int:
        return len(self.records)

    def examples(self, reason: IngestRejection, limit: int = 3) -> tuple[Rejected, ...]:
        """A few concrete rejections of one kind, for reading the report."""
        return tuple(r for r in self.rejected if r.reason is reason)[:limit]

    def check_accounting(self) -> None:
        """Every raw row is either kept or counted. Called by `normalize_all`."""
        if self.n_kept + len(self.rejected) != self.n_raw:
            raise AssertionError(
                f"ingest lost rows: {self.n_raw} raw, {self.n_kept} kept, "
                f"{len(self.rejected)} rejected"
            )

    def summary(self) -> str:
        lines = [f"[{self.source}] {self.n_kept}/{self.n_raw} rows became QuestionRecords"]
        for reason, count in sorted(
            self.rejected_by_reason.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            share = count / self.n_raw if self.n_raw else 0.0
            lines.append(f"  {count:>9,} ({share:6.2%})  {reason.value}")
        return "\n".join(lines)


def _epoch_ms_to_utc(value: Any, field: str) -> datetime:
    """Corpus timestamps are UTC epoch milliseconds.

    Worth stating because it is the opposite of what was expected of ingestion:
    there is no zone or format ambiguity to get wrong here at all. An epoch is an
    instant by construction. The "mixed formats and zones" class of bug belongs
    to sources that publish local wall-clock strings; it does not arise here, and
    the real damage in this corpus turned out to be a field that is populated,
    well-typed, and means something other than its name (see the module
    docstring on Kalshi).
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} is not a number: {value!r}")
    if value <= 0:
        raise ValueError(f"{field} is not a positive epoch: {value!r}")
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{field} is not a representable epoch: {value!r} ({exc})") from exc


def normalize(row: Mapping[str, Any]) -> QuestionRecord | Rejected:
    """One corpus row to a record, or to a counted rejection. Never `None`."""
    raw_id = str(row.get("platform_market_id", "<no id>"))

    if row.get("platform") != SOURCE:
        return Rejected(raw_id, IngestRejection.WRONG_PLATFORM, f"platform={row.get('platform')!r}")

    if row.get("outcome_type") != "binary":
        return Rejected(
            raw_id, IngestRejection.NOT_BINARY, f"outcome_type={row.get('outcome_type')!r}"
        )

    if row.get("status") != "resolved":
        return Rejected(raw_id, IngestRejection.NOT_RESOLVED, f"status={row.get('status')!r}")

    resolution = row.get("resolution")
    if resolution == "CANCEL":
        return Rejected(raw_id, IngestRejection.ANNULLED, "resolution withdrawn (CANCEL)")
    if resolution not in ("YES", "NO", "MKT"):
        return Rejected(raw_id, IngestRejection.UNKNOWN_RESOLUTION, f"resolution={resolution!r}")

    soft_target = row.get("resolution_prob") if resolution == "MKT" else None
    if resolution == "MKT" and soft_target is None:
        return Rejected(
            raw_id,
            IngestRejection.MISSING_SOFT_TARGET,
            "resolution is MKT but resolution_prob is null",
        )

    title = row.get("title")
    if not title:
        return Rejected(raw_id, IngestRejection.MISSING_FIELD, "title is empty")

    times: dict[str, datetime] = {}
    for field, key in (
        ("open_date", "created_ts"),
        ("scheduled_resolve_date", "close_ts"),
        ("resolved_at", "resolve_ts"),
    ):
        value = row.get(key)
        if value is None:
            return Rejected(raw_id, IngestRejection.MISSING_FIELD, f"{key} is absent")
        try:
            times[field] = _epoch_ms_to_utc(value, key)
        except ValueError as exc:
            return Rejected(raw_id, IngestRejection.BAD_TIMESTAMP, str(exc))

    # A MKT resolution is a crowd-derived probability observed at the resolution
    # instant. It is supervision (`docs/05`), just not a binary outcome — so it
    # goes in the soft-target fields and `outcome`/`resolved_at` stay unset.
    is_binary = resolution in ("YES", "NO")
    try:
        return QuestionRecord(
            question_id=f"{SOURCE}:{raw_id}",
            text=title,
            resolution_criteria=row.get("description") or "",
            open_date=times["open_date"],
            scheduled_resolve_date=times["scheduled_resolve_date"],
            resolved_at=times["resolved_at"] if is_binary else None,
            outcome=(1 if resolution == "YES" else 0) if is_binary else None,
            soft_target=None if is_binary else soft_target,
            soft_target_provenance=None if is_binary else SoftTargetProvenance.CROWD_DERIVED,
            soft_target_at=None if is_binary else times["resolved_at"],
            category=row.get("category") or UNCLASSIFIED_CATEGORY,
            base_rate_class=UNASSIGNED_BASE_RATE_CLASS,
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(p) for p in first["loc"]) or "<model>"
        return Rejected(raw_id, IngestRejection.SCHEMA_REJECTED, f"{location}: {first['msg']}")


def normalize_all(rows: Sequence[Mapping[str, Any]]) -> IngestReport:
    """Normalize a batch, deduplicating by question id."""
    records: list[QuestionRecord] = []
    rejected: list[Rejected] = []
    seen: set[str] = set()

    for row in rows:
        result = normalize(row)
        if isinstance(result, Rejected):
            rejected.append(result)
            continue
        if result.question_id in seen:
            rejected.append(
                Rejected(result.question_id, IngestRejection.DUPLICATE_ID, "already ingested")
            )
            continue
        seen.add(result.question_id)
        records.append(result)

    report = IngestReport(records=tuple(records), rejected=tuple(rejected), n_raw=len(rows))
    report.check_accounting()
    return report
