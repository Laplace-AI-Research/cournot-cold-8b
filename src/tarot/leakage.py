# SPDX-License-Identifier: Apache-2.0
"""The leakage detector.

`docs/01`: "The single most important component in the repo. Build and test it
first." Everything downstream trusts this module, so it is written to be read,
not to be clever.

Failure posture, from `docs/01` and restated because it drives every default
below: **when in doubt, drop the evidence doc.** Recall on evidence matters far
less than precision on timestamps. A doc we wrongly drop costs us a little
signal; a doc we wrongly keep can invalidate every number we publish.

The four checks from `docs/01`:

  1. Hard timestamp filter        `_check_timestamp`
  2. Source-level trust tiers     `_check_source`
  3. Outcome-string scan          `scan_for_outcome_strings` (matcher built;
                                  fuzzy threshold tau deliberately unset)
  4. Adversarial probe set        `run_adversarial_probe` + `null_threshold`
                                  (machinery built; the materiality threshold is
                                  derived from a null run, not chosen a priori)

plus timestamp-integrity checks that make (1) meaningful (provenance,
post-publication revision, retrieval ordering) and `check_forecast_time`, which
bounds `as_of` by a question's *actual* resolution time.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from enum import IntEnum, StrEnum
from typing import Protocol

from tarot.types import (
    EvidenceDoc,
    ForecastRequest,
    ForecastResponse,
    Outcome,
    QuestionRecord,
    TimestampProvenance,
)

__all__ = [
    "LEAKAGE_DETECTOR_VERSION",
    "REVISION_PIN_RE",
    "DocVerdict",
    "Exemption",
    "LeakagePolicy",
    "LeakageReport",
    "LeakageStats",
    "NullThresholdMethod",
    "OutcomeScanConfig",
    "OutcomeScanResult",
    "ProbeCase",
    "ProbeCaseResult",
    "ProbeReport",
    "ProbeVerdict",
    "RejectionReason",
    "ScanStrictness",
    "SourceTrustRegistry",
    "TrustTier",
    "check_forecast_time",
    "compare_to_null",
    "filter_evidence",
    "null_threshold",
    "run_adversarial_probe",
    "scan_for_outcome_strings",
    "screen_bundle",
    "screen_doc",
]

#: Recorded in every dataset manifest and every eval manifest (`docs/01`,
#: `docs/07`). Bump on ANY change to accept/reject semantics — including a change
#: to the default policy or the default trust registry. Two runs with different
#: values here are not comparable.
LEAKAGE_DETECTOR_VERSION = "0.4.0"


# --------------------------------------------------------------------------
# Trust tiers
# --------------------------------------------------------------------------


class TrustTier(IntEnum):
    """Source trust tiers (`docs/01`). Lower is more trustworthy."""

    #: Reliable immutable timestamps: SEC filings, central bank releases, wire
    #: services with versioned archives.
    TIER_1 = 1
    #: Ordinary published sources: timestamps usually right, not guaranteed
    #: immutable. Admissible only with explicit timestamp provenance.
    TIER_2 = 2
    #: Continuously revised (wikis, docs pages, live-updating trackers).
    #: Excluded from evidence bundles by default.
    TIER_3 = 3


class SourceTrustRegistry:
    """Maps a doc's `source` string to a trust tier.

    Unknown sources are NOT silently trusted. `tier_for` returns `None` for a
    source it has never heard of and the policy decides what that means; the
    default policy treats it as tier 3.
    """

    def __init__(self, tiers: Mapping[str, TrustTier]) -> None:
        self._tiers = {self.normalize_source(k): v for k, v in tiers.items()}

    @staticmethod
    def normalize_source(source: str) -> str:
        """Public because the revision-pin check needs the same normalization,
        and two spellings of "same source" is how a guard gets bypassed."""
        return source.strip().lower()

    def tier_for(self, source: str) -> TrustTier | None:
        return self._tiers.get(self.normalize_source(source))

    def with_sources(self, tiers: Mapping[str, TrustTier]) -> SourceTrustRegistry:
        """Return a copy with `tiers` added or overridden."""
        return SourceTrustRegistry({**self._tiers, **dict(tiers)})

    def __len__(self) -> int:
        return len(self._tiers)


# Ratified 2026-08-14 (the internal decisions log). No longer provisional for the
# sources listed here; adding a source is a decision-log entry, and any change to
# this mapping bumps LEAKAGE_DETECTOR_VERSION.
_DEFAULT_TIERS: dict[str, TrustTier] = {
    # Tier 1 — immutable, publisher-stamped, versioned archives.
    "sec.gov": TrustTier.TIER_1,  # EDGAR: acceptance datetime, amendments filed as new documents
    "federalreserve.gov": TrustTier.TIER_1,
    "ecb.europa.eu": TrustTier.TIER_1,
    "bls.gov": TrustTier.TIER_1,
    "bea.gov": TrustTier.TIER_1,
    "fred": TrustTier.TIER_1,
    "reuters": TrustTier.TIER_1,
    "apnews": TrustTier.TIER_1,
    # Ratified 2026-08-20. An exchange print is publisher-issued, immutable and
    # exactly dated — the properties that put sec.gov at tier 1, and stronger
    # than a news wire. Scoped to SETTLED closes: the vintage risk lives on
    # RevisionPolicy, which refuses a revised-in-place series at any tier, so
    # tier 1 here does not import it.
    "coinbase": TrustTier.TIER_1,
    # Tier 1 ONLY as pinned revisions. See `LeakagePolicy.revision_pinned_sources`:
    # a page is tier 3 and is rejected; a revision is immutable and exactly dated.
    "wikipedia": TrustTier.TIER_1,
    "wikinews": TrustTier.TIER_1,
}

#: A URL that pins a specific wiki revision rather than the live page.
REVISION_PIN_RE = re.compile(r"[?&]oldid=\d+", re.IGNORECASE)

DEFAULT_TRUST_REGISTRY = SourceTrustRegistry(_DEFAULT_TIERS)


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakagePolicy:
    """Every knob the detector has. Defaults implement the `docs/01` posture."""

    max_trust_tier: TrustTier = TrustTier.TIER_2
    """Tiers strictly worse than this are excluded. Default excludes tier 3."""

    unknown_source_tier: TrustTier = TrustTier.TIER_3
    """Tier assigned to a source not in the registry. Default: treat an unknown
    source as continuously-revised, i.e. exclude it. An unknown source is a
    source whose timestamp behaviour nobody has checked."""

    disqualifying_provenance: frozenset[TimestampProvenance] = frozenset(
        {TimestampProvenance.CRAWL, TimestampProvenance.INFERRED}
    )
    """Provenances that are never acceptable at any tier. A crawl date is not a
    publication date, and an inferred date is a guess — `docs/01` names both."""

    tiers_allowing_unknown_provenance: frozenset[TrustTier] = frozenset({TrustTier.TIER_1})
    """UNKNOWN provenance is ambiguous, and ambiguity means drop — except at
    tier 1, where membership in the tier *is* the assertion that the source's
    timestamps are reliable and immutable (`docs/01`).

    This is an exception that could quietly become the main path, so every doc
    admitted through it is tagged `Exemption.TIER1_UNKNOWN_PROVENANCE` and
    counted; see `LeakageStats.tier1_unknown_provenance_share`. If that share is
    large, the source tier is doing all the work and per-doc provenance is
    decorative — which is a thing to know before publishing an eval number, not
    after someone asks."""

    revision_pinned_sources: frozenset[str] = frozenset({"wikipedia", "wikinews"})
    """Sources that are only admissible as a pinned revision, never as a page.

    A live wiki page is continuously revised and is tier 3 by `docs/01`. A
    *specific revision* of it is immutable by construction and carries an exact
    timestamp, which is tier 1. The distinction is entirely in the URL, so it is
    enforced on the URL: a doc from one of these sources whose `url` does not pin
    a revision is rejected outright, not demoted.

    Rejected rather than demoted deliberately. Demotion would leave a
    convenience fetcher that drops page URLs in silently producing tier-3 docs
    that get filtered somewhere else for some other reason; rejection names what
    went wrong at the point it went wrong.
    """

    reject_if_modified_at_or_after_as_of: bool = True
    """Drop a doc whose `last_modified` is not strictly before `as_of`, even if
    its publication timestamp is. Covers silent post-publication updates."""

    reject_if_timestamp_after_retrieval: bool = True
    """Drop a doc claiming to be published after we fetched it. That record is
    internally inconsistent, so neither field can be trusted."""


DEFAULT_POLICY = LeakagePolicy()


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


class RejectionReason(StrEnum):
    """Why a doc was dropped. Counted per-reason into the dataset manifest."""

    MISSING_TIMESTAMP = "missing_timestamp"
    TIMESTAMP_NOT_BEFORE_AS_OF = "timestamp_not_before_as_of"
    AMBIGUOUS_PROVENANCE = "ambiguous_provenance"
    MODIFIED_AT_OR_AFTER_AS_OF = "modified_at_or_after_as_of"
    TIMESTAMP_AFTER_RETRIEVAL = "timestamp_after_retrieval"
    UNKNOWN_SOURCE = "unknown_source"
    UNTRUSTED_SOURCE_TIER = "untrusted_source_tier"
    UNPINNED_REVISION = "unpinned_revision"
    """A revision-addressed source cited by page rather than by revision."""
    OUTCOME_STRING_MATCH = "outcome_string_match"
    #: Not a doc-level reason — see `check_forecast_time`.
    AS_OF_AFTER_RESOLUTION = "as_of_after_resolution"


class Exemption(StrEnum):
    """A rule a doc was admitted *despite*, recorded so exceptions stay visible.

    An exemption is not a warning about a particular doc — each one is a case the
    policy deliberately allows. It is counted so the aggregate can be checked: a
    rule that is bypassed for most of the corpus is not a rule.
    """

    TIER1_UNKNOWN_PROVENANCE = "tier1_unknown_provenance"


@dataclass(frozen=True)
class Rejection:
    reason: RejectionReason
    detail: str


@dataclass(frozen=True)
class DocVerdict:
    """The result of screening one doc against one `as_of`.

    A verdict is only meaningful for the `as_of` it was computed at: the same doc
    is admissible at a later forecast time and inadmissible at an earlier one.
    """

    doc_id: str
    as_of: datetime
    rejections: tuple[Rejection, ...]
    tier: TrustTier | None
    """Resolved tier, or `None` if the source was unknown to the registry."""

    exemptions: tuple[Exemption, ...] = ()
    """Rules this doc was admitted despite. Only meaningful when `admissible`."""

    @property
    def admissible(self) -> bool:
        return not self.rejections

    @property
    def reasons(self) -> tuple[RejectionReason, ...]:
        return tuple(r.reason for r in self.rejections)


#: Share of admitted tier-1 evidence riding on the unknown-provenance exemption
#: above which the exemption is doing structural work rather than covering edge
#: cases. Not a hard gate — a number to look at before publishing.
EXEMPTION_ATTENTION_SHARE = 0.20


@dataclass(frozen=True)
class LeakageStats:
    """Aggregatable counts. This is what goes in the dataset manifest.

    Separate from `LeakageReport` because a corpus build screens millions of docs
    across many bundles and only the counts survive; and because the exemption
    share is only interpretable in aggregate — a single-doc bundle riding the
    exemption is 100% of itself and means nothing.
    """

    n_input: int
    n_kept: int
    rejected_by_reason: Mapping[RejectionReason, int]
    admitted_by_exemption: Mapping[Exemption, int]
    admitted_by_tier: Mapping[TrustTier | None, int]
    detector_version: str = LEAKAGE_DETECTOR_VERSION

    @property
    def n_rejected(self) -> int:
        return self.n_input - self.n_kept

    @property
    def tier1_unknown_provenance_share(self) -> float | None:
        """Fraction of admitted tier-1 docs that only got in via the exemption.

        `None` when no tier-1 evidence was admitted, which is different from 0.
        """
        n_tier1 = self.admitted_by_tier.get(TrustTier.TIER_1, 0)
        if n_tier1 == 0:
            return None
        return self.admitted_by_exemption.get(Exemption.TIER1_UNKNOWN_PROVENANCE, 0) / n_tier1

    @property
    def exemption_needs_attention(self) -> bool:
        share = self.tier1_unknown_provenance_share
        return share is not None and share > EXEMPTION_ATTENTION_SHARE

    def merge(self, other: LeakageStats) -> LeakageStats:
        if other.detector_version != self.detector_version:
            raise ValueError(
                "refusing to merge stats from different detector versions: "
                f"{self.detector_version} vs {other.detector_version}"
            )
        return LeakageStats(
            n_input=self.n_input + other.n_input,
            n_kept=self.n_kept + other.n_kept,
            rejected_by_reason=dict(
                Counter(self.rejected_by_reason) + Counter(other.rejected_by_reason)
            ),
            admitted_by_exemption=dict(
                Counter(self.admitted_by_exemption) + Counter(other.admitted_by_exemption)
            ),
            admitted_by_tier=dict(Counter(self.admitted_by_tier) + Counter(other.admitted_by_tier)),
            detector_version=self.detector_version,
        )

    def summary(self) -> str:
        share = self.tier1_unknown_provenance_share
        share_text = "n/a" if share is None else f"{share:.1%}"
        flag = " ATTENTION" if self.exemption_needs_attention else ""
        return (
            f"kept {self.n_kept}/{self.n_input}, "
            f"tier1 admitted={self.admitted_by_tier.get(TrustTier.TIER_1, 0)}, "
            f"tier1 unknown-provenance exemption={share_text}{flag} "
            f"[detector {self.detector_version}]"
        )


def merge_stats(stats: Iterable[LeakageStats]) -> LeakageStats:
    """Fold per-bundle stats into the corpus-level numbers for the manifest."""
    merged: LeakageStats | None = None
    for s in stats:
        merged = s if merged is None else merged.merge(s)
    if merged is None:
        return LeakageStats(0, 0, {}, {}, {})
    return merged


@dataclass(frozen=True)
class LeakageReport:
    """Bundle-level result, carrying the docs themselves plus the counts.

    `docs/01` requires the dataset manifest to carry "the count of docs rejected
    by each check", which is why rejections are accumulated rather than
    short-circuited: a doc that fails three checks is counted under all three.
    Consequently `sum(rejected_by_reason.values()) >= n_rejected`.
    """

    as_of: datetime
    kept: tuple[EvidenceDoc, ...]
    verdicts: tuple[DocVerdict, ...]
    stats: LeakageStats

    @property
    def n_input(self) -> int:
        return len(self.verdicts)

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_rejected(self) -> int:
        return self.n_input - self.n_kept

    @property
    def rejected_by_reason(self) -> Mapping[RejectionReason, int]:
        return self.stats.rejected_by_reason

    @property
    def detector_version(self) -> str:
        return self.stats.detector_version


# --------------------------------------------------------------------------
# Check 1 + timestamp integrity
# --------------------------------------------------------------------------


def _check_timestamp(doc: EvidenceDoc, as_of: datetime, policy: LeakagePolicy) -> list[Rejection]:
    """The hard timestamp filter: `doc.timestamp < as_of`, strictly.

    On the boundary, `timestamp == as_of` is REJECTED. `docs/01` states the rule
    as a strict inequality, and the tie is genuinely ambiguous: publication
    timestamps are commonly truncated to the second, minute, or day, so an equal
    timestamp usually means "same bucket", not "same instant" — and half of that
    bucket is after `as_of`. Dropping is the cheap side of the error.
    """
    out: list[Rejection] = []

    if doc.timestamp is None:
        # Nothing else in this function can be evaluated without a timestamp.
        return [
            Rejection(
                RejectionReason.MISSING_TIMESTAMP,
                "doc has no publication timestamp",
            )
        ]

    if doc.timestamp >= as_of:
        out.append(
            Rejection(
                RejectionReason.TIMESTAMP_NOT_BEFORE_AS_OF,
                f"timestamp {doc.timestamp.isoformat()} is not strictly before "
                f"as_of {as_of.isoformat()}",
            )
        )

    if (
        policy.reject_if_modified_at_or_after_as_of
        and doc.last_modified is not None
        and doc.last_modified >= as_of
    ):
        out.append(
            Rejection(
                RejectionReason.MODIFIED_AT_OR_AFTER_AS_OF,
                f"last_modified {doc.last_modified.isoformat()} is not strictly before "
                f"as_of {as_of.isoformat()}; snippet may reflect post-as_of revisions",
            )
        )

    if (
        policy.reject_if_timestamp_after_retrieval
        and doc.retrieved_at is not None
        and doc.timestamp > doc.retrieved_at
    ):
        out.append(
            Rejection(
                RejectionReason.TIMESTAMP_AFTER_RETRIEVAL,
                f"timestamp {doc.timestamp.isoformat()} is after retrieved_at "
                f"{doc.retrieved_at.isoformat()}; record is internally inconsistent",
            )
        )

    return out


def _check_provenance(
    doc: EvidenceDoc, tier: TrustTier, policy: LeakagePolicy
) -> tuple[list[Rejection], list[Exemption]]:
    """Ambiguous timestamp provenance means drop the doc.

    Returns the rejections and any exemption the doc was admitted under, so the
    exception can be counted rather than merely trusted.
    """
    if doc.provenance in policy.disqualifying_provenance:
        return (
            [
                Rejection(
                    RejectionReason.AMBIGUOUS_PROVENANCE,
                    f"provenance {doc.provenance.value!r} does not establish publication time",
                )
            ],
            [],
        )

    if doc.provenance is TimestampProvenance.UNKNOWN:
        if tier in policy.tiers_allowing_unknown_provenance:
            return [], [Exemption.TIER1_UNKNOWN_PROVENANCE]
        return (
            [
                Rejection(
                    RejectionReason.AMBIGUOUS_PROVENANCE,
                    f"provenance is unrecorded and source tier {int(tier)} does not "
                    "assert reliable timestamps",
                )
            ],
            [],
        )

    return [], []


# --------------------------------------------------------------------------
# Check 2
# --------------------------------------------------------------------------


def _check_revision_pin(doc: EvidenceDoc, policy: LeakagePolicy) -> list[Rejection]:
    """A revision-addressed source must be cited by revision, not by page.

    Structural rather than documented: the failure mode is a convenience fetcher
    added six months from now that takes a page URL, and a comment does not stop
    that. `docs/01`'s tier-3 rule for continuously-revised sources is exactly what
    a live wiki page is; the pinned revision is a different artifact that happens
    to share a hostname.
    """
    if SourceTrustRegistry.normalize_source(doc.source) not in policy.revision_pinned_sources:
        return []
    if doc.url and REVISION_PIN_RE.search(doc.url):
        return []
    return [
        Rejection(
            RejectionReason.UNPINNED_REVISION,
            f"source {doc.source!r} is admissible only as a pinned revision, but "
            f"url {doc.url!r} does not pin one (expected an 'oldid=' parameter); "
            "a live page is continuously revised and is tier 3",
        )
    ]


def _check_source(
    doc: EvidenceDoc, registry: SourceTrustRegistry, policy: LeakagePolicy
) -> tuple[TrustTier | None, list[Rejection]]:
    """Resolve the source's tier and reject anything worse than the policy allows.

    Returns the *registry's* tier (None if unknown) alongside the rejections, so
    a report can distinguish "known tier 3" from "never heard of it" even though
    both are dropped.
    """
    registry_tier = registry.tier_for(doc.source)
    out: list[Rejection] = []

    if registry_tier is None:
        effective = policy.unknown_source_tier
        out.append(
            Rejection(
                RejectionReason.UNKNOWN_SOURCE,
                f"source {doc.source!r} is not in the trust registry; "
                f"treated as tier {int(effective)}",
            )
        )
    else:
        effective = registry_tier

    if effective > policy.max_trust_tier:
        out.append(
            Rejection(
                RejectionReason.UNTRUSTED_SOURCE_TIER,
                f"source {doc.source!r} is tier {int(effective)}, "
                f"policy admits tier <= {int(policy.max_trust_tier)}",
            )
        )

    return registry_tier, out


# --------------------------------------------------------------------------
# Forecast-time admissibility
# --------------------------------------------------------------------------


def check_forecast_time(record: QuestionRecord, as_of: datetime) -> Rejection | None:
    """Reject an `as_of` at or after the question's supervision becoming available.

    Validated against `resolved_at`, or against `soft_target_at` for a question
    that resolved to a probability rather than an event. Never against
    `scheduled_resolve_date`: questions slip constantly, and an expected-March
    question that resolved in June makes `as_of = April` an ordinary forecast.
    Skipped entirely for a question carrying no supervision, which has no bound.

    This lives here rather than on `ForecastRequest` because the request cannot
    carry either time — when the answer became available is precisely what the
    model must not see.

    TODO: a record carrying BOTH signals is bounded by the earlier of the two
    here, which is right for scoring against either. An eval runner scoring
    specifically against the soft target of a record that also resolved should
    bound by `soft_target_at` directly; that path does not exist yet.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")

    bounds = [t for t in (record.resolved_at, record.soft_target_at) if t is not None]
    if not bounds:
        return None

    # The earliest signal binds: once any target information existed, a forecast
    # made after it is not a forecast.
    earliest = min(bounds)
    if as_of >= earliest:
        which = "resolved_at" if earliest == record.resolved_at else "soft_target_at"
        return Rejection(
            RejectionReason.AS_OF_AFTER_RESOLUTION,
            f"as_of {as_of.isoformat()} is at or after {which} "
            f"{earliest.isoformat()} for {record.question_id}",
        )
    return None


# --------------------------------------------------------------------------
# Check 3 — outcome-string scan
# --------------------------------------------------------------------------


class ScanStrictness(StrEnum):
    """Rungs of the outcome-string matcher, each subsuming the one below."""

    #: Case-sensitive substring of the raw snippet. Catches copy-paste leakage.
    EXACT = "exact"
    #: Substring after casefolding, punctuation stripping, whitespace collapse.
    NORMALIZED = "normalized"
    #: Normalized, plus best windowed similarity >= `fuzzy_threshold`.
    FUZZY = "fuzzy"


@dataclass(frozen=True)
class OutcomeScanConfig:
    """Configuration for the outcome-string scan.

    `docs/01` says "verbatim or near-verbatim" without defining near-verbatim.
    The ladder below is the mechanism; `fuzzy_threshold` (tau) is the part that
    genuinely needs a decision and is therefore left unset. FUZZY without a tau
    raises rather than defaulting, so nobody inherits a made-up number.

    TODO(decision): tau must be tuned on a labelled set of known-leaky snippets,
    not picked by feel, and traded off explicitly — a pre-resolution doc can
    legitimately quote a forecast phrased like the outcome, so this check has a
    real false-positive rate. Record in the internal decisions log.
    """

    strictness: ScanStrictness = ScanStrictness.NORMALIZED
    fuzzy_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.strictness is ScanStrictness.FUZZY and self.fuzzy_threshold is None:
            raise ValueError(
                "ScanStrictness.FUZZY requires an explicit fuzzy_threshold (tau); "
                "there is no defensible default — see the internal decisions log"
            )
        if self.fuzzy_threshold is not None and not (0.0 < self.fuzzy_threshold <= 1.0):
            raise ValueError(f"fuzzy_threshold must be in (0, 1], got {self.fuzzy_threshold}")


DEFAULT_OUTCOME_SCAN_CONFIG = OutcomeScanConfig()


@dataclass(frozen=True)
class OutcomeScanResult:
    """Which surface forms matched, and how strongly."""

    matched: bool
    matched_forms: tuple[str, ...] = ()
    max_similarity: float = 0.0
    """1.0 for an exact or normalized substring hit; the best windowed similarity
    ratio otherwise. Always computed under FUZZY, so a near-miss is visible."""


def _normalize_for_scan(text: str) -> str:
    """Casefold, drop punctuation that is not part of a number, collapse space.

    `%` and `.` survive so that "3.4%" stays one token — numeric outcomes are the
    common case for the synthetic corpus, and stripping them would turn "rose
    3.4%" and "rose 34%" into the same string.
    """
    kept = [c if (c.isalnum() or c in ".%") else " " for c in text.casefold()]
    return " ".join("".join(kept).split())


def _best_windowed_ratio(form: str, snippet: str) -> float:
    """Best similarity between `form` and any same-length window of `snippet`.

    Windowed rather than whole-snippet, because a snippet is far longer than a
    surface form and a whole-string ratio would be near zero even for a verbatim
    hit. Widths of +/- one token tolerate a dropped or inserted word.
    """
    form_tokens = form.split()
    snippet_tokens = snippet.split()
    if not form_tokens or not snippet_tokens:
        return 0.0

    width = len(form_tokens)
    best = 0.0
    for start in range(len(snippet_tokens)):
        for w in (width - 1, width, width + 1):
            if w <= 0 or start + w > len(snippet_tokens):
                continue
            window = " ".join(snippet_tokens[start : start + w])
            best = max(best, SequenceMatcher(None, form, window).ratio())
    return best


def scan_for_outcome_strings(
    snippet: str,
    outcome_surface_forms: Sequence[str],
    *,
    config: OutcomeScanConfig = DEFAULT_OUTCOME_SCAN_CONFIG,
) -> OutcomeScanResult:
    """Flag a snippet that states the resolution outcome.

    `outcome_surface_forms` are the natural-language ways this question's actual
    outcome could be written — "Biden wins Pennsylvania", "CPI rose 3.4%", "the
    merger was blocked".

    Producing those forms is split in two (the internal decisions log, 2026-08-13):
    synthetic auto-resolved questions can generate them mechanically from the
    same template that built the question, which covers the bulk of corpus
    volume and needs no decision; real market questions with prose resolution
    criteria need model-generated forms with human spot-check, and generator
    accuracy is measured before it gates anything. Neither generator lives here —
    this module matches forms, it does not invent them.

    Note the check's standing (`docs/01`): it "catches the obvious cases" and is
    "not sufficient alone — leakage is usually subtler". A clean scan is not
    evidence of no leakage. Check 4 is the real test.
    """
    forms = [f for f in outcome_surface_forms if f.strip()]
    if not forms:
        return OutcomeScanResult(matched=False)

    matched: list[str] = []
    best = 0.0

    normalized_snippet = _normalize_for_scan(snippet)
    for form in forms:
        if form in snippet:  # EXACT — raw substring
            matched.append(form)
            best = 1.0
            continue

        if config.strictness is ScanStrictness.EXACT:
            continue

        normalized_form = _normalize_for_scan(form)
        if normalized_form and normalized_form in normalized_snippet:
            matched.append(form)
            best = 1.0
            continue

        if config.strictness is not ScanStrictness.FUZZY:
            continue

        ratio = _best_windowed_ratio(normalized_form, normalized_snippet)
        best = max(best, ratio)
        # `fuzzy_threshold` is not None here: OutcomeScanConfig enforces it.
        if config.fuzzy_threshold is not None and ratio >= config.fuzzy_threshold:
            matched.append(form)

    return OutcomeScanResult(
        matched=bool(matched),
        matched_forms=tuple(matched),
        max_similarity=best,
    )


# --------------------------------------------------------------------------
# Public screening API
# --------------------------------------------------------------------------


def screen_doc(
    doc: EvidenceDoc,
    as_of: datetime,
    *,
    policy: LeakagePolicy = DEFAULT_POLICY,
    registry: SourceTrustRegistry = DEFAULT_TRUST_REGISTRY,
    outcome_surface_forms: Sequence[str] = (),
    scan_config: OutcomeScanConfig = DEFAULT_OUTCOME_SCAN_CONFIG,
) -> DocVerdict:
    """Screen one doc for use at forecast time `as_of`.

    All checks run; none short-circuits. A doc that fails several ways should say
    so, both for the manifest counts and because a doc failing three checks is a
    different kind of problem from one failing a boundary condition.

    The outcome-string scan (check 3) runs only when `outcome_surface_forms` is
    non-empty, because it is the one check that is per-(question, doc) rather
    than per-(doc, as_of).
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")

    registry_tier, source_rejections = _check_source(doc, registry, policy)
    effective_tier = registry_tier if registry_tier is not None else policy.unknown_source_tier

    provenance_rejections, exemptions = _check_provenance(doc, effective_tier, policy)
    revision_rejections = _check_revision_pin(doc, policy)

    rejections = [
        *_check_timestamp(doc, as_of, policy),
        *provenance_rejections,
        *source_rejections,
        *revision_rejections,
    ]

    if outcome_surface_forms:
        scan = scan_for_outcome_strings(doc.snippet, outcome_surface_forms, config=scan_config)
        if scan.matched:
            rejections.append(
                Rejection(
                    RejectionReason.OUTCOME_STRING_MATCH,
                    f"snippet states the outcome: {list(scan.matched_forms)} "
                    f"(similarity {scan.max_similarity:.2f}, {scan_config.strictness.value})",
                )
            )

    return DocVerdict(
        doc_id=doc.doc_id,
        as_of=as_of,
        rejections=tuple(rejections),
        tier=registry_tier,
        exemptions=tuple(exemptions) if not rejections else (),
    )


def screen_bundle(
    docs: Iterable[EvidenceDoc],
    as_of: datetime,
    *,
    policy: LeakagePolicy = DEFAULT_POLICY,
    registry: SourceTrustRegistry = DEFAULT_TRUST_REGISTRY,
    outcome_surface_forms: Sequence[str] = (),
    scan_config: OutcomeScanConfig = DEFAULT_OUTCOME_SCAN_CONFIG,
) -> LeakageReport:
    """Screen an evidence bundle. Kept docs preserve input order."""
    verdicts: list[DocVerdict] = []
    kept: list[EvidenceDoc] = []
    rejected: Counter[RejectionReason] = Counter()
    exempted: Counter[Exemption] = Counter()
    admitted_tiers: Counter[TrustTier | None] = Counter()

    for doc in docs:
        verdict = screen_doc(
            doc,
            as_of,
            policy=policy,
            registry=registry,
            outcome_surface_forms=outcome_surface_forms,
            scan_config=scan_config,
        )
        verdicts.append(verdict)
        if verdict.admissible:
            kept.append(doc)
            admitted_tiers[verdict.tier] += 1
            exempted.update(verdict.exemptions)
        else:
            rejected.update(verdict.reasons)

    return LeakageReport(
        as_of=as_of,
        kept=tuple(kept),
        verdicts=tuple(verdicts),
        stats=LeakageStats(
            n_input=len(verdicts),
            n_kept=len(kept),
            rejected_by_reason=dict(rejected),
            admitted_by_exemption=dict(exempted),
            admitted_by_tier=dict(admitted_tiers),
        ),
    )


def filter_evidence(
    docs: Iterable[EvidenceDoc],
    as_of: datetime,
    *,
    policy: LeakagePolicy = DEFAULT_POLICY,
    registry: SourceTrustRegistry = DEFAULT_TRUST_REGISTRY,
) -> tuple[EvidenceDoc, ...]:
    """Convenience wrapper returning only the admissible docs.

    Prefer `screen_bundle` on any path that writes a manifest — this one throws
    away the rejection counts and the exemption counts, and an uncounted drop or
    an uncounted exception is invisible.
    """
    return screen_bundle(docs, as_of, policy=policy, registry=registry).kept


# --------------------------------------------------------------------------
# Check 4 — adversarial probe set
# --------------------------------------------------------------------------


class Forecaster(Protocol):
    """Anything implementing the `docs/00` interface contract."""

    def __call__(self, request: ForecastRequest) -> ForecastResponse: ...


@dataclass(frozen=True)
class ProbeCase:
    """One matched pair: the same question, screened evidence vs. injected evidence.

    The pair is identical except for `injected_docs`, so the difference in score
    is attributable to those docs and nothing else.
    """

    question_id: str
    request: ForecastRequest
    """The clean request: evidence that passed screening at `request.as_of`."""

    injected_docs: tuple[EvidenceDoc, ...]
    """Post-`as_of` docs added to build the poisoned variant.

    For the ADVERSARIAL arm these carry outcome information. For the NULL arm
    they are post-`as_of` but outcome-free — see `run_adversarial_probe`.
    """

    outcome: Outcome


@dataclass(frozen=True)
class ProbeCaseResult:
    question_id: str
    outcome: Outcome
    p_clean: float
    p_injected: float

    @property
    def brier_clean(self) -> float:
        return (self.p_clean - self.outcome) ** 2

    @property
    def brier_injected(self) -> float:
        return (self.p_injected - self.outcome) ** 2

    @property
    def delta(self) -> float:
        """Improvement from the injected docs. Positive means the model scored
        better once it was handed post-`as_of` evidence."""
        return self.brier_clean - self.brier_injected


@dataclass(frozen=True)
class ProbeReport:
    """Clean vs. injected performance, per case and in aggregate."""

    label: str
    results: tuple[ProbeCaseResult, ...]
    samples: int
    model_version: str = ""
    """Which checkpoint produced these forecasts. A null belongs to the
    checkpoint it was measured on; `compare_to_null` refuses a mismatch."""

    detector_version: str = LEAKAGE_DETECTOR_VERSION

    @property
    def n_cases(self) -> int:
        return len(self.results)

    @property
    def deltas(self) -> tuple[float, ...]:
        return tuple(r.delta for r in self.results)

    @property
    def brier_clean(self) -> float:
        return statistics.fmean(r.brier_clean for r in self.results)

    @property
    def brier_injected(self) -> float:
        return statistics.fmean(r.brier_injected for r in self.results)

    @property
    def mean_delta(self) -> float:
        return statistics.fmean(self.deltas)

    @property
    def stdev_delta(self) -> float:
        """Spread of a SINGLE case's delta. This is what a threshold is built
        from — see `null_threshold`."""
        return statistics.stdev(self.deltas) if self.n_cases > 1 else 0.0

    @property
    def standard_error(self) -> float:
        """Spread of the MEAN delta, `SD / sqrt(n)`.

        Answers "is this null well enough estimated to gate on", and nothing
        else. It must not appear in a threshold: it shrinks with more null
        cases, so a threshold built on it would tighten as the null got better
        characterized, raising false alarms for doing more work.
        """
        return self.stdev_delta / (self.n_cases**0.5) if self.n_cases > 0 else 0.0

    def quantile(self, q: float) -> float:
        """Empirical quantile of the per-case deltas, linearly interpolated."""
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0, 1], got {q}")
        ordered = sorted(self.deltas)
        if not ordered:
            raise ValueError("no cases")
        pos = q * (len(ordered) - 1)
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (pos - low) * (ordered[high] - ordered[low])

    def summary(self) -> str:
        return (
            f"[{self.label}] n={self.n_cases} samples={self.samples} "
            f"brier clean={self.brier_clean:.4f} injected={self.brier_injected:.4f} "
            f"delta={self.mean_delta:+.4f} (sd {self.stdev_delta:.4f})"
        )


def run_adversarial_probe(
    cases: Sequence[ProbeCase],
    forecaster: Forecaster,
    *,
    samples: int = 1,
    label: str = "adversarial",
) -> ProbeReport:
    """Score each case clean and injected, and report the gap.

    `docs/01`: "This is the real test; run it every training cycle, not once."
    So this belongs in the per-cycle gate, not in a human's hands.

    **Reading the number requires a null run.** A materiality threshold cannot be
    chosen a priori: adding *any* documents moves a forecast, so some of the gap
    is the model reacting to extra text rather than hunting for outcomes. Run
    this same function over cases whose `injected_docs` are post-`as_of` but
    carry no outcome information (`label="null"`), and derive the threshold from
    that distribution with `null_threshold`. Then `compare_to_null`.

    The null must be re-run for every checkpoint: it measures how much added
    context moves *this model*, and that changes as the model does.

    `samples > 1` averages the forecast probability over repeated calls, so the
    gap is not confounded with decode noise. Averaging probabilities (not Brier
    scores) is deliberate: it is the ensemble forecast that is being scored.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")

    results: list[ProbeCaseResult] = []
    versions: set[str] = set()
    for case in cases:
        poisoned = case.request.model_copy(
            update={"evidence": tuple(case.request.evidence) + tuple(case.injected_docs)}
        )
        p_clean, clean_versions = _mean_probability(forecaster, case.request, samples)
        p_injected, injected_versions = _mean_probability(forecaster, poisoned, samples)
        versions |= clean_versions | injected_versions
        results.append(
            ProbeCaseResult(
                question_id=case.question_id,
                outcome=case.outcome,
                p_clean=p_clean,
                p_injected=p_injected,
            )
        )

    if len(versions) > 1:
        # Otherwise the gap is partly between two models rather than between two
        # evidence bundles, and the null stops belonging to a checkpoint.
        raise ValueError(f"probe run mixes model versions: {sorted(versions)}")

    return ProbeReport(
        label=label,
        results=tuple(results),
        samples=samples,
        model_version=next(iter(versions), ""),
    )


def _mean_probability(
    forecaster: Forecaster, request: ForecastRequest, samples: int
) -> tuple[float, set[str]]:
    responses = [forecaster(request) for _ in range(samples)]
    return (
        statistics.fmean(r.probability for r in responses),
        {r.model_version for r in responses},
    )


class NullThresholdMethod(StrEnum):
    """How the single-case threshold is read off the null distribution."""

    #: Empirical quantile of the null's per-case deltas. Default: it assumes
    #: nothing about the shape, and the null is not Gaussian — it is bounded
    #: below by how much removing context can hurt, and skewed.
    EMPIRICAL_QUANTILE = "empirical_quantile"
    #: `mean + z * SD`. For null sets too small for a stable empirical quantile,
    #: where interpolating between the top two observations is mostly noise.
    NORMAL = "normal"


#: Quantile of the null's per-case delta distribution used as the single-case
#: threshold. 0.95 means a case has to beat 95% of what outcome-free injected
#: context achieves before it counts as an exceedance.
DEFAULT_NULL_QUANTILE = 0.95

#: Standard-normal multiplier for NullThresholdMethod.NORMAL. Applied to the
#: null's standard DEVIATION, not its standard error — see `null_threshold`.
DEFAULT_NULL_Z = 1.96

#: Run-level false-alarm rate for the exceedance test in `compare_to_null`.
#: Kept sensitive on purpose: a missed leak is catastrophic and a false alarm
#: costs a re-run. See `ProbeVerdict` for what a trip actually means.
DEFAULT_PROBE_ALPHA = 0.05

#: Below this, the null mean is too poorly estimated to hang a gate on. Reported
#: via `ProbeReport.standard_error`, which is what standard error is FOR here.
MIN_NULL_CASES = 20


def null_threshold(
    null: ProbeReport,
    *,
    method: NullThresholdMethod = NullThresholdMethod.EMPIRICAL_QUANTILE,
    quantile: float = DEFAULT_NULL_QUANTILE,
    z: float = DEFAULT_NULL_Z,
) -> float:
    """Derive the single-case materiality threshold from a null run.

    The question this answers is "how large a delta does a SINGLE case plausibly
    reach when the injected documents carry no outcome information" — so the
    spread that matters is the null's standard deviation, or better, its
    empirical quantile.

    NOT the standard error. SE is the spread of the null *mean*, and it shrinks
    as `1/sqrt(n_null)`: using it would mean that characterizing the null more
    carefully tightens the threshold and raises the false-alarm rate. More effort
    understanding the null would make the check worse, which is backwards.
    Standard error has a job here, but it is `ProbeReport.standard_error`,
    reporting whether the null itself is well enough estimated to gate on.

    No number was written into the internal decisions log up front because any
    a-priori threshold would have been arbitrary. This one is measured, and it is
    re-measured per checkpoint — see `compare_to_null`.
    """
    if null.n_cases == 0:
        raise ValueError("cannot derive a threshold from an empty null run")
    if null.n_cases == 1:
        raise ValueError("cannot estimate null spread from a single case")

    if method is NullThresholdMethod.EMPIRICAL_QUANTILE:
        return null.quantile(quantile)
    return null.mean_delta + z * null.stdev_delta


def _binomial_upper_count(n: int, p: float, alpha: float) -> int:
    """Largest k with P(X > k) <= alpha for X ~ Binomial(n, p).

    Exact rather than normal-approximated: n is the probe-set size and p is
    small (0.05 by default), which is precisely where the normal approximation
    is worst.
    """
    cumulative = 0.0
    for k in range(n + 1):
        cumulative += math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
        if cumulative >= 1.0 - alpha:
            return k
    return n


@dataclass(frozen=True)
class ProbeVerdict:
    """Result of the exceedance test against the null.

    **What a failure means.** Not an automatic block. At a 5% run-level
    false-alarm rate against a probe that runs every training cycle, a
    hard-blocking gate trips roughly once every twenty runs on nothing, and a
    gate that blocks releases on nothing is a gate people learn to wave through.

    A failure means: **the checkpoint is not published or promoted until the trip
    is explained.** First step is a re-run with an enlarged probe set — a real
    leak persists and a fluke does not — which is cheap, so the sensitive setting
    costs little. What is not allowed is shipping past an unexplained trip.
    """

    passed: bool
    threshold: float
    method: NullThresholdMethod
    n_exceeding: int
    """Probe cases whose delta beat the null threshold."""

    max_expected_exceeding: int
    """Most exceedances consistent with the null at this `alpha`."""

    observed_exceedance_rate: float
    expected_exceedance_rate: float
    alpha: float
    probe_mean_delta: float
    """Reported for continuity with the Brier numbers. It does NOT gate: a mean
    hides a few badly-leaking cases inside many clean ones."""

    null_mean_delta: float
    null_standard_error: float
    """Spread of the null MEAN. Says whether the null is well estimated; it is
    deliberately not part of the threshold."""

    n_probe: int
    n_null: int
    model_version: str
    null_is_well_estimated: bool
    detector_version: str = LEAKAGE_DETECTOR_VERSION

    def summary(self) -> str:
        state = "PASS" if self.passed else "FAIL — investigate before publishing"
        caveat = "" if self.null_is_well_estimated else f" [null n={self.n_null} is thin]"
        return (
            f"{state}: {self.n_exceeding}/{self.n_probe} cases beat the null "
            f"threshold {self.threshold:+.4f} ({self.method.value}), "
            f"at most {self.max_expected_exceeding} expected at alpha={self.alpha} "
            f"[model {self.model_version or 'unknown'}]{caveat}"
        )


def compare_to_null(
    probe: ProbeReport,
    null: ProbeReport,
    *,
    method: NullThresholdMethod = NullThresholdMethod.EMPIRICAL_QUANTILE,
    quantile: float = DEFAULT_NULL_QUANTILE,
    z: float = DEFAULT_NULL_Z,
    alpha: float = DEFAULT_PROBE_ALPHA,
) -> ProbeVerdict:
    """Gate an adversarial run against its null, per case.

    Counts how many probe cases beat the null's single-case threshold and tests
    that count against what the null itself would produce. Under the null the
    exceedance rate is `1 - quantile` by construction, so the comparison is
    like-for-like: single-case deltas against a single-case threshold.

    Counting exceedances rather than comparing means is deliberate. Leakage is
    not uniform — a handful of questions whose injected docs state the outcome
    outright will move a long way while the rest do not, and averaging buries
    them among the cases where nothing happened.

    **The null belongs to this checkpoint.** It measures how much any added
    context moves *this model's* output, and a checkpoint that grew more
    context-sensitive during RL has a different null. Mismatched model versions
    are refused rather than silently compared, so the null cannot decay into a
    cached constant.
    """
    if probe.n_cases == 0:
        raise ValueError("cannot judge an empty probe run")
    if probe.model_version != null.model_version:
        raise ValueError(
            "probe and null were produced by different checkpoints "
            f"({probe.model_version!r} vs {null.model_version!r}); the null is a "
            "property of the checkpoint and must be re-derived for each one"
        )

    threshold = null_threshold(null, method=method, quantile=quantile, z=z)
    expected_rate = (
        1.0 - quantile
        if method is NullThresholdMethod.EMPIRICAL_QUANTILE
        else (1.0 - _normal_cdf(z))
    )
    n_exceeding = sum(1 for d in probe.deltas if d > threshold)
    max_expected = _binomial_upper_count(probe.n_cases, expected_rate, alpha)

    return ProbeVerdict(
        passed=n_exceeding <= max_expected,
        threshold=threshold,
        method=method,
        n_exceeding=n_exceeding,
        max_expected_exceeding=max_expected,
        observed_exceedance_rate=n_exceeding / probe.n_cases,
        expected_exceedance_rate=expected_rate,
        alpha=alpha,
        probe_mean_delta=probe.mean_delta,
        null_mean_delta=null.mean_delta,
        null_standard_error=null.standard_error,
        n_probe=probe.n_cases,
        n_null=null.n_cases,
        model_version=probe.model_version,
        null_is_well_estimated=null.n_cases >= MIN_NULL_CASES,
    )


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Confidence-toward-truth gate (the internal decisions log, 2026-08-21)
# --------------------------------------------------------------------------

#: A case "reads as leaked" when the model ends this confident in the truth.
#: A chosen number, not a measured one, so `confidence_gate` reports across a
#: range and the conclusion should never rest on this single value.
DEFAULT_CONFIDENCE_BAR = 0.9


def confidence_toward_truth(probability: float, outcome: Outcome) -> float:
    """How much probability the forecast put on what actually happened.

    The Brier delta `brier_clean - brier_injected` is bounded above by
    `brier_clean`, so on a case the model already had right there is no error
    left to recover and *no leak, however blatant, can produce a large delta*.
    Measured on the 2026-08-21 probe that ceiling was 15% of cases -- the gate
    was structurally incapable of firing on the other 85%, and the observed
    exceedance had already saturated it.

    This statistic has no such ceiling. A document that states the outcome
    should drive the model to near-certainty *wherever it started*, so a perfect
    leak scores 1.0 on every case and the gate can reach 100%.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0,1], got {probability}")
    return probability if outcome == 1 else 1.0 - probability


@dataclass(frozen=True)
class ConfidenceVerdict:
    """Confident-and-correct counts for one arm against its null.

    Read `n_confident` against `max_expected_confident`, exactly as with the
    delta gate: the null fixes how often this model lands confidently on the
    truth from added context alone, and the arm has to beat that.
    """

    bar: float
    passed: bool
    n_cases: int
    n_confident: int
    """Cases where the model ended at or above `bar` on the true outcome."""

    null_rate: float
    """Share of NULL cases that did, which is the floor this arm must clear."""

    max_expected_confident: int
    alpha: float
    mean_confidence: float
    null_mean_confidence: float

    @property
    def rate(self) -> float:
        return self.n_confident / self.n_cases if self.n_cases else 0.0


def confidence_gate(
    probe: ProbeReport,
    null: ProbeReport,
    *,
    bar: float = DEFAULT_CONFIDENCE_BAR,
    alpha: float = DEFAULT_PROBE_ALPHA,
) -> ConfidenceVerdict:
    """Gate an arm on how often it ends confidently correct, versus its null.

    The null supplies the floor rather than a per-case threshold. That avoids
    the failure the delta gate hit: the null's 95th percentile of a
    ceiling-bounded statistic was 0.72 Brier, a bar most cases could not reach
    even in principle, so the count could not discriminate.

    `bar` is a choice and is not defended here -- callers should sweep it (see
    `confidence_curve`) and confirm the verdict is not an artifact of one value.
    """
    if not 0.0 < bar <= 1.0:
        raise ValueError(f"bar must be in (0, 1], got {bar}")
    if probe.model_version != null.model_version:
        raise ValueError(
            f"null was run on {null.model_version!r} but the probe on "
            f"{probe.model_version!r}; the null belongs to the checkpoint"
        )

    def confident(report: ProbeReport) -> list[float]:
        return [confidence_toward_truth(r.p_injected, r.outcome) for r in report.results]

    arm_conf, null_conf = confident(probe), confident(null)
    n_confident = sum(1 for c in arm_conf if c >= bar)
    null_rate = (sum(1 for c in null_conf if c >= bar) / len(null_conf)) if null_conf else 0.0
    max_expected = _binomial_upper_count(len(arm_conf), null_rate, alpha)
    return ConfidenceVerdict(
        bar=bar,
        passed=n_confident <= max_expected,
        n_cases=len(arm_conf),
        n_confident=n_confident,
        null_rate=null_rate,
        max_expected_confident=max_expected,
        alpha=alpha,
        mean_confidence=sum(arm_conf) / len(arm_conf) if arm_conf else 0.0,
        null_mean_confidence=sum(null_conf) / len(null_conf) if null_conf else 0.0,
    )


def confidence_curve(
    probe: ProbeReport,
    null: ProbeReport,
    *,
    bars: Sequence[float] = (0.7, 0.8, 0.9, 0.95, 0.99),
    alpha: float = DEFAULT_PROBE_ALPHA,
) -> tuple[ConfidenceVerdict, ...]:
    """`confidence_gate` across several bars, so no conclusion rests on one."""
    return tuple(confidence_gate(probe, null, bar=b, alpha=alpha) for b in bars)
