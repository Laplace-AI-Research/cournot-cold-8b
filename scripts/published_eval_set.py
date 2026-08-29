# SPDX-License-Identifier: Apache-2.0
"""Build `published` eval set from the refresh artifacts. The only source of an external number.

    uv run python scripts/published_eval_set.py --out data/published_eval.json

**This script did not exist until 2026-08-27.** `data/published_eval.json` (n=132)
was committed in a repo-hygiene commit with no build path recorded anywhere, which
means the artifact the model card's headline rests on could not be regenerated,
audited, or extended by anyone including us. That gap is what this closes.

Reconstruction is checked, not asserted: `--verify-against` rebuilds the shipped
132 from the same candidates and refuses to agree unless every field matches.

## `scheduled_resolve_date` is not `resolved_at`

The prior tier conditions on question + resolution criteria + **resolution date**
+ `as_of`. The resolution date a forecaster has at `as_of` is the market's
*scheduled close*, not the moment it actually resolved — those differ on 62% of
`dev` questions. All 132 rows of the shipped set had them equal, because the
refresh never captured `closeTime` and the builder substituted `resolved_at`.

That substitution puts a post-`as_of` fact into the prompt (non-negotiable #1)
and makes `dev` and `published` non-commensurable, since `dev` is built from the
parquet corpus and carries real close times. See the internal decisions log, 2026-08-27o.

So: **a row with no `close_ts` is refused, not defaulted.** Defaulting is exactly
how the defect entered, and a builder that silently fills a leakage-relevant
field is the shape `docs/11` calls a guard that permits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cournot.splits import FREEZE


def _iso(ms: float) -> str:
    """Epoch milliseconds to the timezone-aware ISO form the eval sets use."""
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


def build(candidates: list[dict[str, Any]], *, require_close: bool = True) -> dict[str, Any]:
    """Candidates -> the `published` eval payload, with every drop counted."""
    questions: list[dict[str, Any]] = []
    drops: dict[str, int] = {
        "unresolved_or_other": 0,
        "no_created_ts": 0,
        "no_close_ts": 0,
        "not_after_freeze": 0,
        "close_before_open": 0,
    }
    freeze_ms = FREEZE.timestamp() * 1000
    seen: set[str] = set()

    for row in candidates:
        market_id = row.get("id")
        if not isinstance(market_id, str) or market_id in seen:
            continue
        if row.get("resolution") not in ("YES", "NO"):
            drops["unresolved_or_other"] += 1
            continue
        created = row.get("created_ts")
        if not isinstance(created, int | float):
            drops["no_created_ts"] += 1
            continue

        resolved_at = row.get("resolved_at")
        if not isinstance(resolved_at, str):
            drops["unresolved_or_other"] += 1
            continue
        resolve_ms = datetime.fromisoformat(resolved_at).timestamp() * 1000
        if resolve_ms <= freeze_ms:
            drops["not_after_freeze"] += 1
            continue

        close = row.get("close_ts")
        if not isinstance(close, int | float):
            if require_close:
                drops["no_close_ts"] += 1
                continue
            close = resolve_ms
        if close <= created:
            drops["close_before_open"] += 1
            continue

        seen.add(market_id)
        questions.append(
            {
                "question_id": f"manifold:{market_id}",
                "text": row["title"],
                "open_date": _iso(created),
                "scheduled_resolve_date": _iso(close),
                "resolved_at": resolved_at,
                "outcome": 1 if row["resolution"] == "YES" else 0,
                "quarter": 0,
            }
        )

    questions.sort(key=lambda q: (q["resolved_at"], q["question_id"]))
    payload: dict[str, Any] = {
        "split": "published",
        "source": "refresh-artifacts",
        "freeze": FREEZE.isoformat(),
        "n": len(questions),
        "base_rate": (
            sum(q["outcome"] for q in questions) / len(questions) if questions else None
        ),
        "dropped": drops,
        "questions": questions,
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps([q["question_id"] for q in questions], sort_keys=True).encode()
    ).hexdigest()
    return payload


def _verify(payload: dict[str, Any], reference_path: str) -> int:
    """Refuse unless every shipped row is reproduced field-for-field.

    Compares against the *shipped* file rather than recomputing from this
    builder's own output — a check that reused this arithmetic would fail
    together with it, which `docs/11` rule 1 exists to prevent.
    """
    with open(reference_path, encoding="utf-8") as handle:
        reference = {q["question_id"]: q for q in json.load(handle)["questions"]}
    built = {q["question_id"]: q for q in payload["questions"]}

    missing = sorted(set(reference) - set(built))
    # `scheduled_resolve_date` is expected to differ: the shipped file carries the
    # defect this builder refuses to reproduce. Every other field must match.
    fields = ("text", "open_date", "resolved_at", "outcome")
    mismatched = [
        qid
        for qid in set(reference) & set(built)
        if any(reference[qid][f] != built[qid][f] for f in fields)
    ]
    corrected = [
        qid
        for qid in set(reference) & set(built)
        if reference[qid]["scheduled_resolve_date"] != built[qid]["scheduled_resolve_date"]
    ]

    print(f"reference {len(reference):,} rows, rebuilt {len(built):,} rows")
    print(f"  reproduced field-for-field : {len(set(reference) & set(built)) - len(mismatched):,}")
    print(f"  mismatched on {fields}: {len(mismatched):,}")
    print(f"  scheduled_resolve_date corrected: {len(corrected):,}")
    if missing:
        print(f"  MISSING from rebuild: {len(missing):,}  e.g. {missing[:5]}")
    if mismatched:
        print(f"  MISMATCHED: {mismatched[:5]}")
        return 1
    if missing:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="data/manifold/published_candidates.jsonl")
    parser.add_argument("--out", default="data/published_eval.json")
    parser.add_argument(
        "--verify-against",
        default="",
        help="rebuild and compare against a shipped eval set instead of writing",
    )
    parser.add_argument(
        "--allow-missing-close",
        action="store_true",
        help=(
            "reproduce the pre-2026-08-27o behaviour of defaulting "
            "scheduled_resolve_date to resolved_at. For reproducing the old "
            "artifact only; never for building one that ships."
        ),
    )
    args = parser.parse_args(argv)

    candidates: list[dict[str, Any]] = []
    with open(args.candidates, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"{len(candidates):,} candidates", file=sys.stderr)

    payload = build(candidates, require_close=not args.allow_missing_close)

    if args.verify_against:
        return _verify(payload, args.verify_against)

    if not payload["questions"]:
        print("\nrefusing to write an empty published set. dropped:", file=sys.stderr)
        for reason, count in sorted(payload["dropped"].items()):
            if count:
                print(f"  {reason}: {count:,}", file=sys.stderr)
        print(
            "\nif every row dropped on no_close_ts, run "
            "scripts/published_closetime.py first.",
            file=sys.stderr,
        )
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
    print(f"\nwrote {payload['n']:,} questions -> {args.out}")
    print(f"base rate {payload['base_rate']:.4f}")
    print(f"content hash {str(payload['content_hash'])[:16]}")
    for reason, count in sorted(payload["dropped"].items()):
        if count:
            print(f"  dropped {reason}: {count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
