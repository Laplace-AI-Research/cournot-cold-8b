"""Build the Kalshi transfer evaluation set from a rule that is written down.

    uv run python scripts/kalshi_eval_set.py --out data/kalshi_score_778_rebuilt.json

## Why this exists

`data/kalshi_score_778.json` has no builder anywhere in this repository. Its
metadata states a filter -- *"judgment categories only; numeric/threshold titles
removed; lifetime>=7d; one market per event"* -- but no committed code implements
it, so the population behind every Kalshi transfer number on three public model
cards cannot be rebuilt or checked. §9.1 of the technical report discloses this;
this script is the promise being kept.

`polymarket_eval_set.py` is the precedent, including its discipline: **the rule
below was written before this was run and was not adjusted afterwards.** No
stage of it was chosen by looking at what it does to the row count or to any
model comparison.

## The rule, stated in full

The artifact's four clauses, each made executable, plus two the artifact's own
`filter` string omits and which are needed for the rule to determine a set at
all. Those two are marked ADDED.

1. **Judgment categories only.** A judgment question resolves on a human or
   institutional decision, or on a world event a forecaster reasons about. It is
   NOT a price or index level, a sporting result, a physical measurement, or a
   mechanical count of utterances. On Kalshi's own 18-category series taxonomy
   that admits: Politics, Elections, Economics, Science and Technology, World,
   Companies, Health, Social, Transportation, Education, Entertainment, Exotics.
   It excludes: Sports, Financials, Crypto, Commodities, Climate and Weather,
   Mentions.

   Entertainment is admitted on principle -- an award outcome is a judgment --
   and its numeric titles are removed by clause 2 rather than by category.

2. **Numeric / threshold titles removed.** Any title expressing a threshold on a
   continuous quantity. Matched on the title, not the category, because a
   judgment category still carries "above $X" markets.

3. **Lifetime >= 7 days**, measured `settlement_ts - open_time`. `docs/11`'s
   domain prior: time to resolution is the property that has mattered twice, and
   a market open for under a week is a nowcast.

4. **One market per event.** Deterministic tiebreak: the market with the LONGEST
   lifetime, ties broken by ticker ascending. Stated because "one per event"
   alone does not say which one, and any unstated tiebreak is a silent
   researcher degree of freedom.

5. **ADDED -- multivariate combo markets excluded.** `docs/10` records that
   `/markets` has no default for `mve_filter`, so unfiltered pulls return
   multi-leg parlays whose titles are the legs comma-joined and whose timestamps
   track generation rather than events. They are not forecasting questions. The
   artifact's filter string does not mention them; without excluding them the
   rule admits 27M rows of noise.

6. **ADDED -- a resolution window.** The artifact's filter string names none, yet
   the shipped set spans 2026-06-20 to 2026-08-27, so a window was applied and
   not recorded. `--resolved-from` / `--resolved-to` make it explicit and
   default to the shipped set's own span so the comparison is like-for-like.

**Clauses 5 and 6 are the finding, not an implementation detail.** The stated
filter under-determines the set: two choices that materially change the
population were never written down.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime

JUDGMENT = {
    "Politics", "Elections", "Economics", "Science and Technology", "World",
    "Companies", "Health", "Social", "Transportation", "Education",
    "Entertainment", "Exotics",
}
MECHANICAL = {
    "Sports", "Financials", "Crypto", "Commodities", "Climate and Weather",
    "Mentions",
}

# Clause 2. Thresholds on a continuous quantity, however phrased.
NUMERIC_TITLE = re.compile(
    r"(above|below|over|under|at least|at most|more than|less than|greater than|"
    r"exceed|reach|between)\s+[\$€£]?[\d,]+(\.\d+)?"
    r"|[\$€£][\d,]+(\.\d+)?"
    r"|\b\d+(\.\d+)?\s*(%|percent|bps|basis points)"
    r"|\bhow many\b|\bnumber of\b",
    re.IGNORECASE,
)


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def series_of(event_ticker: str) -> str:
    """Kalshi event tickers are SERIES-SUFFIX. The series is the part before the
    first hyphen; a ticker with no hyphen is its own series."""
    return event_ticker.split("-", 1)[0] if event_ticker else ""


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="data/kalshi/sweep-20260824.jsonl.gz")
    ap.add_argument("--categories", default="data/kalshi/series_category.json")
    ap.add_argument("--out", default="data/kalshi_score_778_rebuilt.json")
    ap.add_argument("--resolved-from", default="2026-06-20")
    ap.add_argument("--resolved-to", default="2026-08-28")
    ap.add_argument("--min-lifetime-days", type=int, default=7)
    ap.add_argument("--compare", default="data/kalshi_score_778.json")
    args = ap.parse_args(argv)

    cats: dict[str, str] = json.load(open(args.categories, encoding="utf-8"))
    lo = parse_ts(args.resolved_from + "T00:00:00+00:00")
    hi = parse_ts(args.resolved_to + "T00:00:00+00:00")

    drop = {
        "not_finalized": 0, "mve": 0, "no_event": 0, "unknown_series": 0,
        "mechanical_category": 0, "numeric_title": 0, "outside_window": 0,
        "short_lifetime": 0, "bad_timestamps": 0, "no_result": 0,
    }
    best: dict[str, dict] = {}
    seen = 0

    op = gzip.open if args.sweep.endswith(".gz") else open
    with op(args.sweep, "rt", encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            seen += 1
            r = json.loads(line)

            if r.get("status") != "finalized":
                drop["not_finalized"] += 1; continue
            if r.get("mve_collection_ticker"):
                drop["mve"] += 1; continue
            ev = r.get("event_ticker") or ""
            if not ev:
                drop["no_event"] += 1; continue
            cat = cats.get(series_of(ev))
            if cat is None:
                drop["unknown_series"] += 1; continue
            if cat in MECHANICAL or cat not in JUDGMENT:
                drop["mechanical_category"] += 1; continue
            title = r.get("title") or ""
            if NUMERIC_TITLE.search(title):
                drop["numeric_title"] += 1; continue
            res = r.get("result")
            if res not in ("yes", "no"):
                drop["no_result"] += 1; continue

            opened = parse_ts(r.get("open_time"))
            settled = parse_ts(r.get("settlement_ts"))
            if opened is None or settled is None:
                drop["bad_timestamps"] += 1; continue
            if not (lo <= settled < hi):
                drop["outside_window"] += 1; continue
            life = (settled - opened).days
            if life < args.min_lifetime_days:
                drop["short_lifetime"] += 1; continue

            row = {
                # keyed on the MARKET ticker, not the event ticker: the shipped
                # set uses kalshi:KXNEWGLENN-262-JUL where the event is
                # KXNEWGLENN-262. Keying on the event made 642 genuine matches
                # look like 30 and nearly produced a wrong finding about a
                # public artifact.
                "question_id": f"kalshi:{r.get('ticker') or ev}",
                "text": title,
                "open_date": opened.isoformat(),
                "scheduled_resolve_date": (parse_ts(r.get("close_time")) or settled).isoformat(),
                "resolved_at": settled.isoformat(),
                "outcome": 1 if res == "yes" else 0,
                "quarter": 0,
                "event_ticker": ev,
                "category": cat,
                "_lifetime": life,
                "_ticker": r.get("ticker") or "",
            }
            # clause 4: longest lifetime wins, ties by ticker ascending
            cur = best.get(ev)
            if cur is None or (row["_lifetime"] > cur["_lifetime"]) or (
                row["_lifetime"] == cur["_lifetime"] and row["_ticker"] < cur["_ticker"]
            ):
                best[ev] = row

    rows = sorted(best.values(), key=lambda r: r["question_id"])
    for r in rows:
        r.pop("_lifetime", None); r.pop("_ticker", None)

    print(f"scanned {seen:,} sweep rows")
    print("dropped:")
    for k, v in sorted(drop.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>10,}  {k}")
    print(f"\nkept {len(rows):,} rows, one per event")

    payload = {
        "content_hash": hashlib.sha256(
            json.dumps([r["question_id"] for r in rows], sort_keys=True).encode()
        ).hexdigest(),
        "n": len(rows),
        "split": "transfer_test",
        "platform": "kalshi",
        "builder": "scripts/kalshi_eval_set.py",
        "filter": ("judgment categories only; numeric/threshold titles removed; "
                   f"lifetime>={args.min_lifetime_days}d; one market per event "
                   "(longest lifetime, ties by ticker); MVE combos excluded; "
                   f"resolved in [{args.resolved_from}, {args.resolved_to})"),
        "judgment_categories": sorted(JUDGMENT),
        "mechanical_categories": sorted(MECHANICAL),
        "dropped": drop,
        "questions": rows,
    }
    with open(args.out, "w", encoding="utf-8") as h:
        json.dump(payload, h, indent=1, sort_keys=True)
    print(f"wrote {args.out}")

    if args.compare:
        try:
            old = json.load(open(args.compare, encoding="utf-8"))
        except FileNotFoundError:
            return 0
        a = {q["question_id"] for q in old["questions"]}
        b = {r["question_id"] for r in rows}
        print(f"\n== against the shipped set ({args.compare}) ==")
        print(f"  shipped {len(a):,}   rebuilt {len(b):,}   overlap {len(a & b):,}")
        print(f"  in shipped only: {len(a - b):,}")
        print(f"  in rebuilt only: {len(b - a):,}")
        if a == b:
            print("  IDENTICAL -- the stated rule reproduces the shipped set.")
        else:
            print("  NOT identical. The stated filter does not determine the set;")
            print("  see clauses 5 and 6 in this script's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
