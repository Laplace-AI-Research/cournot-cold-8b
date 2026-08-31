"""Fetch yes_sub_title for the rebuilt Kalshi set, so its prompts can be rebuilt.

    uv run --with requests python scripts/kalshi_subtitles.py

## Why this exists

The shipped eval set's prompt is not the market title. It is

    title (trailing "?" removed) + " -- resolves YES if: " + yes_sub_title

and `yes_sub_title` is in NO archived source we hold: not the sweep, not
settlements, not the corpus parquet. The rebuilt set therefore scored the models
on *different prompts*, and the reproduction check caught it at
max|diff| 0.700 against a 0.02 tolerance (docs/10 2026-08-30g).

That is the most damaging way the shipped artifact is unreproducible. Wrong rows
can be argued about; a prompt that cannot be reconstructed means nobody can
recover what the model was actually shown.

Resumable, rate-limited, and a 404 is recorded as a transport outcome rather
than an empty subtitle -- `tarot.outcomes` exists because this project has
three times counted a failed request as a content result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence

API = "https://api.elections.kalshi.com/trade-api/v2/markets/"
UA = "Laplace-Research/tarot (non-commercial research eval split)"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="data/kalshi_eval_rebuilt.json")
    ap.add_argument("--out", default="data/kalshi/subtitles.jsonl")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args(argv)

    import requests

    rows = json.load(open(args.set, encoding="utf-8"))["questions"]
    tickers = [r["question_id"].split(":", 1)[1] for r in rows]

    done: set[str] = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["ticker"])
        print(f"resuming: {len(done):,} already fetched")

    s = requests.Session()
    s.headers["User-Agent"] = UA
    counts = {"ok": len(done), "missing_field": 0, "not_found": 0, "error": 0}

    with open(args.out, "a", encoding="utf-8") as out:
        for i, t in enumerate(tickers):
            if t in done:
                continue
            try:
                r = s.get(API + t, timeout=25)
            except Exception:
                counts["error"] += 1
                continue
            if r.status_code == 404:
                counts["not_found"] += 1
                out.write(json.dumps({"ticker": t, "status": "not_found"}) + "\n")
                out.flush()
                continue
            if not r.ok:
                counts["error"] += 1
                time.sleep(1.0)
                continue
            m = r.json().get("market", {})
            sub = m.get("yes_sub_title")
            if sub is None:
                counts["missing_field"] += 1
            else:
                counts["ok"] += 1
            out.write(json.dumps({
                "ticker": t, "status": "ok",
                "title": m.get("title"), "yes_sub_title": sub,
            }, sort_keys=True) + "\n")
            out.flush()
            if (i + 1) % 200 == 0:
                print(f"  {i+1:,}/{len(tickers):,}  {counts}", flush=True)
            time.sleep(args.sleep)

    print(f"\n{counts}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
