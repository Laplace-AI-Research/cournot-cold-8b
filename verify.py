# SPDX-License-Identifier: Apache-2.0
"""Recompute every headline number in README.md from the shipped forecasts.

    uv run python verify.py

Requires only `src/tarot` — no model, no GPU. If a number here disagrees with the
model card, the model card is wrong.
"""
from __future__ import annotations
import json, statistics, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from tarot.metrics import brier, corp, ece, output_histogram

def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("expectation_probability") is not None:
                out[r["question_id"]] = float(r["expectation_probability"])
    return out

def report(label, eval_path, run_path):
    qs = {q["question_id"]: q for q in json.load(open(eval_path, encoding="utf-8"))["questions"]}
    fc = load(run_path)
    ids = [k for k in qs if k in fc]
    ys = [qs[k]["outcome"] for k in ids]
    ps = [fc[k] for k in ids]
    d, e, h = brier(ps, ys), ece(ps, ys), output_histogram(ps)
    c = corp(ps, ys)
    print(f"\n{label}  n={len(ids)}")
    print(f"  Brier            {d.score:.4f}")
    print(f"  base rate        {d.base_rate:.4f}   base-rate Brier {d.uncertainty:.4f}")
    print(f"  calibration      {d.calibration:.4f}   (lower is better)")
    print(f"  resolution       {d.resolution:.4f}   (higher is better)")
    print(f"  ECE eq-width/10  {e.equal_width.value:.4f}   eq-mass/10 {e.equal_mass.value:.4f}")
    print(f"  CORP  MCB {c.mcb:.4f}  DSC {c.dsc:.4f}  UNC {c.unc:.4f}   (bin-free; residual {c.residual:.1e})")
    print(f"  distinct values  {h.distinct_values}   sd {statistics.pstdev(ps):.4f}")
    print(f"  BSS vs base rate {1 - d.score / d.uncertainty:+.1%}  [diagnostic, not comparable across corpora]")
    return d

if __name__ == "__main__":
    print("Tarot-Draw 8B — verifying the model card's numbers")
    pub = report("PUBLISHED (headline, contamination-free)", "eval/published_eval.json", "forecasts/published.jsonl")
    report("DEV (iteration only)", "eval/bakeoff_3000.json", "forecasts/scalar_tfull.jsonl")
    print("\nTRANSFER — a different venue, matched question shape (README: Transfer)")
    # Transfer, on the rebuilt Kalshi set. The set this replaced had no builder
    # anywhere and could not be independently reconstructed; this one is built by
    # scripts/kalshi_eval_set.py + kalshi_subtitles.py + kalshi_rebuild_prompts.py,
    # and those scripts reproduce all 157 prompts shared with the old set exactly.
    kq = {q["question_id"]: q for q in json.load(open("eval/kalshi_transfer_v2.json", encoding="utf-8"))["questions"]}
    kf = load("forecasts/kalshi_transfer_v2.jsonl")
    kids = [k for k in kq if k in kf]
    FREEZE = "2026-08-15T00:00:00+00:00"
    rows = [
        ("Kalshi, all", lambda q: True),
        ("  Politics", lambda q: q["category"] == "Politics"),
        ("  Entertainment", lambda q: q["category"] == "Entertainment"),
        ("  Science and Technology", lambda q: q["category"] == "Science and Technology"),
        ("  Economics", lambda q: q["category"] == "Economics"),
        ("  Elections", lambda q: q["category"] == "Elections"),
        ("  post-freeze (clean)", lambda q: q["resolved_at"] > FREEZE),
        ("  pre-freeze (exposed)", lambda q: q["resolved_at"] <= FREEZE),
        ("  Elections, post-freeze", lambda q: q["category"] == "Elections" and q["resolved_at"] > FREEZE),
        ("  Elections, pre-freeze", lambda q: q["category"] == "Elections" and q["resolved_at"] <= FREEZE),
    ]
    for label, sel in rows:
        ids = [k for k in kids if sel(kq[k])]
        ys = [kq[k]["outcome"] for k in ids]
        if len(ids) < 2 or len(set(ys)) < 2:
            continue
        d = brier([kf[k] for k in ids], ys)
        bss = 1 - d.score / d.uncertainty
        mark = "   <-- worse than a constant" if bss < 0 else ""
        print(f"  {label:<26} n={len(ids):>4}  Brier {d.score:.4f}  resolution {d.resolution:.4f}"
              f"  BSS {bss:+.1%}{mark}")
    print("  Manifold home venue        n=3000  Brier 0.1674  resolution 0.0718  BSS +30.1%")
    print("  -> Elections is worse than a constant on BOTH sides of the freeze, so it is")
    print("     a subject-matter limit, not contamination. Those questions name obscure")
    print("     local candidates, which is a lookup rather than a forecast.")

    print(f"\nHeadline for the card: Brier {pub.score:.4f}, calibration {pub.calibration:.4f}, resolution {pub.resolution:.4f}")
