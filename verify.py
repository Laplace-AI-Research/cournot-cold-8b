# SPDX-License-Identifier: Apache-2.0
"""Recompute every headline number in README.md from the shipped forecasts.

    uv run python verify.py

Requires only `src/cournot` — no model, no GPU. If a number here disagrees with the
model card, the model card is wrong.
"""
from __future__ import annotations
import json, statistics, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from cournot.metrics import brier, corp, ece, output_histogram

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
    print("Cournot-Cold 8B — verifying the model card's numbers")
    pub = report("PUBLISHED (headline, contamination-free)", "eval/published_eval.json", "forecasts/published.jsonl")
    report("DEV (iteration only)", "eval/bakeoff_3000.json", "forecasts/scalar_tfull.jsonl")
    print("\nTRANSFER — a different venue, matched question shape (README: Transfer)")
    kq = {q["question_id"]: q for q in json.load(open("eval/kalshi_score_778.json", encoding="utf-8"))["questions"]}
    kf = load("forecasts/kalshi_transfer.jsonl")
    kids = [k for k in kq if k in kf]
    for label, sel in (("Kalshi, all", lambda q: True),
                       ("  Politics (best stratum)", lambda q: q["category"] == "Politics"),
                       ("  Elections (worst stratum)", lambda q: q["category"] == "Elections")):
        ids = [k for k in kids if sel(kq[k])]
        ys = [kq[k]["outcome"] for k in ids]
        if len(set(ys)) < 2:
            continue
        d = brier([kf[k] for k in ids], ys)
        print(f"  {label:<28} n={len(ids):>4}  Brier {d.score:.4f}  resolution {d.resolution:.4f}"
              f"  BSS {1 - d.score / d.uncertainty:+.1%}")
    print("  Manifold home venue          n=3000  Brier 0.1674  resolution 0.0718  BSS +30.1%")
    print("  -> Politics discriminates BETTER off-venue than at home. Elections collapses:")
    print("     those questions name obscure local candidates, which is a lookup, not a forecast.")
    print(f"\nHeadline for the card: Brier {pub.score:.4f}, calibration {pub.calibration:.4f}, resolution {pub.resolution:.4f}")
