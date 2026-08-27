---
license: apache-2.0
base_model: Qwen/Qwen3-8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-classification
tags: [forecasting, calibration, prediction-markets, brier-score, lora]
model-index:
  - name: pari-prior-8b
    results:
      - task: {type: forecasting, name: Binary event forecasting}
        dataset:
          name: Pari published split (Manifold, resolved after 2026-08-15 freeze)
          type: manifold-published
        metrics:
          - {type: brier, value: 0.1800, name: Brier score}
          - {type: calibration, value: 0.0241, name: Murphy calibration}
          - {type: resolution, value: 0.0900, name: Murphy resolution}
          - {type: ece, value: 0.1037, name: ECE (equal-width, 10 bins)}
---

# Pari Prior 8B

A **calibrated probability estimator** for binary questions about future events.
Given a question, its resolution criteria, its scheduled resolution date and an
`as_of` date, it returns a probability in [0, 1]. It conditions on the **question
text alone** — no retrieval, no news, no market price.

LoRA adapter + scalar regression head on `Qwen/Qwen3-8B`.

---

## The headline number

**Brier 0.1800** on 132 questions that resolved after a freeze date committed in
writing before it passed.

Reported the way the field's benchmark maintainers ask for it — a raw Brier with
its corpus, base rate and horizon attached, plus the decomposition — rather than
as a skill score against our own baseline. (ForecastBench built a
difficulty-adjustment methodology specifically because naive Brier/BSS
comparisons across different question sets have a rank correlation to true skill
of only ~0.64.)

| | published (headline) | dev (iteration only) |
|---|---:|---:|
| n | 132 | 3,000 |
| **Brier** | **0.1800** | 0.1674 |
| base rate | 0.4621 | 0.3970 |
| base-rate Brier (uncertainty) | 0.2486 | 0.2394 |
| Murphy calibration ↓ | 0.0241 | 0.0010 |
| Murphy resolution ↑ | 0.0900 | 0.0718 |
| ECE, equal-width / equal-mass (10 bins) | 0.1037 / 0.1086 | 0.0233 / 0.0260 |
| distinct output values | 70 | 99 |
| forecast sd | 0.268 | 0.283 |
| BSS vs base rate *(diagnostic, not a comparability claim)* | +27.6% | +30.1% |

**Calibration is materially worse on the published split than on dev** (ECE 0.104
vs 0.023). We do not think this is a fluke to be explained away — see
[Known weaknesses](#known-weaknesses), where the responsible bin is identified.

### For context, not as a comparison

ForecastBench's live *baseline* leaderboard (no tools, no retrieval) at time of
writing: superforecaster median 0.103, o3 0.145, Qwen3-235B-A22B 0.167,
Llama-3-8B-chat 0.178, zero-shot GPT-4 without retrieval 0.206. **These are
different questions, a different base rate, and a different horizon.** They tell
you the band we are in, not a ranking. We intend to submit to that leaderboard
so a real difficulty-adjusted comparison exists.

---

## Contamination

**Zero of the 3,000 dev questions and zero of the 132 published questions
resolved before the base model was released.** Outcome memorisation is therefore
closed by construction, not by argument.

- **Freeze: 2026-08-15**, committed in a dated, public git history *before* it
  passed. Anyone with the repository and a clock can check that nothing in the
  published split resolved before we committed to the freeze.
- The split is gated on **`Qwen3-8B`'s public release date (2025-04-29)**, not on
  a stated pretraining cutoff. This is deliberate: **Qwen3-8B publishes no data
  cutoff** — not in the technical report, not on the model card. A release date
  is externally checkable; a cutoff is a vendor claim.
- **88% of published questions (116/132) also *opened* after that release date**,
  and score **BSS +25.0% on their own** — so the result does not rest on
  questions that existed before the base model shipped.

**What this does not establish.** The published split is Manifold-only. It shows
our Manifold skill is not memorised Manifold *outcomes*. It cannot show that
skill is not a **Manifold-specific artifact** — pattern-matching on that
platform's phrasing, creator conventions or topic base rates. That is an open
question, stated here rather than in a footnote, and the experiment that would
settle it (a published-side evaluation on a second venue) has not been run.

---

## Where it works, and where it does not

Forecasts are made at question **open** and compared against the Manifold crowd
price read at a fraction φ through each question's life. This is deliberately
unfair to us — the crowd has seen news we have not.

| horizon | crossover φ\* | our BSS at φ=0.10 | crowd BSS at φ=0.10 |
|---|---:|---:|---:|
| under 7 days | **never beats the crowd** | +6.9% | +15.6% |
| 7–30 days | 0.31 | +26.6% | +21.9% |
| 30–120 days | 0.46 | +33.6% | +20.8% |
| 120+ days | **0.62** | **+44.3%** | +19.4% |

**The supported claim:** on questions with a horizon beyond 30 days, this model
beats the Manifold crowd from question open through roughly the first half of the
question's life; past 120 days, through about 62% of it, at more than twice the
crowd's skill against the base rate. **On questions resolving within a week it
never beats the crowd at any point.**

The mechanism: the crowd's skill is roughly horizon-invariant (+15.6% → +19.4%
across the whole range) while ours rises steeply (+6.9% → +44.3%). We do not win
because the crowd degrades on long-horizon questions — we win because we improve
and it stays flat.

**This is a Manifold claim.** "Beats the market" is not a claim this model makes.

---

## Known weaknesses

Named specifically, because a limitations section that hedges generically is not
a limitations section.

**1. It does not transfer to another venue.** On 3,000 Polymarket questions,
resolution collapses from 0.0718 to **0.0048** — about 6% retained — and Brier
(0.2059) is **worse than always predicting the base rate** (0.1873). Recalibration
cannot fix this: an in-sample decile recalibration, fit on the test set and
therefore an optimistic ceiling, reaches only 0.1825.

**2. It cannot do mechanical threshold or counting questions.** Questions of the
form *"Will Trump say 'X' this week?"* or *"Will TSA passengers on Feb 15 be
between 1,500,000 and 1,700,000?"* need a time series and precise arithmetic, not
a judgment prior. Frontier models score **13.5–16.0%** on date-duration
arithmetic against 58.6–76.3% on date addition ([Test of
Time](https://arxiv.org/abs/2406.09170), Table 8). This is the largest single
component of the transfer failure above.

**3. It is weakest on sports.** On a keyword-identified sports subset (236
questions, 8% of dev): Brier 0.2379, resolution 0.0320, **BSS +4.8%** — against
+32.0% on everything else. Manifold publishes no category or tag metadata, so
this subset was identified by text keywords and **under-recalls**; treat 8% as a
lower bound and the subset as approximate. We report it because the opposite
worry (that the headline was propped up by sports) is a reasonable one and the
data refutes it — excluding sports *raises* the aggregate.

**4. Calibration degrades on the published split**, ECE 0.104 vs 0.023 on dev.
Both binning schemes agree on the culprit: forecasts in the **0.52–0.59** band
(n=13) had an observed rate of 0.154 against a mean forecast of 0.557, a gap of
+0.40. At n=13 this is one unlucky bin, not a diagnosed defect — but it is the
reason the headline calibration term is 0.0241, and it will be re-checked as the
published split accumulates.

**5. It is trained at a single `as_of`.** Every training example uses
`as_of = open_date`, so the model is off-distribution at any later `as_of` and
degrades as `as_of` moves toward resolution. A varied-`as_of` training arm was
run and **failed**: it flattened the horizon curve by hedging, losing resolution
at every φ. The defect stands, unfixed.

**6. Single model, no ensemble.** Calibration under distribution shift is the
known fragile regime for single models ([Ovadia et
al.](https://arxiv.org/abs/1906.02530)); ensembling was the only method they
found robust. We do not ensemble.

---

## Intended use

**In scope.** Estimating base rates for binary questions about future events,
especially at long horizon; a cheap prior to be updated by other evidence;
cold-start estimates where no market exists; research on calibration and
forecasting.

**Out of scope.** Questions resolving within days. Mechanical threshold, counting
or arithmetic questions. Venues other than Manifold without re-validation. Any
setting where a miscalibrated probability causes harm — this model has **no**
evidence channel and cannot know anything that happened after its base model was
trained.

**Never** use it as the sole input to a consequential decision. It is a prior.

---

## Evaluation data

- **`published`** (headline, n=132) — Manifold questions resolving after
  2026-08-15. Never trained on, contamination-free by construction. Accumulates
  forward; the only source of an external number.
- **`dev`** (n=3,000) — resolving 2025-08-15 to 2026-08-15. Gates iteration.
  **Never published as a headline claim.**
- **Parity** (n=1,741) — the subset with a usable price series, for crowd
  comparison at five values of φ. Each question appears once per φ; rows are
  **not independent** and must be compared within φ, paired on question id.

All intervals are paired, question-clustered bootstraps (10,000 resamples).
Seed-to-seed noise on this setup is **±0.003 Brier**, so smaller differences are
not findings.

---

## Training

- **Base:** `Qwen/Qwen3-8B` (Apache 2.0), LoRA r=32 α=64 dropout=0.05,
  `all-linear`, plus a trainable scalar head (`modules_to_save=["score"]`).
- **Objective:** Brier/MSE loss on a scalar output against the **terminal 0/1
  outcome**. Under a proper scoring rule a 0/1 label is an unbiased Bernoulli
  observation; on our corpus it beat a lower-variance market-price target on
  resolution, because the price target is biased.
- **Corpus:** 81,870 resolved Manifold questions, all resolving before
  2025-08-14.
- **Footing:** chat template on **both** train and score. This is load-bearing —
  a mismatch here silently invalidated an earlier headline result — and was
  re-established empirically by reproducing stored forecasts (mean abs diff
  0.0018 with template, 0.2015 without).
- **Padding:** right. The head pools the last non-pad token — the opposite
  convention from generation.
- **Calibration:** beta calibration, fit on `dev`, never on `published`.

**Reproduction note:** `transformers >= 4.51.0` is required for
`Qwen3ForSequenceClassification`; `config.pad_token_id` must be set explicitly.

---

## Training data provenance and licensing

The corpus is derived from the Manifold Markets public API.

**Manifold's terms restrict bulk API data to personal and non-commercial use, and
state that it may not be used to train machine learning models for commercial
purposes without a data licence** (data@manifold.markets). This adapter is
released for **research and non-commercial use** on that basis. Anyone intending
commercial use must obtain that licence independently.

The raw corpus is **not redistributed**. A datasheet describing its composition,
collection and known defects is published in its place.

The base model's own licence (Apache 2.0) is unaffected and permits commercial
use of the base weights.

---

## Reproducing

**Check the numbers without a GPU.** Every figure in this card is recomputed
from the shipped forecasts by:

```bash
uv run python verify.py
```

If it disagrees with this card, the card is wrong.

**Regenerate the forecasts** (needs the adapter and a GPU):

```bash
uv run python scripts/scalar_score.py \
  --adapter <path-to-adapter> \
  --set eval/published_eval.json \
  --out forecasts/published.jsonl \
  --chat-template
```

`--chat-template` is not optional — see Footing above.

Every eval run writes a manifest carrying model hash, data snapshot hash, eval
split id and git SHA. Numbers without a manifest are not comparable.

## What is in this repository

| path | |
|---|---|
| `README.md` | this model card |
| `DATASHEET.md` | corpus provenance, per Gebru et al. |
| `verify.py` | recomputes every headline number from the shipped forecasts |
| `eval/` | the evaluation splits behind every claim |
| `forecasts/` | this model's raw outputs, including the Polymarket transfer failure |
| `src/pari/`, `scripts/` | the metric and scoring code |

**Weights are not in this repository.** The LoRA adapter is 349 MB and belongs on
the Hugging Face Hub, not in git.

**The research record — every experiment, including the arms that failed — is in
[`Laplace-AI-Research/pari-research`](https://github.com/Laplace-AI-Research/pari-research).**
This repository is the artifact; that one is how it was arrived at.

---

## Citation and provenance of claims

Every number in this card is derived in `docs/10-decisions.md`, which also
records the results that **failed** — a varied-`as_of` arm that hedged, a
mixed-venue arm that traded Manifold resolution for Polymarket robustness, two
claims that died under a paired bootstrap, and a citation we retracted after
failing to find it in the paper we had attributed it to.
