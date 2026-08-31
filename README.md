---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-8B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-classification
tags:
  - forecasting
  - probabilistic-forecasting
  - calibration
  - uncertainty-quantification
  - regression
  - prediction-markets
  - brier-score
  - lora
  - peft
  - qwen3
model-index:
  - name: cournot-cold-8b
    results:
      - task: {type: forecasting, name: Binary event forecasting}
        dataset:
          name: Cournot published split (Manifold, resolved after 2026-08-15 freeze)
          type: manifold-published
        metrics:
          - {type: brier, value: 0.1893, name: Brier score}
          - {type: calibration, value: 0.0048, name: Murphy calibration}
          - {type: resolution, value: 0.0627, name: Murphy resolution}
          - {type: ece, value: 0.0558, name: ECE (equal-width, 10 bins)}
---

# Cournot-Cold 8B

A **calibrated probability estimator** for binary questions about future events.
Given a question, its resolution criteria, its scheduled resolution date and an
`as_of` date, it returns a probability in [0, 1]. It conditions on the **question
text alone** — no retrieval, no news, no market price.

LoRA adapter + scalar regression head on `Qwen/Qwen3-8B`.

![Cournot-Cold 8B architecture](assets/architecture.svg)

**Weights:** [`Laplace-AI-Research/cournot-cold-8b`](https://huggingface.co/Laplace-AI-Research/cournot-cold-8b)
on the Hugging Face Hub — LoRA adapter, 349 MB.
**Smaller siblings:** [`Cournot-Cold 4B`](https://huggingface.co/Laplace-AI-Research/cournot-cold-4b)
and [`Cournot-Cold 1.7B`](https://huggingface.co/Laplace-AI-Research/cournot-cold-1-7b).
The 1.7B is measurably worse — +0.0119 Brier [+0.0080, +0.0158] against the 4B on
dev — and exists for the case where neither of the larger two will fit. Its card
carries a **training-variance** measurement that applies to this model too: two
runs of an identical recipe differ by more than a question-bootstrap interval
implies, so a model-vs-model difference below roughly **0.008 Brier** should be
treated as unresolved, including some of the intervals quoted on this page.

The 4B is trained on the same 81,870 questions with the same targets, seed and footing,
and **statistically indistinguishable from this model on four independent axes**:
dev (n=3,000, Brier delta +0.0010 [−0.0020, +0.0040]), the published split
(n=277), off-venue on Kalshi (n=1,219, +0.0018 [−0.0035, +0.0071]), and across
seeds. Its transfer results are now measured on that model rather than inherited
from this one. **Take the 4B** — half the base model, fits a 16 GB card, and on
accuracy they are the same model.
**Evidence:** [`Laplace-AI-Research/cournot-cold-8b`](https://github.com/Laplace-AI-Research/cournot-cold-8b)
on GitHub — the eval splits behind every claim, this model's raw forecasts
(including the venue transfers where it *failed*), the metric code, and
`verify.py`, which recomputes every number below without a model or a GPU.

The training corpus is **not** redistributed; `DATASHEET.md` documents its
composition, collection and known defects in its place. See
[Training data provenance and licensing](#training-data-provenance-and-licensing).

---

## The headline number

**Brier 0.1893** on 277 questions that resolved after a freeze date committed in
writing before it passed.

Reported the way the field's benchmark maintainers ask for it — a raw Brier with
its corpus, base rate and horizon attached, plus the decomposition — rather than
as a skill score against our own baseline. (ForecastBench built a
difficulty-adjustment methodology specifically because naive Brier/BSS
comparisons across different question sets have a rank correlation to true skill
of only ~0.64.)

| | published (headline) | dev (iteration only) |
|---|---:|---:|
| n | 277 | 3,000 |
| **Brier** | **0.1893** | 0.1674 |
| base rate | 0.4946 | 0.3970 |
| base-rate Brier (uncertainty) | 0.2500 | 0.2394 |
| Murphy calibration ↓ | 0.0048 | 0.0010 |
| Murphy resolution ↑ | 0.0627 | 0.0718 |
| ECE, equal-width / equal-mass (10 bins) | 0.0558 / 0.0525 | 0.0233 / 0.0260 |
| distinct output values | 83 | 99 |
| forecast sd | 0.250 | 0.283 |
| BSS vs base rate *(diagnostic, not a comparability claim)* | +24.3% | +30.1% |

Calibration is somewhat worse on published than on dev (ECE 0.056 vs 0.023) and
resolution is lower (0.063 vs 0.072). Both are what a smaller, harder, forward-
accumulating split tends to look like; neither currently rests on a single bin.
See [Known weaknesses](#known-weaknesses).

> **These numbers replaced an earlier set on 2026-08-27, and the reason is worth
> your attention.** The published split was found to be feeding the model each
> question's *actual resolution date* in place of its *scheduled close* —
> information not available at `as_of`. The split was rebuilt and the model
> re-scored. Full disclosure below under
> [The defect we found in our own evaluation](#the-defect-we-found-in-our-own-evaluation).

### For context, not as a comparison

ForecastBench's live *baseline* leaderboard (no tools, no retrieval) at time of
writing: superforecaster median 0.103, o3 0.145, Qwen3-235B-A22B 0.167,
Llama-3-8B-chat 0.178, zero-shot GPT-4 without retrieval 0.206. **These are
different questions, a different base rate, and a different horizon.** They tell
you the band we are in, not a ranking. We intend to submit to that leaderboard
so a real difficulty-adjusted comparison exists.

---

### Calibration without a binning choice

Expected calibration error depends on a binning scheme, and **neither available
scheme is defensible**: equal-width leaves most bins too sparse to estimate a
frequency on a skewed question set, and equal-mass has edges that move between
checkpoints, so two models get scored against different boundaries. We have been
publishing **both** numbers (0.0233 and 0.0260 here) because we could not choose.

**CORP removes the choice.** The reliability curve is the isotonic (PAV) fit, so
there are no bins at all:

| | value |
|---|---:|
| **MCB** — miscalibration ↓ | **0.0095** |
| **DSC** — discrimination ↑ | **0.0702** |
| UNC — uncertainty (a property of the split) | 0.2500 |

`score = MCB − DSC + UNC` holds **exactly**, with a residual of zero. The binned
Murphy decomposition carries a residual that depends on the scheme; this does
not.

Method: Dimitriadis, Gneiting & Jordan, *Stable reliability diagrams for
probabilistic classifiers*, PNAS 2021 ([arXiv:2008.03033](https://arxiv.org/abs/2008.03033)).
Recomputed from the shipped forecasts by `verify.py`; the ECE pair is retained
above for continuity with earlier versions of this card.

---

## Contamination

**Zero of the 3,000 dev questions and zero of the 277 published questions
resolved before the base model was released.** Outcome memorisation is therefore
closed by construction, not by argument.

- **Freeze: 2026-08-15**, committed in a dated, public git history *before* it
  passed. Anyone with the repository and a clock can check that nothing in the
  published split resolved before we committed to the freeze.
- The split is gated on **`Qwen3-8B`'s public release date (2025-04-29)**, not on
  a stated pretraining cutoff. This is deliberate: **Qwen3-8B publishes no data
  cutoff** — not in the technical report, not on the model card. A release date
  is externally checkable; a cutoff is a vendor claim.
- **88% of published questions (243/277) also *opened* after that release date**,
  and score **BSS +20.9% on their own** — so the result does not rest on
  questions that existed before the base model shipped.

**What this does not establish, and what has since been tested.** The published
split is Manifold-only, so on its own it shows the skill is not memorised
Manifold *outcomes* but cannot rule out a **Manifold-specific artifact**.

That second question has now been tested directly on a different venue — see
[Transfer](#transfer) — and the venue-specific reading is **refuted**: on Kalshi
Politics the model discriminates better than it does at home. What remains is not
a venue limit but a subject-matter one.

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

**The crowd-parity claim above is Manifold-only** — we have no price-series
comparison on another venue, so "beats the market" is not a claim this model
makes. The *skill* claim is broader and is set out under [Transfer](#transfer);
the *parity* claim is not.

---

## Transfer

**Skill crosses venues. It does not cross question types, and it does not reach
subjects without a public track record.**

Tested on **1,219 Kalshi judgment questions** — a different venue, real money,
long-horizon, matched to the question *shape* this model is for.

> **This section was rebuilt on 2026-08-30 and its numbers changed.** The set it
> replaced had **no builder anywhere**: its selection rule was stated in a
> metadata string that no code implemented, so nobody outside could reconstruct
> the population, and its prompts turned out to depend on a Kalshi API field
> present in no archive we kept. This set is built by
> [`scripts/kalshi_eval_set.py`](scripts/kalshi_eval_set.py),
> [`kalshi_subtitles.py`](scripts/kalshi_subtitles.py) and
> [`kalshi_rebuild_prompts.py`](scripts/kalshi_rebuild_prompts.py), which
> together reproduce **all 157 prompts shared with the old set exactly** and
> refuse to write otherwise. `verify.py` recomputes every number below.

| stratum | n | Brier | resolution | BSS |
|---|---:|---:|---:|---:|
| Manifold dev (home venue) | 3,000 | 0.1674 | 0.0718 | +30.1% |
| **Kalshi — Politics** | 170 | 0.0751 | **0.1328** | **+61.4%** |
| Kalshi — Entertainment | 500 | 0.1772 | — | +21.7% |
| Kalshi — Science & Technology | 97 | 0.1931 | — | +20.3% |
| Kalshi — Economics | 67 | 0.2597 | — | −6.8% |
| **Kalshi — Elections** | 382 | 0.2144 | **0.0198** | **−3.5%** |
| Kalshi — all | 1,219 | 0.1800 | — | +19.4% |

**Read the aggregate with care.** It is a composition artifact. Elections is
**31% of this corpus and scores worse than a constant**; Politics is 14% and
beats the home venue. Neither number alone is honest, so both are given.

**Calibration does not transfer even where discrimination does.** A new venue
needs its own calibration mapping, fit on that venue's own development split.

### Elections is a subject-matter limit, not contamination

This card previously carried a section titled *"the contamination-free Kalshi
subset, where this model does not beat a constant."* **That framing was wrong**,
and it was wrong in the direction that made us look careful.

That subset was **75% Elections**. The model failed on it because it cannot
forecast obscure local elections — which this same section already said about
the aggregate — not because the freeze boundary revealed anything. Holding
category fixed shows it directly:

| slice | n | BSS |
|---|---:|---:|
| Elections, resolved **after** the freeze | 69 | −5.8% |
| Elections, resolved **before** the freeze | 313 | −3.3% |

**Post-freeze is not better.** If pre-freeze performance were inflated by the
base model having seen those questions, the clean subset would be worse. It is
not. The limit is the subject, not the contamination.

For completeness, and because it is the least explained number here: the
post-freeze slice as a whole scores Brier 0.1406 (BSS +43.7%) against 0.1870
(+12.7%) pre-freeze. That gap is driven by **Entertainment at n=84**, not by
the freeze, and it has no explanation yet. **It should not be quoted as evidence
about contamination in either direction.**

## Known weaknesses

Named specifically, because a limitations section that hedges generically is not
a limitations section.

**1. It needs the question's subject to have a public reference class.** This is
the sharpest limit, and it is not a venue limit — see [Transfer](#transfer).

On 461 Kalshi *Elections* questions the model collapses to resolution **0.0080**
[0.0044, 0.0215], BSS **−23.1%**. Those questions name obscure individuals in
local races — *"Will Peter Chatzky be the Democratic nominee for NY-17?"*, *"Who
will win the 2026 Ann Arbor Democratic mayoral primary?"* There is no reference
class for such a name in a text-only prior. **That is a lookup, not a forecast,
and this model cannot perform it.**

The same limit explains most of the Polymarket result below.

**2. It cannot do mechanical threshold or counting questions.** Questions of the
form *"Will Trump say 'X' this week?"* or *"Will TSA passengers on Feb 15 be
between 1,500,000 and 1,700,000?"* need a time series and precise arithmetic, not
a judgment prior. Frontier models score **13.5–16.0%** on date-duration
arithmetic against 58.6–76.3% on date addition ([Test of
Time](https://arxiv.org/abs/2406.09170), Table 8). Independently corroborated off-venue: **48% of a real-money venue's own
judgment-category markets are numeric-threshold questions**, and they are the
half we cannot do.

**3. It is weakest on sports.** On a keyword-identified sports subset (236
questions, 8% of dev): Brier 0.2379, resolution 0.0320, **BSS +4.8%** — against
+32.0% on everything else. Manifold publishes no category or tag metadata, so
this subset was identified by text keywords and **under-recalls**; treat 8% as a
lower bound and the subset as approximate. We report it because the opposite
worry (that the headline was propped up by sports) is a reasonable one and the
data refutes it — excluding sports *raises* the aggregate.

**4. Calibration is worse on published than on dev**, ECE 0.056 vs 0.023, and
resolution is lower, 0.063 vs 0.072. No single bin is responsible: the largest
equal-width reliability gap is **−0.184 in the 0.2–0.3 band at n=13**, and the
mid-range bands that carry most of the mass are close — 0.5–0.6 is n=47 at
−0.030, 0.4–0.5 is n=63 at −0.097.

An earlier version of this card attributed the published-split calibration gap to
one 13-question bin at 0.52–0.59. **That bin does not survive the corrected
split** (see below); the gap is now smaller and diffuse rather than concentrated.
We are recording the change rather than quietly restating the conclusion.

The honest reading at n=277 is that published is a smaller and somewhat harder
sample than dev, not that a specific defect has been located. It will be
re-checked as the split accumulates — the forward accrual is the only thing that
settles it.

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

**In scope.** Binary questions about future events that are (a) **long-horizon**,
beyond ~30 days, and (b) about **subjects with a public track record** — an
institution, an office, a well-known organisation or person. A cheap prior to be
updated by other evidence; cold-start estimates where no market exists; research
on calibration and forecasting. Validated on two independent venues (Manifold,
Kalshi) under those two conditions.

**Out of scope.**
- Questions resolving **within a week** — it never beats the crowd there.
- **Mechanical threshold, counting or arithmetic** questions.
- Questions about **subjects with no public reference class** — a local primary
  candidate, a private individual, an obscure entity. This is the sharpest limit
  and the model gives *worse than base-rate* answers in this regime.
- Any **new venue without a fresh calibration mapping**. Discrimination transfers;
  calibration does not.
- Any setting where a miscalibrated probability causes harm — this model has **no**
  evidence channel and cannot know anything that happened after its base model was
  trained.

**Never** use it as the sole input to a consequential decision. It is a prior.

---

## Evaluation data

- **`published`** (headline, n=277) — Manifold questions resolving after
  2026-08-15. Never trained on, contamination-free by construction. Accumulates
  forward; the only source of an external number. Built by
  `scripts/published_eval_set.py`, which refuses any question whose scheduled
  close time is unknown rather than substituting a default — see below for why
  that refusal exists.
- **`dev`** (n=3,000) — resolving 2025-08-15 to 2026-08-15. Gates iteration.
  **Never published as a headline claim.**
- **Parity** (n=1,741) — the subset with a usable price series, for crowd
  comparison at five values of φ. Each question appears once per φ; rows are
  **not independent** and must be compared within φ, paired on question id.

All intervals are paired, question-clustered bootstraps (10,000 resamples).
Seed-to-seed noise on this setup is **±0.003 Brier**, so smaller differences are
not findings.

---

## The defect we found in our own evaluation

On **2026-08-27**, before this model was released publicly, we found that the
published evaluation split had been feeding the model information it could not
have had.

**What it was.** Every forecast prompt carries a resolution date. The date a
forecaster actually has at `as_of` is the question's **scheduled close**. Our
published split was supplying the date the question **actually resolved** — on
132 of 132 rows. The corresponding figure on `dev`, which is built from a
different source, was 37.7%.

**How it happened.** The script that collects post-freeze questions from the
Manifold API recorded the title, resolution and creation time but **dropped
`closeTime`**, a field present in the same API response. With no close time
available, whatever built the eval set filled the gap with the resolution time.

**Why it went unnoticed.** `published_eval.json` had **no builder script**. It
entered the repository as a committed JSON file with no code behind it, so it
could not be regenerated, diffed or reviewed. The defect lived in the one
artifact in the pipeline nobody could reproduce. That is the part we consider
most instructive.

**What we fixed.** The collector now captures `closeTime`; a backfill recovered
it for questions already gathered; and the split is now built by
`scripts/published_eval_set.py`, which **refuses a question with no close time
rather than defaulting one**, because defaulting is how the defect entered. The
builder reproduces the previously shipped rows field-for-field and is
mutation-tested.

**What changed, and what we cannot claim.** The corrected split is larger (277 vs
132, partly from forward accrual) and its fields now match `dev`'s semantics
(41.9% vs 37.7% coincidental matches), so the two are commensurable for the first
time. Headline Brier moved 0.1800 → 0.1893, calibration 0.0241 → 0.0048, ECE
0.104 → 0.056.

**We cannot attribute those changes to the fix.** On the 132 questions common to
both, holding weights and adapter constant and changing only the prompt field, the
paired question-clustered interval is **+0.0070 Brier [−0.0155, +0.0330]** and
**−0.0118 resolution [−0.0376, +0.0143]** — both spanning zero, and underpowered:
the Brier half-width is 3.5× the effect, and detecting it would need n≈1,584. The
difference between the old and new headline is dominated by the split containing
different questions, not by the correction.

So the honest statement is not "the old number was inflated." It is: **the old
number was produced under a defective input, and the magnitude of that defect is
below what this sample can measure.** The fix is justified by the leak itself,
not by its measured effect.

Derivation sits in an internal decisions log which is **not public**. Everything
needed to check the claim is in this repository: the corrected split, the
regenerated forecasts, and `verify.py`.

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
- **Calibration: none applied.** The head's raw `sigmoid(logit)` is the shipped
  probability -- `scalar_score.py` applies no post-hoc mapping, and no fitted
  calibrator is distributed. An earlier version of this line claimed beta
  calibration fit on `dev`; that step was never part of the pipeline. The
  calibration figures above are therefore what the head produces **untuned**,
  which is a stronger result than the original claim, not a weaker one.
- **Seed: replicated.** A second full-corpus run at seed 20260828 differs by
  **+0.0012 Brier [−0.0016, +0.0041]** — not significant, half-width at the
  ±0.003 noise floor. The shipped number is not one seed's luck.

**Reproduction note:** `transformers >= 4.51.0` is required for
`Qwen3ForSequenceClassification`; `config.pad_token_id` must be set explicitly.

---

## Licensing, in two parts

| what | licence | commercial use |
|---|---|---|
| **code** — `src/`, `scripts/`, `verify.py` | **Apache-2.0** (`LICENSE-CODE`) | **permitted** |
| adapter weights | CC BY-NC 4.0 (`LICENSE`) | not permitted |
| forecasts | CC BY-NC 4.0 (`LICENSE`) | not permitted |
| eval metadata — ids, dates, outcomes | CC BY-NC 4.0 (`LICENSE`) | not permitted |
| question text | **not redistributed here** | not ours to license |
| base model | Apache-2.0, by its authors | unaffected |

**The split is deliberate.** The evaluation code contains no third-party rights
and is permissively licensed, including for commercial use. The data-derived
artifacts cannot be, because the corpus they come from restricts it.

**Corrected 2026-08-29:** an earlier version of this repository shipped the code
with **no licence grant at all**, which under copyright means all rights reserved
— published, but not usable by anyone. That was not intended and is fixed here.

---

## Training data provenance and licensing

The corpus is derived from the Manifold Markets public API.

**Manifold's terms restrict bulk API data to personal and non-commercial use, and
state that it may not be used to train machine learning models for commercial
purposes without a data licence** (data@manifold.markets). This adapter is
released under **CC BY-NC 4.0** on that basis — research and non-commercial use.
Anyone intending commercial use must obtain that licence from Manifold
independently; it is not ours to grant.

The raw corpus is **not redistributed**. A datasheet describing its composition,
collection and known defects is published in its place.

The base model's own licence (Apache 2.0) is unaffected and permits commercial
use of the base weights.

---

## Reproducing

Every headline number in this card is recomputed from the shipped forecasts by
`verify.py` — **no model, no GPU, no network**:

```bash
uv run python verify.py
```

To regenerate those forecasts from the weights instead:

```bash
uv run python scripts/scalar_score.py \
  --adapter Laplace-AI-Research/cournot-cold-8b \
  --set eval/published_eval.json \
  --out published.jsonl \
  --chat-template
```

Every eval run writes a manifest carrying model hash, data snapshot hash, eval
split id and git SHA. Numbers without a manifest are not comparable.

---

## Citation and provenance of claims

Every number in this card is recomputed from the shipped forecasts by
`verify.py` — no model, no GPU, no network. **That is the check that matters,
and it is the one you can run.**

Behind it sits an internal decisions log, **not public**, which also records the
results that failed: a varied-`as_of` arm that hedged, a mixed-venue arm that
traded Manifold resolution for Polymarket robustness, two claims that died under
a paired bootstrap, a leak found in this project's own published split, and a
citation retracted after we could not find it in the paper we had attributed it
to. Nothing in this card depends on that log — every claim here is either
reproducible from the files in this repository or stated as unverified.
