# Datasheet — Cournot-Cold 8B training and evaluation corpora

Follows Gebru et al., *Datasheets for Datasets* (arXiv:1803.09010). Published in
place of the raw corpus, which is **not redistributed** — see Distribution.

## Motivation

Built to train and evaluate a text-only probability estimator for binary
questions about future events. Assembled by Laplace AI Research; not funded by a
third party.

## Composition

| corpus | n | window | role |
|---|---:|---|---|
| training | 81,870 | resolved before 2025-08-14 | fine-tuning |
| `dev` | 3,000 | resolved 2025-08-15 → 2026-08-15 | iteration only, never a published claim |
| `published` | 132 | resolved after 2026-08-15 | the only source of an external number |
| parity | 1,741 | subset of dev with a usable price series | crowd comparison at five values of φ |
| Polymarket transfer | 3,000 | dev window | out-of-venue check only, never trained on |

Each instance is a question: text, resolution criteria, open date, scheduled
resolution date, realised resolution date, binary outcome. **No user data, no
personal data, no free-text user content.**

Base rates differ by split (train 0.4163, dev 0.3970, published 0.4621) and are
stated wherever a score is quoted, because Brier is not comparable across base
rates.

## Collection

Manifold Markets public API, normalised into a common schema shared across
venues. Resolution outcomes are taken from the platform's own settlement, not
inferred. Timestamps are UTC, timezone-aware.

**Known defects, recorded rather than smoothed:**
- Manifold publishes **no category or tag metadata** — both fields are null for
  every question. Any category breakdown in the model card is therefore a text
  keyword heuristic that under-recalls, not ground truth.
- The Polymarket comparison corpus required filtering: 17.6% of an unfiltered
  sample had titles stating no YES condition at all (e.g. "ITF Women Merzig:
  Completed Match: X vs Y") and carried a base rate of 0.799 against 0.230 for
  the rest. Those are excluded and the filter is recorded in the eval metadata.
- Polymarket volume and liquidity are null for every candidate in our window, so
  no liquidity-matched comparison was possible; Manifold's set required ≥10
  forecasters. **This asymmetry favours neither side cleanly and must be stated
  with any cross-venue comparison.**

## Preprocessing

Questions with malformed or missing outcomes are dropped and **counted**, not
repaired. Targets are the terminal binary outcome. No evidence documents are
attached at any stage — this is a no-retrieval model.

## Splits and contamination

The split keys on `resolved_at`. The freeze (2026-08-15) was committed in a dated
public git history **before it passed**, so "nothing in `published` resolved
before we committed to the freeze" is externally checkable.

**Zero questions in either evaluation split resolved before the base model's
public release (Qwen3-8B, 2025-04-29)**, so outcome memorisation is closed by
construction. 88% of `published` also *opened* after that date.

The gate is release date, not a stated pretraining cutoff — **Qwen3-8B publishes
no cutoff anywhere**, and a release date is externally checkable while a vendor
cutoff is not.

## Uses

Used to train and evaluate Cournot-Cold 8B. Not suitable for: studying Manifold
users or trading behaviour (no user data retained); any claim about venues other
than Manifold (transfer fails — see the model card); mechanical threshold or
counting questions.

## Distribution

**Neither the raw corpus nor any venue's question text is redistributed.**
Manifold's terms restrict bulk API data to personal and non-commercial use and
prohibit training ML models for commercial purposes without a data licence
(data@manifold.markets). Kalshi's terms have not been reviewed by us and no
position is asserted about them.

The evaluation splits in `eval/` are published so results are checkable, and they
carry **question ids, dates and outcomes only — no question titles.** An earlier
version of these files shipped verbatim titles, which contradicted this section;
they were removed on 2026-08-28 and every number in the model card still
reproduces, because `verify.py` joins on `question_id` and never reads the text.

Each row's `question_id` is the venue's own stable identifier, so anyone may
retrieve the corresponding text from that venue directly, subject to that venue's
terms rather than ours.

## Maintenance

`published` accumulates forward as questions resolve. Its statistics will move,
and the model card's headline is expected to be re-derived rather than frozen.
Corrections and retractions are recorded in an internal decisions log, which is
**not public**. Where a correction affects a number in this repository, the
corrected number and the reason are carried here — see the model card.
