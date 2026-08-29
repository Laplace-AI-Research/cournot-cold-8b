# SPDX-License-Identifier: Apache-2.0
"""Train a scalar probability head under Brier loss. `docs/14`, Phase 2's stage.

    uv run python scripts/scalar_train.py --corpus data/sft_corpus_v4.jsonl \
        --eval-set data/bakeoff_3000.json --out runs_scalar/arm_price

Replaces `docs/05` RL, deferred 2026-08-24. The argument is in the internal decisions log
(direction re-scope); what matters here is the three properties this shape has
and text SFT does not:

1. **The objective is the metric.** Brier loss on a probability output is a
   proper scoring rule evaluated directly. Next-token cross-entropy over a digit
   string is not -- `0.31` and `0.29` share no gradient relationship.
2. **Minimum target variance**, which is the `brier` experiment's finding applied
   at the smallest scale that tests it.
3. **No decoder to collapse.** v1 lost 438 of 500 forecasts to argmax; a scalar
   head has no argmax. the internal decisions log (2026-08-24g) showed the sign of the SFT effect
   depends on the decoder. Here there is no such degree of freedom.

## What the head is

`AutoModelForSequenceClassification` with `num_labels=1`, which for a causal
model pools the **last non-pad token** -- so the prompt's final hidden state
becomes one logit, and `sigmoid(logit)` is the forecast. LoRA over the base plus
`modules_to_save=["score"]`, because the head is newly initialised and adapters
alone would leave it untrained on reload.

**Right padding, deliberately.** Sequence classification reads the last non-pad
token, so left padding would put the pad run *after* the content and pool the
wrong position. This is the opposite of the generation case, and getting it
backwards is silent: the model trains, the loss falls, and every forecast is
read off a pad token.

## What this run answers

Whether resolution moves against the **thinking** base -- the baseline
2026-08-24e/g established as the one that binds, and the one every earlier
comparison in this repo failed to use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sft_train import PROMPT, install_teardown, load_corpus

from cournot.parsing import render_forecast

#: Targets are probabilities. A 0 or 1 is legal under a proper scoring rule --
#: unlike text SFT, where it is a string to emit (`docs/04`) -- but it is still
#: worth knowing how many there are, because an all-terminal corpus is the
#: high-variance arm and should be labelled as such rather than discovered.
EXTREME = 1e-6

#: What the scalar head writes in place of a rationale. The number comes from
#: the head; `cournot.parsing.render_forecast` owns the shape, so the scalar arm is
#: scored by the same contract implementation as every text arm.
SCALAR_REASONING = "scalar head"


def brier_loss(predictions, targets):
    """Mean squared error between the prediction and its target.

    Against a **realized outcome** this is exactly Brier. Against a **soft
    target** it is not: it is regression error toward a price, and it is
    numerically much smaller because a price is far easier to predict than a
    coin flip. The first run logged ~0.05 and that is a fit statistic, not a
    forecast score -- the training log says `mse-vs-target` for that reason.

    Forecast skill comes only from `scripts/decode_compare.py`, scoring the
    written forecasts against outcomes through the same harness as every other
    arm.
    """
    return ((predictions - targets) ** 2).mean()


def build_texts(rows: Sequence[dict[str, object]], tokenizer, chat_template: bool) -> list[str]:
    texts: list[str] = []
    for r in rows:
        body = PROMPT.format(
            question=r["question"],
            criteria=r.get("criteria") or "Resolves YES if the event occurs.",
            scheduled=r.get("scheduled") or "unknown",
            as_of=r.get("as_of") or "unknown",
        )
        if chat_template:
            body = tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        texts.append(body)
    return texts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--forecasts", required=True, help="where to write eval-set predictions")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260825,
        help=(
            "seeds torch, so shuffling and head initialisation are reproducible. "
            "Varying only this measures run-to-run spread, which is the check "
            "that separates a real effect from one seed's luck."
        ),
    )
    parser.add_argument("--teardown-pod", default="")
    parser.add_argument(
        "--teardown-after-seconds",
        type=int,
        default=1800,
        help=(
            "on SUCCESS, wait this long before terminating the pod, leaving a "
            "window to fetch results. On failure teardown is immediate, since "
            "there is nothing to fetch. 0 tears down immediately either way.\n\n"
            "This exists because on 2026-08-25 a job finished at 06:37 and the "
            "pod ran until 17:00 -- 11.1 hours billed for 80 minutes of work. "
            "The stall detector did not cover it: that guards a HUNG process, "
            "and this one exited cleanly. Nothing tore down a pod whose work "
            "was done."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    teardown = install_teardown(args.teardown_pod, bool(args.teardown_pod))
    try:
        code = run(args)
        if args.teardown_pod and args.teardown_after_seconds > 0:
            print(
                f"[teardown] job complete; terminating in "
                f"{args.teardown_after_seconds}s -- fetch results now",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(args.teardown_after_seconds)
        return code
    finally:
        teardown.fire("process exit")


def run(args) -> int:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Extreme targets are LEGAL here and illegal in text SFT -- see
    # `cournot.sft.check_extreme_share`. The terminal arm is the high-variance half
    # of the comparison this stage exists to run, so refusing it would refuse
    # the experiment.
    torch.manual_seed(args.seed)
    print(f"torch seed {args.seed}", file=sys.stderr)

    # Two text-SFT assumptions relaxed, both deliberately:
    #   allow_extreme_targets -- a 0/1 label is a proper Bernoulli observation
    #     under Brier loss, and is the PREFERRED target (`docs/14`, amended
    #     2026-08-25e).
    #   require_reasoning -- the head pools the PROMPT's final hidden state and
    #     `build_texts` never passes the reasoning, so demanding it would reject
    #     a corpus this stage can train on perfectly well. That is what makes
    #     the whole 81,870-question train split reachable without generation.
    rows = load_corpus(args.corpus, allow_extreme_targets=True, require_reasoning=False)
    if not rows:
        print("empty corpus", file=sys.stderr)
        return 1
    targets = [float(r["target"]) for r in rows]  # type: ignore[arg-type]
    extreme = sum(1 for t in targets if t <= EXTREME or t >= 1.0 - EXTREME)
    print(
        f"{len(rows):,} examples, mean target {sum(targets) / len(targets):.4f}, "
        f"{extreme:,} at 0 or 1 ({extreme / len(rows):.1%})",
        file=sys.stderr,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding: sequence classification pools the last NON-PAD token, so a
    # left pad run would place the content before the padding and pool a pad.
    tokenizer.padding_side = "right"

    texts = build_texts(rows, tokenizer, args.chat_template)

    class Corpus(Dataset):
        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, i: int):
            return texts[i], targets[i]

    def collate(batch):
        strings = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        encoded = tokenizer(
            strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        )
        return encoded, ys

    if args.dry_run:
        print("dry run: corpus loaded and tokenized, training skipped")
        return 0

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            # The head is newly initialised; without this it is never saved and
            # a reloaded adapter reads from random weights.
            modules_to_save=["score"],
        ),
    )
    model.print_trainable_parameters()
    # Sequence classification keeps hidden states for the whole batch, so this
    # is heavier per example than causal SFT at the same batch size. A first run
    # at batch 8 OOM'd at step 800 of 922 on a 44 GiB A40 -- late enough to
    # waste the whole run, because nothing is saved until the end.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.cuda()

    loader = DataLoader(
        Corpus(), batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    steps = int(len(loader) * args.epochs)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, steps // args.grad_accum), pct_start=0.05
    )

    model.train()
    seen, running = 0, 0.0
    optimizer.zero_grad()
    for step, (encoded, ys) in enumerate(loader):
        if step >= steps:
            break
        encoded = {k: v.cuda() for k, v in encoded.items()}
        logits = model(**encoded).logits.squeeze(-1).float()
        loss = brier_loss(torch.sigmoid(logits), ys.cuda())
        (loss / args.grad_accum).backward()
        running += loss.item()
        seen += 1
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        if seen % 100 == 0:
            print(
                f"  step {step + 1}/{steps}  mse-vs-target {running / seen:.4f}",
                file=sys.stderr,
            )
            running, seen = 0.0, 0

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved to {args.out}", file=sys.stderr)

    # Score the eval set from the same process -- no second engine, and no
    # opportunity for a footing to drift between training and scoring.
    with open(args.eval_set, encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    eval_rows = [
        {
            "question": q["text"],
            "criteria": "Resolves YES if the event described occurs.",
            "scheduled": q["scheduled_resolve_date"][:10],
            "as_of": q["open_date"][:10],
        }
        for q in questions
    ]
    eval_texts = build_texts(eval_rows, tokenizer, args.chat_template)

    model.eval()
    predictions: list[float] = []
    with torch.no_grad():
        for start in range(0, len(eval_texts), args.batch_size):
            chunk = eval_texts[start : start + args.batch_size]
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
            )
            encoded = {k: v.cuda() for k, v in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1).float()
            predictions.extend(torch.sigmoid(logits).cpu().tolist())

    with open(args.forecasts, "w", encoding="utf-8") as sink:
        for q, p in zip(questions, predictions, strict=True):
            sink.write(
                json.dumps(
                    {
                        "question_id": q["question_id"],
                        "quarter": q.get("quarter", 0),
                        "outcome": q["outcome"],
                        # No text to parse: the head IS the forecast. Written to
                        # the same field decode_compare reads so the scalar arm
                        # goes through the identical scoring path.
                        "response": render_forecast(SCALAR_REASONING, p),
                        "expectation_probability": p,
                        "expectation_greedy": p,
                        "expectation_tenths_mass": 1.0,
                        "seconds": 0.0,
                        "usage": {},
                        "error": None,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(predictions):,} forecasts -> {args.forecasts}", file=sys.stderr)
    print("Score with scripts/decode_compare.py and scripts/bootstrap_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
