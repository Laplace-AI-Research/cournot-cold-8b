"""Score an existing scalar-head adapter on an eval set.

    uv run python scripts/scalar_score.py --adapter runs_sft/adapter_tfull \
        --set data/asof_phi50.json --out forecasts.jsonl --chat-template

`scalar_train.py` scores in-process straight after training, which is right for
a fresh arm and useless for re-scoring a saved one. This is the separate path:
same prompt construction, same `render_forecast`, so a scalar arm still goes
through the identical scoring code as every text arm.

The footing flags must match whatever the adapter was TRAINED with
(`/preflight-config-check`). Scoring a chat-templated adapter on raw prompts is
the defect the internal decisions log (2026-08-24c) records, and it is silent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scalar_train import SCALAR_REASONING, build_texts

from cournot.parsing import render_forecast


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--chat-template", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(args.set, encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    print(f"{len(questions):,} questions from {args.set}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding: sequence classification pools the last NON-PAD token, so a
    # left pad run would pool a pad. Opposite of the generation case.
    tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, args.adapter)
    model.cuda()
    model.eval()

    rows = [
        {
            "question": q["text"],
            "criteria": "Resolves YES if the event described occurs.",
            "scheduled": q["scheduled_resolve_date"][:10],
            "as_of": q["open_date"][:10],
        }
        for q in questions
    ]
    texts = build_texts(rows, tokenizer, args.chat_template)

    predictions: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start : start + args.batch_size]
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
            if (start // args.batch_size) % 50 == 0:
                print(f"  {start + len(chunk)}/{len(texts)}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as sink:
        for q, p in zip(questions, predictions, strict=True):
            sink.write(
                json.dumps(
                    {
                        "question_id": q["question_id"],
                        "quarter": q.get("quarter", 0),
                        "outcome": q["outcome"],
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
    print(f"wrote {len(predictions):,} -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
