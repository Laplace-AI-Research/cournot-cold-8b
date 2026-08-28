"""LoRA SFT for Cournot-Cold 8B, run on a Runpod pod that tears itself down.

    uv run python scripts/sft_train.py --corpus data/sft_corpus.jsonl --out runs_sft/arm0

`docs/04`: this stage locks the output format and teaches "reason, then commit to
a number". **Not forecasting skill** — that is `docs/05`. So the thing to watch
is not whether Brier improves; it is whether the four gates hold:

- format compliance > 99%
- ECE no worse than the base model's
- probability histogram not collapsed — distinct values and entropy, not the mean
- no systematic drift toward 0 or 1 relative to base

## The one number this run exists to produce

Post-hoc calibration already takes Qwen3-8B from 0.2763 to 0.2288 without
training anything, so SFT cannot justify itself on Brier. The question is
**resolution**, currently 0.0154: calibration cannot create discrimination, it
can only stop you giving it away.

If resolution does not move, Phase 2's premise needs revisiting rather than more
arms. That is why one arm runs before the four-arm mixture sweep.

## Self-teardown

the internal decisions log (2026-08-23) permits unattended provisioning only if the job stops its
own meter. `--teardown-pod` terminates the pod in a `finally` block, so it fires
on success, on exception, and on SIGTERM. The one case it cannot cover is the
process being killed outright, which is what the caller's spend ceiling is for.

LoRA rather than full fine-tuning, for a measured reason: the base model has
resolution and loses it to calibration, which is an output-distribution problem
and where LoRA is strongest. Full FT also risks catastrophic forgetting, which
would surface precisely in the gates above.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from types import FrameType

from cournot.sft import check_extreme_share

PROMPT = """You are a careful forecaster. Estimate the probability that the question \
below resolves YES.

Question: {question}
Resolution criteria: {criteria}
Scheduled resolution: {scheduled}
You are forecasting as of: {as_of}

Reply with exactly this and nothing else:
<reasoning>two to four sentences</reasoning>
<probability>0.XX</probability>"""

COMPLETION = "<reasoning>{reasoning}</reasoning>\n<probability>{probability:.2f}</probability>"


@dataclass(frozen=True)
class Teardown:
    """Terminates the pod exactly once, however the process ends.

    Registered for SIGTERM and SIGINT as well as the `finally` path: Runpod's
    stop button and a `kill` both arrive as signals, and a handler that only
    covered the happy path would leave the meter running in precisely the cases
    where nobody is watching.
    """

    pod_id: str
    enabled: bool
    _done: list[bool]

    def fire(self, reason: str) -> None:
        if not self.enabled or self._done:
            return
        self._done.append(True)
        print(f"[teardown] terminating pod {self.pod_id} ({reason})", file=sys.stderr)
        try:
            import urllib.request

            key = os.environ.get("RUNPOD_API_KEY", "")
            if not key:
                print(
                    "[teardown] RUNPOD_API_KEY not set — CANNOT stop the pod. "
                    "Terminate it manually.",
                    file=sys.stderr,
                )
                return
            # v2, not v1. Verified from a pod on 2026-08-24: the same key
            # returns 200 on api.runpod.io/v2 and **403** on
            # rest.runpod.io/v1, which is deprecated. Pointing teardown at v1
            # would have failed silently at exactly the moment nobody is watching.
            request = urllib.request.Request(
                f"https://api.runpod.io/v2/pods/{self.pod_id}",
                method="DELETE",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                print(f"[teardown] HTTP {response.status}", file=sys.stderr)
        except Exception as exc:  # teardown must never raise, whatever went wrong
            print(f"[teardown] FAILED: {exc!r} — terminate the pod manually", file=sys.stderr)


def install_teardown(pod_id: str, enabled: bool) -> Teardown:
    teardown = Teardown(pod_id=pod_id, enabled=enabled, _done=[])

    def handler(signum: int, _frame: FrameType | None) -> None:
        teardown.fire(f"signal {signum}")
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)
    return teardown


def load_corpus(
    path: str, *, allow_extreme_targets: bool = False, require_reasoning: bool = True
) -> list[dict[str, object]]:
    """Read the assembled SFT corpus, refusing anything malformed.

    `docs/04`: "Parse strictly at train time. Malformed examples are dropped, not
    repaired." Dropped **and counted** — a corpus that quietly shrank is a
    different experiment from the one that was assembled.
    """
    rows: list[dict[str, object]] = []
    dropped = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = row.get("target")
            if require_reasoning and row.get("reasoning") is None:
                dropped += 1
                continue
            if not isinstance(target, (int, float)):
                dropped += 1
                continue
            if not 0.0 <= float(target) <= 1.0:
                dropped += 1
                continue
            rows.append(row)
    print(f"corpus: {len(rows):,} usable, {dropped:,} dropped as malformed", file=sys.stderr)

    targets = [float(r["target"]) for r in rows]  # type: ignore[arg-type]
    if allow_extreme_targets:
        # `docs/14`'s scalar head. Under Brier loss a 0/1 label is a proper
        # Bernoulli observation the model averages over -- the opposite of text
        # SFT, where `0.00` is a string to emit. Blocking it here would refuse
        # the high-variance arm of the comparison Phase 2 exists to run, which
        # is what `check_extreme_share`'s own docstring says must not happen.
        extreme = sum(1 for value in targets if value <= 0.0 or value >= 1.0)
        share = extreme / len(targets) if targets else 0.0
        print(
            f"  targets at exactly 0 or 1: {share:.1%} (permitted: scalar head)",
            file=sys.stderr,
        )
    else:
        share = check_extreme_share(targets)
        print(f"  targets at exactly 0 or 1: {share:.1%}", file=sys.stderr)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--teardown-pod",
        default="",
        help=(
            "Runpod pod id to terminate when this process ends, however it ends. "
            "Required by the internal decisions log (2026-08-23) for unattended runs; leave empty "
            "only when running somewhere that is not billing by the hour."
        ),
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help=(
            "wrap the prompt in the model's chat template. MUST match what "
            "vllm_score.py is given, or training and scoring are on different "
            "footings -- the defect the internal decisions log records (2026-08-24c)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="load and report, train nothing")
    args = parser.parse_args(argv)

    teardown = install_teardown(args.teardown_pod, bool(args.teardown_pod))
    try:
        rows = load_corpus(args.corpus)
        if not rows:
            print("empty corpus — nothing to train on", file=sys.stderr)
            return 1

        soft = sum(1 for r in rows if r.get("kind") == "soft")
        print(f"  soft targets {soft:,} ({soft / len(rows):.1%}), terminal {len(rows) - soft:,}")
        # Terminal targets ARE 0.0 or 1.0, so overall extreme mass can never fall
        # below (1 - soft_fraction) and says nothing about the corpus. The
        # informative figure is the extreme mass among the SOFT targets, which is
        # where a soft target could have been moderate and was not.
        soft_targets = [float(r["target"]) for r in rows if r.get("kind") == "soft"]
        overall = [float(r["target"]) for r in rows]  # type: ignore[arg-type]

        def extreme_share(values: list[float]) -> float:
            return sum(1 for v in values if v <= 0.05 or v >= 0.95) / len(values) if values else 0.0

        print(f"  extreme mass, whole corpus : {extreme_share(overall):.1%}")
        print(
            f"    of which unavoidable      : {1 - soft / len(rows):.1%} "
            "(terminal targets are 0 or 1 by definition)"
        )
        print(f"  extreme mass, SOFT only    : {extreme_share(soft_targets):.1%}  <- the signal")
        print(
            "  (`docs/04` calls the overconfidence trap this stage's dominant "
            "failure. A high SOFT figure means the mitigation is not mitigating.)"
        )

        if args.dry_run:
            print("\ndry run — corpus loads, nothing trained")
            return 0

        os.makedirs(args.out, exist_ok=True)
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        def build_prompt(r: dict[str, object]) -> str:
            """Train on exactly the string that scoring will present.

            Until 2026-08-24 this emitted the raw `PROMPT` while the base model
            was scored through Ollama, which templates server-side -- so every
            base-vs-adapter table compared a templated base against
            raw-prompted adapters (the internal decisions log, 2026-08-24c). `--chat-template`
            closes that, and it must be set to whatever `vllm_score.py` is given.
            """
            body = PROMPT.format(
                question=r["question"],
                criteria=r.get("criteria") or "Resolves YES if the event occurs.",
                scheduled=r.get("scheduled") or "unknown",
                as_of=r.get("as_of") or "unknown",
            )
            if not args.chat_template:
                return body
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        texts = [
            build_prompt(r)
            + "\n"
            + COMPLETION.format(reasoning=r["reasoning"], probability=float(r["target"]))  # type: ignore[arg-type]
            # EOS, explicitly. Without it the model never learns to stop: on the
            # 2026-08-24 v2 run, 20 of 23 compliance failures emitted thousands
            # of newlines and never reached <probability> at all (4,649 chars
            # against 540 for a good response). Greedy decoding then has nothing
            # to halt on, so a minority of prompts fall into a newline loop.
            + tokenizer.eos_token
            for r in rows
        ]
        dataset = Dataset.from_dict({"text": texts})

        trainer = SFTTrainer(
            model=args.base_model,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
            args=SFTConfig(
                output_dir=args.out,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                max_length=args.max_seq_len,
                bf16=torch.cuda.is_available(),
                logging_steps=25,
                save_strategy="epoch",
                report_to=[],
            ),
        )
        trainer.train()
        trainer.save_model(args.out)
        print(f"\nadapter written to {args.out}")
        print("\nNEXT: score it through the harness and compare RESOLUTION against")
        print("the base model's 0.0154. Brier alone cannot answer what this run asked.")
        return 0
    finally:
        teardown.fire("process exit")


if __name__ == "__main__":
    sys.exit(main())
