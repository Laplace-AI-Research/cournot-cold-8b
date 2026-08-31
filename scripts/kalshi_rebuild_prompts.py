"""Rebuild the rebuilt set's prompts to the shipped construction, and prove it.

    uv run python scripts/kalshi_rebuild_prompts.py

The shipped eval set's prompt is

    title with a trailing "?" removed + " -- resolves YES if: " + yes_sub_title

(with an em dash, not two hyphens). `yes_sub_title` is in no archived source we
hold, so it was re-fetched from the live API by `kalshi_subtitles.py`
(docs/10 2026-08-30g).

## The check that makes this trustworthy

157 question ids appear in both the shipped 778 and the rebuilt 1,259. If the
construction rule is right, it reproduces all 157 shipped texts **exactly**.
This script refuses to write unless it does. A rule that reproduces most of them
is a rule that is wrong somewhere, and "most" is how the prompt defect survived
the first run.

## Rows whose subtitle could not be fetched

DROPPED AND COUNTED, never defaulted to the bare title. Defaulting would
silently reintroduce the exact defect the reproduction gate caught, on a subset,
where it would be invisible.
"""

from __future__ import annotations

import json
import sys

SEP = " — resolves YES if: "   # em dash


def build(title: str, sub: str) -> str:
    """The construction is CONDITIONAL, and the condition had to be recovered.

    When `yes_sub_title` already appears in the title it adds nothing, and the
    shipped set keeps the title unchanged, question mark and all. When it does
    not, the title's trailing "?" is dropped and the subtitle is appended.

    That is not a guess. On the 157 questions shared with the shipped set the
    split is perfect: 92 of 92 unsuffixed rows have the subtitle inside the
    title, and 0 of 65 suffixed rows do.
    """
    if sub.lower() in title.lower():
        return title
    return title.rstrip().rstrip("?").rstrip() + SEP + sub


def main() -> int:
    subs: dict[str, dict] = {}
    for line in open("data/kalshi/subtitles.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            subs[r["ticker"]] = r

    ev = json.load(open("data/kalshi_eval_rebuilt.json", encoding="utf-8"))
    old = {q["question_id"]: q["text"]
           for q in json.load(open("data/kalshi_score_778.json", encoding="utf-8"))["questions"]}

    kept, dropped = [], {"no_subtitle": 0, "not_found": 0, "no_title": 0}
    for q in ev["questions"]:
        t = q["question_id"].split(":", 1)[1]
        rec = subs.get(t)
        if rec is None or rec.get("status") == "not_found":
            dropped["not_found"] += 1
            continue
        sub, title = rec.get("yes_sub_title"), rec.get("title")
        if not title:
            dropped["no_title"] += 1
            continue
        if sub is None:
            dropped["no_subtitle"] += 1
            continue
        q = dict(q)
        q["text"] = build(title, sub)
        kept.append(q)

    shared = [q for q in kept if q["question_id"] in old]
    exact = sum(1 for q in shared if q["text"] == old[q["question_id"]])
    print(f"rebuilt {len(kept):,} rows; dropped {sum(dropped.values())} {dropped}")
    print(f"shared with the shipped set: {len(shared)}")
    print(f"prompts reproduced EXACTLY:  {exact} / {len(shared)}")

    if shared and exact != len(shared):
        print("\nREFUSING to write. The construction rule does not reproduce every")
        print("shipped prompt, so it is wrong somewhere. Examples:")
        for q in shared:
            if q["text"] != old[q["question_id"]]:
                print(f"  id      : {q['question_id']}")
                print(f"  shipped : {old[q['question_id']]}")
                print(f"  rebuilt : {q['text']}")
                break
        return 1

    ev["questions"] = sorted(kept, key=lambda q: q["question_id"])
    ev["n"] = len(kept)
    ev["prompt_construction"] = (
        "if yes_sub_title appears in title: title unchanged; "
        "else title (trailing '?' removed) + ' — resolves YES if: ' + yes_sub_title. "
        "yes_sub_title re-fetched from the live API; it is in no archived source.")
    ev["dropped_prompt_rebuild"] = dropped
    import hashlib
    ev["content_hash"] = hashlib.sha256(
        json.dumps([q["question_id"] for q in ev["questions"]], sort_keys=True).encode()
    ).hexdigest()
    with open("data/kalshi_eval_rebuilt.json", "w", encoding="utf-8") as h:
        json.dump(ev, h, indent=1, sort_keys=True)
    print(f"\nwrote data/kalshi_eval_rebuilt.json  n={ev['n']:,}  hash {ev['content_hash'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
