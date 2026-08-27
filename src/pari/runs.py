"""Reading forecast run files.

A run file is one JSON object per line, written by `scripts/vllm_score.py` and
read by every scorer. This module owns the parts of that format that more than
one caller depends on, so they cannot drift.

Currently one function, and it exists because of a specific failure shape.
Generation writes in chunks, so a crash leaves a **truncated final line**. A
resume that counted that line as finished would silently drop the question from
the run -- a quieter version of the whole-run loss that chunking exists to
prevent. The truncated line is therefore discarded rather than salvaged.
"""

from __future__ import annotations

import json
import os

__all__ = ["already_scored"]


def already_scored(path: str) -> set[str]:
    """`question_id`s already present in a run file.

    A missing file is an empty set, not an error: the first run of a job and a
    resumed one take the same path. Malformed and partial lines are skipped, so
    the caller re-runs those questions rather than losing them.
    """
    if not os.path.exists(path):
        return set()
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                # Redundant with the handler below -- `json.loads("")` raises
                # JSONDecodeError, which is already caught. Verified by mutation
                # on 2026-08-24: removing this changes no behaviour and fails no
                # test, because it is an EQUIVALENT mutation rather than an
                # untested guard. Kept for legibility, recorded as not
                # load-bearing so nobody later mistakes its silence for coverage.
                continue
            try:
                seen.add(json.loads(stripped)["question_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return seen
