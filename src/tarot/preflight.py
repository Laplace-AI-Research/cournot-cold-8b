# SPDX-License-Identifier: Apache-2.0
"""Preconditions for long-running unattended jobs.

Two jobs died on `OSError: [Errno 28] No space left on device` on 2026-08-20 —
a 106,416-market label fetch at row 33,498, and a 500-question bake-off at 418.
Neither checked free space before starting, and both had been running for hours.
The cost was not the crash; it was that the crash happened *late*, after the work
that could not be resumed had already been paid for.

    require_free_space("data/manifold", gigabytes=2.0, job="label fetch")

**The caller supplies the estimate, and it is deliberately not inferred.** This
module cannot know how much a job will write, and a guess dressed up as a check
is worse than no check — it would pass reliably and fail exactly when a job grew
beyond whatever the guess assumed. What it does know is the current free space,
which is the half that is actually measurable.

**Checks the filesystem holding `path`, not the current directory.** `~/Laplace`
and a scratch directory can sit on different volumes, and a job that writes to one
while a guard reads the other is a guard that reports on the wrong disk.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = [
    "InsufficientDiskSpaceError",
    "existing_ancestor",
    "free_gigabytes",
    "require_free_space",
]

#: Bytes per GiB. Binary, matching what `df -h` prints on macOS and Linux.
GIB = 1 << 30


class InsufficientDiskSpaceError(OSError):
    """Raised before a job starts, rather than partway through by the kernel."""


def existing_ancestor(path: str | Path) -> Path:
    """The nearest existing directory at or above `path`.

    A job's output directory often does not exist yet — `os.makedirs` runs after
    the preflight. Walking up finds the filesystem it *will* live on, so the
    check works before the first write rather than only after it.
    """
    resolved = Path(path).expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.exists():
            return candidate
    return Path(os.sep)


def free_gigabytes(path: str | Path) -> float:
    """Free space, in GiB, on the filesystem holding `path`."""
    return shutil.disk_usage(existing_ancestor(path)).free / GIB


def require_free_space(path: str | Path, *, gigabytes: float, job: str = "this job") -> float:
    """Return free GiB, or raise if it is below `gigabytes`.

    Returns the measurement so a caller can log what it saw — a job that records
    "started with 41.2 GiB free" makes a later out-of-space failure diagnosable
    rather than mysterious.
    """
    if gigabytes < 0:
        raise ValueError(f"gigabytes must be non-negative, got {gigabytes}")
    free = free_gigabytes(path)
    if free < gigabytes:
        target = existing_ancestor(path)
        raise InsufficientDiskSpaceError(
            f"{job} needs about {gigabytes:.1f} GiB free on the filesystem holding "
            f"{target}, and {free:.1f} GiB is available. Refusing to start: this "
            "job is long enough that running out partway costs more than not "
            "starting. Free space, or lower the estimate deliberately."
        )
    return free
