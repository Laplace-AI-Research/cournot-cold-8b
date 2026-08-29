# SPDX-License-Identifier: Apache-2.0
"""Assemble the SFT question pool at a stated synthetic share.

The mixture ratio was adopted 2026-08-22 as a **gated sweep** — 0 / 25 / 50 /
74.1% synthetic, each arm gated on the event partition (`cournot.eventgate`), taking
the highest ratio that does not degrade it. This module builds one arm's pool.

## Arms must be the same size, or the sweep measures two things at once

`size` fixes the assembled pool's row count so arms differ **only** in their
ratio. Without it the natural construction -- keep every real record, add
synthetic to taste -- gives 81,874 rows at 0% and 316,116 at 74.1%, and a
difference between those arms could be the mixture or could be 3.9x the data.
The sweep could not tell you which, which is the one thing it exists to do.

The common size is capped by the 0% arm, which must be all-real: **81,874**. That
sits below `docs/04`'s stated 100-200k volume, and the trade is deliberate -- a
clean comparison at 82k beats a confounded one at 316k. Recorded as a spec
deviation rather than absorbed quietly.

## The share is of the pool, and it is exact or it raises

`synthetic_share` is the fraction of the *assembled pool* that is synthetic, not
a ratio against the real count and not a target to approximate. A caller asking
for 50% gets 50% or an error, because a silently-approximated mixture makes the
sweep's rungs incomparable — and the whole point of the sweep is comparing rungs.

## Why 75% is not a rung

There are 234,483 synthetic questions and 81,874 real ones, so the largest
reachable synthetic share is **74.1%** — every synthetic question against every
real one. A 75% arm would need 245,622 synthetic rows. `max_synthetic_share`
computes this rather than trusting a number typed into a plan.

That ceiling is also the naive pool: `docs/13` warned the ratio would be set by
availability if nobody set it, and 74.1% is what "use everything" means. The
sweep's top rung is therefore the do-nothing case, which makes the sweep measure
what the default would have cost.

## Real questions are filtered, synthetic ones are not

Non-predictive contamination (lotteries, personal goals, site meta) is a property
of the *scraped* corpus; synthetic questions are constructed from a series and
cannot be non-predictive. Passing them through the same filter would be a no-op
that implies a check happened.

The filter matters here for a measured reason: the 2026-08-22 surface-form
spot-check found the generator's only failures were on a lottery question and a
market titled "Daily market" — the contaminated stratum degrades everything built
on top of it, not just the eval.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from cournot.nonpredictive import DropPolicy, HasNonPredictiveFlag, filter_nonpredictive

R = TypeVar("R", bound=HasNonPredictiveFlag)
S = TypeVar("S")


class MixtureInfeasibleError(ValueError):
    """The requested share cannot be built from the pools supplied."""


@dataclass(frozen=True)
class MixturePool(Generic[R, S]):
    """One arm's assembled pool, plus what it took to build it."""

    real: tuple[R, ...]
    synthetic: tuple[S, ...]
    requested_share: float
    #: Real records dropped by the non-predictive policy, before mixing.
    dropped_nonpredictive: int
    #: Real records KEPT but never classified -- 12.9% of the Manifold corpus has
    #: no group slug. The residual contamination lives here, and counting these
    #: as clean is the error that understated the corpus figure on 2026-08-20.
    unassessed: int
    drop_policy: DropPolicy

    @property
    def size(self) -> int:
        return len(self.real) + len(self.synthetic)

    @property
    def achieved_share(self) -> float:
        return len(self.synthetic) / self.size if self.size else 0.0


def max_synthetic_share(n_real_available: int, n_synthetic_available: int) -> float:
    """Largest synthetic share reachable using every real record.

    Computed rather than assumed. The adopted ladder's top rung was written as
    75%, which is not reachable: it would need 245,622 synthetic questions
    against the 234,483 that exist.
    """
    total = n_real_available + n_synthetic_available
    return n_synthetic_available / total if total else 0.0


def _hash_order(items: Sequence[S], salt: str) -> list[S]:
    """Deterministic order that is not the corpus's own.

    Taking a prefix of file order would correlate the sample with whatever the
    ingest sorted by — creation time, id — and make the arms differ in more than
    their ratio.
    """
    return sorted(
        items,
        key=lambda item: hashlib.blake2b(f"{salt}:{item!r}".encode(), digest_size=8).hexdigest(),
    )


def assemble(
    real: Sequence[R],
    synthetic: Sequence[S],
    *,
    synthetic_share: float,
    drop_policy: DropPolicy,
    size: int | None = None,
    salt: str = "mixture",
    tolerance: float = 0.005,
) -> MixturePool[R, S]:
    """Build one arm's pool at exactly `synthetic_share`.

    Both keyword arguments are required and have no default. `drop_policy` in
    particular: `cournot.nonpredictive` refuses to guess what a caller wants
    excluded, and an assembled training corpus is precisely where that silence
    would be costly.

    `size` pins the assembled row count so a sweep's arms are comparable; see the
    module docstring. Without it, real records are kept in full and synthetic ones
    subsampled to hit the share, which makes arms differ in size as well as ratio.

    Real records are kept in full and synthetic ones subsampled to hit the share,
    unless the share demands more synthetic than exists — in which case the real
    side is subsampled instead.

    `tolerance` is how far the achieved share may sit from the requested one
    before this raises. Integer pool sizes cannot hit every share exactly: 90%
    from 5 synthetic questions would need 5/9 of a real record. The assembled
    share is therefore **checked, not assumed** — an early version returned
    0.833 for a requested 0.900 without complaint, which is exactly the silent
    approximation that makes the sweep's rungs incomparable.
    """
    if not 0.0 <= synthetic_share < 1.0:
        raise ValueError(f"synthetic_share must be in [0, 1), got {synthetic_share}")

    if size is not None and size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    kept = filter_nonpredictive(real, policy=drop_policy)
    real_pool = list(kept.kept)
    n_real, n_syn = len(real_pool), len(synthetic)

    if size is not None:
        want_real = round(size * (1.0 - synthetic_share))
        want_syn = size - want_real
        if want_real > n_real or want_syn > n_syn:
            raise MixtureInfeasibleError(
                f"size {size:,} at share {synthetic_share:.3f} needs "
                f"{want_real:,} real and {want_syn:,} synthetic, against "
                f"{n_real:,} and {n_syn:,} available."
            )
        pinned = MixturePool(
            real=tuple(_hash_order(real_pool, salt)[:want_real]),
            synthetic=tuple(_hash_order(synthetic, salt)[:want_syn]),
            requested_share=synthetic_share,
            dropped_nonpredictive=kept.n_dropped,
            unassessed=kept.unassessed,
            drop_policy=drop_policy,
        )
        if abs(pinned.achieved_share - synthetic_share) > tolerance:
            raise MixtureInfeasibleError(
                f"share {synthetic_share:.3f} requested but "
                f"{pinned.achieved_share:.3f} assembled at size {size:,}, "
                f"outside tolerance {tolerance}."
            )
        return pinned

    if synthetic_share == 0.0:
        return MixturePool(
            real=tuple(real_pool),
            synthetic=(),
            requested_share=0.0,
            dropped_nonpredictive=kept.n_dropped,
            unassessed=kept.unassessed,
            drop_policy=drop_policy,
        )

    # Keep all real, take the synthetic the share implies.
    want_syn = round(n_real * synthetic_share / (1.0 - synthetic_share))
    if want_syn <= n_syn:
        chosen_real, chosen_syn = real_pool, _hash_order(synthetic, salt)[:want_syn]
    else:
        # Not enough synthetic to reach the share with every real record, so the
        # real side gives way. Raising instead would make the top rung
        # unbuildable; silently returning a lower share would make the rungs
        # incomparable, which is worse.
        want_real = round(n_syn * (1.0 - synthetic_share) / synthetic_share)
        if want_real > n_real:
            raise MixtureInfeasibleError(
                f"share {synthetic_share:.3f} needs {want_syn:,} synthetic against "
                f"{n_syn:,} available, and backing off to {want_real:,} real exceeds "
                f"the {n_real:,} available. Max reachable share is "
                f"{max_synthetic_share(n_real, n_syn):.3f}."
            )
        chosen_real = _hash_order(real_pool, salt)[:want_real]
        chosen_syn = list(synthetic)

    pool = MixturePool(
        real=tuple(chosen_real),
        synthetic=tuple(chosen_syn),
        requested_share=synthetic_share,
        dropped_nonpredictive=kept.n_dropped,
        unassessed=kept.unassessed,
        drop_policy=drop_policy,
    )
    if abs(pool.achieved_share - synthetic_share) > tolerance:
        raise MixtureInfeasibleError(
            f"share {synthetic_share:.3f} requested but {pool.achieved_share:.3f} "
            f"assembled ({len(pool.synthetic):,} synthetic, {len(pool.real):,} real), "
            f"outside tolerance {tolerance}. The pools are too small to hit this "
            f"share exactly. Max reachable share is "
            f"{max_synthetic_share(n_real, n_syn):.3f}."
        )
    return pool
