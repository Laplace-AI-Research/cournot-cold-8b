"""Run manifests. `docs/07`: an eval number without one is not citable.

    manifest = build_manifest(
        model_hash=hash_file(weights),
        data_snapshot_hash=hash_file(eval_set),
        label="cournot-cold-v1",
    )
    card = certify(model_card_metrics(slice_), manifest)

`cournot.splits` already refuses to build a card-bound artifact from `dev`. That
guard answers *which questions*; it says nothing about *which model*, *which
data*, or *which code* produced the number. A `published` slice scored by an
unknown checkpoint against an unrecorded snapshot is exactly as uncitable as a
dev number, and looks considerably more respectable.

**The seven fields are `docs/07`'s list, not a convenient subset**, and every one
of them is required at construction. There is no partial manifest: the failure
this prevents is a manifest that exists, is missing the one field that mattered,
and is trusted because a manifest exists.

**`certify` is the only route to a `CertifiedCard`.** Same reasoning as
`model_card_metrics` having no `force=True` — a renderer that accepts bare
`CardMetrics` makes the manifest optional in practice, and under deadline it
would be omitted.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cournot.leakage import LEAKAGE_DETECTOR_VERSION
from cournot.splits import CardMetrics, Split

__all__ = [
    "CertifiedCard",
    "DecodeConfig",
    "DirtyWorktreeError",
    "IncompleteManifestError",
    "RunManifest",
    "build_manifest",
    "certify",
    "git_sha",
    "hash_file",
]


class IncompleteManifestError(ValueError):
    """A manifest field was absent, blank, or not timezone-aware."""


class DirtyWorktreeError(RuntimeError):
    """The recorded git SHA would not describe the code that ran."""


def hash_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed.

    Streamed rather than read whole because the eval set and a checkpoint are
    both large enough that reading them into memory to hash them is a real cost.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_sha(*, allow_dirty: bool = False) -> str:
    """The commit that produced this run.

    Refuses a dirty worktree by default. A SHA recorded next to uncommitted
    changes names code that was never what ran, which is worse than no SHA: it
    is a field that looks like provenance and is not. `allow_dirty` exists for
    tests and exploratory runs and appends a marker so the result can never be
    mistaken for a clean one.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise DirtyWorktreeError(f"could not read git state: {exc}") from exc
    if not dirty:
        return sha
    if allow_dirty:
        return f"{sha}-dirty"
    raise DirtyWorktreeError(
        f"worktree has uncommitted changes, so {sha[:12]} does not describe the "
        "code that ran. Commit first, or pass allow_dirty=True and accept that "
        "the run is not reproducible from this SHA."
    )


def _require(name: str, value: str) -> str:
    if not value or not value.strip():
        raise IncompleteManifestError(
            f"{name} is required and was empty. docs/07 lists seven fields; a "
            "manifest missing one is trusted because a manifest exists."
        )
    return value


@dataclass(frozen=True)
class DecodeConfig:
    """How the forecasts were produced. Every field is load-bearing.

    On 2026-08-24 the same Qwen3-8B weights, on the same questions, produced
    calibrated Briers of **0.2288 and 0.2374** depending only on fields recorded
    here -- thinking mode was on by default in one serving stack and off in the
    other, and nothing in the manifest could tell the two runs apart. A day was
    spent attributing the difference to training, then to padding, before the
    cause turned out to be a default nobody had written down (the internal decisions log,
    2026-08-24c/e).

    `docs/11` rule 3: a load-bearing field carries an external invariant. The
    invariant here is that two runs differing in any of these are **not
    comparable**, and a manifest that cannot express the difference cannot
    enforce it.

    No defaults, deliberately -- the same reason `as_of` has none. Every value
    that has ever caused a silent divergence was a default.
    """

    serving_stack: str
    """Engine and version, e.g. `vllm==0.11.0` or `ollama`. Two stacks batch,
    pad and position differently."""

    prompt_construction: str
    """`chat_template` or `raw`. Scoring a templated base against raw-prompted
    adapters is the defect the internal decisions log (2026-08-24c) records."""

    temperature: float
    """0.0 is greedy. Ollama defaults to 0.8 and does not say so."""

    thinking: bool
    """Qwen3 enables it by default. Worth -0.0294 of Brier [-0.0384, -0.0205],
    which is more than any training arm this project has run."""

    probability_read: str
    """`expectation`, `argmax` or `scalar_head`. Against the thinking base the
    SIGN of the SFT effect depends on the first two, and both signs are
    significant (the internal decisions log, 2026-08-24g).

    `scalar_head` is the Phase 2 path (`docs/14`): a regression output read
    directly off a sequence-classification head, generating no tokens at all.
    Added 2026-08-27 because `DecodeConfig` could not describe the artifact
    about to ship, so `/preflight-config-check` could not gate its scoring run
    (the internal decisions log, 2026-08-27t)."""

    max_new_tokens: int
    """A cap short enough to truncate a thinking block changes the answer.

    Must be > 0 for the generative reads. Must be exactly 0 for `scalar_head`,
    which generates nothing — a nonzero cap there would be a footing recorded
    that does not exist."""

    def __post_init__(self) -> None:
        _require("serving_stack", self.serving_stack)
        if self.prompt_construction not in ("chat_template", "raw"):
            raise IncompleteManifestError(
                f"prompt_construction must be 'chat_template' or 'raw', got "
                f"{self.prompt_construction!r}"
            )
        if self.probability_read not in ("expectation", "argmax", "scalar_head"):
            raise IncompleteManifestError(
                f"probability_read must be 'expectation', 'argmax' or 'scalar_head', "
                f"got {self.probability_read!r}"
            )
        if self.temperature < 0.0:
            raise IncompleteManifestError(f"temperature must be >= 0, got {self.temperature}")
        if self.probability_read == "scalar_head":
            # The head generates nothing. A nonzero cap would record a decode
            # parameter that never applied, which is the class of silent
            # mis-description this type exists to prevent.
            if self.max_new_tokens != 0:
                raise IncompleteManifestError(
                    f"max_new_tokens must be 0 for scalar_head (it generates no "
                    f"tokens), got {self.max_new_tokens}"
                )
        elif self.max_new_tokens <= 0:
            raise IncompleteManifestError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")

    def as_dict(self) -> dict[str, str]:
        return {
            "serving_stack": self.serving_stack,
            "prompt_construction": self.prompt_construction,
            "temperature": f"{self.temperature:g}",
            "thinking": "true" if self.thinking else "false",
            "probability_read": self.probability_read,
            "max_new_tokens": str(self.max_new_tokens),
        }

    def comparable_to(self, other: DecodeConfig) -> bool:
        """Whether two runs may be compared at all. Exact equality, no tolerance."""
        return self.as_dict() == other.as_dict()


@dataclass(frozen=True)
class RunManifest:
    """The seven fields `docs/07` requires, all mandatory."""

    model_hash: str
    data_snapshot_hash: str
    eval_split_id: Split
    git_sha: str
    leakage_detector_version: str
    freeze: datetime
    run_date: datetime
    decode: DecodeConfig
    """How the forecasts were produced. See `DecodeConfig` -- identical weights
    gave 0.2288 and 0.2374 across two settings of these fields."""

    base_model_release_date: datetime
    """When the base model was RELEASED, not the cutoff it claims.

    `Split.PUBLISHED` asserts contamination-freedom "since the freeze is after
    every base model's pretraining cutoff". That was a docstring and nothing
    checked it: swapping in a model released after the freeze voids every
    published number at once, silently, leaving no trace in any leakage counter
    because no document is involved.

    Release date rather than stated cutoff, deliberately. A cutoff is
    self-reported, frequently understated, and unverifiable from outside; a
    release date is a fact. A model released after the freeze may have seen
    post-freeze events whatever cutoff it advertises.
    """

    def __post_init__(self) -> None:
        _require("model_hash", self.model_hash)
        _require("data_snapshot_hash", self.data_snapshot_hash)
        _require("git_sha", self.git_sha)
        _require("leakage_detector_version", self.leakage_detector_version)
        for field in ("freeze", "run_date", "base_model_release_date"):
            stamp = getattr(self, field)
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise IncompleteManifestError(
                    f"{field} must be timezone-aware (CLAUDE.md: naive datetimes "
                    "are a bug). A naive freeze silently shifts the split boundary."
                )

    def as_dict(self) -> dict[str, str]:
        return {
            "model_hash": self.model_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "eval_split_id": self.eval_split_id.value,
            "git_sha": self.git_sha,
            "leakage_detector_version": self.leakage_detector_version,
            "freeze": self.freeze.isoformat(),
            "run_date": self.run_date.isoformat(),
            # Both of these were validated and then silently dropped here, so a
            # serialized manifest could not reproduce the check it had passed.
            "base_model_release_date": self.base_model_release_date.isoformat(),
            **{f"decode_{k}": v for k, v in self.decode.as_dict().items()},
        }


def build_manifest(
    *,
    model_hash: str,
    data_snapshot_hash: str,
    eval_split_id: Split = Split.PUBLISHED,
    freeze: datetime,
    base_model_release_date: datetime,
    decode: DecodeConfig,
    allow_dirty: bool = False,
    now: datetime | None = None,
) -> RunManifest:
    """Assemble a manifest, reading git and the leakage version from source.

    Those two are read rather than passed so a caller cannot record a version
    that was not the one loaded — the conflation `docs/11` rule 3 warns about.

    `decode` is keyword-only and has no default, so a caller cannot omit it and
    inherit whatever the serving stack happened to do. That omission is what let
    0.2288 and 0.2374 coexist for a day (the internal decisions log, 2026-08-24e).
    """
    return RunManifest(
        model_hash=model_hash,
        data_snapshot_hash=data_snapshot_hash,
        base_model_release_date=base_model_release_date,
        decode=decode,
        eval_split_id=eval_split_id,
        git_sha=git_sha(allow_dirty=allow_dirty),
        leakage_detector_version=LEAKAGE_DETECTOR_VERSION,
        freeze=freeze,
        run_date=now or datetime.now(UTC),
    )


@dataclass(frozen=True)
class CertifiedCard:
    """Card metrics with provenance attached. The only publishable artifact."""

    card: CardMetrics
    manifest: RunManifest

    def summary(self) -> str:
        provenance = (
            f"model={self.manifest.model_hash[:12]} "
            f"data={self.manifest.data_snapshot_hash[:12]} "
            f"git={self.manifest.git_sha[:12]} "
            f"leakage={self.manifest.leakage_detector_version}"
        )
        return f"{self.card.summary()}\n{provenance}"


def certify(card: CardMetrics, manifest: RunManifest) -> CertifiedCard:
    """Bind a manifest to card metrics, refusing any disagreement.

    The two carry `split` and `freeze` independently — the card from the slice
    it scored, the manifest from what the caller recorded. Checking them against
    each other catches a manifest built for a different run and pasted onto this
    one, which is the realistic way provenance goes wrong: not fabricated, reused.
    """
    if manifest.eval_split_id is not card.split:
        raise IncompleteManifestError(
            f"manifest records split {manifest.eval_split_id.value!r} but the card "
            f"was scored on {card.split.value!r}. One of them belongs to a "
            "different run."
        )
    if manifest.freeze != card.freeze:
        raise IncompleteManifestError(
            f"manifest freeze {manifest.freeze.isoformat()} disagrees with the "
            f"card's {card.freeze.isoformat()}. The split boundary is not what "
            "the manifest claims."
        )
    # The claim `Split.PUBLISHED` makes, enforced. A base model released after
    # the freeze may have trained on the very questions the published split is
    # built from, and nothing else in this pipeline would notice: the leakage
    # detector screens documents, and this failure involves no document.
    # No split condition: `model_card_metrics` refuses anything but `published`,
    # so every `CardMetrics` reaching here is publishable by construction. A
    # `card.split is PUBLISHED` guard would be an unreachable branch, which this
    # project treats as dead code rather than as defence.
    if manifest.base_model_release_date > manifest.freeze:
        raise IncompleteManifestError(
            f"base model was released {manifest.base_model_release_date.date().isoformat()}, "
            f"after the freeze {manifest.freeze.date().isoformat()}. `Split.PUBLISHED` "
            "claims contamination-freedom on the grounds that the freeze precedes "
            "every base model's pretraining; that does not hold here, so this "
            "number is not publishable."
        )
    return CertifiedCard(card=card, manifest=manifest)
