# SPDX-License-Identifier: Apache-2.0
"""Tarot — a conditional probability estimator.

Phase 1 surface: types, the leakage detector, and metrics. Per `CLAUDE.md`
build order, no data loaders, model code, or training code exist yet.
"""

from tarot.types import (
    EvidenceDoc,
    ForecastRequest,
    ForecastResponse,
    PricePoint,
    QuestionRecord,
    TimestampProvenance,
)

__all__ = [
    "EvidenceDoc",
    "ForecastRequest",
    "ForecastResponse",
    "PricePoint",
    "QuestionRecord",
    "TimestampProvenance",
    "__version__",
]

__version__ = "0.1.0"
