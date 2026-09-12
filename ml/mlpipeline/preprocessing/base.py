"""Preprocessing op protocol (Section 7.1).

An op is a pure, deterministic function of (image, params). Randomness is banned
here by design — augmentation (train-only) lives in ml/mlpipeline/training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class OpContext:
    """Per-run context threaded through the op chain.

    ``extras`` lets ops communicate downstream state (e.g. resize records the
    scale factors consumed by post-processing). Warnings accumulate without
    failing the run.
    """

    source_image_id: str | None = None
    stage_index: int = 0
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PreprocessOp(Protocol):
    """Every preprocessing op implements this minimal surface."""

    name: str  # stable registry key, e.g. "clahe"

    def apply(self, img: np.ndarray, params: dict[str, Any], ctx: OpContext) -> np.ndarray:
        """Apply the op. Must not mutate ``img`` in place. Must be deterministic."""
        ...
